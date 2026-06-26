from functools import partial
from pathlib import Path
import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader
import numpy as np

from stable_worldmodel.wm.tdmpc2 import (
    TDMPC2,
    tdmpc2_forward,
)


class ModelObjectCallBack(Callback):
    """Periodically saves the entire model object (not just state dict) to disk.

    Saves ``*_epoch_<N>_object.ckpt`` every ``epoch_interval`` epochs (and the
    final epoch). Mirrors the JEPA ``SaveCkptCallback`` for DENSE checkpoints so
    the v3 probe protocol works on TD-MPC2:

      * ``step_interval`` > 0  -> uniform-cadence ``*_step_<global_step>_object.ckpt``
        across all epochs (long-run coverage).
      * ``intra_epoch_steps`` -> one-shot global-step thresholds saved while
        ``current_epoch < intra_epoch_max_epoch`` (dense early-training ckpts).

    Filenames keep the ``epoch_<N>`` / ``step_<N>`` tokens so the probe's
    ``_parse_ckpt`` regex picks them up.
    """

    def __init__(
        self,
        dirpath,
        filename='model_object',
        epoch_interval=1,
        step_interval=0,
        intra_epoch_steps=None,
        intra_epoch_max_epoch=0,
    ):
        super().__init__()
        self.dirpath, self.filename, self.epoch_interval = (
            Path(dirpath),
            filename,
            epoch_interval,
        )
        self.step_interval = step_interval
        self.intra_epoch_steps = sorted(intra_epoch_steps or [])
        self.intra_epoch_max_epoch = intra_epoch_max_epoch
        self._saved_targets = set()
        self._next_step_target = step_interval if step_interval else None

    def _save(self, model, tag):
        path = self.dirpath / f'{self.filename}_{tag}_object.ckpt'
        torch.save(model, path)
        logging.info(f'Saved world model to {path}')

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        if not trainer.is_global_zero:
            return
        step = trainer.global_step
        # Dense early-training one-shot thresholds (gated to early epochs).
        if trainer.current_epoch < self.intra_epoch_max_epoch:
            for target in self.intra_epoch_steps:
                if step >= target and target not in self._saved_targets:
                    self._saved_targets.add(target)
                    self._save(pl_module.model, f'step_{target}')
        # Uniform cadence across all epochs; resume-safe (realign above resume).
        if (
            self._next_step_target is not None
            and step >= self._next_step_target
        ):
            self._save(pl_module.model, f'step_{step}')
            self._next_step_target = step + self.step_interval

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_interval == 0 or epoch == trainer.max_epochs:
            self._save(pl_module.model, f'epoch_{epoch}')


class _ZScore:
    """Picklable z-score callable (mean/std fixed at construction).

    A module-level class (not a closure/lambda) so it survives pickling to
    DataLoader workers when ``num_workers > 0`` / ``persistent_workers=True``.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean.to(x.device)) / self.std.to(x.device)).float()


def get_column_normalizer(dataset, source, target):
    """Z-score normalization transform computed from the full dataset column."""
    data = torch.from_numpy(dataset.get_col_data(source)[:])
    data = data[~torch.isnan(data).any(dim=1)]
    mean, std = (
        data.mean(0, keepdim=True).clone(),
        data.std(0, keepdim=True).clone(),
    )
    mean, std = mean.squeeze(), std.squeeze() + 1e-2

    return spt.data.transforms.WrapTorchTransform(
        _ZScore(mean, std), source=source, target=target
    )


def get_img_preprocessor(source, target, img_size=64):
    """ImageNet-normalized + resized image preprocessing pipeline."""
    stats = spt.data.dataset_stats.ImageNet
    return spt.data.transforms.Compose(
        spt.data.transforms.ToImage(**stats, source=source, target=target),
        spt.data.transforms.Resize(img_size, source=source, target=target),
    )


@hydra.main(version_base=None, config_path='./config', config_name='tdmpc2')
def run(cfg):
    """
    Main training entry point for the TD-MPC2 model.

    Uses dataset rewards directly.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """
    torch.set_float32_matmul_precision('high')

    encoding_keys = list(cfg.wm.get('encoding', {}).keys())
    if not encoding_keys:
        raise ValueError('No encoding modalities defined in cfg.wm.encoding!')

    use_pixels = 'pixels' in encoding_keys
    goal_obs_key = cfg.get(
        'goal_obs_key'
    )  # if set, concatenate episode goal into this key
    # Cube path: the per-step GOAL state lives in its own column (e.g. `target`,
    # the 28-d goal observation), NOT in the episode-final obs. When `goal_col`
    # is set we augment `goal_obs_key` (observation) with that column per step
    # instead of with the episode-final obs (TwoRoom behavior, goal_col unset).
    goal_col = cfg.get('goal_col')
    extra_keys = [k for k in encoding_keys if k != 'pixels']

    keys_to_load = list(encoding_keys) + ['action', 'reward']
    if goal_col is not None and goal_col not in keys_to_load:
        keys_to_load.append(goal_col)

    # Optional column aliasing (default-absent -> unchanged TwoRoom/PushT
    # behavior). Cube vision needs it: the logical `pixels` key is an alias for
    # the on-disk front-camera column (`pixels_front_pixels`) — the SAME view the
    # JEPA Cube zoo and the live env render — so the goal-image-MSE cost matches.
    extra_load = {}
    aliases = cfg.get('column_aliases', None)
    if aliases is not None:
        extra_load['column_aliases'] = OmegaConf.to_container(
            aliases, resolve=True
        )

    # Format-agnostic load (auto-detects .lance vs .h5 from the resolved path).
    # The JEPA zoo uses tworoom_expert.lance; load_dataset resolves the dataset
    # path under <cache_dir>/datasets/ and dispatches to the matching reader.
    # (LanceDataset ignores keys_to_cache — it has efficient random access — and
    # does not accept a `cache_dir` reader kwarg, so cache_dir is only passed to
    # load_dataset for path resolution.)
    base_dataset = swm.data.load_dataset(
        cfg.dataset_name,
        cache_dir=cfg.get('cache_dir'),
        num_steps=cfg.wm.horizon + 1,
        keys_to_load=keys_to_load,
        keys_to_cache=keys_to_load if cfg.get('cache_dataset', True) else [],
        **extra_load,
    )

    if goal_obs_key is not None:
        if goal_obs_key not in encoding_keys:
            raise ValueError(
                f'cfg.goal_obs_key="{goal_obs_key}" must be one of the encoding keys {encoding_keys}.'
            )
        _raw_obs = base_dataset.get_col_data(goal_obs_key)[:]

        if goal_col is not None:
            # Cube path: the goal is the per-step `goal_col` column directly.
            # Augment observation s -> [s, g] where g = target[t] (same row).
            goals_by_step = base_dataset.get_col_data(goal_col)[:]
            if goals_by_step.shape != _raw_obs.shape:
                raise ValueError(
                    f'goal_col="{goal_col}" shape {goals_by_step.shape} must '
                    f'match goal_obs_key="{goal_obs_key}" shape '
                    f'{_raw_obs.shape} for per-step goal augmentation.'
                )
            _src = f'per-step "{goal_col}" column'
        else:
            # TwoRoom path: append each episode's FINAL obs of goal_obs_key.
            # Episode structure: prefer the reader's computed offsets/lengths
            # (works for both Lance and HDF5); fall back to ep_offset/ep_len
            # columns (older HDF5 layout) only if attributes are unavailable.
            if hasattr(base_dataset, 'offsets') and hasattr(
                base_dataset, 'lengths'
            ):
                _ep_off = (
                    np.asarray(base_dataset.offsets).flatten().astype(int)
                )
                _ep_len = (
                    np.asarray(base_dataset.lengths).flatten().astype(int)
                )
            else:
                _ep_off = (
                    base_dataset.get_col_data('ep_offset')[:]
                    .flatten()
                    .astype(int)
                )
                _ep_len = (
                    base_dataset.get_col_data('ep_len')[:]
                    .flatten()
                    .astype(int)
                )
            _goal_idx = np.clip(_ep_off + _ep_len - 1, 0, len(_raw_obs) - 1)
            goals_by_step = np.empty_like(_raw_obs)
            for _ep, (_off, _len) in enumerate(
                zip(_ep_off.tolist(), _ep_len.tolist())
            ):
                goals_by_step[_off : _off + _len] = _raw_obs[_goal_idx[_ep]]
            _src = 'episode-final obs'

        base_dataset._cache[goal_obs_key] = np.concatenate(
            [_raw_obs, goals_by_step], axis=-1
        )
        logging.info(
            f'Goal augmentation: appended {_src} to "{goal_obs_key}" '
            f'(dim {_raw_obs.shape[-1]} → {base_dataset._cache[goal_obs_key].shape[-1]})'
        )

    raw_actions = base_dataset.get_col_data('action')[:]
    valid_actions = raw_actions[~np.isnan(raw_actions).any(axis=1)]
    act_max = valid_actions.max()
    act_min = valid_actions.min()

    if act_max > 1.01 or act_min < -1.01:
        logging.error(
            f'Dataset actions fall outside the [-1, 1] range! (Min: {act_min:.2f}, Max: {act_max:.2f}).\n'
            'TD-MPC2 uses a Tanh actor and strictly requires actions to be bounded between [-1, 1].\n'
            'Please normalize your dataset actions.'
        )
        raise ValueError(
            'Unnormalized actions detected in the dataset. Training aborted.'
        )

    with open_dict(cfg):
        cfg.action_dim = base_dataset.get_dim('action')
        cfg.extra_dims = {'action': cfg.action_dim}

        for key in extra_keys:
            if goal_obs_key is not None and key == goal_obs_key:
                cfg.extra_dims[key] = base_dataset._cache[key].shape[-1]
            else:
                cfg.extra_dims[key] = base_dataset.get_dim(key)

    transforms = []
    if use_pixels:
        transforms.append(
            get_img_preprocessor('pixels', 'pixels', cfg.image_size)
        )

    # Stashed for model.set_state_norm so eval/probe normalize the goal-MSE
    # augmented state identically to training (the plan config does not z-score
    # 'state'/'goal_state', so the model must carry these stats itself).
    aug_state_mean = aug_state_std = None

    for key in extra_keys:
        if goal_obs_key is not None and key == goal_obs_key:
            aug_data = torch.from_numpy(base_dataset._cache[key]).float()
            aug_clean = aug_data[~torch.isnan(aug_data).any(dim=1)]
            _mean = aug_clean.mean(0).clone()
            _std = aug_clean.std(0).clone() + 1e-2
            aug_state_mean, aug_state_std = _mean, _std
            transforms.append(
                spt.data.transforms.WrapTorchTransform(
                    _ZScore(_mean, _std),
                    source=key,
                    target=key,
                )
            )
        else:
            transforms.append(get_column_normalizer(base_dataset, key, key))

    base_dataset.transform = spt.data.transforms.Compose(*transforms)

    train_set, val_set = spt.data.random_split(
        base_dataset, [cfg.train_split, 1 - cfg.train_split]
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    model = TDMPC2(cfg)

    # Persist the augmented-state z-score stats so the goal-MSE planning/probe
    # cost can normalize the raw eval-time state/goal_state identically.
    if aug_state_mean is not None:
        model.set_state_norm(aug_state_mean, aug_state_std)
        logging.info(
            f'Stored augmented-state z-score stats on model '
            f'(dim {aug_state_mean.numel()}).'
        )

    # Total optimizer steps for the LR schedule (one step per train batch).
    total_steps = int(cfg.trainer.max_epochs) * len(train_loader)

    def add_opt(module_regex, lr, eps=1e-8):
        opt_cfg = dict(cfg.optimizer)
        opt_cfg['lr'] = lr
        opt_cfg['eps'] = eps
        # spt.Module defaults a missing scheduler to CosineAnnealingLR (which
        # needs T_max); provide an explicit warmup+cosine schedule (mirrors the
        # JEPA train scripts) so the optimizer group is fully specified.
        return {
            'modules': module_regex,
            'optimizer': opt_cfg,
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        }

    module = spt.Module(
        model=model,
        forward=partial(tdmpc2_forward, cfg=cfg),
        hparams=OmegaConf.to_container(cfg, resolve=True),
        optim={
            'enc_opt': add_opt(
                r'model\.(cnn|pixel_encoder|extra_encoders|sim_norm).*',
                cfg.optimizer.lr * cfg.get('enc_lr_scale', 0.3),
            ),
            'wm_opt': add_opt(
                r'model\.(dynamics|reward|qs).*',
                cfg.optimizer.lr,
            ),
            'pi_opt': add_opt(
                r'model\.pi.*', cfg.optimizer.lr * 0.1, eps=1e-5
            ),
        },
    )
    subdir = cfg.subdir
    # Save under <STABLEWM_HOME>/checkpoints/<subdir> so the probe/eval loaders
    # (grad_dump / AutoCostModel) — which resolve cfg.policy relative to the
    # 'checkpoints' subfolder — can find the *_object.ckpt files, matching the
    # JEPA zoo layout.
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), subdir
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.wandb.enable:
        logger = WandbLogger(
            name=f'{cfg.wm.name}_{cfg.dataset_name}_{subdir}',
            project=cfg.wandb.project,
            resume='allow' if subdir else None,
            id=subdir or None,
            log_model=False,
        )
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    trainer = pl.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=[
            ModelObjectCallBack(
                dirpath=run_dir,
                filename=cfg.output_model_name,
                epoch_interval=cfg.get('epoch_interval', 1),
                step_interval=cfg.get('step_interval', 0),
                intra_epoch_steps=cfg.get('intra_epoch_steps', None),
                intra_epoch_max_epoch=cfg.get('intra_epoch_max_epoch', 0),
            )
        ],
    )
    spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
    )()


if __name__ == '__main__':
    run()
