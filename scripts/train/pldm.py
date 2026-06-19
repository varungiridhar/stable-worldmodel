import os
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from functools import partial

from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import PLDMLoss, TemporalStraighteningLoss
from lightning.pytorch.callbacks import Callback
from stable_worldmodel.wm.utils import save_pretrained


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class SaveCkptCallback(Callback):
    """Save model checkpoints via ``save_pretrained``.

    Saves every ``epoch_interval`` epochs (and always the final epoch). When
    ``intra_epoch_steps`` is non-empty, also saves the first time the global
    optimizer step crosses each listed threshold, but only while
    ``current_epoch < intra_epoch_max_epoch`` -- i.e. dense early-training
    checkpoints used to resolve the ESNR U-shape (whose minimum lands very
    early in training). Intra-epoch files are named ``weights_step_<N>.pt``;
    per-epoch files ``weights_epoch_<N>.pt``.
    """

    def __init__(
        self,
        run_name,
        cfg,
        epoch_interval=1,
        intra_epoch_steps=None,
        intra_epoch_max_epoch=0,
        step_interval=0,
    ):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval
        self.intra_epoch_steps = sorted(intra_epoch_steps or [])
        self.intra_epoch_max_epoch = intra_epoch_max_epoch
        self.step_interval = step_interval
        self._saved_targets = set()
        # Next global step at which the uniform-cadence save fires. Realigned
        # above the resumed step on requeue (see on_train_batch_end).
        self._next_step_target = step_interval if step_interval else None

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        if not trainer.is_global_zero:
            return
        step = trainer.global_step
        # Dense early-training saves: one-shot global-step thresholds, gated to
        # early epochs so the ESNR U-shape minimum is resolvable.
        if trainer.current_epoch < self.intra_epoch_max_epoch:
            for target in self.intra_epoch_steps:
                if step >= target and target not in self._saved_targets:
                    self._saved_targets.add(target)
                    self._save(pl_module.model, f'weights_step_{target}.pt')
        # Uniform cadence across ALL epochs (long-run U-shape coverage).
        # Resume-safe: a requeued run gets a fresh callback, so we realign the
        # target above the resumed step rather than re-dumping every past
        # multiple. Files are named by the actual global step, so post-resume
        # saves never collide with pre-resume ones.
        if (
            self._next_step_target is not None
            and step >= self._next_step_target
        ):
            self._save(pl_module.model, f'weights_step_{step}.pt')
            self._next_step_target = step + self.step_interval

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_interval == 0 or epoch == trainer.max_epochs:
            self._save(pl_module.model, f'weights_epoch_{epoch}.pt')

    def _save(self, model, filename):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=filename,
        )


def pldm_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""
    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    output = self.model.encode(batch)

    emb = output['emb']  # (B, T, D)
    act_emb = output['act_emb']

    inpt_emb = emb[:, : cfg.wm.history_size]  # (B, T-1, D)
    inpt_act = act_emb[:, : cfg.wm.history_size]
    tgt_emb = emb[:, cfg.wm.num_preds :]  # (B, T-1, patches, dim)
    pred_emb = self.model.predict(inpt_emb, inpt_act)

    output['idm_emb'] = torch.cat([emb[:, 1:], emb[:, :-1]], dim=-1)
    output['act_label'] = batch['action'][:, :-1].detach()
    output['act_pred'] = self.idm(output['idm_emb'])
    output['pred_loss'] = (pred_emb - tgt_emb).square().mean()
    output['temp_straight_loss'] = self.path_straight(emb)
    output.update(self.pldm(emb, output['act_pred'], output['act_label']))

    output['loss'] = output['pred_loss']
    for k, v in cfg.loss.items():
        loss_key = f'{k}_loss'
        if not v.enabled or (loss_key not in output):
            continue
        output['loss'] = output['loss'] + v.weight * output[loss_key]

    # log all losses
    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)

    return output


@hydra.main(version_base=None, config_path='./config', config_name='pldm')
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    print(
        f'Loading dataset "{dataset_name}" from {"local cache: " + cache_dir if cache_dir else "default location"}'
    )
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    img_processor = get_img_preprocessor('pixels', 'pixels', cfg.img_size)

    extra_transforms = []
    for col in cfg.data.dataset.keys_to_load:
        if col in ['pixels']:
            continue
        normalizer = get_column_normalizer(dataset, col, col)
        extra_transforms.append(normalizer)

    if hasattr(cfg.data.dataset, 'keys_to_merge'):
        for col in cfg.data.dataset.keys_to_merge:
            normalizer = get_column_normalizer(dataset, col, col)
            extra_transforms.append(normalizer)

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col in ['pixels']:
                continue
            setattr(cfg.wm, f'{col}_dim', dataset.get_dim(col))

        effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim
        cfg.model.action_encoder.input_dim = effective_act_dim
        cfg.idm.input_dim = 2 * cfg.wm.embed_dim
        cfg.idm.output_dim = effective_act_dim

    transform = spt.data.transforms.Compose(img_processor, *extra_transforms)

    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    train = DataLoader(train_set, **cfg.loader, generator=rnd_gen)
    val_cfg = {**cfg.loader}
    val_cfg['shuffle'] = False
    val_cfg['drop_last'] = False
    val = DataLoader(val_set, **val_cfg)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)
    idm = hydra.utils.instantiate(cfg.idm)

    models = {
        'model': world_model,
        'idm': idm,
    }

    losses = {
        'pldm': PLDMLoss(),
        'path_straight': TemporalStraighteningLoss(),
    }

    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {}
    for model_name in models.keys():
        optimizers[f'{model_name}_opt'] = {
            'modules': str(model_name),
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        **models,
        **losses,
        forward=partial(pldm_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )
    logging.info(f'🫆🫆🫆 Run ID: {run_id} 🫆🫆🫆')

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=cfg.get('ckpt', {}).get('epoch_interval', 5),
        intra_epoch_steps=cfg.get('ckpt', {}).get('intra_epoch_steps', None),
        intra_epoch_max_epoch=cfg.get('ckpt', {}).get(
            'intra_epoch_max_epoch', 0
        ),
        step_interval=cfg.get('ckpt', {}).get('step_interval', 0),
    )

    # NOTE: resumable checkpointing for embers preemption is handled by spt's
    # Manager (it installs its own 'last' requeue ModelCheckpoint, redirects it
    # to the cache run dir, and resumes via a SLURM-job-id index on requeue).
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    # spt 0.1.7's Manager.init_and_sync_wandb() assumes a *previous* completed
    # run dir to reload config from on requeue/resume. For a fresh OFFLINE run
    # that file doesn't exist, so it raises and aborts training. The sync is a
    # non-essential resume convenience, so wrap it to no-op on failure while
    # keeping wandb offline. (Guard for spt<=0.1.7; harmless if upstream fixes.)
    _orig_sync = spt.Manager.init_and_sync_wandb

    def _safe_init_and_sync_wandb(self):
        try:
            _orig_sync(self)
        except Exception as e:  # noqa: BLE001
            logging.warning(
                f'Skipping spt wandb resume-sync (fresh offline run): {e!r}'
            )

    spt.Manager.init_and_sync_wandb = _safe_init_and_sync_wandb

    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == '__main__':
    run()
