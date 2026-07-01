import os
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from functools import partial
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.utils import save_pretrained
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoVideoProcessor


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def get_img_preprocessor(source, target, img_size=224):
    stats = spt.data.dataset_stats.ImageNet
    return spt.data.transforms.Compose(
        spt.data.transforms.ToImage(**stats, source=source, target=target),
        spt.data.transforms.Resize(img_size, source=source, target=target),
    )


class VideoPipeline(spt.data.transforms.Transform):
    def __init__(self, processor, source='image', target='image'):
        super().__init__()
        self.processor, self.source, self.target = processor, source, target

    def __call__(self, x):
        frames = self.nested_get(x, self.source)
        self.nested_set(
            x,
            self.processor(frames, return_tensors='pt')[
                'pixel_values_videos'
            ].squeeze(0),
            self.target,
        )
        return x


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


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
        # Uniform cadence across ALL epochs (bounds loss to <= step_interval
        # steps on an embers/inferno kill). Resume-safe: a requeued run gets a
        # fresh callback, so we realign the target above the resumed step rather
        # than re-dumping every past multiple. Files are named by the actual
        # global step, so post-resume saves never collide with pre-resume ones.
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


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def _strip_action_dims(tensor, action_range):
    """Remove the action dimensions from the last axis."""
    return torch.cat(
        [tensor[..., : action_range[0]], tensor[..., action_range[1] :]],
        dim=-1,
    )


def dinowm_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, compute losses."""
    for key in self.model.extra_encoders:
        batch[key] = torch.nan_to_num(batch[key], 0.0).squeeze()

    batch = self.model.encode(
        batch,
        target='emb',
        is_video=cfg.backbone.get('is_video_encoder', False),
    )

    embedding = batch['emb'][:, : cfg.wm.history_size, ...]
    pred_embedding = self.model.predict(embedding)
    target_embedding = batch['emb'][:, cfg.wm.num_preds :, ...].detach()

    # Per-modality losses
    pixels_dim = batch['pixels_emb'].size(-1)
    batch['pixels_loss'] = F.mse_loss(
        pred_embedding[..., :pixels_dim], target_embedding[..., :pixels_dim]
    )

    start, action_range = pixels_dim, [0, 0]
    for key in self.model.extra_encoders:
        dim = batch[f'{key}_emb'].size(-1)
        lo, hi = start, start + dim
        if key == 'action':
            action_range = [lo, hi]
        else:
            batch[f'{key}_loss'] = F.mse_loss(
                pred_embedding[..., lo:hi],
                target_embedding[..., lo:hi].detach(),
            )
        start = hi

    # Actionless embeddings (for probes and total loss)
    batch['actionless_emb'] = _strip_action_dims(batch['emb'], action_range)
    batch['actionless_prev_emb'] = _strip_action_dims(embedding, action_range)
    batch['actionless_pred_emb'] = _strip_action_dims(
        pred_embedding, action_range
    )
    batch['actionless_target_emb'] = _strip_action_dims(
        target_embedding, action_range
    )

    batch['loss'] = F.mse_loss(
        batch['actionless_pred_emb'],
        batch['actionless_target_emb'].detach(),
    )

    if batch['loss'].isnan():
        raise ValueError('NaN loss encountered!')

    self.log_dict(
        {f'{stage}/{k}': v.detach() for k, v in batch.items() if '_loss' in k},
        on_step=True,
        sync_dist=True,
    )
    return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path='./config', config_name='prejepa')
def run(cfg):
    # Seed init + dropout + dataloader together for reproducible training.
    # (prejepa/spt only seeded the dataloader generator, leaving the predictor's
    # weight init and dropout masks drawing from an unseeded global RNG.)
    pl.seed_everything(cfg.seed, workers=True)

    # --- Dataset ---
    encoding_keys = list(cfg.wm.get('encoding', {}).keys())
    keys_to_load = ['pixels'] + encoding_keys

    # Optional column aliasing (default-absent -> unchanged PushT/TwoRoom
    # behavior). Cube needs it: `pixels` is an alias for the on-disk
    # front-camera column (`pixels_front_pixels`). Encoding keys for Cube are
    # just {action} (pixels + action), so no proprio merge is needed here.
    extra_load = {}
    aliases = cfg.get('column_aliases', None)
    if aliases is not None:
        extra_load['column_aliases'] = OmegaConf.to_container(
            aliases, resolve=True
        )

    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    print(
        f'Loading dataset "{cfg.dataset_name}" from {"local cache: " + cache_dir if cache_dir else "default location"}'
    )
    dataset = swm.data.load_dataset(
        cfg.dataset_name,
        num_steps=cfg.n_steps,
        frameskip=cfg.frameskip,
        transform=None,
        cache_dir=cache_dir,
        keys_to_load=keys_to_load,
        keys_to_cache=encoding_keys,
        **extra_load,
    )

    normalizers = [
        get_column_normalizer(dataset, col, col)
        for col in cfg.wm.get('encoding', {})
    ]

    if cfg.backbone.get('is_video_encoder', False):
        processor = AutoVideoProcessor.from_pretrained(cfg.backbone.name)
        transform = spt.data.transforms.Compose(
            VideoPipeline(processor, source='pixels', target='pixels'),
            spt.data.transforms.Resize(
                cfg.image_size, source='pixels', target='pixels'
            ),
            *normalizers,
        )
    else:
        transform = spt.data.transforms.Compose(
            get_img_preprocessor('pixels', 'pixels', cfg.image_size),
            *normalizers,
        )
    dataset.transform = transform

    with open_dict(cfg) as cfg:
        cfg.extra_dims = {}
        for key in cfg.wm.get('encoding', {}):
            if key not in dataset.column_names:
                raise ValueError(
                    f"Encoding key '{key}' not found in dataset columns."
                )
            dim = dataset.get_dim(key)
            cfg.extra_dims[key] = (
                dim if key != 'action' else dim * cfg.frameskip
            )

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, [cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        persistent_workers=True,
        pin_memory=True,
        shuffle=True,
        generator=rnd_gen,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # --- Model ---
    encoder = hydra.utils.instantiate(cfg.model.encoder)
    encoder.eval()
    encoder.requires_grad_(False)

    is_cnn = hasattr(encoder.config, 'hidden_sizes')
    embed_dim = (
        encoder.config.hidden_sizes[-1]
        if is_cnn
        else encoder.config.hidden_size
    )
    num_patches = 1 if is_cnn else (cfg.image_size // cfg.patch_size) ** 2
    embed_dim += sum(cfg.wm.get('encoding', {}).values())

    if cfg.backbone.get('is_video_encoder', False):
        num_patches += num_patches * (cfg.n_steps // 4)

    with open_dict(cfg):
        cfg.model.predictor.dim = embed_dim
        cfg.model.predictor.num_patches = num_patches
        cfg.model.extra_encoders = {
            '_target_': 'torch.nn.ModuleDict',
            'modules': {
                key: {
                    '_target_': 'stable_worldmodel.wm.prejepa.module.Embedder',
                    'in_chans': cfg.extra_dims[key],
                    'emb_dim': int(cfg.wm.encoding[key]),
                }
                for key in cfg.wm.get('encoding', {})
            },
        }

    world_model = hydra.utils.instantiate(cfg.model, encoder=encoder)

    world_model = spt.Module(
        model=world_model,
        forward=partial(dinowm_forward, cfg=cfg),
        optim={
            'model_opt': {'modules': 'model', 'optimizer': dict(cfg.optimizer)}
        },
    )

    # --- Training ---
    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f'Run ID: {run_id}')

    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    # CPUOffloadCallback was removed in stable_pretraining>=0.1.7. It is only a
    # VRAM-saving optimization and is unnecessary for a frozen dinov2_small +
    # small predictor on a 24 GB GPU, so include it only if the installed spt
    # still provides it (keeps this script portable across spt versions).
    extra_cbs = (
        [spt.callbacks.CPUOffloadCallback()]
        if hasattr(spt.callbacks, 'CPUOffloadCallback')
        else []
    )
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            *extra_cbs,
            SaveCkptCallback(
                run_name=cfg.output_model_name,
                cfg=cfg.model,
                epoch_interval=cfg.get('ckpt', {}).get('epoch_interval', 5),
                intra_epoch_steps=cfg.get('ckpt', {}).get(
                    'intra_epoch_steps', None
                ),
                intra_epoch_max_epoch=cfg.get('ckpt', {}).get(
                    'intra_epoch_max_epoch', 0
                ),
                step_interval=cfg.get('ckpt', {}).get('step_interval', 0),
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval='step'),
        ],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    # spt 0.1.7's Manager.init_and_sync_wandb() assumes a *previous* completed
    # run dir (files/wandb-config.json) to reload config from on requeue/resume.
    # For a fresh OFFLINE run that file doesn't exist, so it raises
    # FileNotFoundError/TypeError and aborts training. The sync is a non-essential
    # resume convenience, so wrap it to no-op on failure while keeping wandb
    # offline. (Guard for spt<=0.1.7; harmless if upstream fixes it.)
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
        data=spt.data.DataModule(train=train_loader, val=val_loader),
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )
    manager()


if __name__ == '__main__':
    run()
