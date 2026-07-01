"""Random-policy data collection for ANY DMControl domain.

Generalizes ``collect_reacher.py`` so Phase-4a's reach-target (``qpos_match``)
tasks get data with no SAC expert (none are on disk; ``collect_dmc.py`` needs
them). A random policy visits diverse reachable states, which is exactly what the
offset-goal protocol needs: the GCS / eval goal is the true qpos
``goal_offset_steps`` ahead in a collected trajectory, so it is reachable by
construction. ``World.collect`` writes every ``info`` key as a column, so the
privileged ``qpos`` / ``qvel`` the metric reads are saved automatically — the
dataset is GCS-ready with no extra wiring.

Reuses the ``reacher`` collect config (world settings / num_traj / seed); pass
``env_name=`` to pick the domain::

    python scripts/data/collect_dmc_random.py env_name=swm/CartpoleDMControl-v0
    # -> $STABLEWM_HOME/datasets/dmc/cartpole_random.h5
"""

import os
from pathlib import Path

from omegaconf import OmegaConf

os.environ.setdefault('MUJOCO_GL', 'egl')  # headless-friendly (GPU node); overridable

import hydra
import numpy as np
from loguru import logger as logging

import stable_worldmodel as swm
from stable_worldmodel.policy import RandomPolicy

# swm env id -> short dataset name used in dmc/<name>_random.lance
_NAME = {
    'swm/ReacherDMControl-v0': 'reacher',
    'swm/CartpoleDMControl-v0': 'cartpole',
    'swm/CheetahDMControl-v0': 'cheetah',
    'swm/FingerDMControl-v0': 'finger',
    'swm/PendulumDMControl-v0': 'pendulum',
    'swm/BallInCupDMControl-v0': 'ballincup',
    'swm/WalkerDMControl-v0': 'walker',
    'swm/HopperDMControl-v0': 'hopper',
}


@hydra.main(version_base=None, config_path='./config', config_name='reacher')
def run(cfg):
    """Collect random trajectories from a DMControl domain."""

    world = swm.World(cfg.env_name, **cfg.world)

    options = cfg.get('options')
    options = OmegaConf.to_object(options) if options is not None else None

    rng = np.random.default_rng(cfg.seed)
    name = _NAME.get(
        cfg.env_name, cfg.env_name.split('/')[-1].split('-')[0].lower()
    )

    world.set_policy(RandomPolicy(seed=rng.integers(0, 1_000_000).item()))

    out = (
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
        / 'datasets'
        / f'dmc/{name}_random.h5'
    )
    # HDF5 (not Lance): dm_control binds its EGL render context to the MAIN
    # thread, but the LanceWriter consumes the (rendering) episode generator on
    # an async worker thread -> "context already current on another thread"
    # crash. The HDF5 writer consumes eagerly on the main thread, which is why
    # the validated reacher_random.h5 is HDF5. Downstream train/eval/GCS read
    # .h5 unmodified.
    world.collect(
        out,
        episodes=cfg.num_traj,
        format='hdf5',
        seed=rng.integers(0, 1_000_000).item(),
        options=options,
    )

    # The HDF5 writer emits ep_offset/ep_len/step_idx but NOT the per-row
    # episode_idx that GCS-Align (gcs_align._episode_col) and eval need;
    # reconstruct it from the episode boundaries so the dataset matches the
    # validated reacher_random.h5 schema.
    import h5py

    with h5py.File(out, 'r+') as h5:
        if 'episode_idx' not in h5 and {'ep_offset', 'ep_len'} <= set(h5):
            eo = h5['ep_offset'][:]
            el = h5['ep_len'][:]
            n_rows = h5['step_idx'].shape[0]
            epi = np.zeros(n_rows, dtype=np.int64)
            for e, (o, length) in enumerate(zip(eo, el)):
                epi[o : o + length] = e
            h5.create_dataset('episode_idx', data=epi)
            logging.info(f'Added per-row episode_idx ({len(eo)} eps) for GCS/eval.')

    logging.success(f'Completed random data collection for {name} -> {out}')


if __name__ == '__main__':
    run()
