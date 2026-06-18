"""Offline ESNR probe for a single world-model checkpoint.

Computes the planning-ESNR (paper Code 1 / Eq. 10) of ``model.get_cost`` for a
checkpoint, on the SAME TwoRoom start/goal states the MPC evaluation uses. It
reuses ``eval_wm.py``'s dataset/transform/process/policy construction and the
identical start-state sampling, then -- instead of running the planner --
captures the preprocessed ``info_dict`` at the solver boundary and backprops the
cost objective to the action trajectory.

Run (low-mem GPU; ESNR is computed in fp32, never bf16):

    python scripts/plan/esnr_probe.py --config-name tworoom \
      policy=p1a_prejepa_s42/weights_epoch_20.pt \
      eval.dataset_name=tworoom_expert.lance \
      eval.num_eval=50 eval.goal_offset_steps=75 eval.eval_budget=150 \
      solver.num_samples=100 solver.n_steps=10 \
      esnr.num_samples=300 esnr.var_scale=1.0 esnr.obs_batch=1 \
      esnr.out_csv=/path/to/esnr_rows/prejepa_s42_e20.csv

Note: ``solver.*`` here only configures planning shape (horizon/action_block);
the probe does not run the CEM iterations -- it captures the info_dict at the
first solver call and computes ESNR with ``esnr.num_samples`` action samples.
"""

import csv
import os
import re
from pathlib import Path

os.environ['MUJOCO_GL'] = 'egl'

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm
import stable_worldmodel.metrics  # noqa: F401  (register swm.metrics)


# ---- helpers copied from eval_wm.py so the probe sees identical inputs -------


def img_transform(cfg, dtype=torch.float32):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(dtype, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def _episode_col(dataset):
    for c in ('episode_idx', 'ep_idx'):
        try:
            dataset.get_col_data(c)
            return c
        except Exception:  # noqa: BLE001
            continue
    raise KeyError('no episode-index column (episode_idx/ep_idx) found')


def get_episodes_length(dataset, episodes):
    col_name = _episode_col(dataset)
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data('step_idx')
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    return swm.data.load_dataset(
        dataset_name,
        cache_dir=cfg.get('cache_dir', None),
        keys_to_cache=list(cfg.dataset.keys_to_cache),
    )


class _CaptureDone(BaseException):
    """Raised once the solver's info_dict has been captured, to stop eval."""


def _parse_ckpt(policy: str):
    """('p1a_prejepa_s42', 'epoch'|'step', N) from '<dir>/weights_<kind>_<N>.pt'."""
    model_dir = policy.split('/')[0]
    stem = Path(policy).stem
    m_e = re.search(r'epoch_(\d+)', stem)
    m_s = re.search(r'step_(\d+)', stem)
    if m_e:
        return model_dir, 'epoch', int(m_e.group(1))
    if m_s:
        return model_dir, 'step', int(m_s.group(1))
    return model_dir, 'unknown', -1


@hydra.main(version_base=None, config_path='./config', config_name='tworoom')
def run(cfg: DictConfig):
    assert cfg.get('policy', 'random') != 'random', (
        'ESNR needs a real checkpoint'
    )

    # --- world + transforms + dataset (mirror eval_wm.py) ---
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    transform = {
        'pixels': img_transform(cfg, torch.float32),
        'goal': img_transform(cfg, torch.float32),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = _episode_col(dataset)
    ep_indices, _ = np.unique(
        dataset.get_col_data(col_name), return_index=True
    )

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ['pixels']:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != 'action':
            process[f'goal_{col}'] = process[col]

    # --- model in FP32 (no bf16): a variance ratio needs full precision ---
    model = swm.wm.utils.load_pretrained(cfg.policy)
    model = model.to('cuda').eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform
    )

    # --- start-state sampling identical to eval_wm.py (same seed => same states) ---
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_indices = np.nonzero(
        dataset.get_col_data('step_idx') <= max_start_per_row
    )[0]

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])
    eval_episodes = dataset.get_col_data(col_name)[random_episode_indices]
    eval_start_idx = dataset.get_col_data('step_idx')[random_episode_indices]

    world.set_policy(policy)

    # --- capture the preprocessed info_dict at the first solver call ---
    captured = {}

    def _capturing_solve(info_dict, init_action=None):
        captured['info'] = {
            k: (v.detach().clone() if torch.is_tensor(v) else v)
            for k, v in info_dict.items()
        }
        raise _CaptureDone()

    policy.solver.solve = _capturing_solve

    try:
        world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(
                cfg.eval.get('callables'), resolve=True
            ),
            video=None,
        )
    except _CaptureDone:
        pass

    if 'info' not in captured:
        raise RuntimeError('Solver was never called; could not capture info.')

    info = captured['info']
    print(
        f'[esnr] captured info_dict with B={cfg.eval.num_eval} obs; '
        f'horizon={solver.horizon} action_dim={solver.action_dim}'
    )

    # --- compute ESNR (Code 1) by backprop through get_cost ---
    result = swm.metrics.run_planning_esnr(
        model,
        info,
        horizon=solver.horizon,
        action_dim=solver.action_dim,
        num_samples=cfg.esnr.num_samples,
        var_scale=cfg.esnr.var_scale,
        seed=cfg.seed,
        device='cuda',
        obs_batch=cfg.esnr.get('obs_batch', 1),
        sample_batch=cfg.esnr.get('sample_batch', 16),
        proposal=cfg.esnr.get('proposal', 'prior'),
    )
    print(f'[esnr] {result}')

    # --- write one tidy CSV row (per-job file; merged later) ---
    model_dir, ckpt_kind, ckpt_num = _parse_ckpt(cfg.policy)
    row = {
        'policy': cfg.policy,
        'model_dir': model_dir,
        'ckpt_kind': ckpt_kind,
        'ckpt_num': ckpt_num,
        'train_seed': cfg.esnr.get('train_seed', ''),
        'global_step': cfg.esnr.get('global_step', ''),
        'esnr': result['esnr'],
        'esnr_log10': result['esnr_log10'],
        'esnr_num': result['esnr_num'],
        'esnr_den': result['esnr_den'],
        'degenerate_frac': result['degenerate_frac'],
        'n_components': result['n_components'],
        'num_obs': result['num_obs'],
        'num_samples': result['num_samples'],
        'sample_batch': result.get('sample_batch', ''),
        'var_scale': result['var_scale'],
        'proposal': result['proposal'],
        'goal_offset': cfg.eval.goal_offset_steps,
        'eval_dataset': cfg.eval.dataset_name,
        'sample_seed': cfg.seed,
    }
    out_csv = Path(cfg.esnr.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f'[esnr] wrote {out_csv.resolve()}')


if __name__ == '__main__':
    run()
