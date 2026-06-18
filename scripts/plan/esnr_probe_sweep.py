"""Batched offline ESNR sweep over all checkpoints of one training run.

Same physics as ``esnr_probe.py`` but amortized: the preprocessed ``info_dict``
captured at the solver boundary is INDEPENDENT of the world model (it's dataset
frames + env renders for fixed start states), so we capture it ONCE and reuse it
for every checkpoint of the run. Each checkpoint then only costs a model load +
the action-trajectory backward.

Writes one CSV per checkpoint to ``esnr.out_dir/<run>__<ckpt>.csv`` and SKIPS any
that already exist, so a preempted job resumes for free on resubmit.

    python scripts/plan/esnr_probe_sweep.py --config-name tworoom \
      esnr.run_name=p1a_prejepa_s0 \
      eval.dataset_name=tworoom_expert.lance eval.num_eval=16 \
      eval.goal_offset_steps=75 eval.eval_budget=150 \
      solver.num_samples=100 solver.n_steps=10 \
      esnr.num_samples=96 esnr.sample_batch=16 \
      esnr.out_dir=/.../results/esnr_rows seed=42 bf16=false
"""

import csv
import glob
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
    raise KeyError('no episode-index column found')


def get_episodes_length(dataset, episodes):
    col = _episode_col(dataset)
    ei = dataset.get_col_data(col)
    si = dataset.get_col_data('step_idx')
    return np.array([np.max(si[ei == e]) + 1 for e in episodes])


class _CaptureDone(BaseException):
    pass


def _parse_ckpt(filename):
    stem = Path(filename).stem
    m_e = re.search(r'epoch_(\d+)', stem)
    m_s = re.search(r'step_(\d+)', stem)
    if m_e:
        return 'epoch', int(m_e.group(1))
    if m_s:
        return 'step', int(m_s.group(1))
    return 'unknown', -1


@hydra.main(version_base=None, config_path='./config', config_name='tworoom')
def run(cfg: DictConfig):
    run_name = cfg.esnr.run_name
    ckpt_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_name
    )
    ckpts = sorted(glob.glob(str(ckpt_dir / 'weights_*.pt')))
    assert ckpts, f'no weights_*.pt in {ckpt_dir}'
    out_dir = Path(cfg.esnr.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[sweep] {run_name}: {len(ckpts)} checkpoints')

    # --- world / transform / process (model-independent) ---
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {'pixels': img_transform(cfg), 'goal': img_transform(cfg)}
    dataset = swm.data.load_dataset(
        cfg.eval.dataset_name,
        cache_dir=cfg.get('cache_dir', None),
        keys_to_cache=list(cfg.dataset.keys_to_cache),
    )
    col = _episode_col(dataset)
    ep_indices, _ = np.unique(dataset.get_col_data(col), return_index=True)
    process = {}
    for c in cfg.dataset.keys_to_cache:
        if c == 'pixels':
            continue
        p = preprocessing.StandardScaler()
        cd = dataset.get_col_data(c)
        p.fit(cd[~np.isnan(cd).any(axis=1)])
        process[c] = p
        if c != 'action':
            process[f'goal_{c}'] = p

    # --- start-state sampling identical to eval (seed -> same B states) ---
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start = episode_len - cfg.eval.goal_offset_steps - 1
    msd = {e: max_start[i] for i, e in enumerate(ep_indices)}
    per_row = np.array([msd[e] for e in dataset.get_col_data(col)])
    valid = np.nonzero(dataset.get_col_data('step_idx') <= per_row)[0]
    g = np.random.default_rng(cfg.seed)
    idx = np.sort(
        valid[g.choice(len(valid) - 1, size=cfg.eval.num_eval, replace=False)]
    )
    eval_episodes = dataset.get_col_data(col)[idx]
    eval_start = dataset.get_col_data('step_idx')[idx]

    def build_policy(policy_path):
        model = swm.wm.utils.load_pretrained(policy_path).to('cuda').eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        config = swm.PlanConfig(**cfg.plan_config)
        pol = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )
        return model, solver, pol

    # --- capture info_dict ONCE (using the first checkpoint's policy) ---
    _, solver0, policy0 = build_policy(f'{run_name}/{Path(ckpts[0]).name}')
    captured = {}

    def _cap(info_dict, init_action=None):
        captured['info'] = {
            k: (v.detach().clone() if torch.is_tensor(v) else v)
            for k, v in info_dict.items()
        }
        raise _CaptureDone()

    policy0.solver.solve = _cap
    world.set_policy(policy0)  # configures the solver (sets _config)
    horizon, action_dim = solver0.horizon, solver0.action_dim
    try:
        world.evaluate(
            dataset=dataset,
            start_steps=eval_start.tolist(),
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
    assert 'info' in captured, 'failed to capture info_dict'
    info = captured['info']
    action_space = solver0._action_space  # stashed by solver.configure()
    del policy0, solver0
    torch.cuda.empty_cache()
    print(
        f'[sweep] captured info: B={cfg.eval.num_eval} horizon={horizon} action_dim={action_dim}'
    )

    proposal = cfg.esnr.get('proposal', 'prior')
    plan_cfg = swm.PlanConfig(**cfg.plan_config)

    # --- loop over checkpoints (skip done) ---
    for cp in ckpts:
        name = Path(cp).name
        out_csv = out_dir / f'{run_name}__{Path(name).stem}.csv'
        if out_csv.exists():
            print(f'[sweep] skip (done): {name}')
            continue
        kind, num = _parse_ckpt(name)
        model = (
            swm.wm.utils.load_pretrained(f'{run_name}/{name}')
            .to('cuda')
            .eval()
        )
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        # CEM-based proposals need a configured solver bound to this model
        solver = None
        if proposal in ('cem_optimized', 'cem_centered'):
            solver = hydra.utils.instantiate(cfg.solver, model=model)
            solver.configure(
                action_space=action_space,
                n_envs=cfg.eval.num_eval,
                config=plan_cfg,
            )
        res = swm.metrics.run_planning_esnr(
            model,
            info,
            horizon=horizon,
            action_dim=action_dim,
            num_samples=cfg.esnr.num_samples,
            var_scale=cfg.esnr.var_scale,
            seed=cfg.seed,
            device='cuda',
            obs_batch=cfg.esnr.get('obs_batch', 1),
            sample_batch=cfg.esnr.get('sample_batch', 16),
            proposal=proposal,
            solver=solver,
        )
        row = {
            'run_name': run_name,
            'ckpt': name,
            'ckpt_kind': kind,
            'ckpt_num': num,
            'esnr': res['esnr'],
            'esnr_log10': res['esnr_log10'],
            'esnr_num': res['esnr_num'],
            'esnr_den': res['esnr_den'],
            'degenerate_frac': res['degenerate_frac'],
            'n_components': res['n_components'],
            'num_obs': res['num_obs'],
            'num_samples': res['num_samples'],
            'sample_batch': res.get('sample_batch', ''),
            'var_scale': res['var_scale'],
            'proposal': res['proposal'],
            'cem_std_mean': res.get('cem_std_mean', ''),
            'goal_offset': cfg.eval.goal_offset_steps,
            'seed': cfg.seed,
        }
        with out_csv.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        del model
        torch.cuda.empty_cache()
        print(
            f'[sweep] {name}: esnr={res["esnr"]:.4g} log10={res["esnr_log10"]:.3f} '
            f'degen={res["degenerate_frac"]:.2f}'
        )

    print(f'[sweep] {run_name} done -> {out_dir}')


if __name__ == '__main__':
    run()
