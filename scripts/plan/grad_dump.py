"""Dump the full per-sample action-gradient stack (N,B,D) for one checkpoint.

Like ``esnr_probe.py`` but instead of reducing to the scalar ESNR it persists
the WHOLE gradient stack (+ per-sample planning costs + the latent
cost-sensitivity residual) to a compressed ``.npz``. This feeds the Phase-2
gradient debugging -- *which transform of the (N,B,D) stack ranks PLDM>LeWM
scale-invariantly?* -- and offline EPGQ scoring, with no GPU re-probing.

Mirrors ``esnr_probe.py`` exactly through the solver-boundary ``info_dict``
capture (so the gradients are taken on the SAME start/goal states the MPC eval
uses), then calls ``collect_planning_grads`` instead of ``run_planning_esnr``.

    python scripts/plan/grad_dump.py --config-name tworoom \
      policy=p1a_pldm_s0/weights_epoch_30.pt \
      eval.dataset_name=tworoom_expert.lance eval.num_eval=16 \
      eval.goal_offset_steps=75 eval.eval_budget=150 \
      solver.num_samples=100 solver.n_steps=10 \
      esnr.num_samples=96 esnr.sample_batch=16 esnr.proposal=prior \
      +grad.out_npz=/.../grads/p1a_pldm_s0__weights_epoch_30__prior.npz seed=42
"""

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
from stable_worldmodel.metrics.esnr import compute_esnr_from_grads


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
    return np.array([np.max(step_idx[episode_idx == e]) + 1 for e in episodes])


class _CaptureDone(BaseException):
    pass


def _parse_ckpt(policy: str):
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
    assert cfg.get('policy', 'random') != 'random', 'grad_dump needs a ckpt'
    out_npz = OmegaConf.select(cfg, 'grad.out_npz')
    assert out_npz, 'pass +grad.out_npz=/path/to.npz'
    proposal = cfg.esnr.get('proposal', 'prior')

    # --- world + transforms + dataset (mirror eval_wm.py / esnr_probe.py) ---
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {
        'pixels': img_transform(cfg, torch.float32),
        'goal': img_transform(cfg, torch.float32),
    }
    dataset = swm.data.load_dataset(
        cfg.eval.dataset_name,
        cache_dir=cfg.get('cache_dir', None),
        keys_to_cache=list(cfg.dataset.keys_to_cache),
    )
    col_name = _episode_col(dataset)
    ep_indices, _ = np.unique(
        dataset.get_col_data(col_name), return_index=True
    )
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

    # --- model in fp32 (no bf16): a variance ratio needs full precision ---
    model = swm.wm.utils.load_pretrained(cfg.policy).to('cuda').eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform
    )

    # --- start-state sampling identical to eval (same seed => same states) ---
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start = episode_len - cfg.eval.goal_offset_steps - 1
    msd = {e: max_start[i] for i, e in enumerate(ep_indices)}
    per_row = np.array([msd[e] for e in dataset.get_col_data(col_name)])
    valid = np.nonzero(dataset.get_col_data('step_idx') <= per_row)[0]
    g = np.random.default_rng(cfg.seed)
    idx = np.sort(
        valid[g.choice(len(valid) - 1, size=cfg.eval.num_eval, replace=False)]
    )
    eval_episodes = dataset.get_col_data(col_name)[idx]
    eval_start = dataset.get_col_data('step_idx')[idx]

    world.set_policy(policy)  # configures the solver

    # --- capture the preprocessed info_dict at the first solver call ---
    captured = {}
    orig_solve = policy.solver.solve  # save to restore for CEM proposals

    def _cap(info_dict, init_action=None):
        captured['info'] = {
            k: (v.detach().clone() if torch.is_tensor(v) else v)
            for k, v in info_dict.items()
        }
        raise _CaptureDone()

    policy.solver.solve = _cap
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
    assert 'info' in captured, 'Solver never called; could not capture info.'
    policy.solver.solve = orig_solve  # restore for CEM-based proposals
    info = captured['info']
    print(
        f'[grad_dump] captured info: B={cfg.eval.num_eval} '
        f'horizon={solver.horizon} action_dim={solver.action_dim} '
        f'proposal={proposal}'
    )

    # --- collect the full per-sample gradient stack ---
    grads, costs, aux = swm.metrics.collect_planning_grads(
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
        proposal=proposal,
        solver=(solver if proposal != 'prior' else None),
        capture_cost_sens=True,
    )
    esnr_sanity = compute_esnr_from_grads(grads)  # cross-check vs sweep CSVs
    print(
        f'[grad_dump] grads={tuple(grads.shape)} esnr={esnr_sanity["esnr"]:.4g} '
        f'log10={esnr_sanity["esnr_log10"]:.3f} '
        f'degen={esnr_sanity["degenerate_frac"]:.2f} '
        f'cost_sens={"yes" if aux["cost_sens"] is not None else "no"}'
    )

    # --- persist .npz (grads/costs fp32 to halve disk; metadata as 0-d arrays) ---
    model_dir, ckpt_kind, ckpt_num = _parse_ckpt(cfg.policy)
    cost_sens = aux['cost_sens']
    payload = dict(
        grads=grads.numpy().astype(np.float32),  # (N, B, D)
        costs=costs.numpy().astype(np.float32),  # (N, B)
        cost_sens=(
            cost_sens.numpy().astype(np.float32)
            if cost_sens is not None
            else np.zeros(0, dtype=np.float32)
        ),
        run_name=model_dir,
        ckpt=Path(cfg.policy).name,
        ckpt_kind=ckpt_kind,
        ckpt_num=ckpt_num,
        proposal=proposal,
        var_scale=float(cfg.esnr.var_scale),
        seed=int(cfg.seed),
        horizon=int(solver.horizon),
        action_dim=int(solver.action_dim),
        cem_std_mean=str(aux['cem_std_mean']),
        esnr=float(esnr_sanity['esnr']),
        esnr_log10=float(esnr_sanity['esnr_log10']),
        degenerate_frac=float(esnr_sanity['degenerate_frac']),
        goal_offset=int(cfg.eval.goal_offset_steps),
    )
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **payload)
    print(f'[grad_dump] wrote {out_npz.resolve()}')


if __name__ == '__main__':
    run()
