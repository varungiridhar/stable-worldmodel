"""GCS-Align -- Goal-Conditioned State-Alignment of a world model's plan cost.

A *task-general*, *scale-free* world-model quality metric. Where ESNR / the
frozen v3 metric (cost-normalized planning-gradient variance) score the GEOMETRY
of the planning-cost surface (and were fooled cross-task by the frozen DINO-WM
encoder), GCS-Align directly asks the only question that matters for planning:
*do the actions the model THINKS are good actually reach the TRUE goal?*

For a trained WM on a task, with ``B`` start states (the offset-25 eval protocol)
and ``N`` action sequences sampled from the CEM prior ``N(0, var_scale)``:

  * ``c_model[n,b]`` -- the model's PLANNING COST for action-sequence ``n`` from
    start ``b``: roll the actions through the MODEL's latent dynamics from the
    start latent and evaluate ``model.get_cost`` against the goal latent. This is
    EXACTLY the scalar the CEM planner minimizes (we reuse
    :func:`stable_worldmodel.metrics.collect_planning_grads`, which samples the
    same prior and computes the same cost; we ask it to also hand back the exact
    sampled actions so the two paths below share one action set).
  * ``d_true[n,b]`` -- the TRUE terminal distance-to-goal: roll the SAME
    action-sequence ``n`` in the REAL env from start ``b`` (un-normalising the
    actions and applying the same frameskip the planner uses), read the terminal
    privileged TRUE state, and take a task-appropriate distance to the goal's
    true state.
  * ``GCS[b] = spearman_n(c_model[:,b], d_true[:,b])`` -- per-start rank
    agreement between the model's cost surface and the true outcome.
  * ``GCS = nanmean_b GCS[b]`` in ``[-1, 1]``; HIGHER (-> +1) is BETTER (the
    actions the model rates cheap really do reach the true goal).

Mirrors ``grad_dump.py`` for model/env/dataset setup and start-state sampling, so
the model-cost path is taken on the SAME start/goal states as the MPC eval.

``+gcs.proposal`` selects where the N actions are drawn from. Default ``prior``
is vanilla GCS above (CEM prior ``N(0, var_scale)``). ``cem_centered`` is
*GCS-at-the-optimum*: it first runs the eval CEM planner to its converged plan,
then draws the N actions from a same-width proposal RE-CENTERED on that optimum
(``N(cem_mean, var_scale)``), so GCS scores the cost<->true-distance agreement in
the neighborhood the optimizer actually moves into. This reproduces
``probe_cost_geometry.py``'s ``gcs_optimized`` (and reuses the SAME
``collect_planning_grads(proposal='cem_centered', solver=...)`` path), and the
result is still written under the ``['gcs']`` key (with a ``proposal`` field).

    python scripts/plan/gcs_align.py --config-name tworoom \
      policy=p1a_prejepa_s0/weights_epoch_20.pt \
      eval.dataset_name=tworoom_expert.lance eval.num_eval=16 \
      eval.goal_offset_steps=25 eval.eval_budget=50 \
      solver.num_samples=100 solver.n_steps=10 solver.topk=30 \
      +gcs.num_samples=96 +gcs.sample_batch=16 \
      +gcs.out=/.../gcs/p1a_prejepa_s0__weights_epoch_20.json seed=42
"""

import json
import os
import re
from functools import partial
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm
import stable_worldmodel.metrics  # noqa: F401  (register swm.metrics)
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal


# --------------------------------------------------------------------------- #
# setup helpers (verbatim from grad_dump.py so the start sampling is identical) #
# --------------------------------------------------------------------------- #
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


def _clear_wm_caches(model):
    """Clear PreJEPA's stateful init/goal embedding caches between solves.

    Byte-identical to ``swm.metrics.esnr._clear_wm_caches`` (and the C-probe's
    ``_clear_caches``); used by the ``cem_centered`` proposal so the inner
    ``solver.solve`` re-derives the eval planner's optimum from clean caches.
    """
    for attr in ('_init_cached_info', '_goal_cached_info'):
        if hasattr(model, attr):
            delattr(model, attr)


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


# --------------------------------------------------------------------------- #
# per-task TRUE state: how to read the terminal privileged state from the real #
# env, which goal_state key holds the goal's true state, and the distance.     #
# Each entry MUST mirror that task's success criterion (privileged columns).   #
# --------------------------------------------------------------------------- #
# Push-T weighting: SE(2) on the block. Success thresholds are pos < 20 px AND
# |angle| < pi/9 rad, so we weight the (wrapped) angle error by
# (pos_threshold / angle_threshold) = 20 / (pi/9) ~= 57.3 px/rad. With that, an
# angle error AT its success threshold contributes the same as a position error
# AT its threshold -- the two SE(2) channels are commensurate.
_PUSHT_ANGLE_W = 20.0 / (np.pi / 9.0)


def _read_true_tworoom(env_u):
    # TwoRoom true state used for success = agent (x, y). env._get_info()['state']
    # == agent_position; we read the attribute directly.
    return np.asarray(env_u.agent_position, dtype=np.float64).reshape(-1)[:2]


def _read_true_reacher(env_u):
    # DMControl reacher: privileged joint angles (nq,). Success = per-joint
    # |qpos - target_qpos| < 0.05 rad (ReacherQPosMatchTask.get_termination).
    return np.copy(env_u.env.physics.data.qpos).astype(np.float64)


def _read_true_pusht(env_u):
    # Push-T state vector: [agent_xy(2), block_xy(2), block_angle, agent_vel(2)].
    return np.asarray(env_u._get_obs(), dtype=np.float64)


def _dist_tworoom(term, goal):
    # agent (x, y) L2; goal = expert agent xy at offset (env's success target).
    return float(
        np.linalg.norm(term[:2] - np.asarray(goal, dtype=np.float64)[:2])
    )


def _dist_reacher(term, goal):
    # ||qpos - goal_qpos|| (raw, no wrap -- matches get_termination's raw diff).
    goal = np.asarray(goal, dtype=np.float64)
    n = min(term.shape[0], goal.shape[0])
    return float(np.linalg.norm(term[:n] - goal[:n]))


def _dist_pusht(term, goal):
    # Weighted SE(2) on the BLOCK: ||block_xy_err|| + W * |angle_err (wrapped)|.
    goal = np.asarray(goal, dtype=np.float64)
    pos_err = np.linalg.norm(term[2:4] - goal[2:4])
    ang = abs(float(term[4]) - float(goal[4]))
    ang = min(ang, 2.0 * np.pi - ang)  # wrap to [0, pi]
    return float(pos_err + _PUSHT_ANGLE_W * ang)


# --- Phase-4a: generic DMControl reach-target (qpos_match) reader / distance.
#     Any DMControl domain wired with the generic qpos_match task
#     (custom_tasks/qpos_match.py) is GCS-eligible with no bespoke reader: its
#     true terminal state is `physics.data.qpos` and the goal is `goal_qpos`
#     (auto-derived from the dataset 'qpos' column at start+offset).
def _read_true_qpos(env_u):
    # Privileged joint state qpos (nq,). Identical to _read_true_reacher; shared
    # across every qpos_match domain.
    return np.copy(env_u.env.physics.data.qpos).astype(np.float64)


def _dist_qpos(term, goal, mask=None, scale=None):
    # ||qpos - goal_qpos|| over the matched DOFs (raw, no wrap -- matches
    # QPosMatchMixin.get_termination's raw per-joint diff). `mask` (bool over nq)
    # drops DOFs no controller can place to the goal (free root slide / free
    # spinner) which would otherwise flatten d_true; `scale` (per-DOF) makes
    # mixed-unit qpos commensurate (e.g. cart metres vs pole radians). Defaults
    # reproduce the plain reacher L2.
    term = np.asarray(term, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    n = min(term.shape[0], goal.shape[0])
    d = term[:n] - goal[:n]
    if mask is not None:
        d = d[np.asarray(mask, dtype=bool)[:n]]
    if scale is not None:
        d = d / np.asarray(scale, dtype=np.float64)[: len(d)]
    return float(np.linalg.norm(d))


# --- OGBench-Cube: TRUE state = block-0 position (3-D). The env success reads
#     obj_pos = _data.joint('object_joint_0').qpos[:3] and checks
#     ||obj_pos - target|| <= 0.04 m (cube_env._compute_successes); the dataset
#     logs exactly that as privileged/block_0_pos. We read the SAME quantity and
#     take the goal from goal_privileged/block_0_pos (block_0_pos at start+offset)
#     so d_true mirrors the success metric.
def _read_true_cube(env_u):
    return np.asarray(
        env_u._data.joint('object_joint_0').qpos[:3], dtype=np.float64
    ).reshape(-1)[:3]


def _dist_cube(term, goal):
    term = np.asarray(term, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    return float(np.linalg.norm(term[:3] - goal[:3]))


# env_name -> (read_terminal_true(env_u), goal_state_key, dist(term, goal_true))
_TASKS = {
    'swm/TwoRoom-v1': (_read_true_tworoom, 'goal_state', _dist_tworoom),
    'swm/ReacherDMControl-v0': (
        _read_true_reacher,
        'goal_qpos',
        _dist_reacher,
    ),
    'swm/PushT-v1': (_read_true_pusht, 'goal_state', _dist_pusht),
    # --- Phase-4a DMControl reach-target tasks (generic qpos-match) ---
    # NB: each mask MUST equal the env wrapper's qpos_mask (so env success and
    # the metric's d_true score the SAME DOFs); free/unbounded DOFs are dropped.
    'swm/CartpoleDMControl-v0': (_read_true_qpos, 'goal_qpos', _dist_qpos),
    'swm/FingerDMControl-v0': (
        _read_true_qpos,
        'goal_qpos',
        partial(_dist_qpos, mask=[True, True, False]),  # drop free spinner
    ),
    'swm/CheetahDMControl-v0': (
        _read_true_qpos,
        'goal_qpos',
        partial(  # 6 bounded leg hinges only; drop free root qpos[0:3]
            _dist_qpos,
            mask=[False, False, False, True, True, True, True, True, True],
        ),
    ),
    'swm/PendulumDMControl-v0': (_read_true_qpos, 'goal_qpos', _dist_qpos),
    # --- OGBench-Cube manipulation (block-0 position; success = L2 <= 0.04 m) ---
    'swm/OGBCube-v0': (
        _read_true_cube,
        'goal_privileged/block_0_pos',
        _dist_cube,
    ),
}


def _spearman(x, y):
    """Spearman rank corr over N; nan if either side is constant (no ranking)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or np.std(x[ok]) == 0.0 or np.std(y[ok]) == 0.0:
        return np.nan
    rho, _ = spearmanr(x[ok], y[ok])
    return float(rho)


@hydra.main(version_base=None, config_path='./config', config_name='tworoom')
def run(cfg: DictConfig):
    assert cfg.get('policy', 'random') != 'random', 'gcs_align needs a ckpt'
    out_path = OmegaConf.select(cfg, 'gcs.out')
    assert out_path, 'pass +gcs.out=/path/to.json'
    env_name = cfg.world.env_name
    assert env_name in _TASKS, (
        f'no TRUE-distance defined for env {env_name}; known: {list(_TASKS)}'
    )
    read_true, goal_key, dist_fn = _TASKS[env_name]

    N = int(OmegaConf.select(cfg, 'gcs.num_samples', default=96))
    var_scale = float(OmegaConf.select(cfg, 'gcs.var_scale', default=1.0))
    sample_batch = int(OmegaConf.select(cfg, 'gcs.sample_batch', default=16))
    obs_batch = int(OmegaConf.select(cfg, 'gcs.obs_batch', default=1))
    # 'prior' (default) == vanilla GCS: draw the N actions from the CEM prior
    # N(0, var_scale). 'cem_centered' == GCS-at-the-optimum: re-center the
    # same-width proposal on the converged CEM optimum N(cem_mean, var_scale),
    # mirroring probe_cost_geometry.py's GCS_optimized (the C-probe).
    proposal = str(OmegaConf.select(cfg, 'gcs.proposal', default='prior'))
    assert proposal in ('prior', 'cem_centered'), (
        f"gcs.proposal must be 'prior' or 'cem_centered', got {proposal!r}"
    )
    # Stage-2 (ADDITIVE): when true, ALSO record the task's TRUE state at EVERY
    # rollout step (via the same read_true used for d_true) and the goal vector,
    # caching them to the sidecar npz so distances can be redefined OFFLINE to
    # mirror the eval success criterion -- without re-running the sim. This does
    # NOT touch c_model, d_true, or the gcs value (the d_true line is untouched);
    # it only appends two arrays to the npz. Default False -> legacy behavior.
    cache_traj = bool(OmegaConf.select(cfg, 'gcs.cache_traj', default=False))

    # --- world + transforms + dataset (mirror grad_dump.py exactly) ---
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {
        'pixels': img_transform(cfg, torch.float32),
        'goal': img_transform(cfg, torch.float32),
    }
    _ds_extra = {}
    _ktl = cfg.dataset.get('keys_to_load', None)
    if _ktl is not None:
        _ds_extra['keys_to_load'] = list(_ktl)
    _aliases = cfg.dataset.get('column_aliases', None)
    if _aliases is not None:
        _ds_extra['column_aliases'] = OmegaConf.to_container(
            _aliases, resolve=True
        )
    dataset = swm.data.load_dataset(
        cfg.eval.dataset_name,
        cache_dir=cfg.get('cache_dir', None),
        keys_to_cache=list(cfg.dataset.keys_to_cache),
        **_ds_extra,
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
    assert 'action' in process, (
        'action StandardScaler required to un-normalise'
    )
    proc_action = process['action']

    # --- model in fp32 (no bf16): the cost path must be full precision ---
    model = swm.wm.utils.load_world_model(cfg.policy).to('cuda').eval()
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
    # OPTIONAL cube-moving start filter (default null = legacy behavior); MUST
    # mirror eval_wm.py exactly so GCS scores the same start states as the MPC
    # eval (same valid set, seed, threshold => identical sampled starts).
    move_thresh = OmegaConf.select(cfg, 'eval.start_move_thresh', default=None)
    if move_thresh is not None:
        pos_col = OmegaConf.select(
            cfg, 'eval.start_move_col', default='privileged/block_0_pos'
        )
        valid = swm.data.utils.filter_moving_starts(
            dataset,
            valid,
            cfg.eval.goal_offset_steps,
            float(move_thresh),
            pos_col,
            col_name,
        )
        print(
            f'[gcs] {len(valid)} start points after move filter '
            f'(thresh={move_thresh} on {pos_col}).'
        )
    g = np.random.default_rng(cfg.seed)
    idx = np.sort(
        valid[g.choice(len(valid) - 1, size=cfg.eval.num_eval, replace=False)]
    )
    eval_episodes = dataset.get_col_data(col_name)[idx]
    eval_start = dataset.get_col_data('step_idx')[idx]

    world.set_policy(policy)

    # --- capture the preprocessed info_dict at the first solver call ---
    # This both gives us the model-cost INFO and leaves world.envs at their B
    # start states (in the SAME b-order as init_state/goal_state below).
    captured = {}

    def _cap(info_dict, init_action=None):
        captured['info'] = {
            k: (v.detach().clone() if torch.is_tensor(v) else v)
            for k, v in info_dict.items()
        }
        raise _CaptureDone()

    orig_solve = policy.solver.solve
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
    finally:
        # Restore the REAL solve: the cem_centered proposal calls solver.solve
        # inside collect_planning_grads to find the CEM optimum to re-center on.
        policy.solver.solve = orig_solve
    assert 'info' in captured, 'Solver never called; could not capture info.'
    info = captured['info']

    horizon = solver.horizon
    action_dim = solver.action_dim  # = single_action_dim * action_block
    action_block = int(cfg.plan_config.action_block)
    single_dim = action_dim // action_block
    n_env_steps = horizon * action_block
    goal_offset = int(cfg.eval.goal_offset_steps)
    B = int(cfg.eval.num_eval)
    if n_env_steps != goal_offset:
        print(
            f'[gcs] WARNING: horizon*action_block={n_env_steps} != '
            f'goal_offset_steps={goal_offset}; the model rollout length does '
            f'not match the goal horizon. GCS assumes they match.'
        )
    print(
        f'[gcs] B={B} N={N} horizon={horizon} action_dim={action_dim} '
        f'(single={single_dim} x block={action_block}) '
        f'n_env_steps={n_env_steps} goal_offset={goal_offset} task={env_name}'
    )

    # --- (1) model-cost path: c_model[n,b] + the EXACT sampled actions -------- #
    # For 'cem_centered' (GCS-at-the-optimum), re-seed the CEM generator and
    # clear the WM caches so the solver.solve run INSIDE collect_planning_grads
    # (which finds the optimum to re-center the proposal on) reproduces the eval
    # planner's optimum -- byte-identical to probe_cost_geometry.py's
    # GCS_optimized preamble. The N-sample draw then uses its own seeded gen.
    if proposal == 'cem_centered':
        solver.torch_gen.manual_seed(int(cfg.seed))
        _clear_wm_caches(model)
    _grads, costs, aux = swm.metrics.collect_planning_grads(
        model,
        info,
        horizon=horizon,
        action_dim=action_dim,
        num_samples=N,
        var_scale=var_scale,
        seed=cfg.seed,
        device='cuda',
        obs_batch=obs_batch,
        sample_batch=sample_batch,
        proposal=proposal,
        solver=solver if proposal == 'cem_centered' else None,
        capture_cost_sens=False,
        return_actions=True,
    )
    c_model = costs.numpy().astype(np.float64)  # (N, B)
    actions = aux['actions'].numpy()  # (N, B, H, A), NORMALISED action space
    assert actions.shape == (N, B, horizon, action_dim), (
        f'unexpected action shape {actions.shape}'
    )

    # --- (2) true-rollout path: d_true[n,b] in the REAL env ------------------- #
    init_state, goal_state, _ = _extract_init_goal(
        dataset,
        eval_episodes.tolist(),
        eval_start.tolist(),
        goal_offset,
    )
    assert goal_key in goal_state, (
        f"goal_state key '{goal_key}' not found; have {list(goal_state)}. "
        f'(It is derived from the per-step true-state column at offset '
        f'{goal_offset}.)'
    )
    goal_true = goal_state[goal_key]  # (B, ...) privileged goal true state
    callables = OmegaConf.to_container(cfg.eval.get('callables'), resolve=True)
    merged = {**init_state, **goal_state}

    d_true = np.full((N, B), np.nan, dtype=np.float64)
    # Stage-2 traj cache (additive; only populated when cache_traj). traj_states
    # is lazily allocated once the per-step true-state dim is known (read_true);
    # goal_vecs holds each start's goal vector (same layout read_true returns).
    traj_states = None  # -> (N, B, n_env_steps, state_dim)
    goal_vecs = None    # -> (B, goal_dim)
    for b in range(B):
        env_u = world.envs.envs[b].unwrapped
        merged_b = {k: v[b] for k, v in merged.items()}
        goal_b = goal_true[b]
        if cache_traj:
            gv = np.asarray(goal_b, dtype=np.float64).reshape(-1)
            if goal_vecs is None:
                goal_vecs = np.full((B, gv.shape[0]), np.nan, dtype=np.float64)
            goal_vecs[b, : gv.shape[0]] = gv
        # actions for this start: (N, H, A) normalised -> per-sample env rollout
        for n in range(N):
            # restore to the SAME physical start as the model-cost path: a fresh
            # reset (fixed seed => deterministic non-callable variations, which
            # are init_value-fixed for these envs) + the eval callables that set
            # the privileged start (and goal) from the dataset. Identical to how
            # _evaluate_from_dataset initialises each episode.
            env_u.reset(seed=int(cfg.seed) * 100003 + b)
            _apply_callables(env_u, callables, merged_b)
            seq_norm = actions[n, b].reshape(n_env_steps, single_dim)
            seq_raw = proc_action.inverse_transform(seq_norm)  # un-normalise
            for t in range(n_env_steps):
                env_u.step(
                    np.asarray(seq_raw[t])
                )  # ignore terminated: full roll
                if cache_traj:
                    # record the SAME true-state read_true uses for d_true, at
                    # every step (anytime/min-over-t distances become free).
                    s_t = np.asarray(
                        read_true(env_u), dtype=np.float64
                    ).reshape(-1)
                    if traj_states is None:
                        traj_states = np.full(
                            (N, B, n_env_steps, s_t.shape[0]),
                            np.nan,
                            dtype=np.float64,
                        )
                    traj_states[n, b, t, : s_t.shape[0]] = s_t
            # UNCHANGED: terminal d_true via the task's faithful dist_fn.
            d_true[n, b] = dist_fn(read_true(env_u), goal_b)

    # --- (3) GCS = nanmean_b spearman_n(c_model[:,b], d_true[:,b]) ------------- #
    gcs_per_b = np.array(
        [_spearman(c_model[:, b], d_true[:, b]) for b in range(B)],
        dtype=np.float64,
    )
    gcs = float(np.nanmean(gcs_per_b))
    n_valid_b = int(np.isfinite(gcs_per_b).sum())

    model_dir, ckpt_kind, ckpt_num = _parse_ckpt(cfg.policy)
    print(
        f'[gcs] GCS={gcs:+.4f}  (proposal={proposal}; nanmean over '
        f'{n_valid_b}/{B} starts; higher=better)  run={model_dir} '
        f'ckpt={Path(cfg.policy).name}'
    )
    print(f'[gcs] per-start GCS[b]: {np.array2string(gcs_per_b, precision=3)}')

    payload = {
        'gcs': gcs,
        'proposal': proposal,
        'gcs_per_b': [None if np.isnan(v) else float(v) for v in gcs_per_b],
        'n_valid_b': n_valid_b,
        'N': N,
        'B': B,
        'task': env_name,
        'run': model_dir,
        'ckpt': Path(cfg.policy).name,
        'ckpt_kind': ckpt_kind,
        'ckpt_num': ckpt_num,
        'dataset': cfg.eval.dataset_name,
        'goal_offset_steps': goal_offset,
        'horizon': int(horizon),
        'action_block': action_block,
        'single_action_dim': single_dim,
        'n_env_steps': n_env_steps,
        'var_scale': var_scale,
        'seed': int(cfg.seed),
        'pusht_angle_weight': float(_PUSHT_ANGLE_W),
        'cache_traj': bool(cache_traj),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    # sidecar npz with the raw (N,B) surfaces for offline re-analysis / plots.
    npz_extra = {}
    if cache_traj and traj_states is not None:
        # (N, B, T, state_dim) per-step true states + (B, goal_dim) goal vectors.
        npz_extra['traj_states'] = traj_states.astype(np.float32)
        npz_extra['goal_state'] = goal_vecs.astype(np.float32)
        print(
            f'[gcs] cache_traj: traj_states{traj_states.shape} '
            f'goal_state{goal_vecs.shape} -> npz'
        )
    np.savez_compressed(
        out_path.with_suffix('.npz'),
        c_model=c_model.astype(np.float32),
        d_true=d_true.astype(np.float32),
        gcs_per_b=gcs_per_b.astype(np.float32),
        eval_episodes=np.asarray(eval_episodes),
        eval_start=np.asarray(eval_start),
        **npz_extra,
    )
    print(f'[gcs] wrote {out_path.resolve()}')
    print(f'[gcs] wrote {out_path.with_suffix(".npz").resolve()}')


if __name__ == '__main__':
    run()
