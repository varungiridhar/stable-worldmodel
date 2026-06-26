"""TRAM -- True-state Reachability Alignment of a world model's latent dynamics.

A *task-general*, *scale-free* world-model quality metric that probes the
**DYNAMICS** aspect of a world model -- complementary to GCS-Align, which probes
the **cost** aspect. Where GCS-Align asks *do the actions the model rates cheap
actually reach the true goal?*, TRAM asks the more primitive question that GCS
takes for granted: *does the model's action -> latent map move the way the
world's action -> true-state map moves?* i.e. is the PREDICTOR's local geometry
(how the terminal latent responds to action perturbations) aligned with the
world's true reachability geometry (how the terminal TRUE state responds to the
same perturbations)?

This is the antidote to the v3 failure mode: v3 (planning-gradient variance) was
fooled cross-task by the magnitude/scale of the frozen DINO-WM encoder. TRAM
compares two *geometries* via CCA, which is invariant to any (isotropic) linear
rescaling of either view -- so the encoder's latent scale cancels out.

For a trained WM on a task, with ``B=16`` start states (the offset-25 eval
protocol) and ``K=64`` action perturbations sampled from the CEM prior
``N(0, var_scale)`` around a zero baseline plan ``a0``:

  * ``dz_k = F(a0 + da_k) - F(a0)`` -- the LATENT effect of perturbation ``k``:
    roll the MODEL's latent dynamics from the start latent under ``a0 + da_k``
    and under ``a0``, take the difference of the *terminal* state-latent vectors
    (``predicted_emb[...,-1,:]`` for LeWM/PLDM; the terminal pixel-patch latent
    ``predicted_pixels_emb[...,-1,:,:]`` flattened for PreJEPA). a0 = zeros, so
    ``a0 + da_k`` are exactly the CEM-prior samples and the model rollout reuses
    the SAME ``get_cost`` latent rollout the CEM planner uses
    (``model.get_cost`` -> ``model.rollout``; the cost scalar is discarded -- we
    read the predicted terminal latent it leaves in the info dict).
  * ``dy_k = phi(true_terminal(a0 + da_k)) - phi(true_terminal(a0))`` -- the TRUE
    effect: step the SAME (un-normalised) action sequences in the REAL env from
    the same start (reset(seed) + ``_apply_callables`` to restore the privileged
    start, then ``env.step`` with the planner's frameskip), read the terminal
    privileged TRUE state, and take the task readout ``phi`` (TwoRoom agent xy;
    Reacher qpos; Push-T block (x, y, angle)). NEEDS SIM.
  * ``TRAM_b = rho_1(CCA([dz_k], [dy_k]))`` -- the top canonical correlation
    between the ``(K x D_lat)`` latent-effect matrix and the ``(K x D_true)``
    true-effect matrix. CCA is the scale-free comparison: it is invariant to a
    global rescale of either view (the encoder-scale confound), and ``rho_1`` is
    the cosine of the smallest principal angle between the two response
    subspaces. Because ``D_lat >> K`` (esp. PreJEPA), ``dz`` is first PCA-reduced
    to ``p = min(D_lat_rank, #comps for 99% var, pca_dim, K - D_true - 2)`` dims
    (SVCCA-style); the last cap keeps ``p + D_true`` safely below ``K - 1`` so
    the canonical correlation cannot saturate to 1 spuriously. The SAME ``pca_dim``
    cap is applied to EVERY architecture so the chance floor is comparable across
    PreJEPA / LeWM / PLDM (fair ranking). A distance-Spearman RSA score (rank
    agreement of pairwise ``||dz_i-dz_j||`` vs ``||dy_i-dy_j||``) is computed as a
    documented robustness fallback (stored as ``rsa`` in the output).
  * ``TRAM = nanmean_b TRAM_b`` in ``[0, 1]``; HIGHER (-> 1) is BETTER (the
    latent action-geometry mirrors the world's reachability geometry).

Probe regime: the FULL planning horizon (``horizon * action_block = 25`` env
steps) to match GCS-Align exactly, so TRAM and GCS are on the same start/goal
states, same prior, same horizon -- directly comparable. (A 1-step probe is the
documented alternative; not selected, so dynamics are scored over the same
horizon the planner optimises.)

Mirrors ``gcs_align.py`` (which mirrors ``grad_dump.py``) for model/env/dataset
setup, start-state sampling, the solver-hook info capture, and the real-env
restore+rollout pattern. The reused helpers below are copied verbatim from
``gcs_align.py`` (cited per block) so the start states line up byte-for-byte with
the GCS run on the SAME checkpoint.

    python scripts/plan/tram.py --config-name tworoom \
      policy=p1a_prejepa_s0/weights_epoch_20.pt \
      eval.dataset_name=tworoom_expert.lance eval.num_eval=16 \
      eval.goal_offset_steps=25 eval.eval_budget=50 \
      solver.num_samples=100 solver.n_steps=10 solver.topk=30 \
      +tram.K=64 +tram.sample_batch=16 +tram.pca_dim=16 \
      +tram.out=/.../tram/p1a_prejepa_s0__weights_epoch_20.json seed=42
"""

import json
import os
import re
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm
import stable_worldmodel.metrics  # noqa: F401  (register swm.metrics)
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal


# --------------------------------------------------------------------------- #
# setup helpers -- COPIED VERBATIM from scripts/plan/gcs_align.py (which copied #
# them from grad_dump.py) so the start sampling is byte-for-byte identical.     #
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
# per-task TRUE-state readout -- the read_true(env_u) functions are COPIED      #
# VERBATIM from gcs_align.py; phi is the task readout used for the distance     #
# (the privileged channels that define success), here used as the response      #
# vector whose action-perturbation deltas dy_k we align against the latent.     #
# --------------------------------------------------------------------------- #
def _read_true_tworoom(env_u):
    # TwoRoom true state used for success = agent (x, y).
    return np.asarray(env_u.agent_position, dtype=np.float64).reshape(-1)[:2]


def _read_true_reacher(env_u):
    # DMControl reacher: privileged joint angles (nq,).
    return np.copy(env_u.env.physics.data.qpos).astype(np.float64)


def _read_true_pusht(env_u):
    # Push-T state vector: [agent_xy(2), block_xy(2), block_angle, agent_vel(2)].
    return np.asarray(env_u._get_obs(), dtype=np.float64)


# env_name -> (read_terminal_true(env_u), phi(true_state) -> response vector).
# phi selects the success-defining channels: TwoRoom agent xy[:2]; Reacher full
# qpos; Push-T the BLOCK pose (x, y, angle) = state[2:5].
_TASKS = {
    'swm/TwoRoom-v1': (
        _read_true_tworoom,
        lambda s: np.asarray(s, dtype=np.float64).reshape(-1)[:2],
    ),
    'swm/ReacherDMControl-v0': (
        _read_true_reacher,
        lambda s: np.asarray(s, dtype=np.float64).reshape(-1),
    ),
    'swm/PushT-v1': (
        _read_true_pusht,
        lambda s: np.asarray(s, dtype=np.float64).reshape(-1)[2:5],
    ),
}

# task -> indices of phi that are wrapped angles (delta wrapped to (-pi, pi]).
# Push-T block angle is phi index 2 (its success metric wraps the angle, mirror
# that here). Reacher qpos uses the RAW diff (its success criterion is raw, no
# wrap), so it is intentionally NOT listed.
_PHI_ANGLE_IDX = {'swm/PushT-v1': (2,)}


# --------------------------------------------------------------------------- #
# PreJEPA stateful-cache clear -- COPIED VERBATIM from                          #
# stable_worldmodel/metrics/esnr.py::_clear_wm_caches. PreJEPA caches the       #
# (detached) init/goal embeddings on the module, expanded to the *sample count* #
# of the call; the count varies on the last (partial) sample mini-batch, so the #
# cache MUST be cleared before every get_cost or the expand dim mismatches.     #
# LeWM/PLDM cache inside the (rebuilt) info dict, so this is a no-op for them.   #
# --------------------------------------------------------------------------- #
def _clear_wm_caches(model) -> None:
    for attr in ('_init_cached_info', '_goal_cached_info'):
        if hasattr(model, attr):
            delattr(model, attr)


# --------------------------------------------------------------------------- #
# TRAM-specific helpers                                                         #
# --------------------------------------------------------------------------- #
def _terminal_latent_from_info(info_chunk) -> torch.Tensor:
    """Terminal *state*-latent left in the info dict by ``model.get_cost``.

    Architecture-agnostic, and STATE-only (no action channels): for PreJEPA the
    pixel-patch latent ``predicted_pixels_emb`` (the same latent the cost is
    built on; ``split_embedding`` already strips the action/proprio channels);
    for LeWM/PLDM the whole ``predicted_emb`` (action is encoded separately, so
    ``emb`` is pure state). Returns ``(B, S, D_lat)``.
    """
    if 'predicted_pixels_emb' in info_chunk:  # PreJEPA: (B, S, T, P, Dp)
        z = info_chunk['predicted_pixels_emb']
        zt = z[:, :, -1]  # (B, S, P, Dp) terminal frame
        return zt.reshape(zt.shape[0], zt.shape[1], -1)  # (B, S, P*Dp)
    if 'predicted_emb' in info_chunk:  # LeWM / PLDM: (B, S, T, D)
        return info_chunk['predicted_emb'][:, :, -1]  # (B, S, D)
    raise KeyError(
        'no predicted latent key (predicted_pixels_emb / predicted_emb) found '
        f'after get_cost; have {list(info_chunk)}'
    )


def _model_terminal_latents(model, sub_info, actions, sample_batch, device):
    """Terminal state-latents for one start over ``S`` action sequences.

    ``sub_info``: dict of ``(1, ...)`` tensors/arrays for ONE start (B'=1).
    ``actions``: ``(1, S, horizon, action_dim)`` NORMALISED actions on ``device``
    (the model's own action space; the planner samples here). Chunks over the
    sample axis (``sample_batch``) and reuses the EXACT ``get_cost`` latent
    rollout the CEM planner uses (the returned cost scalar is discarded -- we
    read the predicted terminal latent it leaves in the per-chunk info dict).
    Returns ``(S, D_lat)`` float64 on CPU.
    """
    S = actions.shape[1]
    outs = []
    for s0 in range(0, S, sample_batch):
        s1 = min(s0 + sample_batch, S)
        _clear_wm_caches(
            model
        )  # see _clear_wm_caches docstring (partial chunk)
        # expand_info_for_samples: (1, ...) -> (1, s1-s0, ...), exactly the shape
        # CEMSolver.solve / action_objective_grads feed get_cost. Returns a fresh
        # dict each call; get_cost only ADDS keys (the pixel views are read-only),
        # so no clone is needed.
        expanded = swm.metrics.expand_info_for_samples(
            sub_info, s1 - s0, device, torch.float32
        )
        acts = actions[:, s0:s1].contiguous()
        with torch.no_grad():
            model.get_cost(expanded, acts)
        zt = _terminal_latent_from_info(expanded)  # (1, s1-s0, D_lat)
        outs.append(zt[0].to('cpu', torch.float64))
    return torch.cat(outs, dim=0)  # (S, D_lat)


def _column_basis(x, tol_scale=1.0):
    """Orthonormal basis (left singular vectors) of centred ``x`` in sample space.

    Returns ``(U, S, rank)`` where ``U`` columns span the column space of the
    centred ``x`` in ``R^K`` and ``S`` are the singular values (variance per
    direction). ``rank`` drops numerically-zero directions.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    U, sv, _ = np.linalg.svd(x, full_matrices=False)
    if sv.size == 0:
        return U, sv, 0
    tol = sv.max() * max(x.shape) * np.finfo(np.float64).eps * tol_scale
    rank = int((sv > tol).sum())
    return U, sv, rank


def _cca_rho1(dz, dy, pca_dim):
    """Top CCA canonical correlation (SVCCA principal-angle form).

    ``dz`` ``(K, D_lat)``, ``dy`` ``(K, D_true)``. PCA-reduce ``dz`` to
    ``p = min(rank(dz), #comps for 99% var, pca_dim, K - rank(dy) - 2)`` left
    singular vectors (the last term is the degeneracy guard: keeps the dz/dy
    subspaces from generically intersecting, which would force rho_1 -> 1), then
    rho_1 = top singular value of ``Qx^T Qy`` (cosine of the smallest principal
    angle), invariant to any isotropic rescale of either view. Returns
    ``(rho_1 in [0,1], p, rank_dy)``; ``nan`` if degenerate/constant.
    """
    dz = np.asarray(dz, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    ok = np.all(np.isfinite(dz), axis=1) & np.all(np.isfinite(dy), axis=1)
    dz, dy = dz[ok], dy[ok]
    K = dz.shape[0]
    if K < 4:
        return np.nan, 0, 0
    Uz, sz, rz = _column_basis(dz)
    Uy, sy, ry = _column_basis(dy)
    if rz == 0 or ry == 0:
        return np.nan, 0, 0
    cum = np.cumsum(sz[:rz] ** 2) / float(np.sum(sz[:rz] ** 2))
    n_var = int(np.searchsorted(cum, 0.99) + 1)
    p_guard = max(1, K - ry - 2)  # keep p + ry < K-1 (no spurious saturation)
    p = int(min(rz, n_var, int(pca_dim), p_guard))
    Qx = Uz[:, :p]
    Qy = Uy[:, :ry]
    s = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
    return float(np.clip(s[0], 0.0, 1.0)), int(p), int(ry)


def _rsa_spearman(dz, dy):
    """RSA fallback: Spearman of pairwise distances d(dz_i,dz_j) vs d(dy_i,dy_j).

    Rank-based (so invariant to any monotone rescale of either distance set;
    isotropic-scale-invariant like CCA, but only sees pairwise geometry). Used as
    a documented cross-check, not the primary statistic.
    """
    dz = np.asarray(dz, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    ok = np.all(np.isfinite(dz), axis=1) & np.all(np.isfinite(dy), axis=1)
    dz, dy = dz[ok], dy[ok]
    if dz.shape[0] < 4:
        return np.nan
    ddz = pdist(dz, 'euclidean')
    ddy = pdist(dy, 'euclidean')
    if np.std(ddz) == 0.0 or np.std(ddy) == 0.0:
        return np.nan
    rho, _ = spearmanr(ddz, ddy)
    return float(rho)


@hydra.main(version_base=None, config_path='./config', config_name='tworoom')
def run(cfg: DictConfig):
    assert cfg.get('policy', 'random') != 'random', 'tram needs a ckpt'
    out_path = OmegaConf.select(cfg, 'tram.out')
    assert out_path, 'pass +tram.out=/path/to.json'
    env_name = cfg.world.env_name
    assert env_name in _TASKS, (
        f'no TRUE readout defined for env {env_name}; known: {list(_TASKS)}'
    )
    read_true, phi_fn = _TASKS[env_name]
    angle_idx = _PHI_ANGLE_IDX.get(env_name, ())

    K = int(OmegaConf.select(cfg, 'tram.K', default=64))
    var_scale = float(OmegaConf.select(cfg, 'tram.var_scale', default=1.0))
    sample_batch = int(OmegaConf.select(cfg, 'tram.sample_batch', default=16))
    pca_dim = int(OmegaConf.select(cfg, 'tram.pca_dim', default=16))
    primary = str(OmegaConf.select(cfg, 'tram.primary', default='cca'))
    assert primary in ('cca', 'rsa'), "tram.primary must be 'cca' or 'rsa'"

    # --- world + transforms + dataset (mirror gcs_align.py / grad_dump.py) ---
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

    # --- model in fp32 (no bf16): latent rollout must be full precision ---
    model = swm.wm.utils.load_world_model(cfg.policy).to('cuda').eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform
    )

    # --- start-state sampling identical to eval/GCS (same seed => same states) ---
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

    world.set_policy(policy)

    # --- capture the preprocessed info_dict at the first solver call (verbatim
    # gcs_align.py): gives the model-rollout INFO on the SAME B starts and leaves
    # world.envs at those starts. ---
    captured = {}

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
            f'[tram] WARNING: horizon*action_block={n_env_steps} != '
            f'goal_offset_steps={goal_offset}; latent/real rollout length does '
            f'not match the goal horizon GCS uses.'
        )
    print(
        f'[tram] B={B} K={K} horizon={horizon} action_dim={action_dim} '
        f'(single={single_dim} x block={action_block}) n_env_steps={n_env_steps} '
        f'pca_dim={pca_dim} primary={primary} task={env_name}'
    )

    # --- the EXACT K action perturbations (CEM prior N(0, var_scale)), sampled
    # ONCE in a model-independent order so every checkpoint sees identical
    # perturbations (fair CCA across models). a0 = zeros baseline prepended as
    # index 0; indices 1..K are the perturbations a0 + da_k = da_k. ---
    gen = torch.Generator(device='cuda').manual_seed(int(cfg.seed))
    samp = swm.metrics.sample_action_trajectories(
        B, K, horizon, action_dim, var_scale, gen, 'cuda'
    )  # (B, K, horizon, action_dim), NORMALISED action space

    # --- restore info for the real-env start (verbatim gcs_align.py path) ---
    init_state, goal_state, _ = _extract_init_goal(
        dataset,
        eval_episodes.tolist(),
        eval_start.tolist(),
        goal_offset,
    )
    callables = OmegaConf.to_container(cfg.eval.get('callables'), resolve=True)
    merged = {**init_state, **goal_state}

    tram_per_b = np.full(B, np.nan, dtype=np.float64)
    rsa_per_b = np.full(B, np.nan, dtype=np.float64)
    p_per_b = np.zeros(B, dtype=np.int64)
    rank_dy_per_b = np.zeros(B, dtype=np.int64)

    for b in range(B):
        # actions for start b: index 0 baseline a0 (zeros), 1..K perturbations.
        base = torch.zeros(1, 1, horizon, action_dim, device='cuda')
        pert = samp[b].unsqueeze(0)  # (1, K, horizon, action_dim)
        actions_b = torch.cat([base, pert], dim=1)  # (1, K+1, H, A)

        # (1) LATENT effect: terminal state-latents F(a0), F(a0+da_k) -> dz_k.
        sub = {
            k: (
                v[b : b + 1]
                if (torch.is_tensor(v) or isinstance(v, np.ndarray))
                else v
            )
            for k, v in info.items()
        }
        z_all = _model_terminal_latents(
            model, sub, actions_b, sample_batch, 'cuda'
        ).numpy()  # (K+1, D_lat)
        dz = z_all[1:] - z_all[0:1]  # (K, D_lat)

        # (2) TRUE effect: phi(true terminal) for each sequence in the REAL env.
        env_u = world.envs.envs[b].unwrapped
        merged_b = {k: v[b] for k, v in merged.items()}
        acts_np = actions_b[0].cpu().numpy()  # (K+1, horizon, action_dim)
        phis = np.full((K + 1, len(phi_fn(read_true(env_u)))), np.nan)
        for j in range(K + 1):
            # restore the SAME physical start as the latent path (verbatim GCS).
            env_u.reset(seed=int(cfg.seed) * 100003 + b)
            _apply_callables(env_u, callables, merged_b)
            seq_norm = acts_np[j].reshape(n_env_steps, single_dim)
            seq_raw = proc_action.inverse_transform(seq_norm)  # un-normalise
            for t in range(n_env_steps):
                env_u.step(
                    np.asarray(seq_raw[t])
                )  # full roll, ignore terminated
            phis[j] = phi_fn(read_true(env_u))
        dy = phis[1:] - phis[0:1]  # (K, D_true)
        for ai in angle_idx:  # wrap angle deltas to (-pi, pi]
            dy[:, ai] = (dy[:, ai] + np.pi) % (2.0 * np.pi) - np.pi

        # (3) per-start alignment.
        rho1, p_used, rank_dy = _cca_rho1(dz, dy, pca_dim)
        rsa = _rsa_spearman(dz, dy)
        rsa_per_b[b] = rsa
        p_per_b[b] = p_used
        rank_dy_per_b[b] = rank_dy
        tram_per_b[b] = rho1 if primary == 'cca' else rsa

    tram = float(np.nanmean(tram_per_b))
    rsa_mean = float(np.nanmean(rsa_per_b))
    n_valid_b = int(np.isfinite(tram_per_b).sum())
    ccastyle = (
        f'svcca_pca{pca_dim}_var99_guard(K-Dtrue-2)'
        if primary == 'cca'
        else 'rsa_spearman_pairwise_l2'
    )

    model_dir, ckpt_kind, ckpt_num = _parse_ckpt(cfg.policy)
    print(
        f'[tram] TRAM={tram:+.4f}  (nanmean over {n_valid_b}/{B} starts; '
        f'higher=better)  rsa={rsa_mean:+.4f}  run={model_dir} '
        f'ckpt={Path(cfg.policy).name}'
    )
    print(
        f'[tram] per-start TRAM[b]: {np.array2string(tram_per_b, precision=3)}'
    )
    print(f'[tram] pca dims used p[b]: {p_per_b.tolist()}')

    payload = {
        'tram': tram,
        'per_b': [None if np.isnan(v) else float(v) for v in tram_per_b],
        'K': K,
        'horizon': int(horizon),
        'ccastyle': ccastyle,
        'task': env_name,
        'run': model_dir,
        'ckpt': Path(cfg.policy).name,
        # --- provenance / robustness extras ---
        'primary': primary,
        'rsa': rsa_mean,
        'rsa_per_b': [None if np.isnan(v) else float(v) for v in rsa_per_b],
        'n_valid_b': n_valid_b,
        'B': B,
        'pca_dim': pca_dim,
        'p_per_b': p_per_b.tolist(),
        'rank_dy_per_b': rank_dy_per_b.tolist(),
        'probe': 'full_horizon_25step',
        'n_env_steps': n_env_steps,
        'action_block': action_block,
        'single_action_dim': single_dim,
        'goal_offset_steps': goal_offset,
        'var_scale': var_scale,
        'seed': int(cfg.seed),
        'ckpt_kind': ckpt_kind,
        'ckpt_num': ckpt_num,
        'dataset': cfg.eval.dataset_name,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f'[tram] wrote {out_path.resolve()}')


if __name__ == '__main__':
    run()
