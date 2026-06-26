"""GLP + TCA -- two SIM-FREE cousins of GCS-Align.

GCS-Align (``scripts/plan/gcs_align.py``) answers "do the actions the model
thinks are good actually reach the TRUE goal?" by rolling CEM-sampled actions in
the REAL simulator and correlating the model's plan cost with the true terminal
distance. It PASSES cross-task -- but it needs a ground-truth SIMULATOR.

GLP and TCA ask a *strictly cheaper, sim-free* version of the SAME question:
does the model's goal-MSE cost, evaluated on STATIC dataset frames with NO
dynamics rollout and NO simulator, rise with the TRUE distance to the goal?
Only the model's encoder + privileged dataset columns are touched -- never the
env, never the predictor.

For a trained WM on a task we sample many (anchor-frame, goal-frame) pairs from
the expert dataset at true separations (goal offsets {5,25,50,75}). For each
frame we encode the IMAGE (and, for PreJEPA, the privileged proprio/observation
channel exactly as ``get_cost`` does) to the SAME latent the model's
``get_cost`` compares against the goal -- the STATIC encoded latent ``z``, NO
rollout. Then:

  * ``c_lat(a, g)`` -- the model's goal-MSE between the encoded anchor and the
    encoded goal, computed with EACH model's own ``criterion`` reduction:
      - PLDM / LeWM (cls-token): ``sum_D (z_a - z_g)^2``  (MSE, sum over feat).
      - PreJEPA (patch tokens): ``mean_{P,d}(pix_a - pix_g)^2 +
        sum_key mean_e(key_a - key_g)^2``  (MSE, mean over feat, per cost key).
    This is the *static* analogue of the goal term ``get_cost`` minimises: the
    goal side is encoded identically (raw backbone / projector); only the anchor
    side uses the encoder instead of the predicted rollout.
  * ``d_true(a, g)`` -- the privileged TRUE distance, read from the dataset's
    per-step true-state column (TwoRoom agent xy L2; Reacher qpos L2; Push-T
    weighted SE(2) on the block), reusing GCS-Align's ``_dist_*`` verbatim.

  GLP = Spearman(c_lat, d_true) over ALL pairs.  POSITIVE = cost rises with true
        distance = good.  One scalar / model / task.

  TCA = local concordance. For consecutive dataset frames (s_t, s_{t+1}) (one
        dataset step apart), each against the SAME offset goal:
          c  = c_lat(s_t,   goal),  c' = c_lat(s_{t+1}, goal)
          d  = d_true(s_t,  goal),  d' = d_true(s_{t+1}, goal)
        concordant iff sign(c'-c) == sign(d'-d).  TCA = |d'-d|-weighted fraction
        concordant.  0.5 = chance, higher = better.  One scalar / model / task.

Reuses the small helpers + per-task TRUE machinery + dist functions from
``scripts/plan/gcs_align.py`` (cited inline). UNLIKE gcs_align this script never
constructs ``swm.World`` and never steps an env: it is purely dataset + encoder.

    python scripts/plan/metric_glp_tca.py --config-name tworoom \
      policy=p1a_prejepa_s0/weights_epoch_20.pt \
      eval.dataset_name=tworoom_expert.lance \
      +metric.out=/.../glp_tca/p1a_prejepa_s0__weights_epoch_20.json \
      +metric.n_pairs=3000 seed=42
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
from scipy.stats import spearmanr
from sklearn import preprocessing
from torchvision import tv_tensors
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm


# --------------------------------------------------------------------------- #
# setup helpers (verbatim from gcs_align.py / grad_dump.py)                    #
# --------------------------------------------------------------------------- #
def img_transform(cfg, dtype=torch.float32):
    # identical pipeline to gcs_align.img_transform -> the encoder sees exactly
    # the pixels the MPC eval feeds it (ImageNet-normalised, 224).
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(dtype, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def _episode_col(dataset):  # verbatim from gcs_align.py
    for c in ('episode_idx', 'ep_idx'):
        try:
            dataset.get_col_data(c)
            return c
        except Exception:  # noqa: BLE001
            continue
    raise KeyError('no episode-index column (episode_idx/ep_idx) found')


def get_episodes_length(dataset, episodes):  # verbatim from gcs_align.py
    col_name = _episode_col(dataset)
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data('step_idx')
    return np.array([np.max(step_idx[episode_idx == e]) + 1 for e in episodes])


def _parse_ckpt(policy: str):  # verbatim from gcs_align.py
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
# per-task TRUE distance -- the dist_fn's are VERBATIM from gcs_align.py.      #
# gcs_align reads the terminal true-state from the live env; here (sim-free)   #
# we read the SAME privileged quantity from the dataset's per-step true-state  #
# COLUMN, so d_true for both the anchor and the goal frame comes straight off  #
# the recorded trajectory. `_TRUE_COL` names that column per task.            #
# --------------------------------------------------------------------------- #
_PUSHT_ANGLE_W = 20.0 / (np.pi / 9.0)  # gcs_align.py


def _dist_tworoom(term, goal):  # verbatim from gcs_align.py
    return float(
        np.linalg.norm(term[:2] - np.asarray(goal, dtype=np.float64)[:2])
    )


def _dist_reacher(term, goal):  # verbatim from gcs_align.py
    goal = np.asarray(goal, dtype=np.float64)
    n = min(term.shape[0], goal.shape[0])
    return float(np.linalg.norm(term[:n] - goal[:n]))


def _dist_pusht(term, goal):  # verbatim from gcs_align.py
    goal = np.asarray(goal, dtype=np.float64)
    pos_err = np.linalg.norm(term[2:4] - goal[2:4])
    ang = abs(float(term[4]) - float(goal[4]))
    ang = min(ang, 2.0 * np.pi - ang)
    return float(pos_err + _PUSHT_ANGLE_W * ang)


# env_name -> (true-state COLUMN, dist(term_true, goal_true)). The dist_fn is
# the SAME as gcs_align._TASKS (which mirrors each task's success criterion).
_TASKS = {
    'swm/TwoRoom-v1': ('state', _dist_tworoom),
    'swm/ReacherDMControl-v0': ('qpos', _dist_reacher),
    'swm/PushT-v1': ('state', _dist_pusht),
}


def _spearman(x, y):  # verbatim from gcs_align.py
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or np.std(x[ok]) == 0.0 or np.std(y[ok]) == 0.0:
        return np.nan
    rho, _ = spearmanr(x[ok], y[ok])
    return float(rho)


# --------------------------------------------------------------------------- #
# model-family handling: the STATIC encoded latent + its goal-MSE reduction.  #
# Detected structurally (PLDM is not re-exported by wm/__init__, so no         #
# isinstance): PreJEPA owns `backbone` + `extra_encoders` (patch tokens +      #
# concat proprio); PLDM/LeWM own `encoder` + `projector` (cls token).          #
# --------------------------------------------------------------------------- #
def _is_prejepa(model) -> bool:
    return hasattr(model, 'backbone') and hasattr(model, 'extra_encoders')


def _prejepa_cost_keys(model):
    """The non-action extra-encoder keys PreJEPA.criterion compares (besides
    pixels). For our models: TwoRoom/Push-T -> ['proprio'], Reacher ->
    ['observation']."""
    return [k for k in model.extra_encoders.keys() if k != 'action']


@torch.no_grad()
def encode_latents(
    model, raw_pix, extra, emb_keys, is_prejepa, transform, device, batch
):
    """Encode U frames to their STATIC latents (NO rollout).

    raw_pix: (U, C, H, W) uint8 CHW tensor (one frame per row, decoded off the
             dataset). extra: {key: (U, in_chans) float32} privileged channels
             (PreJEPA only; already normalised exactly as the policy would).
    Returns latents kept on CPU:
      PreJEPA  -> {'pixels': (U, P, d), <key>: (U, e), ...}
      cls-token-> {'emb': (U, D)}
    The encode call is the model's OWN ``encode`` -- the identical entry
    ``get_cost`` uses to embed the goal frame (gcs_align relies on the same).
    """
    U = raw_pix.shape[0]
    if is_prejepa:
        out = {'pixels': [], **{k: [] for k in emb_keys}}
    else:
        out = {'emb': []}
    for i in range(0, U, batch):
        # transform each CHW uint8 frame exactly as policy._prepare_info does:
        # wrap as tv_tensors.Image then apply img_transform -> (C, 224, 224).
        pix = (
            torch.stack(
                [
                    transform(tv_tensors.Image(raw_pix[j]))
                    for j in range(i, min(i + batch, U))
                ]
            )
            .unsqueeze(1)
            .to(device)
        )  # (b, T=1, C, H, W)
        if is_prejepa:
            info = {'pixels': pix}
            for k in emb_keys:
                info[k] = (
                    extra[k][i : i + batch].unsqueeze(1).to(device)
                )  # (b,1,in)
            info = model.encode(
                info,
                pixels_key='pixels',
                emb_keys=emb_keys,
                prefix='',
                target='emb',
            )
            out['pixels'].append(
                info['pixels_emb'][:, 0].float().cpu()
            )  # (b,P,d)
            for k in emb_keys:
                out[k].append(info[f'{k}_emb'][:, 0].float().cpu())  # (b,e)
        else:
            info = model.encode({'pixels': pix})
            out['emb'].append(info['emb'][:, 0].float().cpu())  # (b,D)
    return {k: torch.cat(v) for k, v in out.items()}


@torch.no_grad()
def c_lat(latents, ai, gi, is_prejepa, emb_keys, device, chunk=2048):
    """Static goal-MSE c_lat(a, g) per pair, with EACH model's criterion
    reduction (PLDM/LeWM: sum over feat; PreJEPA: per-key mean over feat)."""
    n = len(ai)
    out = np.empty(n, dtype=np.float64)
    ai = torch.as_tensor(ai)
    gi = torch.as_tensor(gi)
    for s in range(0, n, chunk):
        a = ai[s : s + chunk]
        g = gi[s : s + chunk]
        if is_prejepa:
            pa = latents['pixels'][a].to(device)
            pg = latents['pixels'][g].to(device)
            c = ((pa - pg) ** 2).mean(dim=(1, 2))  # mean over (P, d)
            for k in emb_keys:
                ka = latents[k][a].to(device)
                kg = latents[k][g].to(device)
                c = c + ((ka - kg) ** 2).mean(dim=1)  # mean over e
        else:
            ea = latents['emb'][a].to(device)
            eg = latents['emb'][g].to(device)
            c = ((ea - eg) ** 2).sum(dim=1)  # sum over D
        out[s : s + chunk] = c.double().cpu().numpy()
    return out


@hydra.main(version_base=None, config_path='./config', config_name='tworoom')
def run(cfg: DictConfig):
    assert cfg.get('policy', 'random') != 'random', (
        'metric_glp_tca needs a ckpt'
    )
    out_path = OmegaConf.select(cfg, 'metric.out')
    assert out_path, 'pass +metric.out=/path/to.json'
    env_name = cfg.world.env_name
    assert env_name in _TASKS, (
        f'no TRUE-distance defined for env {env_name}; known: {list(_TASKS)}'
    )
    true_col, dist_fn = _TASKS[env_name]

    n_pairs = int(OmegaConf.select(cfg, 'metric.n_pairs', default=3000))
    offsets = OmegaConf.select(cfg, 'metric.offsets', default=None)
    offsets = [5, 25, 50, 75] if offsets is None else list(offsets)
    enc_batch = int(OmegaConf.select(cfg, 'metric.encode_batch', default=64))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- dataset + per-key StandardScalers (mirror gcs_align exactly) -------- #
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
    process = {}
    for c in cfg.dataset.keys_to_cache:
        if c == 'pixels':
            continue
        p = preprocessing.StandardScaler()
        cd = dataset.get_col_data(c)
        p.fit(cd[~np.isnan(cd).any(axis=1)])
        process[c] = p  # eval normalises proprio iff it is in keys_to_cache

    col_name = _episode_col(dataset)
    ep_col = np.asarray(dataset.get_col_data(col_name))
    step_col = np.asarray(dataset.get_col_data('step_idx'))
    ep_off = np.asarray(
        dataset.offsets
    )  # storage start row of each episode id
    ep_indices = np.unique(ep_col)
    ep_len_arr = get_episodes_length(dataset, ep_indices)
    ep_len = {int(e): int(ep_len_arr[i]) for i, e in enumerate(ep_indices)}
    per_row_len = np.array(
        [ep_len[int(e)] for e in ep_col]
    )  # episode len per row

    transform = img_transform(cfg, torch.float32)

    # --- model: fp32, frozen (mirror gcs_align) ----------------------------- #
    model = swm.wm.utils.load_world_model(cfg.policy).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    is_prejepa = _is_prejepa(model)
    emb_keys = _prejepa_cost_keys(model) if is_prejepa else []

    # --- privileged true-state column + (PreJEPA) cost extra-channels -------- #
    true_state = np.asarray(dataset.get_col_data(true_col), dtype=np.float64)
    extra_cols = {}  # raw, NORMALISED-as-policy per-row extra channels
    for k in emb_keys:
        col = np.asarray(dataset.get_col_data(k), dtype=np.float64)
        if k in process:  # eval-faithful: normalise only if a scaler exists
            col = process[k].transform(col)
        extra_cols[k] = col.astype(np.float32)

    # --- sample (anchor, goal) global rows at each offset ------------------- #
    g = np.random.default_rng(int(cfg.seed))
    n_per = max(1, n_pairs // len(offsets))
    a_rows, goal_rows, nx_rows, pair_off = [], [], [], []
    for o in offsets:
        # valid anchor rows: anchor+offset stays inside the episode.
        valid = np.nonzero(step_col <= (per_row_len - o - 1))[0]
        if len(valid) == 0:
            continue
        take = min(n_per, len(valid))
        sel = valid[g.choice(len(valid), size=take, replace=False)]
        a_rows.append(sel)
        goal_rows.append(sel + o)  # contiguous global rows within an episode
        nx_rows.append(sel + 1)  # s_{t+1}: one dataset step apart (TCA)
        pair_off.append(np.full(take, o))
    a_rows = np.concatenate(a_rows)
    goal_rows = np.concatenate(goal_rows)
    nx_rows = np.concatenate(nx_rows)
    pair_off = np.concatenate(pair_off)

    # drop pairs with non-finite true-state on any of the 3 frames
    finite = (
        np.isfinite(true_state[a_rows]).all(1)
        & np.isfinite(true_state[goal_rows]).all(1)
        & np.isfinite(true_state[nx_rows]).all(1)
    )
    a_rows, goal_rows, nx_rows, pair_off = (
        a_rows[finite],
        goal_rows[finite],
        nx_rows[finite],
        pair_off[finite],
    )
    n = len(a_rows)
    assert n > 1, 'no valid pairs sampled'

    # --- read + encode the UNIQUE frames once ------------------------------- #
    uniq = np.unique(np.concatenate([a_rows, goal_rows, nx_rows]))
    row2fid = {int(r): i for i, r in enumerate(uniq)}
    # decode pixels off the dataset (CHW uint8) for every unique frame, chunked
    raw_chunks = []
    LOAD = 256
    for s in range(0, len(uniq), LOAD):
        block = uniq[s : s + LOAD]
        eps = ep_col[block]
        # load_chunk indexes by 0-based LOCAL step (off[ep]+local). step_idx is
        # NOT reliable for this (reacher's is 1-based) and would read the frame
        # one ahead of the get_col_data[row] true-state/proprio. Use the true
        # 0-based local = global_row - episode_start so pixels, true-state and
        # the extra channels all come from the SAME storage row.
        locs = block - ep_off[eps]
        chunk = dataset.load_chunk(eps, locs, locs + 1)
        raw_chunks.append(
            torch.stack([d['pixels'][0] for d in chunk])
        )  # (b,C,H,W)
    raw_pix = torch.cat(raw_chunks).to(torch.uint8)  # (U, C, H, W)
    extra = (
        {k: torch.from_numpy(extra_cols[k][uniq]).float() for k in emb_keys}
        if is_prejepa
        else {}
    )
    print(
        f'[glp_tca] task={env_name} family={"PreJEPA" if is_prejepa else "cls"} '
        f'emb_keys={emb_keys} pairs={n} unique_frames={len(uniq)} '
        f'offsets={offsets} run={cfg.policy}'
    )
    latents = encode_latents(
        model,
        raw_pix,
        extra,
        emb_keys,
        is_prejepa,
        transform,
        device,
        enc_batch,
    )

    a_fid = np.array([row2fid[int(r)] for r in a_rows])
    g_fid = np.array([row2fid[int(r)] for r in goal_rows])
    nx_fid = np.array([row2fid[int(r)] for r in nx_rows])

    # --- c_lat (static goal-MSE) for the GLP and TCA pairs ------------------ #
    c_ag = c_lat(latents, a_fid, g_fid, is_prejepa, emb_keys, device)
    c_ng = c_lat(latents, nx_fid, g_fid, is_prejepa, emb_keys, device)

    # --- d_true off the privileged column ----------------------------------- #
    d_ag = np.array(
        [
            dist_fn(true_state[a], true_state[gg])
            for a, gg in zip(a_rows, goal_rows)
        ]
    )
    d_ng = np.array(
        [
            dist_fn(true_state[nx], true_state[gg])
            for nx, gg in zip(nx_rows, goal_rows)
        ]
    )

    # --- GLP = Spearman(c_lat, d_true) over all pairs ----------------------- #
    glp = _spearman(c_ag, d_ag)
    glp_per_off = {
        int(o): _spearman(c_ag[pair_off == o], d_ag[pair_off == o])
        for o in offsets
    }

    # --- TCA = |d'-d|-weighted fraction concordant -------------------------- #
    dc = c_ng - c_ag  # c' - c
    dd = d_ng - d_ag  # d' - d
    w = np.abs(dd)
    ok = np.isfinite(dc) & np.isfinite(dd)
    concordant = (np.sign(dc) == np.sign(dd)).astype(np.float64)
    wsum = float(w[ok].sum())
    tca = (
        float((w[ok] * concordant[ok]).sum() / wsum)
        if wsum > 0
        else float('nan')
    )

    model_dir, ckpt_kind, ckpt_num = _parse_ckpt(cfg.policy)
    print(
        f'[glp_tca] GLP={glp:+.4f} (POS=good)  TCA={tca:.4f} (0.5=chance)  '
        f'n={n}  run={model_dir} ckpt={Path(cfg.policy).name}'
    )
    print(f'[glp_tca] GLP per-offset: {glp_per_off}')

    payload = {
        'glp': glp,
        'tca': tca,
        'n_pairs': int(n),
        'task': env_name,
        'run': model_dir,
        'ckpt': Path(cfg.policy).name,
        'ckpt_kind': ckpt_kind,
        'ckpt_num': ckpt_num,
        'model_family': 'PreJEPA' if is_prejepa else 'cls_token',
        'emb_keys': list(emb_keys),
        'dataset': cfg.eval.dataset_name,
        'offsets': [int(o) for o in offsets],
        'glp_per_offset': {str(k): v for k, v in glp_per_off.items()},
        'tca_weight_sum': wsum,
        'seed': int(cfg.seed),
        'pusht_angle_weight': float(_PUSHT_ANGLE_W),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        out_path.with_suffix('.npz'),
        c_ag=c_ag.astype(np.float32),
        d_ag=d_ag.astype(np.float32),
        c_ng=c_ng.astype(np.float32),
        d_ng=d_ng.astype(np.float32),
        pair_off=pair_off.astype(np.int32),
        a_rows=a_rows.astype(np.int64),
        goal_rows=goal_rows.astype(np.int64),
        nx_rows=nx_rows.astype(np.int64),
    )
    print(f'[glp_tca] wrote {out_path.resolve()}')
    print(f'[glp_tca] wrote {out_path.with_suffix(".npz").resolve()}')


if __name__ == '__main__':
    run()
