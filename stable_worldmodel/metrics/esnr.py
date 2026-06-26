"""ESNR -- Expected Signal-to-Noise Ratio of the policy-objective gradient.

Reference implementation of the metric from *Towards Policy-Aware World Models*
(Eq. 10 + "Code 1"). ESNR measures, per action-dimension, the squared mean of
the policy-objective gradient divided by its variance across stochastic action
samples, averaged over observations:

    ESNR = E_obs[ (E_N[grad])**2 / Var_N(grad) ]            (averaged over dims)

Following the paper, the gradient is taken w.r.t. the **action trajectory**
(not policy parameters), so the metric applies uniformly to online planning
(MPC) and parametric policy extraction. For online planning the objective is
J_act(a_{0:H}) -- here ``model.get_cost(info_dict, action_candidates)``.

Code 1 (paper):

    def compute_esnr(actions, J, grad_f):     # actions: (B, act_dim)
        actions = actions.detach().requires_grad_()
        grads = grad_f(J, actions)            # (N, B, action_dim)
        grad_mean = grads.mean(dim=0)         # (B, action_dim)
        grad_std  = grads.std(dim=0)          # (B, action_dim)
        snrs = grad_mean**2 / (grad_std**2 + 1e-8)
        return snrs.mean()

This module keeps that exact math in :func:`compute_esnr_from_grads` and adds a
chunked, gradient-based driver (:func:`run_planning_esnr`) that computes the
action-trajectory gradient through a world model's ``get_cost``. Everything is
done in float32/float64 (never bf16) -- a variance ratio is numerically
unreliable in low precision.
"""

from __future__ import annotations

import numpy as np
import torch

_EPS = 1e-8


def compute_esnr_from_grads(grads: torch.Tensor, eps: float = _EPS) -> dict:
    """ESNR from a stack of per-sample gradients (paper Code 1, core math).

    Args:
        grads: tensor of shape ``(N, B, D)`` -- ``N`` action samples, ``B``
            observation samples, ``D`` action-trajectory dimensions.
        eps: denominator floor (paper uses 1e-8).

    Returns:
        dict with ``esnr`` (scalar float), ``esnr_log10`` (the paper plots this),
        ``esnr_num`` = mean over (B, D) of grad_mean**2, ``esnr_den`` = mean
        variance, ``degenerate_frac`` = fraction of components whose gradient
        std is < 1e-6 (the paper's grad->0 caveat monitor), and ``n_components``.
    """
    if grads.ndim != 3:
        raise ValueError(f'grads must be (N, B, D); got {tuple(grads.shape)}')
    if grads.shape[0] < 2:
        raise ValueError('need >= 2 action samples (N) to estimate variance')

    g = grads.to(torch.float64)
    grad_mean = g.mean(dim=0)  # (B, D)
    grad_std = g.std(dim=0)  # (B, D), unbiased -- matches Code 1's grads.std
    snrs = grad_mean.pow(2) / (grad_std.pow(2) + eps)  # (B, D)
    esnr = snrs.mean()

    return {
        'esnr': float(esnr),
        'esnr_log10': float(torch.log10(esnr + eps)),
        'esnr_num': float(grad_mean.pow(2).mean()),
        'esnr_den': float(grad_std.pow(2).mean()),
        'degenerate_frac': float((grad_std < 1e-6).double().mean()),
        'n_components': int(grad_mean.numel()),
    }


def compute_esnr(actions: torch.Tensor, J, grad_f, eps: float = _EPS) -> float:
    """Faithful Code 1 entry point (used by the unit tests).

    Args:
        actions: ``(B, act_dim)`` base actions.
        J: objective with signature ``J(actions) -> tensor``.
        grad_f: ``grad_f(J, actions) -> grads`` of shape ``(N, B, action_dim)``.
    """
    actions = actions.detach().requires_grad_(True)
    grads = grad_f(J, actions)
    return compute_esnr_from_grads(grads, eps=eps)['esnr']


def sample_action_trajectories(
    n_obs: int,
    num_samples: int,
    horizon: int,
    action_dim: int,
    var_scale: float,
    generator: torch.Generator,
    device,
    dtype: torch.dtype = torch.float32,
    center: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample ``(n_obs, num_samples, horizon, action_dim)`` trajectories.

    Default (``center=None``) is the CEM prior at iteration 0 (``N(0, var_scale)``
    per component). When ``center``/``scale`` ``(n_obs, horizon, action_dim)`` are
    given (the on-policy / CEM-optimized proposal), samples ``N(center, scale)``
    around the converged plan. Returns a plain (detached) tensor; the caller
    slices it into mini-batches and sets ``requires_grad_`` per mini-batch so the
    backward graph stays bounded.
    """
    a = torch.randn(
        n_obs,
        num_samples,
        horizon,
        action_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    if center is not None:
        sc = scale if scale is not None else var_scale
        if torch.is_tensor(sc):
            sc = sc.unsqueeze(1)
        a = center.unsqueeze(1) + a * sc
    else:
        a = a * var_scale
    return a.detach()


def expand_info_for_samples(
    info: dict, num_samples: int, device, dtype: torch.dtype
) -> dict:
    """Add a sample axis to each tensor: ``(B, ...) -> (B, num_samples, ...)``.

    Mirrors ``CEMSolver.solve``'s ``expanded_infos`` construction so ``get_cost``
    sees exactly the shapes it sees during planning. Float tensors are cast to
    ``dtype`` (float32 for the probe); the expansion is a view (no copy).
    """
    out = {}
    for k, v in info.items():
        if torch.is_tensor(v):
            target_dtype = dtype if v.is_floating_point() else None
            v = (
                v.to(device=device, dtype=target_dtype)
                .unsqueeze(1)
                .expand(v.shape[0], num_samples, *v.shape[1:])
            )
        elif isinstance(v, np.ndarray):
            v = np.repeat(v[:, None, ...], num_samples, axis=1)
        out[k] = v
    return out


def action_objective_grads(model, info_chunk: dict, actions: torch.Tensor):
    """Return ``(grads, cost)`` where grads = d get_cost / d actions.

    ``info_chunk`` tensors are already sample-expanded to ``(B, N, ...)``;
    ``actions`` is ``(B, N, horizon, action_dim)`` requiring grad. The cost of
    candidate ``(b, n)`` depends only on ``actions[b, n]`` (cross-sample terms
    are zero), so a single backward of ``cost.sum()`` yields the per-sample
    gradient.
    """
    cost = model.get_cost(info_chunk, actions)
    if tuple(cost.shape) != tuple(actions.shape[:2]):
        raise ValueError(
            f'get_cost returned {tuple(cost.shape)}; '
            f'expected {tuple(actions.shape[:2])} (B, N)'
        )
    (grad,) = torch.autograd.grad(cost.sum(), actions)
    return grad, cost.detach()


def _clear_wm_caches(model) -> None:
    """Clear PreJEPA's stateful init/goal embedding caches between obs batches.

    PreJEPA caches the (detached) initial-obs and goal embeddings on the module
    instance, keyed by ``id``/``step_idx``. LeWM/PLDM cache inside the passed
    info dict (rebuilt per chunk) so they need nothing here.
    """
    for attr in ('_init_cached_info', '_goal_cached_info'):
        if hasattr(model, attr):
            delattr(model, attr)


def collect_planning_grads(
    model,
    info: dict,
    *,
    horizon: int,
    action_dim: int,
    num_samples: int,
    var_scale: float,
    seed: int,
    device,
    obs_batch: int = 1,
    sample_batch: int = 16,
    proposal: str = 'prior',
    solver=None,
    clear_caches: bool = True,
    capture_cost_sens: bool = False,
    return_actions: bool = False,
):
    """Collect the per-sample action-gradient stack for one checkpoint.

    This is the model-touching core shared by :func:`run_planning_esnr` (which
    reduces the stack to the scalar ESNR) and the Phase-2 EPGQ metrics (which
    operate on the full stack). The backward pass is chunked over BOTH
    observations (``obs_batch``) and action samples (``sample_batch``); since
    the cost of candidate ``(b, n)`` depends only on ``actions[b, n]``,
    mini-batching the gradient is exact (not an approximation) and bounds peak
    GPU memory regardless of architecture or ``num_samples``. The full action
    set is sampled up front, so the result is independent of the chunk sizes
    (depends only on ``seed``/``num_samples``).

    Args:
        model: world model with a differentiable ``get_cost`` (frozen params ok).
        info: dict of ``(B, ...)`` tensors/arrays as passed to ``solver.solve``
            (already preprocessed by ``WorldModelPolicy._prepare_info``).
        horizon, action_dim: solver action shape (action_dim includes the
            action_block frameskip, e.g. 2*5=10 for TwoRoom).
        num_samples: N action samples per observation.
        var_scale: proposal std (CEM prior == 1.0).
        obs_batch: observations held in memory together (default 1).
        sample_batch: action samples per get_cost+backward (default 16; lower
            for heavy encoders like DINOv2, higher for light ones).
        proposal: ``'prior'`` = CEM prior N(0, var_scale); ``'cem_optimized'`` =
            on-policy, samples N(mean, std) around the converged CEM plan;
            ``'cem_centered'`` = optimized mean, fixed var_scale spread (both
            CEM proposals require ``solver``).
        solver: configured CEMSolver, used only for the CEM proposals.
        capture_cost_sens: also capture the last-step latent cost-sensitivity
            residual ``pred_emb - goal_emb`` (``d cost / d pred_emb``, up to the
            factor 2) by wrapping ``model.criterion``. Best-effort: ``None`` if
            the model has no compatible ``criterion``. Off by the ESNR path.
        return_actions: also return the exact sampled action trajectories whose
            costs are ``costs_all``, as ``aux['actions']`` of shape
            ``(N, B, horizon, action_dim)`` float32 on CPU. Off by default
            (``aux['actions']`` is ``None``); enabling it is purely additive and
            does not change ``grads_all``/``costs_all``. Used by GCS-Align, which
            re-rolls these SAME actions in the real env to score the model cost
            surface against the true outcome.

    Returns:
        ``(grads_all, costs_all, aux)`` where ``grads_all`` is ``(N, B, D)``
        float64 on CPU, ``costs_all`` is ``(N, B)`` float64 on CPU, and ``aux``
        is ``{'cem_std_mean', 'cost_sens' (N, B, D_lat) float32 CPU or None,
        'n_obs', 'actions' (N, B, H, A) float32 CPU or None}``.
    """
    if proposal not in ('prior', 'cem_optimized', 'cem_centered'):
        raise NotImplementedError(f"proposal '{proposal}' not implemented")

    n_obs = None
    for v in info.values():
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            n_obs = v.shape[0]
            break
    if n_obs is None:
        raise ValueError('info has no array/tensor to infer batch size from')

    # CEM-based proposals: run CEM to convergence, sample ESNR around the plan.
    #   cem_optimized -> N(mean, converged_std)  (on-policy, but the converged
    #     std collapses to the sharp optimum -> gradients vanish for good models)
    #   cem_centered  -> N(mean, var_scale)      (planning neighborhood at a
    #     fixed, non-degenerate spread around the optimized plan)
    # NOTE: this runs BEFORE the optional criterion wrap below, so the many
    # get_cost calls inside solver.solve do not pollute the cost_sens capture.
    cem_mean = cem_std = None
    cem_std_mean = ''
    if proposal in ('cem_optimized', 'cem_centered'):
        if solver is None:
            raise ValueError(f"proposal='{proposal}' requires a solver")
        with torch.inference_mode():
            out = solver.solve(
                {
                    k: (v.clone() if torch.is_tensor(v) else v)
                    for k, v in info.items()
                }
            )
        cem_mean = out['actions'].detach().clone().to(device)  # (B, H, A)
        if proposal == 'cem_optimized':
            cem_std = (
                out['var'][0].detach().clone().to(device)
            )  # converged std
            cem_std_mean = float(cem_std.mean())
        else:  # cem_centered: fixed var_scale spread around the optimized mean
            cem_std = None
            cem_std_mean = float(var_scale)
        if clear_caches:
            _clear_wm_caches(model)

    # optional cost-sensitivity capture: wrap model.criterion to stash the
    # last-step latent residual (pred_emb - goal_emb). LeWM/PLDM compute
    # cost = sum((pred_emb[...,-1,:] - goal_emb[...,-1,:])**2) with no averaging,
    # so this residual is exactly (1/2) d cost / d pred_emb. Fully isolated and
    # best-effort: any failure leaves cost_sens None and never affects grads.
    cs_holder = {'last': None}
    orig_criterion = None
    if capture_cost_sens and hasattr(model, 'criterion'):
        orig_criterion = model.criterion

        def _wrapped_criterion(
            *args, _orig=orig_criterion, _h=cs_holder, **kw
        ):
            # forward *args/**kw verbatim: criterion's arity differs by arch
            # (LeWM/PLDM take info_dict; PreJEPA takes info_dict, actions).
            cost = _orig(*args, **kw)
            try:
                info_dict = args[0]
                pe = info_dict['predicted_emb']  # (B, S, T, D_lat)
                ge = info_dict['goal_emb']  # (B, T, D_lat)
                resid = (pe[:, :, -1, :] - ge[:, None, -1, :]).detach()
                _h['last'] = resid.to('cpu', torch.float32)  # (B, S, D_lat)
            except Exception:  # noqa: BLE001
                _h['last'] = None
            return cost

        model.criterion = _wrapped_criterion

    gen = torch.Generator(device=device).manual_seed(int(seed))
    grad_chunks = []  # each (N, ob, D)
    cost_chunks = []  # each (N, ob)
    cs_chunks = [] if capture_cost_sens else None  # each (N, ob, D_lat)
    act_chunks = [] if return_actions else None  # each (N, ob, H, A)

    try:
        for start in range(0, n_obs, obs_batch):
            end = min(start + obs_batch, n_obs)
            ob = end - start

            sub = {}
            for k, v in info.items():
                if torch.is_tensor(v) or isinstance(v, np.ndarray):
                    sub[k] = v[start:end]
                else:
                    sub[k] = v

            # sample all N trajectories up front (cheap, chunk-invariant)
            center = cem_mean[start:end] if cem_mean is not None else None
            scale = cem_std[start:end] if cem_std is not None else None
            actions_full = sample_action_trajectories(
                ob,
                num_samples,
                horizon,
                action_dim,
                var_scale,
                gen,
                device,
                center=center,
                scale=scale,
            )
            if act_chunks is not None:
                # (ob, N, H, A) -> (N, ob, H, A); these are the EXACT actions
                # whose costs are accumulated below, in the same (N, ob) order.
                act_chunks.append(
                    actions_full.detach()
                    .permute(1, 0, 2, 3)
                    .to('cpu', torch.float32)
                )

            sb_grads = []  # each (sbn, ob, D)
            sb_costs = []  # each (sbn, ob)
            sb_cs = [] if capture_cost_sens else None  # each (sbn, ob, D_lat)
            for s0 in range(0, num_samples, sample_batch):
                s1 = min(s0 + sample_batch, num_samples)
                # clear PreJEPA's instance cache each mini-batch: its cached goal
                # embedding is expanded to the *sample count*, which varies on the
                # last (partial) mini-batch. LeWM/PLDM cache inside the (rebuilt)
                # dict, so they re-encode automatically.
                if clear_caches:
                    _clear_wm_caches(model)
                actions = actions_full[:, s0:s1].detach().requires_grad_(True)
                expanded = expand_info_for_samples(
                    sub, s1 - s0, device, torch.float32
                )
                with torch.enable_grad():
                    grads, cost = action_objective_grads(
                        model, expanded, actions
                    )
                sb_grads.append(
                    grads.detach()
                    .permute(1, 0, 2, 3)
                    .reshape(s1 - s0, ob, horizon * action_dim)
                    .to('cpu', torch.float64)
                )
                sb_costs.append(
                    cost.transpose(0, 1).to('cpu', torch.float64)  # (sbn, ob)
                )
                if sb_cs is not None:
                    last = cs_holder['last']
                    if last is not None:
                        sb_cs.append(last.permute(1, 0, 2))  # (sbn, ob, D_lat)
                    else:
                        sb_cs = None  # a missing chunk disables cost_sens
            grad_chunks.append(torch.cat(sb_grads, dim=0))  # (N, ob, D)
            cost_chunks.append(torch.cat(sb_costs, dim=0))  # (N, ob)
            if cs_chunks is not None and sb_cs is not None:
                cs_chunks.append(torch.cat(sb_cs, dim=0))  # (N, ob, D_lat)
            else:
                cs_chunks = None
    finally:
        if orig_criterion is not None:
            model.criterion = orig_criterion

    grads_all = torch.cat(grad_chunks, dim=1)  # (N, B, D)
    costs_all = torch.cat(cost_chunks, dim=1)  # (N, B)
    cost_sens = None
    if cs_chunks:
        try:
            cost_sens = torch.cat(cs_chunks, dim=1)  # (N, B, D_lat)
        except Exception:  # noqa: BLE001
            cost_sens = None

    actions_all = None
    if act_chunks:
        actions_all = torch.cat(act_chunks, dim=1)  # (N, B, H, A)

    aux = {
        'cem_std_mean': cem_std_mean,
        'cost_sens': cost_sens,
        'n_obs': int(n_obs),
        'actions': actions_all,
    }
    return grads_all, costs_all, aux


def run_planning_esnr(
    model,
    info: dict,
    *,
    horizon: int,
    action_dim: int,
    num_samples: int,
    var_scale: float,
    seed: int,
    device,
    obs_batch: int = 1,
    sample_batch: int = 16,
    proposal: str = 'prior',
    solver=None,
    clear_caches: bool = True,
) -> dict:
    """Compute planning-ESNR for one checkpoint over a batch of observations.

    Thin wrapper over :func:`collect_planning_grads`: collects the ``(N, B, D)``
    action-gradient stack, then reduces it with :func:`compute_esnr_from_grads`.
    The returned dict is unchanged from before the EPGQ refactor.

    Returns:
        :func:`compute_esnr_from_grads` dict, plus ``num_obs``, ``num_samples``,
        ``var_scale``, ``proposal``, ``sample_batch``, ``cem_std_mean``.
    """
    grads_all, _costs, aux = collect_planning_grads(
        model,
        info,
        horizon=horizon,
        action_dim=action_dim,
        num_samples=num_samples,
        var_scale=var_scale,
        seed=seed,
        device=device,
        obs_batch=obs_batch,
        sample_batch=sample_batch,
        proposal=proposal,
        solver=solver,
        clear_caches=clear_caches,
        capture_cost_sens=False,
    )
    out = compute_esnr_from_grads(grads_all)
    out.update(
        num_obs=int(aux['n_obs']),
        num_samples=int(num_samples),
        var_scale=float(var_scale),
        proposal=proposal,
        sample_batch=int(sample_batch),
        cem_std_mean=aux['cem_std_mean'],
    )
    return out


__all__ = [
    'compute_esnr',
    'compute_esnr_from_grads',
    'sample_action_trajectories',
    'expand_info_for_samples',
    'action_objective_grads',
    'collect_planning_grads',
    'run_planning_esnr',
]
