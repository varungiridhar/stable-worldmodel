"""EPGQ -- bias-aware policy-gradient quality metrics (Phase 2).

Plain ESNR (``esnr.py``) failed on the TwoRoom / CEM-MPC testbed for four
data-backed reasons (see ``experiments/2026-06-18-phase1a`` M1-M4):

  M1  raw |grad| is a 4-orders-of-magnitude *encoder-scale* confound;
  M2  ESNR's scale-invariant mean^2/var ratio then only separates frozen-vs-
      from-scratch encoders -- it CANNOT rank the same-family pair PLDM > LeWM;
  M3  on-policy proposals *invert* ESNR (a good model drives grad -> 0 at its
      sharp optimum);
  M4  no U-shape; prediction loss out-predicts ESNR within-run.

This module provides candidate metrics that target those failures. Every
function is a **pure NumPy** post-processing of the per-sample action-gradient
stack ``grads`` of shape ``(N, B, D)`` (N action samples, B observations, D
flattened action-trajectory dims) produced by
:func:`stable_worldmodel.metrics.esnr.collect_planning_grads`. They take NumPy
arrays (the ``.npz`` grad dumps) so the whole metric zoo can be re-scored
offline on a CPU node with no GPU and no re-probing.

Scaling convention (the cost is latent MSE-to-goal, so ``cost ~ c^2`` under an
encoder/latent rescale by ``c``, hence ``grad ~ c^2``):

  * ``mean(grad)``          ~ c^2     ->  ``mean^2 ~ c^4``
  * ``var_N(grad)``         ~ c^4
  * ``Var_k(mean_k)``       ~ c^4     (the epistemic term)

So any **per-component ratio of (c^4 quantity) / (c^4 quantity)** is exactly
scale-invariant (M1). Directional/cosine and sign statistics are scale-free by
construction. Plain |grad|-magnitude is NOT -- that is exactly M1. The
:func:`epgq` composition keeps the per-component-ratio-then-average form so the
bias-aware metric stays invariant.

Metric families (map to the M they target):

  * :func:`directional_consistency`, :func:`sign_agreement` -- dimensionless
    gradient coherence across samples [M1/M2]; the prime candidates to rank
    PLDM > LeWM where the scalar ESNR ratio is tied.
  * :func:`whitened_snr` -- Mahalanobis mu^T Sigma^-1 mu; the multivariate
    generalization of ESNR that accounts for correlated noise directions [M2].
  * :func:`esnr` with ``degenerate_thresh`` -- masked ESNR that drops the
    near-zero-variance components (up to 80% for PLDM) before averaging [M2].
  * :func:`per_component_stats` + :func:`epistemic_var` + :func:`epgq` -- the
    bias-aware EPGQ = signal / (aleatoric + lambda * epistemic), with the
    epistemic term a checkpoint pseudo-ensemble (1-seed, no retrain) [the novel
    term]. ``epgq`` is scale-invariant by the ratio-then-average rule above.
  * :func:`cost_weighted_directional` -- VaGraM-flavoured: weight per-obs by the
    latent cost scale so observations the model already solves count less [M1].
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12
_ESNR_EPS = 1e-8  # matches esnr.compute_esnr_from_grads denominator floor
_DEGENERATE_STD = 1e-6  # esnr.py's grad->0 caveat threshold


def per_component_stats(grads) -> tuple[np.ndarray, np.ndarray]:
    """Per-component ``(mean, var)`` over the sample axis.

    Args:
        grads: ``(N, B, D)`` per-sample gradient stack.

    Returns:
        ``(mu, var)`` each ``(B, D)``; ``var`` is the unbiased (ddof=1) sample
        variance over the ``N`` action samples (matches ESNR's ``grads.std``).
    """
    g = np.asarray(grads, dtype=np.float64)
    if g.ndim != 3:
        raise ValueError(f'grads must be (N, B, D); got {g.shape}')
    if g.shape[0] < 2:
        raise ValueError('need >= 2 action samples (N) to estimate variance')
    return g.mean(axis=0), g.var(axis=0, ddof=1)


def esnr(
    grads, eps: float = _ESNR_EPS, degenerate_thresh: float | None = None
):
    """Plain ESNR = ``mean_{b,d} mean^2 / (var + eps)`` (paper Code 1).

    With ``degenerate_thresh`` set, components whose gradient std is below it are
    dropped before averaging (**masked ESNR**) -- a cheap M2 probe, since up to
    ~80% of PLDM's components are near-zero-variance and dilute the mean.

    Returns the scalar ESNR (``nan`` if every component is masked out).
    """
    mu, var = per_component_stats(grads)
    snr = mu**2 / (var + eps)
    if degenerate_thresh is not None:
        mask = np.sqrt(var) >= degenerate_thresh
        if not mask.any():
            return float('nan')
        return float(snr[mask].mean())
    return float(snr.mean())


def directional_consistency(grads) -> float:
    """Mean cosine of each per-sample gradient to its per-obs mean direction.

    Scale-free (cosine). High when the action-gradient points the *same way*
    across samples even if its magnitude is noisy -- the coherence signal the
    scalar ESNR ratio averages away (M2). ``(N, B, D) -> scalar in [-1, 1]``.
    """
    g = np.asarray(grads, np.float64)
    mu = g.mean(axis=0)  # (B, D)
    mu_n = mu / (np.linalg.norm(mu, axis=-1, keepdims=True) + _EPS)
    g_n = g / (np.linalg.norm(g, axis=-1, keepdims=True) + _EPS)
    cos = (g_n * mu_n[None]).sum(axis=-1)  # (N, B)
    return float(cos.mean())


def sign_agreement(grads) -> float:
    """Per-component sign agreement to the mean, aggregated as ``mean (2p-1)^2``.

    ``p[b,d] = P_n( sign(grad) == sign(mean) )``. Scale-free, dimensionless;
    another M1/M2 coherence candidate. ``(N, B, D) -> scalar in [0, 1]``.
    """
    g = np.asarray(grads, np.float64)
    mu = g.mean(axis=0)
    agree = (np.sign(g) == np.sign(mu)[None]).astype(np.float64)  # (N, B, D)
    p = agree.mean(axis=0)  # (B, D)
    return float(((2.0 * p - 1.0) ** 2).mean())


def whitened_snr(grads, eps_frac: float = 1.0) -> float:
    """Mahalanobis signal-to-noise ``mu^T Sigma^-1 mu`` per obs, averaged.

    ``Sigma`` is the ``D x D`` sample covariance of the gradient over the ``N``
    samples, regularized by ``eps_frac * (trace(Sigma)/D) * I`` (the noise floor
    that also tames the degenerate directions). The multivariate ESNR: it
    rewards a mean gradient that is large *relative to the noise it actually
    sits in*, accounting for correlated noise (M2). Scale-invariant: under
    ``grad -> c*grad``, ``mu -> c*mu`` and ``Sigma -> c^2*Sigma`` so the
    quadratic form is unchanged. ``(N, B, D) -> scalar >= 0``.
    """
    g = np.asarray(grads, np.float64)
    _, b_n, d = g.shape
    vals = []
    for b in range(b_n):
        gb = g[:, b, :]  # (N, D)
        mu = gb.mean(axis=0)  # (D,)
        cov = np.atleast_2d(np.cov(gb, rowvar=False))  # (D, D), ddof=1
        reg = eps_frac * (np.trace(cov) / max(d, 1))
        cov_r = cov + reg * np.eye(d)
        try:
            sol = np.linalg.solve(cov_r, mu)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(cov_r, mu, rcond=None)[0]
        vals.append(float(mu @ sol))
    return float(np.mean(vals))


def cost_weighted_directional(grads, costs) -> float:
    """Directional consistency weighted by per-obs cost (VaGraM-flavoured).

    Observations the model already solves (low cost) carry an uninformative
    gradient; weighting the per-obs cosine by the mean cost down-weights them.
    Scale-free in ``grads`` (cosine); ``costs`` only sets the per-obs weights.

    Args:
        grads: ``(N, B, D)``.
        costs: ``(N, B)`` per-sample planning costs.
    """
    g = np.asarray(grads, np.float64)
    c = np.asarray(costs, np.float64)
    mu = g.mean(axis=0)
    mu_n = mu / (np.linalg.norm(mu, axis=-1, keepdims=True) + _EPS)
    g_n = g / (np.linalg.norm(g, axis=-1, keepdims=True) + _EPS)
    cos = (g_n * mu_n[None]).sum(axis=-1).mean(axis=0)  # (B,)
    w = c.mean(axis=0)  # (B,)
    w = w / (w.sum() + _EPS)
    return float((cos * w).sum())


def cost_normalized_grad_magnitude(
    grads, costs, rel_floor: float = 1e-12, per_sample: bool = True
) -> float:
    """Mean L2 magnitude of the action-gradient of the LOG planning cost.

    ``m = E[ ||d cost / d a||_2 / cost ] = E[ ||d log cost / d a||_2 ]``. The
    Phase-2 metric (PIVOT from the SNR family -- see
    ``experiments/2026-06-20-phase2-graddbg``). **LOWER = better.**

    Exactly scale-invariant: a latent/cost rescale ``cost -> k*cost`` (which
    also scales ``d cost/d a -> k * d cost/d a``) shifts ``log cost`` by a
    constant, leaving its action-gradient -- and hence this metric -- unchanged.
    Unlike raw ``|grad|`` (the M1 encoder-scale confound), normalizing by the
    cost removes the latent scale while *keeping* the real within-family signal:
    at matched cost scale a steeper planning cost per unit cost is an
    over-confident / biased objective that misleads CEM. Ranks the
    scale-comparable pair correctly (PLDM<LeWM, i.e. PLDM better) in 3/3 seeds at
    both the 30ep and 250ep regimes, where every SNR/coherence metric inverts.

    Args:
        grads: ``(N, B, D)`` per-sample action gradients.
        costs: ``(N, B)`` per-sample planning costs (non-negative).
        per_sample: normalize each sample's gradient by its own cost then average
            (the natural ``||grad log cost||``); else ``mean||grad|| / mean cost``.

    Returns:
        scalar (>= 0); LOWER means a better-conditioned planning objective.
    """
    g = np.asarray(grads, np.float64)
    c = np.abs(np.asarray(costs, np.float64))
    cmax = float(c.max()) if c.size else 0.0
    if cmax <= 0.0:
        return float('nan')
    c = np.maximum(c, rel_floor * cmax)  # relative floor -> scale-invariant
    gnorm = np.sqrt((g**2).sum(axis=-1))  # (N, B) per-sample ||grad||_2
    if per_sample:
        return float((gnorm / c).mean())
    return float(gnorm.mean() / c.mean())


def degenerate_frac(grads, std_thresh: float = 1e-6) -> float:
    """Fraction of (B, D) action-gradient components with near-zero variance.

    The paper's ``grad -> 0`` caveat made quantitative: a collapsed world model
    (latent/dimensional collapse) drives the planning-cost gradient toward a
    constant, so a large fraction of components have ~zero across-sample
    variance. High ``degenerate_frac`` => the gradient has degenerated.
    """
    _, var = per_component_stats(grads)
    return float((np.sqrt(var) < std_thresh).mean())


def cost_norm_gradmag_guarded(
    grads,
    costs,
    degen_thresh: float = 0.5,
    std_thresh: float = 1e-6,
    per_sample: bool = True,
) -> float:
    """:func:`cost_normalized_grad_magnitude` with a DEGENERACY GUARD (Phase-2 v2).

    The raw metric is ``LOWER = better`` and captures the *over-confident*
    failure (a steep cost gradient from a biased model). But a *totally
    collapsed* model drives BOTH ``cost`` and ``||grad||`` to ~0, so the raw
    metric -> 0 and would rank the degenerate model BEST -- the paper's
    ``grad -> 0`` caveat (M3/M4), seen on the LeWM sigreg-off ablation rung
    (cost ~0.02, ||grad|| ~1e-5, 80% degenerate components). The relation
    between planning quality and gradient magnitude is therefore **U-shaped**:
    both steep and flat gradients are bad.

    This guard returns ``+inf`` (WORST, since lower=better) when the gradient
    has degenerated (``degenerate_frac > degen_thresh``), so the metric is
    monotone with planning quality across BOTH failure modes. ``degen_thresh =
    0.5`` (a majority of components collapsed) is well-separated on the ladder
    (flagged rungs ~0.6-0.8; all others 0.0). Callers ranking by the value
    should treat ``+inf`` as worst (e.g. replace with ``max_finite * k`` before
    a rank correlation so it is kept, not dropped).
    """
    if degenerate_frac(grads, std_thresh) > degen_thresh:
        return float('inf')
    return cost_normalized_grad_magnitude(grads, costs, per_sample=per_sample)


def cost_normalized_grad_variance(grads, costs) -> float:
    """Cost-normalized ALEATORIC variance of the policy gradient (Phase-2 v3).

    ``CNGV = E_{b,d}[ Var_n(d cost / d a) ] / E[cost]^2``. **LOWER = better.**
    The reassessed Phase-2 metric (it beats the v2 magnitude form and, unlike it,
    is encoder-agnostic INCLUDING the frozen-encoder PreJEPA -- Spearman with
    success -0.90 over the full 19-model zoo vs -0.70 for magnitude).

    The key finding: it is the gradient **variance** -- not ESNR's
    ``signal/variance`` ratio (which divides the variance out) and not the raw
    magnitude -- that predicts planning quality. ``cost_norm_gradmag`` worked
    only because for noise-dominated models ``||grad|| ~ sqrt(variance)``, so it
    was a disguised, weaker variance metric. Here the variance is explicit.

    Scale-invariant: under a latent/cost rescale, ``Var(grad) ~ c^4`` and
    ``cost^2 ~ c^4``. (``sqrt(Var)/cost`` is monotonically equivalent -- same
    ranking, a 'relative gradient-noise level'.) A collapsed model has
    ``Var -> 0`` (looks best), so this still needs the degeneracy guard.
    """
    g = np.asarray(grads, np.float64)
    c = np.abs(np.asarray(costs, np.float64))
    mc = c.mean()
    if mc <= 0:
        return float('nan')
    _, var = per_component_stats(g)
    return float(var.mean() / (mc**2))


def cost_norm_gradvar_guarded(
    grads, costs, degen_thresh: float = 0.5, std_thresh: float = 1e-6
) -> float:
    """:func:`cost_normalized_grad_variance` with the degeneracy guard (v3).

    Returns ``+inf`` (WORST) when the gradient has collapsed
    (``degenerate_frac > degen_thresh``): a totally-collapsed model drives both
    the gradient variance and the cost to ~0, so the raw ``LOWER=better`` metric
    would rank it best (the paper's ``grad -> 0`` caveat). LOWER = better.
    """
    if degenerate_frac(grads, std_thresh) > degen_thresh:
        return float('inf')
    return cost_normalized_grad_variance(grads, costs)


def epistemic_var(mu_stack) -> np.ndarray:
    """Checkpoint pseudo-ensemble epistemic term: ``Var_k(mean_k)``.

    Args:
        mu_stack: ``(K, B, D)`` per-checkpoint mean gradients over a window of K
            nearby (same-run, same-family) checkpoints, computed on the SAME
            ``info`` + action samples so they are directly comparable.

    Returns:
        ``(B, D)`` cross-checkpoint variance of the per-component mean gradient
        (zeros if ``K < 2``). Scales as ``c^4`` -- same as the numerator and the
        aleatoric term -- so :func:`epgq` stays scale-invariant.
    """
    m = np.asarray(mu_stack, np.float64)
    if m.ndim != 3:
        raise ValueError(f'mu_stack must be (K, B, D); got {m.shape}')
    if m.shape[0] < 2:
        return np.zeros(m.shape[1:], dtype=np.float64)
    return m.var(axis=0, ddof=1)


def epgq(
    mu,
    var,
    epistemic,
    lam: float = 1.0,
    rel_floor: float = 1e-12,
    degenerate_thresh: float | None = None,
) -> float:
    """Bias-aware EPGQ = signal / (aleatoric + lambda * epistemic).

    Computed **per component then averaged** (not a ratio of averages) so the
    metric is scale-invariant: numerator (``mean^2``) and both denominator terms
    (aleatoric ``var`` and epistemic ``Var_k(mean_k)``) all scale the same way
    under a latent/cost rescale, so each per-component ratio is unchanged. A
    *confidently-wrong* model -- low aleatoric variance but high cross-checkpoint
    epistemic disagreement -- is penalized, which plain ESNR cannot do.

    The denominator is floored **relatively** (``rel_floor * max(denom)``) rather
    than by a fixed additive epsilon, so the metric is invariant to *any*
    rescale (a fixed floor would dominate at small scales and break invariance --
    exactly the exit criterion's invariance test). Components at the floor are
    degenerate (zero gradient) and excluded.

    Args:
        mu: ``(B, D)`` per-component mean gradient for THIS checkpoint.
        var: ``(B, D)`` per-component aleatoric variance for this checkpoint.
        epistemic: ``(B, D)`` cross-window epistemic variance
            (:func:`epistemic_var`).
        lam: epistemic weight (default 1.0 -- equal weight, no tuning).
        rel_floor: denominator floor as a fraction of ``max(denom)`` (scale-
            invariant div-0 guard; also drops zero-gradient components).
        degenerate_thresh: optional *absolute* ``sqrt(var)`` mask (diagnostic
            only -- not scale-invariant; leave ``None`` for the frozen metric).

    Returns:
        scalar EPGQ (``nan`` if every component is degenerate/masked).
    """
    mu = np.asarray(mu, np.float64)
    var = np.asarray(var, np.float64)
    epi = np.asarray(epistemic, np.float64)
    denom = var + lam * epi
    dmax = float(denom.max()) if denom.size else 0.0
    if dmax <= 0.0:
        return float('nan')
    keep = denom > rel_floor * dmax  # relative -> scale-invariant
    if degenerate_thresh is not None:
        keep &= np.sqrt(var) >= degenerate_thresh
    if not keep.any():
        return float('nan')
    return float((mu[keep] ** 2 / denom[keep]).mean())


def single_ckpt_metrics(grads, costs=None) -> dict:
    """All single-checkpoint candidate metrics (no epistemic term) in one dict.

    Convenience for the A0 debugging / scoring sweep. The epistemic EPGQ needs a
    checkpoint window and is computed separately via :func:`per_component_stats`
    + :func:`epistemic_var` + :func:`epgq`.
    """
    mu, var = per_component_stats(grads)
    out = {
        'esnr': esnr(grads),
        'esnr_masked': esnr(grads, degenerate_thresh=_DEGENERATE_STD),
        'directional': directional_consistency(grads),
        'sign_agreement': sign_agreement(grads),
        'whitened_snr': whitened_snr(grads),
        'signal': float((mu**2).mean()),
        'noise': float(var.mean()),
        'gradmag': float((mu**2 + var).mean()),
        'degenerate_frac': degenerate_frac(grads),
    }
    if costs is not None:
        out['cost_weighted_directional'] = cost_weighted_directional(
            grads, costs
        )
        # Phase-2 v3 PRIMARY: cost-normalized gradient variance (+ guard)
        out['cost_norm_gradvar'] = cost_normalized_grad_variance(grads, costs)
        out['cost_norm_gradvar_guarded'] = cost_norm_gradvar_guarded(
            grads, costs
        )
        # v2 magnitude form (kept for comparison; superseded by gradvar)
        out['cost_norm_gradmag'] = cost_normalized_grad_magnitude(grads, costs)
        out['cost_norm_gradmag_guarded'] = cost_norm_gradmag_guarded(
            grads, costs
        )
    return out


__all__ = [
    'per_component_stats',
    'esnr',
    'directional_consistency',
    'sign_agreement',
    'whitened_snr',
    'cost_weighted_directional',
    'cost_normalized_grad_magnitude',
    'cost_normalized_grad_variance',
    'degenerate_frac',
    'cost_norm_gradmag_guarded',
    'cost_norm_gradvar_guarded',
    'epistemic_var',
    'epgq',
    'single_ckpt_metrics',
]
