"""Unit tests for the Phase-2 EPGQ candidate metrics.

CPU-only, pure NumPy, no world model. Validates: (1) the analytic reduction of
EPGQ to ESNR when there is no epistemic term; (2) **exact scale-invariance** of
the encoder-agnostic candidates (the exit criterion's invariance test, M1);
(3) that EPGQ penalizes a confidently-wrong (high-epistemic) model; and (4)
sanity of the directional/sign coherence statistics.
"""

import numpy as np
import pytest

from stable_worldmodel.metrics import epgq as E


def _synth(seed=0, n=3000, b=4, d=8, sigma=0.5):
    """A structured grad stack: a fixed mean direction per obs + iid noise."""
    rng = np.random.default_rng(seed)
    mu_true = rng.normal(size=(b, d))
    return mu_true[None] + sigma * rng.normal(size=(n, b, d))


def test_epgq_reduces_to_esnr_without_epistemic():
    """With epistemic == 0 (and negligible relative floor), EPGQ is exactly the
    per-component mean^2/var average -- i.e. ESNR with no denominator floor."""
    g = _synth()
    mu, var = E.per_component_stats(g)
    val_epgq = E.epgq(mu, var, np.zeros_like(var), lam=1.0)
    val_ref = float((mu**2 / var).mean())
    assert abs(val_epgq - val_ref) / (abs(val_ref) + 1e-12) < 1e-6


def test_scale_invariance_exact():
    """directional / sign / whitened are invariant to any positive rescale."""
    g = _synth()
    metrics = {
        'directional': E.directional_consistency,
        'sign': E.sign_agreement,
        'whitened': E.whitened_snr,
    }
    base = {k: f(g) for k, f in metrics.items()}
    for alpha in (1e-6, 1e-2, 13.0, 1e3, 1e6):
        for k, f in metrics.items():
            v = f(alpha * g)
            rel = abs(v - base[k]) / (abs(base[k]) + 1e-12)
            assert rel < 1e-3, (
                f'{k} not invariant at alpha={alpha} (rel={rel})'
            )


def test_epgq_scale_invariance_exact():
    """The composed EPGQ (signal/(aleatoric+epistemic)) is invariant to rescale
    when every checkpoint's gradients are scaled together (the latent/cost
    rescale the exit criterion tests)."""
    gs = [_synth(seed=s) for s in range(4)]
    mus = np.stack([E.per_component_stats(g)[0] for g in gs])  # (K,B,D)
    epi = E.epistemic_var(mus)
    mu0, var0 = E.per_component_stats(gs[0])
    base = E.epgq(mu0, var0, epi, lam=1.0)
    for alpha in (1e-6, 1e-3, 50.0, 1e4, 1e6):
        gs_a = [alpha * g for g in gs]
        mus_a = np.stack([E.per_component_stats(g)[0] for g in gs_a])
        epi_a = E.epistemic_var(mus_a)
        mu0a, var0a = E.per_component_stats(gs_a[0])
        v = E.epgq(mu0a, var0a, epi_a, lam=1.0)
        rel = abs(v - base) / (abs(base) + 1e-12)
        assert rel < 1e-3, f'EPGQ not invariant at alpha={alpha} (rel={rel})'


def test_esnr_scale_invariance_moderate():
    """Paper Code-1 ESNR has a FIXED eps floor, so it is invariant only where
    var >> eps -- check a moderate (realistic) range."""
    g = _synth(sigma=0.5)
    base = E.esnr(g)
    for alpha in (0.5, 2.0, 10.0):
        rel = abs(E.esnr(alpha * g) - base) / (abs(base) + 1e-12)
        assert rel < 1e-3


def test_epgq_penalizes_epistemic():
    """A confidently-wrong model (same signal+aleatoric, higher epistemic)
    scores LOWER -- the bias-aware behavior plain ESNR cannot express."""
    g = _synth()
    mu, var = E.per_component_stats(g)
    epi_lo = np.full_like(var, 1e-6 * var.mean())
    epi_hi = np.full_like(var, 10.0 * var.mean())
    assert E.epgq(mu, var, epi_hi) < E.epgq(mu, var, epi_lo)


def test_directional_aligned_vs_random():
    rng = np.random.default_rng(1)
    b, d, n = 3, 16, 500
    direction = rng.normal(size=(b, d))
    # samples are positive scalar multiples of one direction -> cosine ~ 1
    g_aligned = direction[None] * (1.0 + 0.05 * rng.normal(size=(n, b, 1)))
    assert E.directional_consistency(g_aligned) > 0.95
    g_rand = rng.normal(size=(n, b, d))
    assert abs(E.directional_consistency(g_rand)) < 0.2


def test_sign_agreement_bounds():
    rng = np.random.default_rng(2)
    b, d, n = 3, 16, 500
    mu = rng.normal(size=(b, d))
    g_same = np.abs(rng.normal(size=(n, b, d))) * np.sign(mu)[None]
    assert E.sign_agreement(g_same) > 0.99
    g_rand = rng.normal(size=(n, b, d))
    assert E.sign_agreement(g_rand) < 0.1


def test_whitened_finite_and_nonneg():
    g = _synth()
    v = E.whitened_snr(g)
    assert np.isfinite(v) and v >= 0.0


def test_per_component_stats_requires_two_samples():
    with pytest.raises(ValueError):
        E.per_component_stats(np.random.randn(1, 4, 3))
    with pytest.raises(ValueError):
        E.per_component_stats(np.random.randn(4, 3))  # wrong ndim


def test_epgq_all_degenerate_is_nan():
    mu = np.zeros((4, 8))
    var = np.zeros((4, 8))
    assert np.isnan(E.epgq(mu, var, np.zeros_like(var)))


def test_cost_norm_gradmag_scale_invariant():
    """A latent/cost rescale scales BOTH cost and d cost/d a by the same factor
    (cost = ||.||^2 -> k*cost, grad -> k*grad), so ||grad log cost|| is invariant."""
    rng = np.random.default_rng(3)
    n, b, d = 500, 4, 8
    g = rng.normal(size=(n, b, d))
    c = np.abs(rng.normal(size=(n, b))) + 0.5
    base = E.cost_normalized_grad_magnitude(g, c)
    for k in (1e-6, 1e-2, 17.0, 1e3, 1e6):
        v = E.cost_normalized_grad_magnitude(k * g, k * c)
        assert abs(v - base) / (abs(base) + 1e-12) < 1e-6


def test_cost_norm_gradvar_scale_invariant():
    """Cost-normalized gradient variance is invariant when cost and grad scale
    together (Var(grad) ~ c^4, cost^2 ~ c^4)."""
    rng = np.random.default_rng(11)
    n, b, d = 600, 4, 8
    g = rng.normal(size=(n, b, d))
    c = np.abs(rng.normal(size=(n, b))) + 0.5
    base = E.cost_normalized_grad_variance(g, c)
    for k in (1e-3, 1e-2, 9.0, 1e3):
        v = E.cost_normalized_grad_variance(k * g, k * c)
        assert abs(v - base) / (abs(base) + 1e-12) < 1e-6


def test_cost_norm_gradvar_noisier_is_larger():
    """More gradient variance (at the same cost scale) -> larger metric (worse)."""
    rng = np.random.default_rng(12)
    n, b, d = 600, 4, 8
    c = np.abs(rng.normal(size=(n, b))) + 1.0
    g_quiet = 0.5 * rng.normal(size=(n, b, d))
    g_noisy = 2.0 * rng.normal(size=(n, b, d))
    assert E.cost_normalized_grad_variance(
        g_noisy, c
    ) > E.cost_normalized_grad_variance(g_quiet, c)
    # collapsed -> guarded to +inf
    g_collapsed = np.full((n, b, d), 1e-9)
    assert E.cost_norm_gradvar_guarded(g_collapsed, c) == float('inf')


def test_degeneracy_guard_flags_collapse():
    """A collapsed model (near-zero-variance gradient) is guarded to +inf
    (worst), while a healthy model keeps its finite value."""
    rng = np.random.default_rng(7)
    n, b, d = 200, 4, 8
    c = np.abs(rng.normal(size=(n, b))) + 1.0
    # healthy: real across-sample variance
    g_ok = rng.normal(size=(n, b, d))
    assert np.isfinite(E.cost_norm_gradmag_guarded(g_ok, c))
    assert E.degenerate_frac(g_ok) < 0.5
    # collapsed: gradient ~constant across samples (std ~ 0) -> degenerate
    g_collapsed = np.full((n, b, d), 1e-9) + 1e-12 * rng.normal(size=(n, b, d))
    assert E.degenerate_frac(g_collapsed) > 0.5
    assert E.cost_norm_gradmag_guarded(g_collapsed, c) == float('inf')


def test_cost_norm_gradmag_steeper_is_larger():
    """A steeper cost (bigger gradient at the same cost scale) gives a larger
    metric value -> ranked worse (lower=better)."""
    rng = np.random.default_rng(4)
    n, b, d = 500, 4, 8
    c = np.abs(rng.normal(size=(n, b))) + 1.0
    g_flat = 1.0 * rng.normal(size=(n, b, d))
    g_steep = 3.0 * rng.normal(size=(n, b, d))
    assert E.cost_normalized_grad_magnitude(
        g_steep, c
    ) > E.cost_normalized_grad_magnitude(g_flat, c)
