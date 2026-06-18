"""Unit tests for the ESNR metric (paper Code 1 / Eq. 10).

CPU-only, no world model. Validates the core math against a quadratic
objective with a known closed-form ESNR, the gradient-degeneracy caveats, and
the (B,N,H,A)->(N,B,H*A) reshape contract used by the planning-ESNR driver.
"""

import torch

from stable_worldmodel.metrics.esnr import (
    compute_esnr,
    compute_esnr_from_grads,
)


def test_quadratic_analytic_esnr():
    """J = sum_d (a_d - c_d)^2  =>  grad_d = 2(a_d - c_d).

    With a_d ~ N(mu_d, sigma^2): grad_d has mean 2(mu_d-c_d) and std 2*sigma,
    so per-component ESNR = (mu_d-c_d)^2 / sigma^2. Choosing c = mu + 1 makes
    every component's ESNR exactly 1/sigma^2.
    """
    torch.manual_seed(0)
    B, D, N = 4, 3, 40000
    sigma = 0.7
    mu = torch.zeros(B, D)
    c = mu + 1.0  # (mu - c)^2 == 1 everywhere

    a = mu[None] + sigma * torch.randn(N, B, D)
    grads = 2.0 * (a - c[None])  # (N, B, D)

    out = compute_esnr_from_grads(grads)
    analytic = 1.0 / sigma**2
    assert abs(out['esnr'] - analytic) / analytic < 0.05
    assert out['n_components'] == B * D
    assert out['degenerate_frac'] == 0.0


def test_compute_esnr_paper_signature():
    """Same target, but through the faithful Code-1 entry point with autograd."""
    torch.manual_seed(0)
    B, D, N = 4, 3, 40000
    sigma = 1.0
    mu = torch.zeros(B, D)
    c = mu + 1.0

    def J(a):
        return ((a - c) ** 2).sum(dim=-1)  # (..., B)

    def grad_f(Jfn, actions):
        a = actions[None] + sigma * torch.randn(N, *actions.shape)
        a = a.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(Jfn(a).sum(), a)
        return g  # (N, B, D)

    esnr = compute_esnr(mu, J, grad_f)
    analytic = 1.0 / sigma**2
    assert abs(esnr - analytic) / analytic < 0.05


def test_degenerate_zero_gradient_returns_zero():
    """Idealised grad==0 is the paper's ESNR->inf caveat (0/0); Code 1's +1e-8
    floor makes a true-zero-gradient estimator collapse to 0 instead."""
    grads = torch.zeros(100, 4, 3)
    out = compute_esnr_from_grads(grads)
    assert out['esnr'] == 0.0
    assert out['degenerate_frac'] == 1.0


def test_degenerate_low_variance_high_mean_blows_up():
    """The real blow-up the paper warns about: near-zero variance, non-zero
    mean (a biased-but-smooth estimator) -> ESNR explodes."""
    grads = 5.0 + 1e-9 * torch.randn(100, 4, 3)
    out = compute_esnr_from_grads(grads)
    assert out['esnr'] > 1e6


def test_reshape_contract():
    """(B,N,H,A) --permute--> (N,B,H,A) --reshape--> (N,B,H*A) must place
    g[b,n] at reshaped[n,b] flattened over (H,A) -- the mapping the planning
    driver relies on."""
    torch.manual_seed(1)
    B, N, H, A = 2, 5, 5, 10
    g = torch.randn(B, N, H, A)
    reshaped = g.permute(1, 0, 2, 3).reshape(N, B, H * A)
    for b in range(B):
        for n in range(N):
            assert torch.equal(reshaped[n, b], g[b, n].reshape(-1))


def test_requires_at_least_two_samples():
    import pytest

    with pytest.raises(ValueError):
        compute_esnr_from_grads(torch.randn(1, 4, 3))
    with pytest.raises(ValueError):
        compute_esnr_from_grads(torch.randn(4, 3))  # wrong ndim
