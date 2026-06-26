"""Tests for TD-MPC2's goal-conditioned latent-MSE planning cost.

The Phase-3 Part-B integration plans TD-MPC2 with the shared goal-conditioned
CEM planner using a *differentiable* goal-MSE cost in the AUGMENTED latent
space (``get_cost_goal_mse``), and probes it with the v3 action-gradient metric
(which backprops the cost wrt the action trajectory). These tests assert the
contract the planner and probe rely on:

  1. ``get_cost_goal_mse`` returns ``(B, N)``;
  2. gradients flow from the cost back to the action candidates and are finite
     (the v3 probe needs ``d cost / d actions``);
  3. ``get_cost`` auto-dispatches to the goal-MSE path when a goal is present,
     and to the native reward/value path when it is not (both kept working).
"""

import torch
from omegaconf import OmegaConf

from stable_worldmodel.wm.tdmpc2 import TDMPC2


def _tiny_cfg(action_dim=2, state_dim=2):
    """Minimal TD-MPC2 cfg with a goal-augmented 'state' encoding (TwoRoom-like).

    The augmented state is ``[state, goal]`` -> dim ``2 * state_dim``, which the
    state encoder consumes (matches the train script's extra_dims wiring).
    """
    aug_dim = 2 * state_dim
    return OmegaConf.create(
        {
            'action_dim': action_dim,
            'extra_dims': {'state': aug_dim, 'action': action_dim},
            'goal_obs_key': 'state',
            'wm': {
                'encoding': {'state': 16},
                'enc_dim': 32,
                'mlp_dim': 32,
                'simnorm_dim': 8,
                'num_q': 2,
                'num_bins': 21,
                'vmin': -6,
                'vmax': 2,
                'tau': 0.01,
                'discount': 0.99,
                'uncertainty_penalty': 0.5,
            },
        }
    )


def _make_model(seed=0):
    torch.manual_seed(seed)
    model = TDMPC2(_tiny_cfg()).eval()
    # Give the model non-trivial z-score stats (as the train script does) so the
    # normalization branch is exercised.
    model.set_state_norm(torch.zeros(4), torch.ones(4) * 2.0)
    return model


def _info(B, N, state_dim=2):
    # Raw (un-normalized) current state + goal, shaped like the CEM-expanded
    # info_dict the solver/probe pass to get_cost: (B, N, D).
    g = torch.Generator().manual_seed(123)
    state = torch.rand(B, N, state_dim, generator=g) * 100.0
    goal = torch.rand(B, N, state_dim, generator=g) * 100.0
    return {'state': state, 'goal_state': goal}


def test_goal_mse_returns_B_N():
    model = _make_model()
    B, N, H, A = 3, 5, 4, 2
    info = _info(B, N)
    actions = torch.randn(B, N, H, A)
    cost = model.get_cost_goal_mse(info, actions)
    assert cost.shape == (B, N)
    assert torch.isfinite(cost).all()
    # latent MSE-to-goal is non-negative
    assert (cost >= 0).all()


def test_goal_mse_grads_flow_to_actions():
    model = _make_model()
    B, N, H, A = 2, 6, 4, 2
    info = _info(B, N)
    actions = torch.randn(B, N, H, A, requires_grad=True)
    cost = model.get_cost_goal_mse(info, actions)
    assert cost.shape == (B, N)
    cost.sum().backward()
    assert actions.grad is not None
    assert torch.isfinite(actions.grad).all()
    # the cost must actually depend on the actions (non-zero gradient signal)
    assert actions.grad.abs().sum() > 0


def test_goal_mse_grad_via_autograd_grad():
    """The exact call the v3 probe makes: torch.autograd.grad(cost.sum(), a)."""
    model = _make_model()
    B, N, H, A = 2, 4, 5, 2
    info = _info(B, N)
    actions = torch.randn(B, N, H, A).requires_grad_(True)
    cost = model.get_cost_goal_mse(info, actions)
    (grad,) = torch.autograd.grad(cost.sum(), actions)
    assert grad.shape == actions.shape
    assert torch.isfinite(grad).all()


def test_goal_mse_packed_action_block():
    """The shared CEM planner packs frameskip into the action dim.

    For TwoRoom the per-horizon-step action is dim ``base(2) * block(5) = 10``;
    TD-MPC2's single-step dynamics must unpack it and roll H*block sub-steps.
    Assert the packed shape is accepted, returns (B, N), and grads flow.
    """
    model = _make_model()
    B, N, H, base, block = 2, 4, 5, 2, 5
    info = _info(B, N)
    actions = torch.randn(B, N, H, base * block, requires_grad=True)
    cost = model.get_cost_goal_mse(info, actions)
    assert cost.shape == (B, N)
    cost.sum().backward()
    assert actions.grad is not None
    assert torch.isfinite(actions.grad).all()
    assert actions.grad.abs().sum() > 0


def test_unpack_actions_shape():
    model = _make_model()
    B, N, H, base, block = 2, 3, 5, 2, 5
    packed = torch.randn(B, N, H, base * block)
    unpacked = model._unpack_actions(packed)
    assert unpacked.shape == (B, N, H * block, base)
    # block-1 packing is a no-op reshape
    p1 = torch.randn(B, N, H, base)
    assert model._unpack_actions(p1).shape == (B, N, H, base)


def test_get_cost_dispatches_to_goal_mse_when_goal_present():
    model = _make_model()
    B, N, H, A = 2, 4, 3, 2
    info = _info(B, N)
    actions = torch.randn(B, N, H, A)
    via_dispatch = model.get_cost(info, actions)
    via_direct = model.get_cost_goal_mse(info, actions)
    assert via_dispatch.shape == (B, N)
    assert torch.allclose(via_dispatch, via_direct)


def test_get_cost_native_path_when_no_goal():
    """Without a goal key, get_cost falls back to the native reward/value cost."""
    model = _make_model()
    B, N, H, A = 2, 4, 3, 2
    # Native path encodes only 'state' (the augmented modality) — feed it a
    # full augmented vector directly so the encoder dims match.
    aug_state = torch.randn(B, N, 4)
    info = {'state': aug_state}
    actions = torch.randn(B, N, H, A)
    cost = model.get_cost(info, actions)
    assert cost.shape == (B, N)
    assert torch.isfinite(cost).all()


def test_criterion_alias_matches_goal_mse():
    model = _make_model()
    B, N, H, A = 2, 4, 3, 2
    info = _info(B, N)
    actions = torch.randn(B, N, H, A)
    assert torch.allclose(
        model.criterion(info, actions), model.get_cost_goal_mse(info, actions)
    )


def test_state_norm_round_trips_through_state_dict():
    """The z-score buffers must survive save/load (both ckpt formats use them)."""
    model = _make_model()
    sd = model.state_dict()
    assert 'state_mean' in sd and 'state_std' in sd
    fresh = TDMPC2(_tiny_cfg()).eval()
    fresh.load_state_dict(sd)
    assert torch.allclose(fresh.state_mean, model.state_mean)
    assert torch.allclose(fresh.state_std, model.state_std)


# --------------------------------------------------------------------------- #
#  VISION-ONLY goal-IMAGE-MSE cost (the cross-paradigm-matched planning path)  #
# --------------------------------------------------------------------------- #
#
# The vision TD-MPC2 is trained on the SAME pixel dataset as the JEPA zoo and
# planned/probed with a PLAIN pixel goal-image-MSE cost (no augmented state).
# These tests assert the contract the shared CEM planner and v3 probe rely on:
#   1. get_cost_goal_image_mse returns (B, N) and is non-negative/finite;
#   2. gradients flow from the cost back to the action candidates (the v3 probe
#      needs d cost / d actions) — incl. the frameskip-packed action block;
#   3. get_cost / criterion auto-dispatch to the image-MSE path for a
#      pixel-encoded model presented with a goal image;
#   4. the pixel encoder is robust to the eval transform's image size (it
#      resizes any input to cfg.image_size internally).


def _vision_cfg(
    action_dim=2, pixel_dim=64, image_size=64, plan_cost=None, use_simnorm=None
):
    """Minimal VISION-only TD-MPC2 cfg (encoding {pixels: ...}, no state).

    ``use_simnorm`` left ``None`` keeps the default (SimNorm ON, == TD-MPC2);
    set ``False`` for the TD-MPC1 "regularizer-off" ablation (SimNorm ->
    LayerNorm).
    """
    cfg = {
        'action_dim': action_dim,
        'image_size': image_size,
        'wm': {
            'encoding': {'pixels': pixel_dim},
            'enc_dim': 32,
            'mlp_dim': 32,
            'simnorm_dim': 8,
            'num_q': 2,
            'num_bins': 21,
            'vmin': -6,
            'vmax': 2,
            'tau': 0.01,
            'discount': 0.99,
            'uncertainty_penalty': 0.5,
        },
    }
    if plan_cost is not None:
        cfg['plan_cost'] = plan_cost
    if use_simnorm is not None:
        cfg['wm']['use_simnorm'] = use_simnorm
    return OmegaConf.create(cfg)


def _make_vision_model(seed=0, **kw):
    torch.manual_seed(seed)
    return TDMPC2(_vision_cfg(**kw)).eval()


def _vision_info(B, N, size=64):
    """Sample-expanded (B, N, C, H, W) current + goal images, as the CEM solver
    / probe pass to get_cost. Channels-first, [0, 1]-ranged like the eval
    transform output (content is arbitrary for the gradient contract)."""
    g = torch.Generator().manual_seed(123)
    pixels = torch.rand(B, N, 3, size, size, generator=g)
    goal = torch.rand(B, N, 3, size, size, generator=g)
    return {'pixels': pixels, 'goal': goal}


def test_vision_goal_image_mse_returns_B_N():
    model = _make_vision_model()
    B, N, H, A = 3, 5, 4, 2
    info = _vision_info(B, N)
    actions = torch.randn(B, N, H, A)
    cost = model.get_cost_goal_image_mse(info, actions)
    assert cost.shape == (B, N)
    assert torch.isfinite(cost).all()
    assert (cost >= 0).all()  # latent MSE-to-goal is non-negative


def test_vision_goal_image_mse_grads_flow_to_actions():
    model = _make_vision_model()
    B, N, H, A = 2, 6, 4, 2
    info = _vision_info(B, N)
    actions = torch.randn(B, N, H, A, requires_grad=True)
    cost = model.get_cost_goal_image_mse(info, actions)
    cost.sum().backward()
    assert actions.grad is not None
    assert torch.isfinite(actions.grad).all()
    assert actions.grad.abs().sum() > 0  # cost actually depends on the actions


def test_vision_goal_image_mse_grad_via_autograd_grad():
    """The exact call the v3 probe makes: torch.autograd.grad(cost.sum(), a)."""
    model = _make_vision_model()
    B, N, H, A = 2, 4, 5, 2
    info = _vision_info(B, N)
    actions = torch.randn(B, N, H, A).requires_grad_(True)
    cost = model.get_cost_goal_image_mse(info, actions)
    (grad,) = torch.autograd.grad(cost.sum(), actions)
    assert grad.shape == actions.shape
    assert torch.isfinite(grad).all()


def test_vision_goal_image_mse_packed_action_block():
    """Frameskip-packed action block (base 2 x block 5 = 10) for TwoRoom."""
    model = _make_vision_model()
    B, N, H, base, block = 2, 4, 5, 2, 5
    info = _vision_info(B, N)
    actions = torch.randn(B, N, H, base * block, requires_grad=True)
    cost = model.get_cost_goal_image_mse(info, actions)
    assert cost.shape == (B, N)
    cost.sum().backward()
    assert actions.grad is not None
    assert torch.isfinite(actions.grad).all()
    assert actions.grad.abs().sum() > 0


def test_vision_get_cost_dispatches_to_image_mse_when_goal_image_present():
    """Auto-dispatch: a pixel-encoded model + 'goal' image -> image-MSE path."""
    model = _make_vision_model()
    B, N, H, A = 2, 4, 3, 2
    info = _vision_info(B, N)
    actions = torch.randn(B, N, H, A)
    via_dispatch = model.get_cost(info, actions)
    via_direct = model.get_cost_goal_image_mse(info, actions)
    assert via_dispatch.shape == (B, N)
    assert torch.allclose(via_dispatch, via_direct)


def test_vision_plan_cost_flag_selects_image_mse():
    """Explicit cfg.plan_cost='goal_image_mse' forces the image-MSE path."""
    model = _make_vision_model(plan_cost='goal_image_mse')
    B, N, H, A = 2, 4, 3, 2
    info = _vision_info(B, N)
    actions = torch.randn(B, N, H, A)
    assert torch.allclose(
        model.get_cost(info, actions),
        model.get_cost_goal_image_mse(info, actions),
    )
    assert torch.allclose(
        model.criterion(info, actions),
        model.get_cost_goal_image_mse(info, actions),
    )


def test_vision_encoder_robust_to_eval_image_size():
    """The eval/probe transform may resize to 224 (JEPA default); the CNN was
    built for cfg.image_size=64 and must resize internally, not crash."""
    model = _make_vision_model(image_size=64)
    B, N, H, A = 2, 4, 5, 2
    info = _vision_info(B, N, size=224)  # 224x224 like the JEPA eval transform
    actions = torch.randn(B, N, H, A, requires_grad=True)
    cost = model.get_cost_goal_image_mse(info, actions)
    assert cost.shape == (B, N)
    assert torch.isfinite(cost).all()
    cost.sum().backward()
    assert actions.grad is not None and actions.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
#  TD-MPC1 ablation: use_simnorm=False (SimNorm -> LayerNorm) — the            #
#  "regularizer-OFF" test bed (tdmpc1_vision.yaml).                            #
# --------------------------------------------------------------------------- #
#
# Contract:
#   1. The default (flag absent) is UNCHANGED TD-MPC2 — sim_norm is a SimNorm.
#   2. use_simnorm=False builds, and the latent normalizer (encoder + dynamics
#      activation) becomes a plain LayerNorm, NOT a SimNorm.
#   3. The differentiable planning cost still flows gradients to the actions
#      (the v3 probe's d cost / d actions) under the ablation.


def test_default_keeps_simnorm():
    """Flag absent -> unchanged TD-MPC2: the latent normalizer is a SimNorm."""
    from stable_worldmodel.wm.tdmpc2.module import SimNorm

    model = _make_vision_model()  # no use_simnorm in cfg
    assert getattr(model, 'use_simnorm', True) is True
    assert isinstance(model.sim_norm, SimNorm)
    # The dynamics' final activation is also a SimNorm.
    assert isinstance(model.dynamics[-1].act, SimNorm)


def test_use_simnorm_false_builds_with_layernorm():
    """use_simnorm=False swaps SimNorm -> LayerNorm in encoder AND dynamics."""
    import torch.nn as nn

    from stable_worldmodel.wm.tdmpc2.module import SimNorm

    model = _make_vision_model(use_simnorm=False)
    assert model.use_simnorm is False
    assert isinstance(model.sim_norm, nn.LayerNorm)
    assert not isinstance(model.sim_norm, SimNorm)
    # The dynamics' final activation (NormedLinear.act) is now a LayerNorm.
    assert isinstance(model.dynamics[-1].act, nn.LayerNorm)


def test_use_simnorm_false_cost_grads_flow_to_actions():
    """The v3-probe contract under the ablation: cost is differentiable wrt the
    action candidates (grads flow, finite, non-zero) with SimNorm OFF."""
    model = _make_vision_model(use_simnorm=False, plan_cost='goal_image_mse')
    B, N, H, A = 2, 6, 4, 2
    info = _vision_info(B, N)
    actions = torch.randn(B, N, H, A, requires_grad=True)
    cost = model.get_cost(info, actions)  # dispatches to goal_image_mse
    assert cost.shape == (B, N)
    assert torch.isfinite(cost).all()
    (grad,) = torch.autograd.grad(cost.sum(), actions)
    assert grad.shape == actions.shape
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0  # cost actually depends on the actions


def test_use_simnorm_false_state_model_grads_flow():
    """The state-path goal-MSE cost is also differentiable with SimNorm OFF
    (covers the non-pixel latent normalizer dim, latent_dim == enc dim)."""
    cfg = _tiny_cfg()
    cfg.wm.use_simnorm = False
    torch.manual_seed(0)
    model = TDMPC2(cfg).eval()
    model.set_state_norm(torch.zeros(4), torch.ones(4) * 2.0)
    assert model.use_simnorm is False
    B, N, H, A = 2, 6, 4, 2
    info = _info(B, N)
    actions = torch.randn(B, N, H, A, requires_grad=True)
    cost = model.get_cost_goal_mse(info, actions)
    (grad,) = torch.autograd.grad(cost.sum(), actions)
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0
