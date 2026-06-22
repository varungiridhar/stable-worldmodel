"""Metrics for world-model / policy analysis.

Provides ESNR (Expected Signal-to-Noise Ratio of the policy-objective gradient)
-- the training-time, action-trajectory metric from the Policy-Aware World
Models paper (see :mod:`stable_worldmodel.metrics.esnr`) -- and the Phase-2
bias-aware EPGQ candidate metrics built on the same per-sample gradient stack
(see :mod:`stable_worldmodel.metrics.epgq`).
"""

from . import epgq
from .esnr import (
    compute_esnr,
    compute_esnr_from_grads,
    action_objective_grads,
    collect_planning_grads,
    expand_info_for_samples,
    run_planning_esnr,
    sample_action_trajectories,
)

__all__ = [
    'compute_esnr',
    'compute_esnr_from_grads',
    'action_objective_grads',
    'collect_planning_grads',
    'expand_info_for_samples',
    'run_planning_esnr',
    'sample_action_trajectories',
    'epgq',
]
