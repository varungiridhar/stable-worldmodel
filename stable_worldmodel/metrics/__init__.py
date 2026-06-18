"""Metrics for world-model / policy analysis.

Currently provides ESNR (Expected Signal-to-Noise Ratio of the policy-objective
gradient) -- the training-time, action-trajectory metric from the Policy-Aware
World Models paper. See :mod:`stable_worldmodel.metrics.esnr`.
"""

from .esnr import (
    compute_esnr,
    compute_esnr_from_grads,
    action_objective_grads,
    expand_info_for_samples,
    run_planning_esnr,
    sample_action_trajectories,
)

__all__ = [
    'compute_esnr',
    'compute_esnr_from_grads',
    'action_objective_grads',
    'expand_info_for_samples',
    'run_planning_esnr',
    'sample_action_trajectories',
]
