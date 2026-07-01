"""Generic 'match a target qpos' goal-conditioned termination for DMControl.

This generalizes the Reacher-specific ``ReacherQPosMatchTask`` (see
``custom_tasks/reacher.py``) so any dm_control suite domain can be turned into a
goal-conditioned reach-target task with ~no per-domain code: the episode ends as
a *success* when the current ``physics.data.qpos`` matches an externally-set
target qpos within a per-joint threshold. The target qpos is supplied at eval /
metric time (the GCS-Align protocol uses the true qpos ``goal_offset_steps``
ahead in the dataset), so the goal is reachable by construction.

Usage (per-domain wrapper, in ``compile_model``)::

    from .custom_tasks.qpos_match import make_qpos_match_task
    if self._task_name == 'qpos_match':
        task = make_qpos_match_task(cartpole.Balance)(
            swing_up=True, sparse=True, random=seed, qpos_threshold=0.05
        )
    else:
        task = cartpole.Balance(swing_up=True, sparse=True, random=seed)

The wrapper then injects the goal via the base ``DMControlWrapper.set_target_qpos``
and reports success via ``DMControlWrapper._qpos_match_terminated``.
"""

import numpy as np

_DEFAULT_QPOS_THRESHOLD = 0.05


class QPosMatchMixin:
    """Adds qpos-match success termination to ANY dm_control suite Task.

    ``target_qpos`` is set externally (via the wrapper's ``set_target_qpos``);
    while it is ``None`` the task never terminates early. ``qpos_mask`` (a boolean
    array over ``nq``, or ``None``) selects which DOFs must match — used to drop a
    DOF no controller can place to the goal (e.g. an unbounded free-root slide or
    a free-spinning joint), which would otherwise make success unreachable and
    flatten the true-distance signal GCS ranks against.

    The base task's observation/reward/reset machinery is inherited unchanged;
    only ``get_termination`` is overridden, so the env's success flag becomes the
    qpos-match condition while everything else (rendering, variation space, the
    privileged ``qpos``/``qvel`` info columns) is untouched.
    """

    target_qpos = None
    qpos_threshold = _DEFAULT_QPOS_THRESHOLD
    qpos_mask = None

    def get_termination(self, physics):
        if self.target_qpos is None:
            return None
        diff = np.abs(np.asarray(physics.data.qpos) - self.target_qpos)
        if self.qpos_mask is not None:
            diff = diff[self.qpos_mask]
        return 0.0 if np.all(diff < self.qpos_threshold) else None


def make_qpos_match_task(base_task_cls):
    """Subclass ``base_task_cls`` adding qpos-match termination.

    Per-domain knobs ``qpos_threshold`` (per-joint tolerance) and ``qpos_mask``
    (which DOFs count) are passed at construction; all other args/kwargs forward
    to the base task's ``__init__``.
    """

    class _QPosMatchTask(QPosMatchMixin, base_task_cls):
        def __init__(
            self,
            *args,
            qpos_threshold=_DEFAULT_QPOS_THRESHOLD,
            qpos_mask=None,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.target_qpos = None
            self.qpos_threshold = qpos_threshold
            self.qpos_mask = (
                np.asarray(qpos_mask, dtype=bool)
                if qpos_mask is not None
                else None
            )

    _QPosMatchTask.__name__ = f'QPosMatch{base_task_cls.__name__}'
    _QPosMatchTask.__qualname__ = _QPosMatchTask.__name__
    return _QPosMatchTask
