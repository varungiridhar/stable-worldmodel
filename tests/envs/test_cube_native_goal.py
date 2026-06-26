"""Tests for the OGBench Cube NATIVE-GOAL eval path (``set_native_task``).

These exercise the new ``CubeEnv.set_native_task(task_id)`` method, which wires
a fixed OGBench native task goal (success mocap target + rendered goal image)
into the env WITHOUT mutating the current physics state.

They require a working MuJoCo (EGL/OSMesa) render path and the ``ogbench``
package, which are only available on a GPU node — NOT on the login node. The
module-level fixture skips gracefully when either is missing, so the suite stays
green everywhere and only does real work where rendering is possible.

Run on a GPU node, e.g.::

    MUJOCO_GL=egl pytest tests/envs/test_cube_native_goal.py -q
"""

import os

import numpy as np
import pytest


# MuJoCo needs a GL backend; EGL is what eval_wm.py uses on the cluster.
os.environ.setdefault('MUJOCO_GL', 'egl')

IMG_SIZE = 224
N_TASKS = 5  # CubeEnv 'single' defines 5 native tasks (task_infos).


@pytest.fixture(scope='module')
def cube_env():
    """Build a single-cube pixel CubeEnv, skipping if it can't render.

    The construction + reset compiles the MuJoCo model and populates
    ``task_infos`` (via ``set_tasks``). Anything that prevents that on this host
    (missing ``ogbench``, no GL/EGL, no MuJoCo) yields a skip rather than a
    failure, so the test is a no-op off-GPU.
    """
    try:
        import gymnasium as gym

        import stable_worldmodel  # noqa: F401 — registers swm/OGBCube-v0

        env = gym.make(
            'swm/OGBCube-v0',
            env_type='single',
            ob_type='pixels',
            width=IMG_SIZE,
            height=IMG_SIZE,
        )
        env.reset(seed=0)
    except Exception as exc:  # noqa: BLE001 — any import/GL/compile failure -> skip
        pytest.skip(
            f'CubeEnv unavailable (no MuJoCo render / ogbench): {exc!r}'
        )

    yield env.unwrapped
    env.close()


def test_task_infos_present(cube_env):
    """The native task goals must be defined (set_native_task reads them)."""
    task_infos = getattr(cube_env, 'task_infos', None)
    assert task_infos, 'task_infos must be populated for native-goal eval'
    assert len(task_infos) == N_TASKS
    for t in task_infos:
        assert 'goal_xyzs' in t, "each task must define 'goal_xyzs'"


@pytest.mark.parametrize('task_id', list(range(N_TASKS)))
def test_set_native_task_goal_image_shape(cube_env, task_id):
    """Goal image is an RGB uint8 HWC frame at the configured camera size."""
    goal_img = cube_env.set_native_task(task_id)

    assert isinstance(goal_img, np.ndarray)
    assert goal_img.shape == (IMG_SIZE, IMG_SIZE, 3), (
        f'goal image shape {goal_img.shape} != ({IMG_SIZE}, {IMG_SIZE}, 3)'
    )
    assert goal_img.dtype == np.uint8, f'goal image dtype {goal_img.dtype}'
    # Must match a normal render() frame byte-for-byte in shape/dtype, so the
    # planner / grad-probe goal-image-MSE path is unchanged vs the offset path.
    normal_frame = cube_env.render()
    assert goal_img.shape == np.asarray(normal_frame).shape
    # Cached on the env, exactly like initialize_episode does for the dataset path.
    assert cube_env._cur_goal_rendered.shape == goal_img.shape


@pytest.mark.parametrize('task_id', list(range(N_TASKS)))
def test_set_native_task_sets_mocap_target(cube_env, task_id):
    """The success mocap target == the native task's goal_xyzs for every cube."""
    cube_env.set_native_task(task_id)

    goal_xyzs = np.asarray(
        cube_env.task_infos[task_id]['goal_xyzs'], dtype=np.float64
    )
    for i in range(cube_env._num_cubes):
        mocap_id = cube_env._cube_target_mocap_ids[i]
        np.testing.assert_allclose(
            cube_env._data.mocap_pos[mocap_id],
            goal_xyzs[i],
            atol=1e-9,
            err_msg=f'cube {i} target not at native goal for task {task_id}',
        )


@pytest.mark.parametrize('task_id', list(range(N_TASKS)))
def test_set_native_task_is_state_side_effect_free(cube_env, task_id):
    """qpos/qvel (the start/physics state) are restored after the call.

    Only the GOAL (success mocap target) is allowed to change; the simulator
    physics state (cube + arm) must be byte-for-byte identical, because eval
    resets envs to dataset INIT states and native mode must preserve them.
    """
    qpos_before = cube_env._data.qpos.copy()
    qvel_before = cube_env._data.qvel.copy()

    cube_env.set_native_task(task_id)

    np.testing.assert_array_equal(
        cube_env._data.qpos,
        qpos_before,
        err_msg=f'qpos mutated by set_native_task({task_id})',
    )
    np.testing.assert_array_equal(
        cube_env._data.qvel,
        qvel_before,
        err_msg=f'qvel mutated by set_native_task({task_id})',
    )


def test_set_native_task_records_task_id(cube_env):
    """_cur_task_id reflects the last native task selected."""
    cube_env.set_native_task(3)
    assert cube_env._cur_task_id == 3


def test_set_native_task_rejects_out_of_range(cube_env):
    with pytest.raises(ValueError):
        cube_env.set_native_task(N_TASKS)  # one past the last valid id
    with pytest.raises(ValueError):
        cube_env.set_native_task(-1)
