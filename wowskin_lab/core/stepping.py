"""CPU MuJoCo stepping for collection and teleoperation."""
import numpy as np
import mujoco


def add_arguments(parser):
    """Parsers still call this. Collection has no GPU backend switch."""
    return None


def configuration(args):
    backend = getattr(args, 'physics_backend', 'cpu')
    camera = getattr(args, 'camera_backend', 'opengl')
    gpu_requested = (backend != 'cpu' or camera != 'opengl'
                     or any(getattr(args, name, False) for name in
                            ('gpu_disable_multiccd', 'gpu_disable_nativeccd', 'gpu_disable_noslip'))
                     or getattr(args, 'gpu_solver_iterations', None) is not None
                     or getattr(args, 'gpu_solver_tolerance', None) is not None)
    if gpu_requested:
        raise ValueError('GPU collection was removed; use CPU MuJoCo')
    return dict(physics_backend='cpu', camera_backend='opengl')


def robot_control(env):
    env.control()


def prepare(env):
    if env.cfg.get('physics_backend', 'cpu') != 'cpu':
        raise RuntimeError('GPU collection was removed; use CPU MuJoCo')
    return None


def advance(env, controller=None):
    """Advance one sensor interval and leave synchronized state/contact forces."""
    start = env.data.time
    after_step = getattr(env, 'after_physics_step', None)
    for _ in range(env.steps_per_sample):
        if controller is None:
            env.controls()
        else:
            controller(env)
        mujoco.mj_step(env.model, env.data)
        if after_step is not None:
            after_step()
    expected = start + env.steps_per_sample*env.cfg['timestep_s']
    if (not np.isclose(env.data.time, expected, atol=1e-8)
            or np.any(env.data.warning.number) or not np.isfinite(env.data.qpos).all()):
        raise RuntimeError('MuJoCo instability/warning; refusing to export invalid data')
    mujoco.mj_forward(env.model, env.data)
