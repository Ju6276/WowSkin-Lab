"""Shared physics selection for native CPU and optional CUDA collection."""
import numpy as np
import mujoco


def add_arguments(parser):
    parser.add_argument('--physics-backend', choices=('cpu', 'warp'), default='cpu',
                        help='CPU MuJoCo (Mac/Linux), or NVIDIA CUDA via MuJoCo Warp')
    parser.add_argument('--camera-backend', choices=('opengl', 'warp'), default='opengl',
                        help='Scene RGB-D renderer; Warp reads CUDA state directly and changes image appearance. Source tactile optics are retained.')
    parser.add_argument('--gpu-disable-multiccd', action='store_true',
                        help='Explicit GPU compatibility choice; retain margins but disable MULTICCD')
    parser.add_argument('--gpu-disable-nativeccd', action='store_true',
                        help='Explicitly disable native CCD to retain nonzero mesh contact margins')
    parser.add_argument('--gpu-disable-noslip', action='store_true',
                        help='Explicitly omit the extra no-slip solve unsupported by Warp')
    parser.add_argument('--gpu-solver-tolerance', type=float,
                        help='Explicit positive GPU constraint tolerance; omitted retains source tolerance')
    parser.add_argument('--gpu-solver-iterations', type=int,
                        help='Explicit GPU constraint iteration limit; omitted retains source limit')


def configuration(args):
    backend = getattr(args, 'physics_backend', 'cpu')
    camera = getattr(args, 'camera_backend', 'opengl')
    if camera not in ('opengl', 'warp') or (camera == 'warp' and backend != 'warp'):
        raise ValueError('Warp scene cameras require --physics-backend warp')
    multiccd = bool(getattr(args, 'gpu_disable_multiccd', False))
    nativeccd = bool(getattr(args, 'gpu_disable_nativeccd', False))
    noslip = bool(getattr(args, 'gpu_disable_noslip', False))
    iterations = getattr(args, 'gpu_solver_iterations', None)
    tolerance = getattr(args, 'gpu_solver_tolerance', None)
    if tolerance is not None and (not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('GPU solver tolerance must be finite and positive')
    if backend not in ('cpu', 'warp'):
        raise ValueError('Unknown physics backend')
    if iterations is not None and iterations < 1:
        raise ValueError('GPU solver iterations must be positive')
    if backend == 'cpu' and (multiccd or nativeccd or noslip or iterations is not None or tolerance is not None):
        raise ValueError('GPU options require --physics-backend warp')
    return dict(physics_backend=backend, camera_backend=camera, gpu_disable_multiccd=multiccd, gpu_disable_noslip=noslip,
                gpu_disable_nativeccd=nativeccd,
                gpu_solver_iterations=iterations, gpu_solver_tolerance=tolerance)


def robot_control(env):
    env.control()


def prepare(env):
    if env.cfg.get('physics_backend', 'cpu') == 'cpu':
        return None
    if getattr(env, '_gpu_physics', None) is None:
        try:
            from .gpu.backend import WarpBackend
        except ImportError as error:
            raise RuntimeError('Warp physics requires an NVIDIA CUDA machine and requirements/gpu.txt; use --physics-backend cpu on Mac') from error
        env._gpu_physics = WarpBackend(env)
    return env._gpu_physics


def advance(env, controller=None):
    """Advance one sensor interval and leave synchronized state/contact forces."""
    backend = prepare(env)
    if backend is not None:
        backend.advance(controller)
        return
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
