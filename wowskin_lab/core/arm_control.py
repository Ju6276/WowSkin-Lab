"""Optional pinned SERL++ task-space torque control; default remains IK/PD.

Reset pose/grasp construction retains the existing joint-PD settling procedure.
The selected torque law takes over after the outermost reset has completed.
There is no controller fallback during a rollout or on contact.
"""
from functools import lru_cache, wraps
import hashlib
import json

import mujoco
import numpy as np

from wowskin_lab.paths import ROOT


@lru_cache(maxsize=1)
def upstream_controller():
    directory = ROOT/'assets/reference/serl_controller'
    manifest = json.loads((directory/'manifest.json').read_text())
    path = directory/'impedance.py'
    source = path.read_text()
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Pinned SERL++ controller checksum mismatch')
    old, new = 'mujoco.mj_fullM(model, M, data.qM)', 'mujoco.mj_fullM(model, data, M)'
    if source.count(old) != 1:
        raise ValueError('Unexpected SERL++ mj_fullM call')
    if 'd: mujoco.' in mujoco.mj_fullM.__doc__:
        source = source.replace(old, new)
    namespace = {'__name__': 'wowskin_serl_impedance'}
    exec(compile(source, str(path), 'exec'), namespace)
    return namespace['impedance_control']


def configure_controller(cfg):
    mode = cfg.get('arm_controller', 'ik_pd')
    if mode not in ('ik_pd', 'cartesian_impedance'):
        raise ValueError(f'Unknown arm controller: {mode}')
    if mode == 'cartesian_impedance':
        cfg.setdefault('cartesian_impedance', dict(
            pos_gains=[100., 100., 100.], ori_gains=[2000., 2000., 200.],
            damping_ratio=1., nullspace_stiffness=.5, gravity_comp=True))
        cfg['arm_control_source'] = json.loads(
            (ROOT/'assets/reference/serl_controller/manifest.json').read_text())
        cfg['arm_control_initialization'] = 'Existing IK/joint-PD reset settling; runtime SERL++ torque with reset joint posture as nullspace reference'


def joint_pd_reset(method):
    """Nest correctly across task resets; latch nullspace posture once settled."""
    @wraps(method)
    def reset(self, *args, **kwargs):
        depth = getattr(self, '_arm_reset_depth', 0)
        if depth == 0:
            # Every native reset uses the same source settings, including the
            # first reset before GPU preparation. Reapply GPU options afterward.
            for name, value in getattr(self, '_gpu_source_options', {}).items():
                setattr(self.model.opt, name, value)
            # Re-upload the fully initialized task state after the outer reset.
            # CUDA Graphs retain array pointers; never reuse a stale reset state.
            self._gpu_physics = None
        self._arm_reset_depth = depth+1
        try:
            result = method(self, *args, **kwargs)
            if depth == 0 and self.cfg.get('arm_controller') == 'cartesian_impedance':
                np.testing.assert_array_equal(self.arm_q, self.arm_v)
                self.arm_nullspace_posture = self.data.qpos[self.arm_q].copy()
            return result
        finally:
            self._arm_reset_depth = depth
    return reset


def cartesian_active(env):
    return env.cfg.get('arm_controller') == 'cartesian_impedance' and not getattr(env, '_arm_reset_depth', 0)


def arm_torque(env):
    d = env.data
    if cartesian_active(env):
        return upstream_controller()(env.model, d, env.tcp_site, env.arm_v,
            pos=env.servo_xyz, ori=env.servo_rotation, joint=env.arm_nullspace_posture,
            **env.cfg['cartesian_impedance'])
    return 600*(env.joint_target-d.qpos[env.arm_q])-45*d.qvel[env.arm_v]+d.qfrc_bias[env.arm_v]
