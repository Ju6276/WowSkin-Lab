import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from wowskin_lab.learning.visk_sim import MuJoCoViSkEnv, policy_increment_in_world


class ActionTransportTests(unittest.TestCase):
    def test_viewer_reset_cannot_rewrite_live_robot_state(self):
        from wowskin_lab.ui.physics_worker import PhysicsWorker
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
        data = mujoco.MjData(model)
        data.qpos[:3] = [.1, .2, .3]
        data.time = 4.2
        mujoco.mj_forward(model, data)
        transport = MuJoCoViSkEnv.__new__(MuJoCoViSkEnv)
        transport.env = SimpleNamespace(model=model, data=data)
        transport.display_model, transport.display_data = PhysicsWorker.display(transport.env)
        transport.handle = SimpleNamespace(lock=nullcontext,
            sync=lambda: mujoco.mj_resetData(transport.display_model, transport.display_data))
        transport._sync_viewer()
        self.assertEqual(transport.display_data.time, 0.)
        self.assertEqual(data.time, 4.2)
        np.testing.assert_allclose(data.qpos[:3], [.1, .2, .3])

    def test_rotated_base_and_one_increment_per_policy_query(self):
        base = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        action = np.array([.01, .02, -.03, .04, -.02, .08, .7])
        command = policy_increment_in_world(action, base, 17)
        np.testing.assert_allclose(command[:3]*17, [-.02, .01, -.03], atol=1e-15)
        np.testing.assert_allclose(base.T@command[:3]*17, action[:3], atol=1e-15)
        composed = np.linalg.matrix_power(Rotation.from_rotvec(command[3:6]).as_matrix(), 17)
        np.testing.assert_allclose(composed, Rotation.from_rotvec(action[3:6]).as_matrix(), atol=1e-14)
        self.assertEqual(command[6], .7)  # Absolute closure is never divided by ticks.

    def test_pending_motion_survives_next_query_and_zero_action(self):
        # A deliberately lagging plant: the measured pose does not yet move.
        # Policy targets must accumulate despite that lag, including a hold.
        target = np.zeros(3)
        env = SimpleNamespace(data=SimpleNamespace(time=0.), gripper_fraction=lambda target=False: .8 if target else .4)
        def step(command):
            target[:] += command[:3]
            env.data.time += .01
            return [dict(action=command.copy(), lift_m=0.)]
        env.step = step
        transport = MuJoCoViSkEnv.__new__(MuJoCoViSkEnv)
        transport.env, transport.handle = env, None
        transport.sample_hz, transport.policy_hz = 100, 6
        transport.step_index = transport.tick_count = transport.clipped_ticks = 0
        transport.max_lift, transport.max_steps = 0., 10
        transport.rows, transport.baseline = [], np.zeros(15)
        transport._raw_touch = lambda: np.zeros(15)
        transport._observation = lambda: {'goal_achieved': False}
        def robot(_):
            return dict(cartesian_states=np.zeros(6), commanded_pose=np.r_[target, np.zeros(3)])
        module = 'wowskin_lab.learning.visk_sim'
        with patch(module+'.robot_sample', side_effect=robot), patch(module+'.world_from_base', return_value=np.eye(4)), \
             patch(module+'.hold_current_pose', side_effect=lambda e: target.fill(0)) as reanchor:
            transport.step([.01, 0, 0, 0, 0, 0, .8])
            transport.step([.01, 0, 0, 0, 0, 0, .8])
            transport.step([0, 0, 0, 0, 0, 0, .8])
            reanchor.assert_not_called()
        np.testing.assert_allclose(target, [.02, 0, 0], atol=1e-12)
        self.assertEqual([r['control_ticks'] for r in transport.rows], [17, 17, 16])
        self.assertEqual(transport.tick_count, 50)


if __name__ == '__main__':
    unittest.main()
