"""The explicit sim experiment disables Cartesian clamps, not joint physics."""
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from wowskin_lab.data.robot_state import robot_metadata
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, response_configuration


@unittest.skipUnless(mujoco.__version__ == '3.15.0', 'Requires .venv')
class CartesianLimitTests(unittest.TestCase):
    def test_default_workspace_box_is_disabled(self):
        cfg = response_configuration(SimpleNamespace(task='pick_cube', skin_dof='trilinear', ideal=True))
        env = PandaPickCubeTactileEnv(cfg, pollen=False)
        env.target_xyz[:] = [.579, 0., .2]
        applied = env.apply_action_target([.02, 0, 0, 0, 0, 0, .37])
        self.assertAlmostEqual(env.target_xyz[0], .584)
        self.assertAlmostEqual(applied[0], .005)
        contract = robot_metadata(env)['controller_contract']
        self.assertTrue(contract['cartesian_limits_enabled'])
        self.assertIsNone(contract['workspace_world_m'])

    def test_targets_cross_old_workspace_and_servo_takes_full_target(self):
        for unrestricted in (False, True):
            with self.subTest(unrestricted=unrestricted):
                cfg = response_configuration(SimpleNamespace(task='pick_cube', skin_dof='trilinear',
                    ideal=True, workspace_limits=True, no_cartesian_limits=unrestricted))
                env = PandaPickCubeTactileEnv(cfg, pollen=False)
                env.target_xyz[:] = [.579, 0., .2]
                env.servo_xyz = env.target_xyz.copy()
                before_q = env.data.qpos.copy()
                applied = env.apply_action_target([.02, 0, 0, .1, 0, 0, .37])
                env.solve_ik()
                # No step/teleport; controller only changes its own targets.
                np.testing.assert_array_equal(env.data.qpos, before_q)
                self.assertAlmostEqual(applied[6], .37)
                if unrestricted:
                    self.assertAlmostEqual(env.target_xyz[0], .599)
                    np.testing.assert_allclose(applied[:6], [.02, 0, 0, .1, 0, 0])
                    np.testing.assert_allclose(env.servo_xyz, env.target_xyz)
                    np.testing.assert_allclose(env.servo_rotation, env.target_rotation)
                else:
                    self.assertAlmostEqual(env.target_xyz[0], .58)
                    self.assertAlmostEqual(applied[3], .03)
                    self.assertLess(env.servo_xyz[0], env.target_xyz[0])
                contract = robot_metadata(env)['controller_contract']
                self.assertEqual(contract['cartesian_limits_enabled'], not unrestricted)
                if unrestricted:
                    self.assertIsNone(contract['workspace_world_m'])
                    self.assertIsNone(contract['tcp_speed_m_s'])
                self.assertEqual(contract['joint_limit_margin_rad'], .02)


if __name__ == '__main__':
    unittest.main()
