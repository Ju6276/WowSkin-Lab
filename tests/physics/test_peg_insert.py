"""Physical grasp, release and insertion; no object-pose changes after reset."""
from types import SimpleNamespace
import unittest
import mujoco
import numpy as np

from wowskin_lab.envs.pick_cube import response_configuration
from wowskin_lab.envs.peg_insert import PandaPegInsertEnv


@unittest.skipUnless(mujoco.__version__ == '3.15.0', 'Requires reduced-skin MuJoCo')
class PegPhysicsTests(unittest.TestCase):
    def environment(self, gripper='panda'):
        return PandaPegInsertEnv(response_configuration(SimpleNamespace(
            task='peg_insert', skin_dof='trilinear', ideal=True, gripper=gripper)))

    def test_both_grippers_hold_and_release_free_peg(self):
        for gripper in ('panda', 'robotiq_2f85'):
            with self.subTest(gripper=gripper):
                env = self.environment(gripper)
                initial = env.data.xpos[env.probe_body].copy()
                self.assertEqual(env.model.jnt_type[env.model.joint('peg_joint').id], mujoco.mjtJoint.mjJNT_FREE)
                self.assertNotIn('grasp_weld', env.xml)
                for _ in range(200):
                    row = env.step(env.hold_action())[-1]
                self.assertLess(abs(env.data.xpos[env.probe_body, 2]-initial[2]), .005)
                self.assertGreater(row['normal_force_n'], 1.)
                self.assertGreater(row['passive_normal_force_n'], 1.)
                self.assertGreater(row['min_volume_ratio'], 0.)
                env.command_gripper(False)
                for _ in range(70):
                    env.step(env.hold_action())
                self.assertLess(env.data.xpos[env.probe_body, 2], initial[2]-.04)

    def test_panda_inserts_using_arm_targets_only(self):
        env = self.environment()
        socket = env.data.xpos[env.model.body('hole').id].copy()
        for step in range(500):
            target = socket.copy()
            target[2] = (.22 if step < 180 else socket[2]+
                         env.peg_score.config['tip_local_m'][2]-
                         env.peg_score.config['socket_entry_local_z_m']-.025)
            env.target_xyz += np.clip((target-env.data.xpos[env.probe_body])*.1, -.0005, .0005)
            row = env.step(env.hold_action())[-1]
            if env.success:
                break
        self.assertTrue(env.success, env.task_evaluation())
        self.assertGreaterEqual(row['peg_depth_m'], .02)
        self.assertGreaterEqual(row['peg_dwell_s']+1e-9, .3)
        self.assertFalse(np.any(env.data.warning.number))
        env.reset()
        self.assertFalse(env.success)


if __name__ == '__main__':
    unittest.main()
