"""Source mechanism and local tactile attachment regression (MuJoCo 3.15)."""
from types import SimpleNamespace
import unittest
import numpy as np
import mujoco
from wowskin_lab.envs.pick_cube import response_configuration
from wowskin_lab.envs.cable import PandaCableEnv
from wowskin_lab.envs.robotiq import DRIVER
from wowskin_lab.paths import ROOT

@unittest.skipUnless(mujoco.__version__ == '3.15.0', 'Requires reduced environment')
class RobotiqChecks(unittest.TestCase):
    def test_source_servo_and_contact_release(self):
        cfg=response_configuration(SimpleNamespace(task='cable',skin_dof='trilinear',
            gripper='robotiq_2f85',cable_layout='single_clip',ideal=True,
            magnetic_fit=ROOT/'configs/magnetic_fit_pollen_nonslip.json'))
        e=PandaCableEnv(cfg,pollen=False)
        self.assertIsNone(e.reskin_response)
        aid=e.gripper_control
        self.assertAlmostEqual(e.model.actuator_gainprm[aid,0],300)
        self.assertAlmostEqual(e.model.actuator_biasprm[aid,2],-20)
        np.testing.assert_allclose(e.model.joint(DRIVER).range,[0,.8])
        self.assertEqual(e.gripper_closed_target(), .8)
        self.assertEqual(e.cfg['gripper_control_mode'],'ebim_robotiq_position')
        rows=[]
        for _ in range(600):
            t=e.data.time
            e.target_xyz[:]=[.53,-.12,.185 if t<2.4 else .1275 if t<4.2 else .18]
            if t>=3.3: e.command_gripper(t<5.4)
            rows.append(e.step(e.hold_action())[-1])
        self.assertGreater(max(r['normal_force_n'] for r in rows),.1)
        self.assertGreater(max(r['passive_normal_force_n'] for r in rows),.1)
        self.assertLess(rows[-1]['normal_force_n'],.001)
        self.assertGreater(min(r['min_volume_ratio'] for r in rows),.15)
        self.assertTrue(np.isfinite([r['magnetic_ut'] for r in rows]).all())
        self.assertTrue(all(np.isnan(r['finger_actuator_force_n']).all() for r in rows))
        self.assertIn('gripper_hinge_torque_nm', rows[-1])
        np.testing.assert_array_equal(e.data.warning.number,0)
