"""Cable scene isolation and cross-model channel contract regression checks."""
import json
import hashlib
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from wowskin_lab.paths import ROOT


class CableChecks(unittest.TestCase):
    def test_single_clip_keeps_fixed_adapter_and_sourced_finger_servo(self):
        from wowskin_lab.envs.cable import PandaCableEnv
        from wowskin_lab.envs.pick_cube import response_configuration
        from types import SimpleNamespace
        import mujoco
        cfg = response_configuration(SimpleNamespace(task='cable', cable_layout='single_clip', ideal=True))
        env = PandaCableEnv(cfg, pollen=False)
        root = ET.fromstring(env.xml)
        self.assertIsNotNone(root.find(".//body[@name='adapter_0']"))
        self.assertIsNone(root.find(".//body[@name='round_peg_0']"))
        self.assertIsNone(root.find(".//body[@name='round_peg_1']"))
        self.assertEqual(env.model.body_dofnum[env.model.body('wire_B_first').id], 0)
        source = ET.parse(ROOT/'assets/reference/robosuite_panda/panda_gripper.xml').getroot()
        for i, actuator in enumerate(env.gripper_controls, 1):
            original = source.find(f"actuator/position[@joint='finger_joint{i}']")
            kp = float(original.get('kp'))
            np.testing.assert_allclose(env.model.actuator_forcerange[actuator], [-20, 20])
            self.assertEqual(env.model.actuator_biasprm[actuator, 1], -kp)
        # Identical held opening error: each sourced servo generates 5 N,
        # from a 5 mm position error, with the local travel bound as target.
        env.data.qpos[env.finger_q] = env.model.jnt_range[env.finger_joints[0], 0] + .005
        env.data.qvel[:] = 0
        env.command_gripper(True)
        env.control()
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_allclose(env.data.qfrc_actuator[env.finger_v], [-5, -5], atol=1e-5)
        # Opening reverses force; neither finger is welded or hard locked.
        env.command_gripper(False); env.control()
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_allclose(env.data.qfrc_actuator[env.finger_v], [20, 20], atol=1e-5)

    @unittest.skipUnless((ROOT/'assets/ebim/manifest.json').exists(), 'Download optional EBiM assets first')
    def test_scene_and_seven_action_interface_do_not_export_cube_success(self):
        from wowskin_lab.envs.cable import PandaCableEnv,CABLE_COUNT,CABLE_LENGTH
        env=PandaCableEnv(pollen=False)
        root=ET.fromstring(env.xml)
        self.assertIsNone(root.find(".//body[@name='cube']"))
        self.assertEqual(len(env.contact_geoms),env.cable_count-1)
        self.assertGreater(env.cable_length,.7)
        self.assertIsNotNone(root.find('.//composite').get('vertex'))
        self.assertEqual(env.model.body_dofnum[env.model.body('wire_B_first').id],0)
        self.assertEqual(env.model.nflex,2)
        from wowskin_lab.envs.franka import mount_reference
        open_gap=2*(.04+mount_reference(env.cfg)['position_m'][1]-env.cfg['total_height_m'])
        lower=env.model.jnt_range[env.finger_joints[0],0]
        self.assertLess(open_gap+2*lower,.001)  # No artificial 2 mm closure gap.
        for body in ('round_peg_0','round_peg_1','cclip_0','wowskin_mount','silicone_mount'):
            self.assertIsNotNone(root.find(f".//body[@name='{body}']"))
        start=env.target_xyz.copy()
        action=env.hold_action();action[0]=.001
        obs, reward, _, _, row=env.step(action)
        self.assertNotIn('cube_position_m',obs)
        self.assertNotIn('cube_xyz_m',row)
        self.assertFalse(row['success'])
        self.assertEqual(reward,0.)
        np.testing.assert_allclose(env.target_xyz,start+[.001,0,0])
        np.testing.assert_array_equal(env.data.warning.number,0)
        np.testing.assert_array_equal(env.data.xfrc_applied,0)
        self.assertEqual(row['cable_centerline_m'].shape,(48,3))
        env.reset()
        self.assertEqual(len(env.contact_geoms),48)
        self.assertEqual(env.data.time,0.)
