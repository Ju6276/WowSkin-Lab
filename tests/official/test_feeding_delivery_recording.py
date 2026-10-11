"""Scripted physical delivery -> automatic score -> reviewed LeRobot save.

This exercises the B/start and Enter/finish recorder calls, not a learned policy.
No object teleportation, success flag injection, or grasp weld is used.
"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import mujoco
import numpy as np

from wowskin_lab.data.lerobot_recording import LeRobotSimRecorder, validate_dataset
from wowskin_lab.envs.feeding import PandaFeedingEnv
from wowskin_lab.envs.pick_cube import response_configuration
from wowskin_lab.paths import ROOT


@unittest.skipUnless(mujoco.__version__ == '3.15.0', 'Requires .venv')
class FeedingDeliveryRecordingTests(unittest.TestCase):
    def test_physical_delivery_and_save_all_views(self):
        self.check_delivery('panda')

    def test_robotiq_preload_delivery_recording_and_release(self):
        self.check_delivery('robotiq_2f85')

    def check_delivery(self, gripper):
        cfg = response_configuration(SimpleNamespace(task='feeding', skin_dof='trilinear', ideal=True,
            gripper=gripper, magnetic_fit=ROOT/'configs/magnetic_fit_pollen_nonslip.json'))
        env = PandaFeedingEnv(cfg, pollen=False)
        if gripper == 'robotiq_2f85':
            source = json.loads((ROOT/'assets/ebim_feeding/config.json').read_text())['controller']
            for key, value in source.items():
                if key.startswith('gripper_'):
                    self.assertEqual(env.cfg['feeding_preload'][key], value)
            # Preload is an actual continuous position command. Native travel
            # and servo gains remain sourced, not reduced to fake stability.
            self.assertAlmostEqual(env.gripper_closed_target(), .8)
            self.assertAlmostEqual(env.model.actuator_gainprm[env.gripper_control, 0], 300.)
            self.assertLess(env.grip_target, .735)
        head = env.model.body('configurable_visual_head').id
        c = env.delivery.config
        with tempfile.TemporaryDirectory() as tmp:
            with LeRobotSimRecorder(env, tmp, source='scripted_delivery_regression_not_learned_policy') as recorder:
                self.assertFalse(recorder.active)
                recorder.start()
                self.assertTrue(recorder.active)
                arrived = False
                for _ in range(900):
                    a = env.hold_action()
                    if not arrived:
                        goal = env.data.xpos[head]+env.data.xmat[head].reshape(3, 3)@(
                            np.asarray(c['mouth_center_head_m'])+[0, -c['target_offset_outside_lips_m'], 0])
                        front = env.data.xpos[env.probe_body]+env.data.xmat[env.probe_body].reshape(3, 3)@np.mean(c['delivery_part_bounds_spoon_m'], axis=0)
                        desired = env.data.site_xpos[env.tcp_site]+goal-front
                        delta = desired-env.target_xyz
                        a[:3] = delta*min(1., .0004/max(np.linalg.norm(delta), 1e-12))
                    row = env.step(a)[-1]
                    recorder.step(row)
                    arrived = arrived or row['delivery_eligible']
                    if env.success:
                        break
                self.assertTrue(env.success)
                self.assertEqual(row['beans_in_spoon'], 6)
                # Automatic score must not stop recording. Enter invokes this
                # explicit finish only after the operator reviews the outcome.
                self.assertTrue(recorder.active)
                path = recorder.finish('success')
                self.assertFalse(recorder.active)
            meta = json.loads((path/'metadata.json').read_text())
            self.assertEqual(meta['status'], 'success')
            self.assertTrue(meta['task_evaluation']['success'])
            with h5py.File(path/'measurements.h5') as f:
                self.assertTrue(f['evaluation/delivery_success'][-1])
                self.assertEqual(f['sensor_states'].shape[1], 15)
                self.assertEqual(f['arm_qpos'].shape[1], 7)
                self.assertEqual(f['camera_base_to_cv'].shape[1], 3)
            result = validate_dataset(Path(tmp)/'feeding/lerobot')
            self.assertEqual(result['episodes'], 1)
            self.assertTrue(result['official_loader_passed'])
            self.assertEqual(len(result['depth_frame_pairs']), 1)
            self.assertEqual(len(result['cameras']), 3)
        np.testing.assert_array_equal(env.data.warning.number, 0)
        if gripper == 'robotiq_2f85':
            height = float(env.data.xpos[env.probe_body, 2])
            env.command_gripper(False)
            # The spoon can briefly rest on the lower pad during opening;
            # wait for mechanical release, not a prescribed instantaneous fall.
            for _ in range(100):
                row = env.step(env.hold_action())[-1]
                if (row['normal_force_n'] < .01 and row['passive_normal_force_n'] < .01
                        and float(env.data.xpos[env.probe_body, 2]) < height-.02
                        and float(env.data.qpos[env.finger_q[0]]) < .005):
                    break
            self.assertLess(row['normal_force_n'], .01)
            self.assertLess(row['passive_normal_force_n'], .01)
            self.assertLess(float(env.data.xpos[env.probe_body, 2]), height-.02)
            # B starts the next capture with a fresh physical preload, not
            # the previous episode's empty/open hand or cached force history.
            env.reset()
            restarted = env.step(env.hold_action())[-1]
            self.assertEqual(restarted['beans_in_spoon'], 6)
            self.assertGreater(restarted['normal_force_n'], 0.)
            self.assertGreater(restarted['passive_normal_force_n'], 0.)


if __name__ == '__main__':
    unittest.main()
