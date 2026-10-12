"""Contact-triggered hold is an opt-in controller, not an FEM repair."""
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from wowskin_lab.core.finger_contact import FingerContactStop
from wowskin_lab.envs.cable import PandaCableEnv
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, response_configuration


class FingerContactChecks(unittest.TestCase):
    def test_hold_does_not_ratchet_and_opening_releases(self):
        stop = FingerContactStop()
        self.assertEqual(stop.apply(1., .4, False), 1.)  # An object blocks closure.
        self.assertEqual(stop.apply(1., .8, True), .8)
        self.assertEqual(stop.apply(1., .85, True), .8)
        self.assertEqual(stop.apply(1., .79, False), .8)  # Pads elastically recover.
        self.assertEqual(stop.apply(.7, .8, True), .7)  # Opening while touching.
        self.assertFalse(stop.active)

    @unittest.skipUnless(mujoco.__version__ == '3.15.0', 'Uses mounted reduced skin')
    def test_empty_close_hold_open_and_reclose(self):
        for task, gripper, env_type in (
            ('pick_cube', 'panda', PandaPickCubeTactileEnv),
            ('cable', 'panda', PandaCableEnv),
            ('cable', 'robotiq_2f85', PandaCableEnv),
        ):
            with self.subTest(task=task, gripper=gripper):
                cfg = response_configuration(SimpleNamespace(task=task, gripper=gripper,
                    cable_layout='single_clip', ideal=True, finger_contact_stop=True, pad_backing_contact=True))
                env = env_type(cfg, pollen=False)
                held = []
                for i in range(120):
                    row = env.step([0, 0, 0, 0, 0, 0, 1])[-1]
                    self.assertTrue(np.isfinite(row['magnetic_ut']).all())
                    self.assertGreater(min(row['min_volume_ratio'], row['passive_min_volume_ratio']), 0.)
                    if row['finger_contact_stop_active']:
                        held.append(row['action'][-1])
                        self.assertEqual(row['requested_gripper_closed_fraction'], 1.)
                self.assertGreater(len(held), 10)
                self.assertLess(held[0], 1.)
                np.testing.assert_allclose(held, held[0], atol=1e-12)
                # Let the sourced position servo settle; no qpos/velocity reset.
                for _ in range(120):
                    row = env.step(np.zeros(7))[-1]
                self.assertFalse(row['finger_contact_stop_active'])
                self.assertLess(env.gripper_fraction(), .04)
                for _ in range(120):
                    row = env.step([0, 0, 0, 0, 0, 0, 1])[-1]
                self.assertTrue(row['finger_contact_stop_active'])
                np.testing.assert_array_equal(env.data.warning.number, 0)
                env.reset()
                self.assertFalse(env.finger_contact_stop.active)

    def test_default_is_not_a_hidden_stop(self):
        cfg = response_configuration(SimpleNamespace(task='cable'))
        self.assertFalse(cfg.get('finger_contact_stop', False))
        self.assertFalse(cfg.get('pad_backing_contact', False))

    @unittest.skipUnless(mujoco.__version__ == '3.15.0', 'Uses mounted reduced skin')
    def test_backing_support_under_full_close_without_contact_stop(self):
        # Keep the full original target and motor gains. This regression must
        # not pass by activating the optional hold or weakening the gripper.
        for task, gripper, env_type in (
            ('pick_cube', 'panda', PandaPickCubeTactileEnv),
            ('cable', 'panda', PandaCableEnv),
            ('cable', 'robotiq_2f85', PandaCableEnv),
        ):
            with self.subTest(task=task, gripper=gripper):
                cfg = response_configuration(SimpleNamespace(task=task, gripper=gripper,
                    cable_layout='single_clip', ideal=True, pad_backing_contact=True))
                env = env_type(cfg, pollen=False)
                for name in ('standard_shell', 'silicone_backing'):
                    self.assertEqual(env.model.geom_conaffinity[env.model.geom(name).id] & 2, 2)
                for command, steps in ((1., 120), (0., 120)):
                    for _ in range(steps):
                        row = env.step([0, 0, 0, 0, 0, 0, command])[-1]
                        self.assertFalse(row['finger_contact_stop_active'])
                        self.assertEqual(row['action'][-1], command)
                        self.assertEqual(env.gripper_fraction(target=True), command)
                        self.assertGreater(min(row['min_volume_ratio'], row['passive_min_volume_ratio']), 0.)
                        self.assertTrue(np.isfinite(row['magnetic_ut']).all())
                self.assertLess(env.gripper_fraction(), .04)
                np.testing.assert_array_equal(env.data.warning.number, 0)


if __name__ == '__main__':
    unittest.main()
