"""Saved camera layouts stay reproducible without changing the default rig."""
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from wowskin_lab.data.cameras import camera_profile, add_recording_cameras


class CameraProfileChecks(unittest.TestCase):
    def test_default_has_two_fixed_cameras_and_one_wrist_camera(self):
        world, hand = ET.Element('worldbody'), ET.Element('body')
        add_recording_cameras(world, hand, .09)
        self.assertEqual([c.get('name') for c in world.findall('camera')], ['lab_front', 'lab_side'])
        self.assertEqual([c.get('name') for c in hand.findall('camera')], ['lab_wrist'])
        self.assertEqual([c['id'] for c in camera_profile()], [1, 2, 52])

    def test_four_view_restores_wrist_parent_and_tcp_relative_direction(self):
        world, hand = ET.Element('worldbody'), ET.Element('body')
        add_recording_cameras(world, hand, .09, profile='previous_four_view')
        self.assertEqual(len(world.findall('camera')), 3)
        self.assertEqual(hand.find('camera').get('name'), 'lab_wrist')
        rig = camera_profile('previous_four_view', tcp_z=.09)
        self.assertEqual([c['id'] for c in rig], [1, 2, 51, 52])
        np.testing.assert_allclose(rig[-1]['target_m'], [0., 0., .17])
        np.testing.assert_allclose(rig[-1]['position_m'], [.1, 0., .075])
        self.assertEqual(rig[-1]['fovy_deg'], 90)

    def test_feeding_overrides_do_not_change_cube_reference(self):
        feeding = camera_profile(scene='feeding')
        cube = camera_profile()
        np.testing.assert_allclose(feeding[0]['position_m'], [-2.45, -.75, 1.9])
        np.testing.assert_allclose(cube[0]['position_m'], [.65, -.4, .42])


if __name__ == '__main__':
    unittest.main()
