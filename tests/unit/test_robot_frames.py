"""Frame invariance and rotated-base command semantics."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from wowskin_lab.data.robot_state import command_in_base, pose_in_base


class RobotFrameChecks(unittest.TestCase):
    def test_pose_roundtrip_and_camera_projection_survive_scene_transform(self):
        base = np.eye(4)
        base[:3, :3] = Rotation.from_euler('xyz', [.1, -.2, np.pi/2]).as_matrix()
        base[:3, 3] = [-1.63, -.38, 1.08485019]
        position = np.array([.4, .1, .2])
        rotation = Rotation.from_euler('xyz', [.3, -.4, .2]).as_matrix()
        p_world = base[:3, :3] @ position+base[:3, 3]
        r_world = base[:3, :3] @ rotation
        result = pose_in_base(p_world, r_world, base)
        np.testing.assert_allclose(result[:3], position, atol=1e-12)
        np.testing.assert_allclose(Rotation.from_rotvec(result[3:]).as_matrix(), rotation, atol=1e-12)
        camera_world = np.eye(4)
        camera_world[:3, :3] = Rotation.from_euler('xyz', [.5, .2, -.6]).as_matrix()
        camera_world[:3, 3] = [.2, .8, 2.]
        np.testing.assert_allclose(camera_world @ np.r_[p_world, 1],
                                   camera_world @ base @ np.r_[position, 1], atol=1e-12)

    def test_delta_translation_rotates_without_origin_offset(self):
        base = np.eye(4)
        base[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        base[:3, 3] = [10, -20, 30]
        action = [.001, 0, .002, .01, .02, .03, .7]
        np.testing.assert_allclose(command_in_base(action, base),
                                   [0, -.001, .002, .01, .02, .03, .7], atol=1e-12)
