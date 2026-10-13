"""Frame transforms, control clocks, preprocessing and action chunk bounds."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial.transform import Rotation

from wowskin_lab.learning.optical_act_data import OpticalEpisodes, absolute_command_delta, chunk_indices, control_deadline, image_tensor, relative_rotation_pose
from wowskin_lab.data.episode_review import synchronized_indices


class OpticalTransportChecks(unittest.TestCase):
    def test_incomplete_dataset_is_rejected_before_loading(self):
        with TemporaryDirectory() as directory:
            (Path(directory)/'INCOMPLETE').touch()
            with self.assertRaisesRegex(ValueError, 'INCOMPLETE'):
                OpticalEpisodes(directory)

    def test_orientation_crossing_pi_is_continuous_and_invertible(self):
        poses = np.array([[.4, 0, .2, np.pi-1e-5, 0, 0, .5],
                          [.4, 0, .2, -np.pi+1e-5, 0, 0, .5]])
        reference = [np.pi, 0, 0]
        relative = relative_rotation_pose(poses, reference)
        self.assertLess(abs(relative[0, 3]-relative[1, 3]), 3e-5)
        recovered = relative_rotation_pose(relative, reference, inverse=True)
        np.testing.assert_allclose(Rotation.from_rotvec(recovered[:, 3:6]).as_matrix(),
                                   Rotation.from_rotvec(poses[:, 3:6]).as_matrix(), atol=2e-7)
        np.testing.assert_allclose(recovered[:, [0, 1, 2, 6]], poses[:, [0, 1, 2, 6]])

    def test_absolute_goal_in_translated_rotated_base_is_not_accumulated(self):
        base = np.eye(4)
        base[:3, :3] = Rotation.from_euler('zy', [.7, -.2]).as_matrix()
        base[:3, 3] = [-1.6, .4, .8]
        action = np.array([.4, -.1, .2, .1, .2, -.3, .7])
        current = np.array([-1.3, .2, 1.])
        orientation = Rotation.from_euler('xy', [.2, .6]).as_matrix()
        delta = absolute_command_delta(action, base, current, orientation)
        goal_xyz = current+delta[:3]
        goal_rotation = orientation@Rotation.from_rotvec(delta[3:6]).as_matrix()
        np.testing.assert_allclose(goal_xyz, base[:3, :3]@action[:3]+base[:3, 3])
        np.testing.assert_allclose(goal_rotation, base[:3, :3]@Rotation.from_rotvec(action[3:6]).as_matrix())
        np.testing.assert_allclose(absolute_command_delta(action, base, goal_xyz, goal_rotation),
                                   [0., 0., 0., 0., 0., 0., .7], atol=1e-12)

    def test_exact_deadlines_and_rejection(self):
        np.testing.assert_array_equal(np.diff([control_deadline(i) for i in range(10)]), [4, 3, 3]*3)
        self.assertEqual(control_deadline(30*3600), 100*3600)
        with self.assertRaises(ValueError):
            control_deadline(-1)
        with self.assertRaises(ValueError):
            absolute_command_delta([np.nan]*7, np.eye(4), np.zeros(3), np.eye(3))

    def test_chunk_padding_stays_in_episode(self):
        selected, padded = chunk_indices(2, 4, 5)
        np.testing.assert_array_equal(selected, [2, 3, 3, 3, 3])
        np.testing.assert_array_equal(padded, [False, False, True, True, True])

    def test_letterbox_preserves_rgb_and_aspect(self):
        red = np.zeros((32, 16, 3), dtype='uint8')
        red[..., 0] = 255
        result = image_tensor(red, 32).numpy()
        self.assertEqual(result.shape, (3, 32, 32))
        np.testing.assert_array_equal(result[:, 0, 0], [127]*3)
        np.testing.assert_array_equal(result[:, 16, 16], [255, 0, 0])

    def test_review_uses_actual_camera_clock_and_rejects_misalignment(self):
        times = 4000+np.arange(12)*.01
        cameras = times[[0, 4, 7, 10]]
        np.testing.assert_array_equal(synchronized_indices(times, cameras), [0, 4, 7, 10])
        with self.assertRaisesRegex(ValueError, 'do not align'):
            synchronized_indices(times, cameras+.002)
        with self.assertRaisesRegex(ValueError, 'strictly increase'):
            synchronized_indices(times, cameras[[0, 1, 1, 3]])
