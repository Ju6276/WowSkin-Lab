"""Durable episode boundaries, training selection and schema semantics."""
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

from wowskin_lab.data.recording import EpisodeWriter
from wowskin_lab.legacy.policy_export import export_dataset, load_episode


def fixture(directory, status='success', offset=0., count=14):
    keys = ['pixels1', 'pixels2', 'pixels51', 'pixels52']
    writer = EpisodeWriter(directory, dict(record_hz=10, camera_keys=keys,
        color_order='RGB', sensor_baseline_subtracted=True,
        point_source='mujoco_ground_truth', center_sensor_index=0, source='unit_test'))
    for i in range(count):
        sample = dict(time_s=i*.1, cartesian_states=np.array([offset+i*i*.001, 0, 0, 0, 0, 0]),
                      gripper_states=i/(count-1), sensor_states=np.r_[i+offset, 0., 0., np.ones(12)*100],
                      robot_points_world_m=np.full((9, 3), i*.01),
                      object_points_world_m=np.full((8, 3), i*.01+.5))
        for key in keys:
            image = np.zeros((128, 128, 3), np.uint8)
            image[:, :, 0] = i  # sentinel: no RGB/BGR swap
            sample[key] = image
            sample[f'robot_tracks_{key}'] = np.full((9, 2), i)
            sample[f'object_tracks_{key}'] = np.full((8, 2), i)
        writer.append(sample)
    writer.finish(status)
    return writer.path


class RecordingChecks(unittest.TestCase):
    def test_reset_and_nonfinite_frames_rejected_without_partial_write(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = EpisodeWriter(directory, {})
            sample = dict(time_s=.1, cartesian_states=np.zeros(6), sensor_states=np.zeros(15))
            writer.append(sample)
            with self.assertRaises(ValueError):
                writer.append(dict(sample, time_s=0))
            with self.assertRaises(ValueError):
                writer.append(dict(sample, time_s=.2, sensor_states=np.full(15, np.nan)))
            self.assertEqual(len(list((writer.path/'frames').glob('*.npz'))), 1)
            self.assertEqual(json.loads((writer.path/'metadata.json').read_text())['status'], 'in_progress')
            writer.finish('interrupted')
            self.assertEqual(json.loads((writer.path/'metadata.json').read_text())['frame_count'], 1)
            with self.assertRaises(RuntimeError):
                writer.append(dict(sample, time_s=.2))

    def test_success_filter_selected_stats_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = fixture(root)
            fixture(root, offset=20.)
            fixture(root, status='failure', offset=1000.)
            dst = root/'selected.pkl'
            report = export_dataset(root, dst, 'visk', episode_ids=[first.name])
            self.assertEqual(report['episodes'], 1)
            with dst.open('rb') as f:
                result = pickle.load(f)
            self.assertLess(result['max_cartesian'][0], 1.)
            np.testing.assert_array_equal(result['observations'][0]['pixels1'][3,0,0], [3,0,0])
            self.assertEqual(result['observations'][0]['sensor_states'].shape, (14, 15))
            with self.assertRaises(FileExistsError):
                export_dataset(root, dst, 'visk')
            self.assertEqual(export_dataset(root, root/'all_success.pkl', 'visk')['episodes'], 2)

    def test_ftf_uses_center_only_and_preserves_world_points(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            dst = root/'ftf.pkl'
            export_dataset(root, dst, 'ftf')
            with dst.open('rb') as f:
                result = pickle.load(f)
            obs = result['observations'][0]
            np.testing.assert_allclose(obs['sensor_states'], np.arange(14))
            np.testing.assert_allclose(obs['gripper_states'][[0,-1]], [-1,1])
            self.assertEqual(obs['robot_tracks_3d_pixels1'].shape, (14,9,3))
            np.testing.assert_array_equal(obs['robot_tracks_3d_pixels1'], obs['robot_tracks_3d_pixels2'])
            self.assertNotIn('pixels51', obs)

    def test_incomplete_episode_and_unknown_selection_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = fixture(root)
            with self.assertRaises(ValueError):
                export_dataset(root, root/'bad.pkl', 'visk', episode_ids=['missing'])
            meta = json.loads((path/'metadata.json').read_text())
            meta['frame_count'] += 1
            (path/'metadata.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                load_episode(path)


if __name__ == '__main__':
    unittest.main()
