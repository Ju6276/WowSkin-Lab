"""Keep blocked measured gripper feedback separate from the actual command."""
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import cv2
import h5py
import numpy as np

from wowskin_lab.data.visk_processing import process_visk_sim
from wowskin_lab.learning.official import prepare


class CommandLabelTests(unittest.TestCase):
    def test_export_and_official_loader_keep_command_and_feedback_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root/'data/raw/pick_cube/demonstration_0'
            raw.mkdir(parents=True)
            (raw/'metadata.json').write_text(json.dumps(dict(
                schema='wowskin_robot_dataset_v1', backend='mujoco', pose_frame='robot_base',
                status='success', tactile_channels=15, sensor_baseline_subtracted=False,
                camera_ids=[1, 2, 52], video_files={str(i): f'cam_{i}.mp4' for i in (1, 2, 52)})))
            t = np.arange(240)*.01
            camera_index = np.arange(0, 240, 3)
            measured = np.minimum(t, .4)
            command = np.minimum(t, 1.)
            with h5py.File(raw/'measurements.h5', 'w') as f:
                for name, value in dict(timestamp=t, camera_timestamp=t[camera_index],
                    cartesian_states=np.zeros((240, 6)), gripper_states=measured,
                    commanded_gripper=command, sensor_states=np.zeros((240, 15)), phase=np.ones(240)).items():
                    f[name] = value
            for camera, color in ((1, 70), (2, 140), (52, 210)):
                writer = cv2.VideoWriter(str(raw/f'cam_{camera}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
                self.assertTrue(writer.isOpened())
                for _ in camera_index:
                    writer.write(np.full((480, 640, 3), color, np.uint8))
                writer.release()
            workspace = root/'workspace'
            prepare('visk', workspace, root/'data', 'pick_cube')
            manifest = json.loads((workspace/'manifest.json').read_text())
            self.assertEqual(manifest['action_profile'], 'mujoco-command')
            self.assertEqual(manifest['camera_ids'], [1, 2, 52])
            report = process_visk_sim(workspace)
            self.assertTrue(report['validation']['command_label_check']['next_command_matches'])
            self.assertGreater(report['validation']['command_label_check']['max_measured_command_gap'], .59)
            with (root/'data/processed_data_pkl_aa/pick_cube.pkl').open('rb') as f:
                obs = pickle.load(f)['observations'][0]
            np.testing.assert_allclose(obs['gripper_states'], measured[camera_index], atol=1e-7)
            np.testing.assert_allclose(obs['commanded_gripper'], command[camera_index], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
