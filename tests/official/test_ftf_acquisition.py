"""Actual MuJoCo frames -> unchanged official writers and first preprocessor."""
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import cv2
import h5py
import numpy as np
import pandas as pd

from wowskin_lab.data.ftf_recording import FTFSimRecorder
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv
from wowskin_lab.learning.official import (process_ftf_robot, verify_reference,
                                        convert_ftf_robot_initial, prepare_ftf_annotation)


class FTFAcquisitionChecks(unittest.TestCase):
    def test_real_rendered_data_passes_original_preprocessor_without_point_substitutes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = PandaPickCubeTactileEnv(pollen=False)
            with FTFSimRecorder(env, root/'extracted_data/pick_cube', source='unit_test_acquisition_contract') as recorder:
                recorder.start()
                for i in range(120):
                    action = env.hold_action()
                    action[0] = .0004 if i < 70 else -.00025
                    recorder.step(env.step(action)[-1])
                # Success of this acquisition test is not a pick-cube demonstration.
                path = recorder.finish('success')
                env.reset()
                recorder.start()
                recorder.step(env.step(env.hold_action())[-1])
                rejected = recorder.finish('failure')
                self.assertEqual(rejected.name, 'demonstration_1_fail')
            meta = json.loads((path/'simulation_metadata.json').read_text())
            self.assertEqual(meta['point_source'].split(';')[0], 'none')
            self.assertFalse(meta['training_ready'])
            for i in (1, 2):
                with (path/f'cam_{i}_rgb_video.metadata').open('rb') as stream:
                    camera = pickle.load(stream)
                cap = cv2.VideoCapture(str(path/f'cam_{i}_rgb_video.avi'))
                self.assertTrue(cap.isOpened())
                count = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    self.assertEqual(frame.shape, (480, 640, 3))
                    count += 1
                cap.release()
                self.assertEqual(count, len(camera['timestamps']))
                self.assertGreater(count, 40)
                self.assertTrue(np.all(np.diff(camera['timestamps']) > 0))
            with (path/'states.pkl').open('rb') as stream:
                states = pickle.load(stream)
            self.assertEqual(sum(s.start_teleop for s in states), 120)
            with h5py.File(path/'reskin_sensor_values.h5') as tactile:
                self.assertEqual(tactile['sensor_values'].shape, (len(states), 15))
                self.assertEqual(tactile['sensor_history'].shape, (len(states), 40, 15))
                np.testing.assert_allclose(tactile['timestamp'][:], [s.timestamp for s in states])
                np.testing.assert_allclose(np.diff(tactile['timestamp'][:]), .01, atol=1e-10)
            process_ftf_robot(root, 'pick_cube')
            processed = root/'processed_data/pick_cube/demonstration_0'
            state_rows = pd.read_csv(processed/'states.csv')
            sensor_rows = pd.read_csv(processed/'sensor.csv')
            self.assertGreater(len(state_rows), 10)
            self.assertEqual(len(state_rows), len(sensor_rows))
            self.assertEqual(len(list((root/'processed_data/pick_cube').iterdir())), 1)
            for i in (1, 2):
                cap = cv2.VideoCapture(str(processed/f'videos/camera{i}.mp4'))
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(state_rows))
                cap.release()
            report = convert_ftf_robot_initial(root, 'pick_cube', path/'sim_camera_calibration.npy')
            self.assertFalse(report['training_ready'])
            self.assertEqual(len(report['episodes']), 1)
            self.assertEqual(report['episodes'][0]['pixels1'][0], len(state_rows))
            # Repeated conversion must not silently append initial observations.
            with self.assertRaises(FileExistsError):
                convert_ftf_robot_initial(root, 'pick_cube', path/'sim_camera_calibration.npy')
            annotation = prepare_ftf_annotation(root, 'pick_cube', root/'annotation')
            prepared = json.loads(Path(annotation['notebook']).read_text())
            original = json.loads(Path(annotation['source']).read_text())
            for i, cell in enumerate(original['cells']):
                if i != 2:  # Only the official task/path settings cell may differ.
                    self.assertEqual(cell, prepared['cells'][i])
            self.assertFalse(annotation['points_annotated'])
            verify_reference('ftf-robot')  # Writer/preprocessor sources remain clean.


if __name__ == '__main__':
    unittest.main()
