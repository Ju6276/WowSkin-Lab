"""Real rendered frames and seven-axis control through the official ViSk loader."""
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import h5py
import numpy as np

from wowskin_lab.data.learning_recording import LearningSimRecorder
from wowskin_lab.data.dataset_export import export_dataset
from wowskin_lab.data.media import video_path
from types import SimpleNamespace
from wowskin_lab.data.visk_processing import process_visk_sim
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, response_configuration
from wowskin_lab.learning.official import prepare, verify_reference, verify_workspace


class ViSkAcquisitionChecks(unittest.TestCase):
    def test_paper_single_sensor_recording_reaches_original_converter_and_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = PandaPickCubeTactileEnv(response_configuration(SimpleNamespace(task='pick_cube', skin_dof='trilinear', ideal=True)), pollen=False)
            with LearningSimRecorder(env, root/'capture', source='acquisition_contract_test') as recorder:
                recorder.start()
                for i in range(240):
                    command = env.hold_action()
                    if 80 < i < 180:
                        command[:3] = [.0001, -.0001, .0001]
                        command[3:6] = [.0005, -.0006, .0007]
                        command[-1] = .5
                    recorder.step(env.step(command)[-1])
                # Labels validate the recording path, not physical task success.
                good = recorder.finish('success')
                env.reset()
                recorder.start()
                recorder.step(env.step(env.hold_action())[-1])
                bad = recorder.finish('failure')
            self.assertEqual(json.loads((bad/'metadata.json').read_text())['status'], 'failure')
            self.assertEqual(env.model.ncam, 3)
            np.testing.assert_array_equal(env.model.cam_bodyid[:2], [0, 0])
            self.assertNotEqual(env.model.cam_bodyid[2], 0)
            self.assertEqual(json.loads((good/"metadata.json").read_text())["camera_ids"], [1, 2, 52])
            self.assertFalse((good/'cam_51.avi').exists())
            self.assertTrue(video_path(good, 52).is_file())
            with h5py.File(good/'measurements.h5') as f:
                times = f['timestamp'][:]
                self.assertEqual(f['sensor_states'].shape, (len(times), 15))
                active = np.flatnonzero(f['phase'][:] == 1)
                self.assertEqual(len(active), 240)
                np.testing.assert_allclose(np.diff(times), .01, atol=1e-10)
                np.testing.assert_allclose(f['command_7d'][active[100]], [.0001, -.0001, .0001, .0005, -.0006, .0007, .5])
                self.assertEqual(f['camera_world_to_cv'].shape[1:], (3, 4, 4))
                self.assertFalse(any('object' in k or 'point' in k for k in f))
                raw_poses, raw_tactile = f['cartesian_states'][:], f['sensor_states'][:]
            export_dataset(root/'capture/pick_cube', root/'data', 'visk')
            workspace = root/'workspace'
            prepare('visk', workspace, root/'data', 'pick_cube')
            report = process_visk_sim(workspace)
            self.assertTrue(report['validation']['dataset_loader_validated'])
            self.assertFalse(report['full_training_completed'])
            self.assertEqual(report['demonstrations'], 1)
            self.assertEqual(report['validation']['sample_shapes']['sensor0'], [1, 15])
            self.assertEqual(report['validation']['sample_shapes']['pixels1'], [1, 3, 128, 128])
            self.assertEqual(report['validation']['sample_shapes']['pixels2'], [1, 3, 128, 128])
            self.assertNotIn('pixels51', report['validation']['sample_shapes'])
            self.assertEqual(report['validation']['sample_shapes']['pixels52'], [1, 3, 128, 128])
            self.assertEqual(report['validation']['sample_shapes']['actions'], [1, 10, 7])
            self.assertEqual(report['validation']['batch_size'], 256)
            self.assertEqual(report['validation']['subsample'], 5)
            self.assertGreaterEqual(report['validation']['sensor_scale_min'], 100)
            with Path(report['dataset']).open('rb') as stream:
                episode = pickle.load(stream)['observations'][0]
            indices = np.load(root/'data/processed_data/pick_cube/demonstration_0/acquisition_alignment.npz')['source_state_index']
            np.testing.assert_allclose(episode['sensor_states'], raw_tactile[indices], rtol=1e-6, atol=1e-5)
            np.testing.assert_allclose(episode['cartesian_states'], raw_poses[indices], rtol=1e-6, atol=1e-6)
            manifest = json.loads((workspace/'manifest.json').read_text())
            self.assertEqual(set(manifest['configured_files']), {
                'utils.py', 'cfgs/local_config.yaml', 'cfgs/suite/xarm_env_aa.yaml',
                'read_data/xarm_env_aa.py', 'envs/xarm-env/xarm_env/envs/robot_env.py',
                'convert_to_pkl.py', 'process_xarm_data.py'})
            self.assertTrue(verify_workspace(workspace)['source_matches_prepared_manifest'])
            verify_reference('visk')
            with self.assertRaises(FileExistsError):
                process_visk_sim(workspace)


if __name__ == '__main__':
    unittest.main()
