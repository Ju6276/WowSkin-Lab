"""Real rendering -> dual optical LeRobot videos/lossless arrays -> official reader."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import h5py
import numpy as np

from wowskin_lab.envs.optical import OpticalPandaEnv
from wowskin_lab.data.lerobot_recording import LeRobotSimRecorder, validate_dataset, upstream
from wowskin_lab.data.optical_recording import optical_path, validate_optical
from wowskin_lab.data.dataset_export import export_dataset
from wowskin_lab.data.episode_review import review
from wowskin_lab.learning.optical_act_data import OpticalEpisodes


class OpticalRecordingChecks(unittest.TestCase):
    def test_repeat_capture_and_official_readback(self):
        for backend in ('tacto','digit_fots','gelsight'):
            with self.subTest(backend=backend), TemporaryDirectory(prefix='wowskin_optical_format_') as directory:
                env = OpticalPandaEnv(sensor=backend)
                env.cfg['optical_ideal'] = True
                paths=[]
                for episode in range(2):
                    if episode: env.reset()
                    with LeRobotSimRecorder(env, directory, source='format_test_not_task_evaluation') as recorder:
                        recorder.start()
                        for _ in range(12):
                            a=env.hold_action(); a[2]=.0001
                            recorder.step(env.step(a)[-1])
                        paths.append(recorder.finish('success'))
                root=Path(directory)/'pick_cube'
                result=validate_dataset(root/'lerobot')
                self.assertEqual(result['episodes'],2)
                ds=upstream()('local/wowskin-lab',root=root/'lerobot',video_backend='pyav')
                self.assertNotIn('observation.tactile',ds.features)
                self.assertEqual(len(ds.meta.video_keys),5)
                data=ds.hf_dataset.with_format('numpy')
                for ep,path in enumerate(paths):
                    self.assertEqual({p.name for p in path.iterdir()},{'metadata.json','measurements.h5'})
                    meta=json.loads((path/'metadata.json').read_text())
                    with h5py.File(path/'measurements.h5') as f:
                        self.assertNotIn('sensor_states',f)
                        validate_optical(optical_path(path,meta),f['camera_timestamp'][:])
                        lo,hi=int(ds.episode_data_index['from'][ep]),int(ds.episode_data_index['to'][ep])
                        for i in range(lo,hi):
                            row=data[i];k=int(np.asarray(row['aux.raw_sample_index']).item())
                            np.testing.assert_allclose(row['action'],np.r_[f['commanded_pose'][k+1],f['commanded_gripper'][k+1]],atol=1e-6)
                            np.testing.assert_allclose(row['observation.state'],np.r_[f['cartesian_states'][k],f['gripper_states'][k]],atol=1e-6)
                for algorithm in ('visk', 'ftf'):
                    with self.assertRaisesRegex(ValueError,'optical datasets'):
                        export_dataset(root,root/('invalid_'+algorithm+'_export'),algorithm)
                # Image-tactile data remain valid LeRobot data: selecting a
                # subset must not go through the magnetic-policy rejection.
                selected = root/'optical_subset'
                original_metadata = (paths[1]/'metadata.json').read_bytes()
                report = export_dataset(root, selected, 'lerobot', episodes=[paths[1].name])
                self.assertEqual(report['validation']['episodes'], 1)
                self.assertEqual(len(report['validation']['cameras']), 5)
                meta = json.loads(original_metadata)
                with h5py.File(optical_path(paths[1], meta)) as original, h5py.File(
                        selected/'optical/chunk-000/episode_000000.h5') as subset:
                    np.testing.assert_array_equal(original['timestamp'][:], subset['timestamp'][:])
                    for side in ('left', 'right'):
                        for key in ('rgb', 'indentation_m'):
                            np.testing.assert_array_equal(original[f'frames/000000/{side}/{key}'][:],
                                                          subset[f'frames/000000/{side}/{key}'][:])
                self.assertEqual((paths[1]/'metadata.json').read_bytes(), original_metadata)
                # Offline review follows the recorded clock, including raw
                # pre/post-roll; ACT trains only on the demonstration phase.
                replay = review(selected, root/'review')
                self.assertEqual(replay['backend'], backend)
                reviewed = json.loads((root/'review/review.json').read_text())
                with h5py.File(paths[1]/'measurements.h5') as raw:
                    np.testing.assert_array_equal(reviewed['camera_time_s'], raw['camera_timestamp'][:])
                    self.assertEqual(len(reviewed['state']), len(raw['timestamp']))
                self.assertIn('marker_status', reviewed['optical'][0]['left'])
                learning = OpticalEpisodes(root/'lerobot', image_size=32, chunk_size=5,
                                           cache_dir=root/'test_image_cache')
                train, valid, stats = learning.split(1)
                self.assertTrue(train and valid)
                self.assertTrue(all(learning.rows[i][0] == 0 for i in train))
                self.assertTrue(all(learning.rows[i][0] == 1 for i in valid))
                self.assertEqual(set(learning[train[0]]), set(learning.image_keys) |
                                 {'observation.state', 'action', 'action_is_pad'})
                self.assertEqual(learning[train[-1]]['action_is_pad'].tolist(), [False, True, True, True, True])
                old_mean = stats['observation.state']['mean'].clone()
                learning.episodes[1]['observation.state'] += 1000
                np.testing.assert_array_equal(learning.split(1)[2]['observation.state']['mean'], old_mean)
                # Abort is retained as raw and never silently added to LeRobot.
                with LeRobotSimRecorder(env,directory,source='format_test_not_task_evaluation') as recorder:
                    recorder.start(); aborted=recorder.abort('intentional format test')
                self.assertEqual(json.loads((aborted/'metadata.json').read_text())['status'],'interrupted')
                self.assertEqual(validate_dataset(root/'lerobot')['episodes'],2)
