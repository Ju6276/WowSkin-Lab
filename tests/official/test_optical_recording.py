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
                with self.assertRaisesRegex(ValueError,'optical datasets'):
                    export_dataset(root,root/'invalid_magnetic_export','visk')
                # Abort is retained as raw and never silently added to LeRobot.
                with LeRobotSimRecorder(env,directory,source='format_test_not_task_evaluation') as recorder:
                    recorder.start(); aborted=recorder.abort('intentional format test')
                self.assertEqual(json.loads((aborted/'metadata.json').read_text())['status'],'interrupted')
                self.assertEqual(validate_dataset(root/'lerobot')['episodes'],2)
