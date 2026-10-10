"""One demonstration produces compatible data for both acquisition routes."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import cv2
import numpy as np

from wowskin_lab.data.learning_recording import LearningSimRecorder
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv
from wowskin_lab.learning.official import process_ftf_robot


class JointAcquisitionChecks(unittest.TestCase):
    def test_spacemouse_commands_record_and_fault_aborts_without_postroll(self):
        from unittest.mock import patch
        from wowskin_lab.core.physics import configuration
        from wowskin_lab.envs.cable import PandaCableEnv
        from wowskin_lab.teleop.spacemouse import SpaceMouseController, MouseState
        cfg = configuration()
        cfg.update(timestep_s=.0001, noise_ut=0., drift_ut_per_s=0.)
        controller, state = SpaceMouseController(), MouseState()
        cfg['spacemouse'] = controller.settings()
        env = PandaCableEnv(cfg, pollen=False)
        with tempfile.TemporaryDirectory() as tmp, LearningSimRecorder(env,tmp,source='spacemouse') as recorder:
            recorder.start()
            controller.action(env,state.sample(),135,now=1.)
            for i in range(15):
                state.feed([1,*np.asarray([175,0,-80,0,0,35],dtype='<i2').tobytes()],2+i*.01)
                action=controller.action(env,state.sample(),135,now=2+i*.01)
                recorder.step(env.step(action)[-1])
            result=recorder.finish('success')
            for name,filename in [('visk','metadata.json'),('ftf','simulation_metadata.json')]:
                meta=json.loads((result[name]/filename).read_text())
                self.assertEqual(meta['source'],'spacemouse')
                self.assertEqual(meta['simulation_config']['spacemouse'],controller.settings())
            with h5py.File(result['visk']/'measurements.h5') as stream:
                self.assertEqual(stream['command_7d'].shape,(16,7))
                self.assertTrue(np.any(stream['command_7d'][1:,3:6]))
                self.assertEqual(stream['sensor_states'].shape,(16,15))
            recorder.start()
            before=env.data.time
            with patch.object(env,'step',side_effect=AssertionError('Invalid physics must not step')):
                failed=recorder.abort('test invalid FEM')
            self.assertEqual(env.data.time,before)
            self.assertFalse(recorder.active)
            self.assertFalse(any(t.is_alive() for t in recorder.ftf.workers))
            self.assertTrue(all('_ignore' in p.name for p in failed.values()))

    def test_cable_raw_recording_keeps_task_and_magnetic_samples_separate(self):
        from wowskin_lab.envs.cable import PandaCableEnv
        from wowskin_lab.envs.pick_cube import save_records
        from wowskin_lab.core.physics import configuration
        cfg = configuration()
        cfg.update(noise_ut=0., drift_ut_per_s=0.)
        env = PandaCableEnv(cfg, pollen=False)
        self.assertIsNone(env.reskin_response)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            with LearningSimRecorder(env, root, source='cable_acquisition_contract_test') as recorder:
                recorder.start()
                for _ in range(120):
                    command = env.hold_action()
                    command[:3] = [.00005, -.00005, 0.]
                    row = env.step(command)[-1]
                    recorder.step(row)
                    rows.append(row)
                result = recorder.finish('success')
            for name,filename in [('visk','metadata.json'),('ftf','simulation_metadata.json')]:
                self.assertEqual(result[name].parent.name,'cable')
                meta = json.loads((result[name]/filename).read_text())
                self.assertEqual(meta['task'],'cable')
                self.assertNotIn('reskin_model',meta['simulation_config'])
            with h5py.File(result['visk']/'measurements.h5') as v, h5py.File(result['ftf']/'reskin_sensor_values.h5') as f:
                ts = v['timestamp'][1:]
                indices = np.searchsorted(f['timestamp'][:],ts)
                np.testing.assert_allclose(f['timestamp'][:][indices],ts,atol=1e-10)
                np.testing.assert_allclose(f['sensor_values'][:][indices],v['sensor_states'][1:],atol=1e-4)
                self.assertEqual(v['sensor_states'].shape,(121,15))
                self.assertEqual(v['command_7d'].shape,(121,7))
                self.assertFalse(any('object' in key or 'cube' in key for key in v))
                for camera in (1,2,51,52):
                    stream=cv2.VideoCapture(str(result['visk']/f'cam_{camera}.avi'))
                    try:
                        ok,frame=stream.read()
                        self.assertTrue(ok)
                        self.assertEqual(frame.shape,(480,640,3))
                        self.assertEqual(int(stream.get(cv2.CAP_PROP_FRAME_COUNT)),len(v['camera_timestamp']))
                    finally:
                        stream.release()
            save_records(rows,root/'diagnostic')
            self.assertNotIn('lift_m',(root/'diagnostic/inference.csv').read_text().splitlines()[0])
            self.assertFalse((root/'visk/raw/pick_cube').exists())
            self.assertFalse((root/'ftf-robot/extracted_data/pick_cube').exists())
            process_ftf_robot(root/'ftf-robot','cable')
            self.assertTrue((root/'ftf-robot/processed_data/cable/demonstration_0/states.csv').is_file())
            np.testing.assert_array_equal(env.data.warning.number,0)

    def test_both_recordings_share_actual_samples_and_finish_without_losing_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = PandaPickCubeTactileEnv(pollen=False)
            with LearningSimRecorder(env, root, source='joint_acquisition_contract_test') as recorder:
                recorder.start()
                for _ in range(120):
                    command = env.hold_action()
                    command[:3] = [.0001, 0, .0001]
                    recorder.step(env.step(command)[-1])
                result = recorder.finish('success')
                self.assertFalse(recorder.active)
                env.reset()
                recorder.start()
                recorder.step(env.step(env.hold_action())[-1])
                failed = recorder.finish('failure')
                self.assertTrue(all(p.name.endswith('_fail') for p in failed.values()))
            vi = json.loads((result['visk']/'metadata.json').read_text())
            ft = json.loads((result['ftf']/'simulation_metadata.json').read_text())
            self.assertEqual(vi['paired_recording_id'], ft['paired_recording_id'])
            self.assertEqual(vi['simulation_config'], ft['simulation_config'])
            self.assertEqual(vi['simulation_config'], env.cfg)
            self.assertIn('seed', vi['simulation_config'])
            self.assertEqual(vi['simulation_episode_id'], ft['simulation_episode_id'])
            with h5py.File(result['visk']/'measurements.h5') as a, h5py.File(result['ftf']/'reskin_sensor_values.h5') as b:
                # FTF has additional prefix/suffix; every actual manual step
                # in ViSk must have one identical sample in the FTF recording.
                vi_time, ft_time = a['timestamp'][1:], b['timestamp'][:]
                indices = np.searchsorted(ft_time, vi_time)
                np.testing.assert_allclose(ft_time[indices], vi_time, atol=1e-10)
                np.testing.assert_allclose(b['sensor_values'][:][indices], a['sensor_states'][1:], atol=1e-4)
                np.testing.assert_allclose(np.diff(ft_time), .01, atol=1e-10)
            process_ftf_robot(root/'ftf-robot', 'pick_cube')
            self.assertTrue((root/'ftf-robot/processed_data/pick_cube/demonstration_0/states.csv').is_file())


if __name__ == '__main__':
    unittest.main()
