"""GPU state -> real RGB-D/tactile files -> offline review, without success labels."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import cv2
import h5py
import numpy as np

from wowskin_lab.data.raw_recording import RawRobotRecorder
from wowskin_lab.data.media import video_path, depth_path
from wowskin_lab.data.episode_review import review
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, response_configuration
from wowskin_lab.envs.feeding import PandaFeedingEnv
from wowskin_lab.paths import ROOT


def setUpModule():
    try:
        import warp as wp
        import mujoco_warp
    except ImportError:
        raise unittest.SkipTest('Optional NVIDIA dependencies: requirements/gpu.txt')
    if not wp.is_cuda_available():
        raise unittest.SkipTest('NVIDIA CUDA device required')


class GpuRecordingChecks(unittest.TestCase):
    def test_sensors_write_real_synchronized_observations(self):
        for sensor in ('anyskin', 'tacto', 'digit_fots', 'gelsight'):
            with self.subTest(sensor=sensor), TemporaryDirectory() as tmp:
                args = dict(task='pick_cube' if sensor == 'anyskin' else 'feeding', ideal=True, physics_backend='warp',
                    gpu_disable_multiccd=True, gpu_disable_nativeccd=True, gpu_disable_noslip=True)
                if sensor == 'anyskin':
                    args['magnetic_fit'] = ROOT/'configs/magnetic_fit_pollen_nonslip.json'
                else:
                    args['sensor_backend'] = sensor
                factory = PandaPickCubeTactileEnv if sensor == 'anyskin' else PandaFeedingEnv
                env = factory(response_configuration(SimpleNamespace(**args)), pollen=False)
                with RawRobotRecorder(env, Path(tmp)/'capture', source='gpu_transport_regression') as recorder:
                    recorder.start()
                    for _ in range(20):
                        action = env.hold_action()
                        action[2] = .0001
                        recorder.step(env.step(action)[-1])
                    path = recorder.finish('unreviewed')
                meta = json.loads((path/'metadata.json').read_text())
                self.assertEqual(meta['status'], 'unreviewed')
                self.assertEqual(meta['simulation_config']['physics_execution']['backend'], 'mujoco_warp')
                self.assertEqual(meta['physics_diagnostics']['overflow_flags'], 0)
                with h5py.File(path/'measurements.h5') as samples:
                    np.testing.assert_allclose(np.diff(samples['timestamp'][:]), .01, atol=1e-10)
                    np.testing.assert_array_equal(samples['arm_qpos'][-1], env.data.qpos[env.arm_q])
                    frame_count = len(samples['camera_timestamp'])
                    self.assertGreater(frame_count, 3)
                    if sensor != 'anyskin':
                        # Pregrasped spoon supplies actual loaded contacts. Empty
                        # cube baselines cannot detect lost GPU contact IDs.
                        with h5py.File(path/meta['optical']['path']) as optical:
                            times = samples['timestamp'][:]
                            recorded = []
                            for i, timestamp in enumerate(optical['timestamp']):
                                index = np.flatnonzero(np.isclose(times, timestamp, atol=1e-9, rtol=0))
                                self.assertEqual(len(index), 1)
                                forces = [optical[f'frames/{i:06d}/{side}'].attrs['normal_force_n']
                                          for side in ('left', 'right')]
                                np.testing.assert_allclose(forces, samples['optical_contact_force_n'][index[0]],
                                                           rtol=1e-6, atol=1e-8)
                                recorded.append(forces)
                            self.assertGreater(np.max(recorded), .05)
                self.assertTrue(depth_path(path, meta).exists())
                for camera in meta['camera_ids']:
                    cap = cv2.VideoCapture(str(video_path(path, camera, meta)))
                    self.assertTrue(cap.isOpened())
                    count = 0
                    while True:
                        ok, rgb = cap.read()
                        if not ok:
                            break
                        self.assertEqual(rgb.shape, (480, 640, 3))
                        self.assertGreater(float(np.std(rgb)), 1.)
                        count += 1
                    cap.release()
                    self.assertEqual(count, frame_count)
                if sensor != 'anyskin':
                    from .test_environment import check_contact_force_readback
                    check_contact_force_readback(env)
                    review(path, Path(tmp)/'review')
                    self.assertTrue((Path(tmp)/'review/index.html').is_file())
