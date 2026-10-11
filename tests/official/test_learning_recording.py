"""Shared dataset -> official exports, with real MuJoCo samples and images."""
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from wowskin_lab.data.dataset_export import export_dataset
from wowskin_lab.data.learning_recording import LearningSimRecorder
from wowskin_lab.data.media import video_path
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, response_configuration
from wowskin_lab.envs.feeding import PandaFeedingEnv
from wowskin_lab.learning.official import prepare, process_ftf_robot
from wowskin_lab.data.visk_processing import process_visk_sim
from wowskin_lab.paths import ROOT


@unittest.skipUnless(mujoco.__version__ == '3.15.0', 'Run in .venv')
class JointAcquisitionChecks(unittest.TestCase):
    def env(self, task):
        cfg = response_configuration(SimpleNamespace(task=task, skin_dof='trilinear', ideal=True,
            magnetic_fit=ROOT/'configs/magnetic_fit_pollen_nonslip.json'))
        return (PandaFeedingEnv if task == 'feeding' else PandaPickCubeTactileEnv)(cfg, pollen=False)

    def test_single_dataset_joints_base_coordinates_and_both_official_routes(self):
        for task in ('pick_cube', 'feeding'):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as tmp:
                root, env = Path(tmp), self.env(task)
                joints, velocities, world_positions = [], [], []
                with LearningSimRecorder(env, root/'episodes', source='acquisition_contract_test') as recorder:
                    recorder.start()
                    for _ in range(240):
                        command = env.hold_action()
                        command[2] = .0001  # Actual >1 cm motion, as required by FTF's original pause detector.
                        recorder.step(env.step(command)[-1])
                        joints.append(env.data.qpos[env.arm_q].copy())
                        velocities.append(env.data.qvel[env.arm_v].copy())
                        world_positions.append(env.data.site_xpos[env.tcp_site].copy())
                    path = recorder.finish('success')
                self.assertFalse((root/'visk').exists())
                self.assertFalse((root/'ftf-robot').exists())
                self.assertFalse(list(path.glob('*.avi')))
                self.assertTrue(all(video_path(path, camera).is_file() for camera in (1, 2)))
                meta = json.loads((path/'metadata.json').read_text())
                self.assertEqual(meta['pose_frame'], 'robot_base')
                self.assertEqual(meta['arm_joint_names'], [f'joint{i}' for i in range(1, 8)])
                self.assertEqual(meta['magnetic_source'], 'fitted_dipole')
                with h5py.File(path/'measurements.h5') as f:
                    times, poses = f['timestamp'][:], f['cartesian_states'][:]
                    phase = f['phase'][:]
                    np.testing.assert_array_equal(f['arm_qpos'][:][phase == 1], joints)
                    np.testing.assert_array_equal(f['arm_qvel'][:][phase == 1], velocities)
                    np.testing.assert_allclose(np.diff(times), .01, atol=1e-9)
                    transforms = f['world_from_base'][:][phase == 1]
                    reconstructed = np.einsum('nij,nj->ni', transforms[:, :3, :3], poses[phase == 1, :3])+transforms[:, :3, 3]
                    np.testing.assert_allclose(reconstructed, world_positions, atol=1e-10)
                    for i in (0, 100, 200):
                        transform = f['world_from_base'][i]
                        rbase = Rotation.from_rotvec(poses[i, 3:]).as_matrix()
                        rworld = Rotation.from_rotvec(f['cartesian_states_world'][i, 3:]).as_matrix()
                        np.testing.assert_allclose(transform[:3, :3] @ rbase, rworld, atol=1e-10)
                    camera_base = f['camera_base_to_cv'][:]
                    np.testing.assert_allclose(camera_base, f['camera_world_to_cv'][:] @ f['world_from_base'][0], atol=1e-10)
                    if task == 'feeding':
                        self.assertGreater(np.linalg.norm(poses[0, :3]-f['cartesian_states_world'][0, :3]), 1.)
                originals = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}
                ftf = export_dataset(path.parent, root/'ftf_export', 'ftf')
                exported = Path(ftf['exported_episodes'][0])
                with (exported/'states.pkl').open('rb') as stream:
                    states = pickle.load(stream)
                np.testing.assert_allclose([s.pos for s in states], poses[40:, :3])
                np.testing.assert_allclose([s.timestamp for s in states], times[40:])
                self.assertEqual(sum(s.start_teleop for s in states), 240)
                calibration = np.load(exported/'sim_camera_calibration.npy', allow_pickle=True).item()
                for cid in (1, 2):
                    np.testing.assert_allclose(calibration[f'cam_{cid}']['ext'], camera_base[0, cid-1])
                with h5py.File(path/'measurements.h5') as raw, h5py.File(exported/'reskin_sensor_values.h5') as tactile:
                    np.testing.assert_allclose(tactile['sensor_values'][:], raw['sensor_states'][40:], rtol=1e-6)
                    np.testing.assert_allclose(tactile['sensor_history'][0], raw['sensor_states'][1:41], rtol=1e-6)
                process_ftf_robot(root/'ftf_export', task)
                self.assertTrue((root/'ftf_export/processed_data'/task/'demonstration_0/states.csv').exists())
                export_dataset(path.parent, root/'visk_export', 'visk')
                if task == 'feeding':
                    prepare('visk', root/'workspace', root/'visk_export', task)
                    report = process_visk_sim(root/'workspace')
                    self.assertTrue(report['validation']['dataset_loader_validated'])
                    self.assertEqual(report['episodes'][0]['pose_frame'], 'robot_base')
                self.assertEqual(originals, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()})
                np.testing.assert_array_equal(env.data.warning.number, 0)

    def test_abort_never_advances_physics_and_failed_episode_is_not_exported(self):
        env = self.env('pick_cube')
        with tempfile.TemporaryDirectory() as tmp, LearningSimRecorder(env, tmp) as recorder:
            recorder.start()
            before = env.data.time
            with patch.object(env, 'step', side_effect=AssertionError('Must not step invalid physics')):
                path = recorder.abort('deliberate acquisition failure')
            self.assertEqual(env.data.time, before)
            self.assertFalse(recorder.active)
            with self.assertRaises(ValueError):
                export_dataset(path.parent, Path(tmp)/'export', 'ftf', [path.name])
            self.assertFalse((Path(tmp)/'export').exists())
