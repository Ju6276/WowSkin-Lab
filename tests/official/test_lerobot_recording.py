"""Real rendered samples -> original LeRobot writer/reader, including append."""
import importlib.util
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import mujoco
import numpy as np

from wowskin_lab.data.lerobot_recording import ANNOTATION, LeRobotSimRecorder, append_raw_episode, upstream, validate_dataset
from wowskin_lab.data.depth import load_depth_frame
from wowskin_lab.data.media import video_path, depth_path
from wowskin_lab.data.dataset_export import export_dataset
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, response_configuration
from wowskin_lab.paths import ROOT


@unittest.skipUnless(mujoco.__version__ == '3.15.0' and importlib.util.find_spec('lerobot'),
                     'Requires .venv-reduced and scripts/setup_capture.py')
class LeRobotRecordingChecks(unittest.TestCase):
    def test_failed_final_verification_keeps_originals_and_blocks_append(self):
        cfg = response_configuration(SimpleNamespace(task='pick_cube', skin_dof='trilinear', ideal=True,
            magnetic_fit=ROOT/'configs/magnetic_fit_pollen_nonslip.json'))
        env = PandaPickCubeTactileEnv(cfg, pollen=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with LeRobotSimRecorder(env, root, source='verification_failure_test') as recorder:
                recorder.start()
                for _ in range(12):
                    recorder.step(env.step(env.hold_action())[-1])
                with patch('wowskin_lab.data.lerobot_recording.verified_media',
                           side_effect=ValueError('injected decoder failure')):
                    with self.assertRaisesRegex(ValueError, 'injected decoder'):
                        recorder.finish('success')
                self.assertNotEqual(recorder.progress[0], 'SAVED')
            raw = root/'pick_cube/raw/capture_000000'
            native = root/'pick_cube/lerobot'
            self.assertTrue((native/'INCOMPLETE').exists())
            self.assertTrue((raw/'measurements.h5').exists())
            self.assertTrue(depth_path(raw).is_file())
            self.assertTrue(all(video_path(raw, i).is_file() for i in (1, 2, 52)))
            self.assertNotIn('lerobot_episode_index', json.loads((raw/'metadata.json').read_text()))
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                validate_dataset(native)
            with LeRobotSimRecorder(env, root, source='verification_failure_test') as recorder:
                with self.assertRaisesRegex(RuntimeError, 'incomplete export'):
                    recorder.start()

    def test_robotiq_continuous_command_and_measured_feedback_are_distinct(self):
        # Format/servo contract only, not a successful pick or strong grasp.
        cfg = response_configuration(SimpleNamespace(task='pick_cube', gripper='robotiq_2f85',
            skin_dof='trilinear', ideal=True, magnetic_fit=ROOT/'configs/magnetic_fit_pollen_nonslip.json'))
        env = PandaPickCubeTactileEnv(cfg, pollen=False)
        with tempfile.TemporaryDirectory() as tmp:
            with LeRobotSimRecorder(env, tmp, source='format_contract_test_not_task_evaluation',
                                    record_depth=False) as recorder:
                recorder.start()
                for step in range(45):
                    a = env.hold_action()
                    a[-1] = .25 if step < 20 else .35
                    recorder.step(env.step(a)[-1])
                path = recorder.finish('success')
            meta = json.loads((path/'metadata.json').read_text())
            self.assertTrue(all(u == 'rad' for u in meta['gripper_driver_position_units']))
            with h5py.File(path/'measurements.h5') as f:
                phase = f['phase'][:] == 1
                np.testing.assert_allclose(f['commanded_gripper'][:][phase], [.25]*20+[.35]*25)
                measured = f['gripper_states'][:][phase]
                self.assertGreater(np.unique(np.round(measured, 5)).size, 5)
                self.assertGreater(np.max(abs(measured-f['commanded_gripper'][:][phase])), .01)
                np.testing.assert_allclose(measured, np.clip(f['gripper_driver_qpos'][:][phase].mean(axis=1)/.8, 0, 1))
                ds = upstream()('local/wowskin-lab', root=Path(tmp)/'pick_cube/lerobot', video_backend='pyav')
                for row in ds.hf_dataset.with_format('numpy'):
                    i = int(row['aux.raw_sample_index'])
                    self.assertAlmostEqual(float(row['observation.state'][-1]), float(f['gripper_states'][i]), places=6)
                    self.assertAlmostEqual(float(row['action'][-1]), float(f['commanded_gripper'][i+1]), places=6)

    def test_two_episodes_exact_raw_alignment_resume_and_official_video_loader(self):
        cfg = response_configuration(SimpleNamespace(task='pick_cube', skin_dof='trilinear', ideal=True,
            magnetic_fit=ROOT/'configs/magnetic_fit_pollen_nonslip.json'))
        env = PandaPickCubeTactileEnv(cfg, pollen=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            instructions = ['Lift the cube and move it across the table.', '夹起方块，移动后放回桌面。']
            # Recreate recorder between episodes: exercise on-disk resume.
            for episode in range(2):
                env.reset(seed=episode)
                with ThreadPoolExecutor(max_workers=1) as pool, LeRobotSimRecorder(
                        env, root, source='format_contract_test_not_task_evaluation',
                        instruction=instructions[episode]) as recorder:
                    env._viewer_run_io = lambda function, *args: pool.submit(function, *args).result()
                    progress_events = []
                    original_progress = recorder.set_progress
                    def progress(stage, detail):
                        progress_events.append((stage, threading.get_ident()))
                        original_progress(stage, detail)
                    recorder.set_progress = progress
                    recorder.start()
                    self.assertEqual(recorder.progress[0], 'RECORDING')
                    # Two windows must not allocate the same capture/episode.
                    with LeRobotSimRecorder(env, root, source='format_contract_test_not_task_evaluation') as other:
                        with self.assertRaisesRegex(RuntimeError, 'Another window'):
                            other.start()
                    for _ in range(36):
                        a = env.hold_action()
                        a[2] = .00015
                        recorder.step(env.step(a)[-1])
                    paths.append(recorder.finish('success'))
                    self.assertEqual(recorder.progress[0], 'SAVED')
                    self.assertTrue(any(stage == 'VERIFYING' and thread != threading.get_ident()
                                        for stage, thread in progress_events))
                    self.assertFalse(recorder.active)
                env._viewer_run_io = None
            native = root/'pick_cube/lerobot'
            result = validate_dataset(native)
            self.assertEqual(result['episodes'], 2)
            self.assertTrue(result['official_loader_passed'])
            self.assertTrue(result['instruction_validated'])
            self.assertEqual(len(result['depth_frame_pairs']), 2)
            ds = upstream()('local/wowskin-lab', root=native, video_backend='pyav')
            data = ds.hf_dataset.with_format('numpy')
            for episode, source in enumerate(paths):
                lo, hi = int(ds.episode_data_index['from'][episode]), int(ds.episode_data_index['to'][episode])
                self.assertEqual(ds[lo]['task'], instructions[episode])
                self.assertEqual({p.name for p in source.iterdir()}, {'measurements.h5', 'metadata.json'})
                self.assertTrue(all(video_path(source, c).suffix == '.mp4' for c in (1, 2, 52)))
                self.assertEqual(set(ds.meta.video_keys), {f'observation.images.camera{i}' for i in (1, 2, 52)})
                with h5py.File(source/'measurements.h5') as f:
                    self.assertEqual(f['camera_base_to_cv'].shape[1:], (3, 4, 4))
                    self.assertGreater(np.linalg.norm(f['camera_base_to_cv'][-1, 2]-f['camera_base_to_cv'][0, 2]), .001)
                    np.testing.assert_allclose(f['camera_base_to_cv'][0, :2], f['camera_base_to_cv'][-1, :2])
                    self.assertIn('evaluation/cube_success', f)
                    for k in range(lo, hi):
                        row = data[k]
                        i, j = int(row['aux.raw_sample_index']), int(row['aux.raw_camera_index'])
                        self.assertEqual(int(row['aux.acquisition_phase']), f['phase'][i])
                        np.testing.assert_allclose(row['observation.state'], np.r_[f['cartesian_states'][i], f['gripper_states'][i]], atol=1e-6)
                        np.testing.assert_allclose(row['observation.joint_position'], f['arm_qpos'][i], atol=1e-6)
                        np.testing.assert_allclose(row['observation.joint_velocity'], f['arm_qvel'][i], atol=1e-6)
                        np.testing.assert_allclose(row['observation.tcp_velocity'], f['tcp_velocity'][i], atol=1e-6)
                        np.testing.assert_allclose(row['observation.tactile'], f['sensor_states'][i], rtol=1e-6, atol=1e-6)
                        np.testing.assert_allclose(row['action'], np.r_[f['commanded_pose'][i+1], f['commanded_gripper'][i+1]], atol=1e-6)
                        np.testing.assert_allclose(row['aux.sim_timestamp'], [f['camera_timestamp'][j]], atol=1e-8)
                        np.testing.assert_allclose(row['aux.action_timestamp'], row['aux.sim_timestamp'], atol=1e-8)
                        self.assertEqual(int(row[ANNOTATION]), episode)
                        depth = load_depth_frame(native, episode, j)
                        self.assertEqual(depth['timestamp'], f['camera_timestamp'][j])
                        np.testing.assert_array_equal(depth['camera_base_to_cv'], f['camera_base_to_cv'][j])
                        with h5py.File(depth_path(source)) as raw_depth:
                            for camera in (1, 2, 52):
                                np.testing.assert_array_equal(depth[f'camera{camera}'], raw_depth[f'camera{camera}'][j])
                # Metric convention check against the visible table plane,
                # using rendered depths and saved calibration, not mock data.
                for camera in (1, 2):
                    ext = depth['camera_world_to_cv'][camera-1]
                    K = depth[f'intrinsic_camera{camera}']
                    # Back-project all valid pixels to world coordinates.
                    v, u = np.mgrid[0:480, 0:640]
                    rays = np.stack([(u-K[0, 2])/K[0, 0], (v-K[1, 2])/K[1, 1], np.ones_like(u)], -1)
                    world = (rays*depth[f'camera{camera}'][..., None]-ext[:3, 3]) @ ext[:3, :3]
                    patch = ((world[..., 0] > .25) & (world[..., 0] < .5) &
                             (world[..., 1] > -.17) & (world[..., 1] < -.08) &
                             depth[f'valid_camera{camera}'])
                    self.assertGreater(patch.sum(), 100)
                    # Tabletop z = .10 m in the cube scene; optical depth
                    # treated as range or mirrored vertically fails this.
                    self.assertLess(float(np.median(abs(world[..., 2][patch]-.10))), .001)
            with self.assertRaisesRegex(ValueError, 'already'):
                append_raw_episode(paths[0], native)
            self.assertFalse((native/'INCOMPLETE').exists())
            with LeRobotSimRecorder(env, root, source='format_contract_test_not_task_evaluation', record_depth=False) as changed:
                with self.assertRaisesRegex(ValueError, 'Changed recording'):
                    changed.start()
            env.cfg['camera_profile'] = 'two_fixed'
            with LeRobotSimRecorder(env, root, source='format_contract_test_not_task_evaluation') as changed:
                with self.assertRaisesRegex(ValueError, 'Changed recording'):
                    changed.start()
            env.cfg['camera_profile'] = 'three_view'
            # Same camera IDs with a different fixed wrist mount are a new rig.
            wrist = env.model.camera('lab_wrist').id
            original_mount = env.model.cam_pos[wrist].copy()
            env.model.cam_pos[wrist, 0] += .01
            try:
                with LeRobotSimRecorder(env, root, source='format_contract_test_not_task_evaluation') as changed:
                    with self.assertRaisesRegex(ValueError, 'Changed recording'):
                        changed.start()
            finally:
                env.model.cam_pos[wrist] = original_mount
            # An abort stays raw and cannot silently enter the training set.
            with LeRobotSimRecorder(env, root, source='format_contract_test_not_task_evaluation') as recorder:
                recorder.start()
                failed = recorder.abort('intentional test')
            with self.assertRaises(ValueError):
                append_raw_episode(failed, native)
            self.assertEqual(validate_dataset(native)['episodes'], 2)
            # Algorithm export resolves the shared raw attachment, not a
            # second independent capture or loss of the 100 Hz tactile history.
            exported = export_dataset(root/'pick_cube', root/'visk_export', 'visk')
            self.assertEqual(len(exported['exported_episodes']), 2)

    def test_measured_tcp_velocity_matches_moving_kinematics(self):
        from scipy.spatial.transform import Rotation
        from wowskin_lab.data.robot_state import robot_sample
        cfg = response_configuration(SimpleNamespace(task='pick_cube', skin_dof='trilinear', ideal=True,
            magnetic_fit=ROOT/'configs/magnetic_fit_pollen_nonslip.json'))
        env = PandaPickCubeTactileEnv(cfg, pollen=False)
        env.data.qvel[:] = 0
        env.data.qvel[env.arm_v] = [.1, -.07, .02, -.12, .04, .06, -.03]
        mujoco.mj_forward(env.model, env.data)
        sample = robot_sample(env)
        before = env.data.site_xpos[env.tcp_site].copy()
        rotation = env.data.site_xmat[env.tcp_site].reshape(3, 3).copy()
        dt = 1e-6
        mujoco.mj_integratePos(env.model, env.data.qpos, env.data.qvel, dt)
        mujoco.mj_forward(env.model, env.data)
        linear = (env.data.site_xpos[env.tcp_site]-before)/dt
        angular = Rotation.from_matrix(env.data.site_xmat[env.tcp_site].reshape(3, 3) @ rotation.T).as_rotvec()/dt
        np.testing.assert_allclose(sample['tcp_velocity'], np.r_[linear, angular], atol=1e-6)
