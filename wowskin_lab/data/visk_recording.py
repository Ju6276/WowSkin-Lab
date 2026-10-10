"""Franka/MuJoCo acquisition for the single-AnySkin setup in ViSk IV-A.

This is an acquisition backend, not an upstream training implementation.
Keep raw measured state, commands and timestamps; never export object points.
"""
import json
from pathlib import Path
import re
from types import SimpleNamespace

import cv2
import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from wowskin_lab.data.cameras import CAMERAS
from wowskin_lab.data.recording import magnetic_baseline, magnetic_metadata

CAMERA_IDS = (1, 2)


class ViSkSimRecorder:
    def __init__(self, env, directory, source='keyboard'):
        if not np.isclose(env.cfg['sample_hz'], 100):
            raise ValueError('ViSk paper acquisition requires 100 Hz tactile samples')
        self.env, self.directory, self.source = env, Path(directory), source
        self.renderer = mujoco.Renderer(env.model, width=640, height=480)
        self.writer, self.h5, self.videos = None, None, {}

    @property
    def active(self):
        return self.writer is not None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _metadata(self):
        path = self.writer.path/'metadata.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.metadata, indent=2, ensure_ascii=False)+'\n')
        temporary.replace(path)

    def start(self):
        if self.active:
            raise RuntimeError('Finish the current demonstration first')
        self.directory.mkdir(parents=True, exist_ok=True)
        ids = [int(m.group(1)) for p in self.directory.glob('demonstration_*')
               if (m := re.match(r'^demonstration_(\d+)', p.name))]
        path = self.directory/f'demonstration_{max(ids, default=-1)+1}'
        path.mkdir()
        self.writer = SimpleNamespace(path=path, count=0)
        self.metadata = dict(schema='visk_mujoco_raw_v2', status='in_progress',
            backend='mujoco', source=self.source, task=self.directory.name,
            simulation_config=dict(self.env.cfg), simulation_episode_id=self.env.episode_id,
            paper='https://arxiv.org/html/2410.17246v2#S4.SS1',
            tactile_sensors=1, tactile_channels=15, tactile_hz=100,
            camera_ids=list(CAMERA_IDS), camera_names=CAMERAS, camera_size=[640, 480],
            camera_hz=30, camera_clock='simulation seconds; captures on first 100 Hz tick at/after deadline',
            camera_model='MuJoCo pinhole; two fixed views of the same tabletop workspace',
            color_order='BGR in OpenCV video', pose='world xyz metres + rotvec radians; tactile_tcp',
            gripper='measured closure fraction: 0 open, 1 closed',
            control='world dxyz metres + local drotvec radians + closure target 0..1',
            tactile='simulated raw magnetic field in uT; measured delta plus saved baseline',
            sensor_baseline_subtracted=False, magnetic_baseline_ut=magnetic_baseline(self.env).tolist(),
            uncalibrated=True, object_points_recorded=False,
            collection_method=f'{self.source} teleoperation; no extra directional-noise injection',
            task_is_paper_benchmark=False, training_validated=False)
        self.metadata.update(magnetic_metadata(self.env))
        self._metadata()
        try:
            self.h5 = h5py.File(path/'measurements.h5', 'w')
            for i in CAMERA_IDS:
                video = cv2.VideoWriter(str(path/f'cam_{i}.avi'), cv2.VideoWriter_fourcc(*'XVID'),
                                        30, (640, 480))
                if not video.isOpened():
                    raise RuntimeError(f'Could not open camera {i} output')
                self.videos[i] = video
            self.next_camera_time = float(self.env.data.time)
            self.previous_time = None
            self.capture(None)
        except BaseException:
            self.finish('interrupted')
            raise
        print(f'Recording ViSk paper single-sensor data: {path}', flush=True)

    def _append(self, key, value):
        value = np.asarray(value)
        if not np.isfinite(value).all():
            raise ValueError(f'Nonfinite raw recording: {key}')
        if key not in self.h5:
            self.h5.create_dataset(key, shape=(0, *value.shape), maxshape=(None, *value.shape),
                                   dtype=value.dtype, chunks=True)
        dataset = self.h5[key]
        dataset.resize(dataset.shape[0]+1, axis=0)
        dataset[-1] = value

    def capture(self, row):
        e, d = self.env, self.env.data
        t = float(d.time)
        if self.previous_time is not None and not np.isclose(t-self.previous_time, .01, atol=1e-8):
            raise ValueError('ViSk recording lost a 100 Hz sample or simulation reset during recording')
        self._append('timestamp', t)
        self._append('cartesian_states', np.r_[d.site_xpos[e.tcp_site],
                     Rotation.from_matrix(d.site_xmat[e.tcp_site].reshape(3, 3)).as_rotvec()])
        self._append('gripper_states', e.gripper_fraction())
        self._append('commanded_pose', np.r_[e.target_xyz, Rotation.from_matrix(e.target_rotation).as_rotvec()])
        self._append('commanded_gripper', e.gripper_fraction(target=True))
        self._append('command_7d', e.hold_action() if row is None else row['action'])
        self._append('command_timestamp', t if row is None else row['action_time_s'])
        self._append('command_valid', row is not None)
        self._append('sensor_states', np.asarray(e.observation()['magnetic_ut'])+magnetic_baseline(e))
        if getattr(e, 'reskin_response', None):
            response_row = e.last_row or {}
            query = np.asarray(response_row.get('reskin_query_mm', [np.nan]*3))
            self._append('reskin_query_mm', np.nan_to_num(query))
            self._append('reskin_query_present', np.isfinite(query).all())
            self._append('reskin_input_in_training_domain', response_row.get('reskin_in_training_domain', False))
        self._append('arm_qpos', d.qpos[e.arm_q].copy())
        self._append('arm_qvel', d.qvel[e.arm_v].copy())
        if t+1e-9 >= self.next_camera_time:
            while self.next_camera_time <= t+1e-9:
                self.next_camera_time += 1/30
            self._append('camera_timestamp', t)
            extrinsics = []
            for i in CAMERA_IDS:
                name = CAMERAS[f'pixels{i}']
                cid = e.model.camera(name).id
                self.renderer.update_scene(d, camera=name)
                self.videos[i].write(self.renderer.render()[..., ::-1].copy())
                ext = np.eye(4)
                ext[:3, :3] = np.diag([1., -1., -1.]) @ d.cam_xmat[cid].reshape(3, 3).T
                ext[:3, 3] = -ext[:3, :3] @ d.cam_xpos[cid]
                extrinsics.append(ext)
                if self.writer.count == 0:
                    focal = 240/np.tan(np.deg2rad(e.model.cam_fovy[cid])/2)
                    self.h5.create_dataset(f'camera_intrinsic_{i}',
                        data=[[focal, 0, 319.5], [0, focal, 239.5], [0, 0, 1]])
            self._append('camera_world_to_cv', np.stack(extrinsics))
            self.writer.count += 1
        self.previous_time = t
        self.h5.flush()

    def step(self, row):
        if self.active:
            self.capture(row)

    def finish(self, status='unreviewed'):
        if status not in ('success', 'failure', 'unreviewed', 'interrupted'):
            raise ValueError(status)
        if not self.active:
            return None
        path = self.writer.path
        for video in self.videos.values():
            video.release()
        self.videos.clear()
        samples = len(self.h5['timestamp']) if self.h5 and 'timestamp' in self.h5 else 0
        if self.h5:
            self.h5.close()
        self.h5 = None
        self.metadata.update(status=status, state_samples=samples, camera_frames=self.writer.count)
        self._metadata()
        self.writer = None
        if status != 'success':
            renamed = path.with_name(path.name+('_fail' if status == 'failure' else '_ignore'))
            path.rename(renamed)
            path = renamed
        print(f'Saved ViSk raw demonstration ({status}): {path}', flush=True)
        return path

    def close(self):
        try:
            self.finish('interrupted')
        finally:
            self.renderer.close()

    def abort(self, reason):
        if self.active:
            self.metadata['error'] = reason
        return self.finish('interrupted')
