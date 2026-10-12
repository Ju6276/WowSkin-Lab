"""One raw camera/state/tactile stream, independent of the training algorithm.

This is an acquisition backend, not an upstream training implementation.
Keep raw measured state, commands and timestamps; never export object points.
"""
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import cv2
import h5py
import mujoco
import numpy as np

from wowskin_lab.data.cameras import CAMERAS, active_cameras, capture_options, camera_configuration
from wowskin_lab.data.depth import DepthWriter
from wowskin_lab.data.instructions import episode_instruction
from wowskin_lab.data.recording import magnetic_baseline, magnetic_metadata, require_training_tactile
from wowskin_lab.data.robot_state import append_h5, camera_from_world, command_in_base, robot_metadata, robot_sample

CAMERA_IDS = (1, 2, 52)


class RawRobotRecorder:
    schema = 'wowskin_robot_dataset_v1'

    def __init__(self, env, directory, source='keyboard', *, instruction=None, record_depth=True):
        if not np.isclose(env.cfg['sample_hz'], 100):
            raise ValueError('Raw robot acquisition requires 100 Hz tactile samples')
        self.env, self.directory, self.source = env, Path(directory), source
        self.cameras = active_cameras(env)
        self.camera_ids = tuple(int(key.removeprefix('pixels')) for key in self.cameras)
        task = getattr(env, 'task_name', 'pick_cube')
        if task == 'cable' and env.cfg.get('cable_layout') == 'single_clip':
            task = 'cable_insert'
        self.instruction = episode_instruction(task, instruction)
        self.instruction_source = 'task_template' if instruction is None else 'user_provided'
        self.record_depth, self.depth = record_depth, None
        self.renderer = mujoco.Renderer(env.model, width=640, height=480)
        self.writer, self.h5, self.videos = None, None, {}
        self.progress = ('IDLE', 'B starts recording')

    def set_progress(self, stage, detail):
        # Assign one immutable value; the UI can read it while disk work runs.
        self.progress = (stage, detail)

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
        self.set_progress('PREPARING', 'Opening camera, depth and 100 Hz state files')
        require_training_tactile(self.env)
        if self.active:
            raise RuntimeError('Finish the current demonstration first')
        self.directory.mkdir(parents=True, exist_ok=True)
        ids = [int(m.group(1)) for p in self.directory.iterdir()
               if (m := re.match(r'^(?:capture|demonstration)_(\d+)', p.name))]
        path = self.directory/f'capture_{max(ids, default=-1)+1:06d}'
        path.mkdir()
        collection = self.directory.parent if self.directory.name == 'raw' else self.directory
        media = collection/'.pending_media'/path.name
        media.mkdir(parents=True)
        self.writer = SimpleNamespace(path=path, count=0)
        self.metadata = dict(schema=self.schema, status='in_progress',
            backend='mujoco', source=self.source, task=self.directory.name,
            instruction=self.instruction, instruction_source=self.instruction_source,
            simulation_config=dict(self.env.cfg), simulation_episode_id=self.env.episode_id,
            paper='https://arxiv.org/html/2410.17246v2#S4.SS1',
            tactile_sensors=1, tactile_channels=15, tactile_hz=100,
            camera_ids=list(self.camera_ids), camera_names=self.cameras, camera_size=[640, 480],
            camera_configuration=camera_configuration(self.env),
            camera_hz=30, camera_clock='simulation seconds; captures on first 100 Hz tick at/after deadline',
            camera_model='MuJoCo pinhole; fixed tabletop views plus wrist when present; per-frame extrinsics recorded',
            color_order='BGR in OpenCV video', pose='robot-base xyz metres + rotvec radians; tactile_tcp',
            gripper='measured closure fraction: 0 open, 1 closed',
            gripper_action='commanded_gripper is the actual absolute closure target (0 open, 1 closed); never substitute gripper_states',
            state_action_contract='State is measured feedback; action is a controller command. Contact can keep measured closure below the commanded closure.',
            control='base dxyz metres + local TCP drotvec radians + closure target 0..1',
            control_world='command_7d_world retains the controller world dxyz + local TCP drotvec + closure',
            tactile='simulated raw magnetic field in uT; measured delta plus saved baseline',
            sensor_baseline_subtracted=False, magnetic_baseline_ut=magnetic_baseline(self.env).tolist(),
            uncalibrated=True, object_points_recorded=False,
            collection_method=f'{self.source}; no extra directional-noise injection',
            task_is_paper_benchmark=False, training_validated=False)
        self.metadata['video_files'] = {str(i): os.path.relpath(media/f'cam_{i}.mp4', path) for i in self.camera_ids}
        self.metadata['depth'] = dict(enabled=self.record_depth,
            path=os.path.relpath(media/'depth.h5', path) if self.record_depth else None,
            dtype='float32', units='metres', shape=[480, 640], camera_ids=list(self.camera_ids),
            convention='OpenCV optical +Z, not Euclidean ray range',
            synchronized_with='camera_timestamp and corresponding RGB frame',
            compression='lossless LZF; original float values and validity masks',
            source='MuJoCo renderer, no real depth-camera noise model')
        self.metadata.update(magnetic_metadata(self.env))
        self.metadata['mesh_quality'] = dict(
            keys=['skin_min_volume_ratio', 'passive_min_volume_ratio', 'mesh_quality_warning'],
            clock='100 Hz timestamp', warning_ratio=.15,
            policy='Positive volume: continue recording fresh computed fields; warning does not drop frames or end episodes. Nonpositive/nonfinite: physics fault.',
            meaning='Local numerical diagnostic, not a calibrated material limit or magnetometer saturation')
        self.metadata.update(robot_metadata(self.env))
        if hasattr(self.env, 'gripper_feedback_sample'):
            self.metadata['gripper_force_feedback'] = dict(
                config=self.env.force_grip.config,
                raw_group='gripper_force_feedback',
                action='7D actions remain inner-servo closure targets; force requests and measured pad forces are separate 100 Hz fields',
                source='MuJoCo simulated contact force, not a calibrated AnySkin force estimate')
        self._metadata()
        try:
            self.h5 = h5py.File(path/'measurements.h5', 'w')
            if self.record_depth:
                self.depth = DepthWriter(media/'depth.h5', self.env.model)
            for i in self.camera_ids:
                video = cv2.VideoWriter(str(media/f'cam_{i}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
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
        print(f'Recording shared raw robot data: {path}', flush=True)
        print(f'Instruction: {self.instruction} | synchronized depth: {self.record_depth}', flush=True)

    def _append(self, key, value):
        append_h5(self.h5, key, value)

    def capture(self, row):
        require_training_tactile(self.env)
        e, d = self.env, self.env.data
        t = float(d.time)
        if self.previous_time is not None and not np.isclose(t-self.previous_time, .01, atol=1e-8):
            raise ValueError('Recording lost a 100 Hz sample or simulation reset during recording')
        self._append('timestamp', t)
        robot = robot_sample(e)
        for key, value in robot.items():
            self._append(key, value)
        self._append('gripper_states', e.gripper_fraction())
        self._append('commanded_gripper', e.gripper_fraction(target=True))
        if hasattr(e, 'gripper_feedback_sample'):
            for key, value in e.gripper_feedback_sample().items():
                self._append('gripper_force_feedback/'+key, value)
        action_world = e.hold_action() if row is None else row['action']
        self._append('command_7d', command_in_base(action_world, robot['world_from_base']))
        self._append('command_7d_world', action_world)
        self._append('command_timestamp', t if row is None else row['action_time_s'])
        self._append('command_valid', row is not None)
        self._append('sensor_states', np.asarray(e.observation()['magnetic_ut'])+magnetic_baseline(e))
        quality = e.last_row or e.tactile_quality()
        self._append('skin_min_volume_ratio', quality['min_volume_ratio'])
        self._append('passive_min_volume_ratio', quality['passive_min_volume_ratio'])
        self._append('mesh_quality_warning', min(quality['min_volume_ratio'], quality['passive_min_volume_ratio']) <= .15)
        if hasattr(e, 'evaluation_metrics'):
            for key, value in e.evaluation_metrics(row).items():
                self._append('evaluation/'+key, value)
        if getattr(e, 'reskin_response', None):
            response_row = e.last_row or {}
            query = np.asarray(response_row.get('reskin_query_mm', [np.nan]*3))
            self._append('reskin_query_mm', query)
            self._append('reskin_query_present', np.isfinite(query).all())
            self._append('reskin_input_in_training_domain', response_row.get('reskin_in_training_domain', False))
        if t+1e-9 >= self.next_camera_time:
            while self.next_camera_time <= t+1e-9:
                self.next_camera_time += 1/30
            self._append('camera_timestamp', t)
            extrinsics = []
            for i in self.camera_ids:
                name = self.cameras[f'pixels{i}']
                cid = e.model.camera(name).id
                self.renderer.update_scene(d, camera=name, scene_option=capture_options())
                self.videos[i].write(self.renderer.render()[..., ::-1].copy())
                ext = camera_from_world(e, cid)
                extrinsics.append(ext)
                if self.writer.count == 0:
                    focal = 240/np.tan(np.deg2rad(e.model.cam_fovy[cid])/2)
                    self.h5.create_dataset(f'camera_intrinsic_{i}',
                        data=[[focal, 0, 319.5], [0, focal, 239.5], [0, 0, 1]])
                if self.depth:
                    self.renderer.enable_depth_rendering()
                    try:
                        depth = self.renderer.render().copy()
                    finally:
                        self.renderer.disable_depth_rendering()
                    self.depth.camera(i, depth, self.h5[f'camera_intrinsic_{i}'][:])
            self._append('camera_world_to_cv', np.stack(extrinsics))
            self._append('camera_base_to_cv', np.stack(extrinsics) @ robot['world_from_base'])
            if self.depth:
                self.depth.finish_frame(t, np.stack(extrinsics) @ robot['world_from_base'], np.stack(extrinsics))
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
        self.set_progress('SAVING', 'Closing original video, depth and state files')
        path = self.writer.path
        for video in self.videos.values():
            video.release()
        self.videos.clear()
        samples = len(self.h5['timestamp']) if self.h5 and 'timestamp' in self.h5 else 0
        if self.h5:
            self.h5.close()
        self.h5 = None
        if self.depth:
            self.depth.close()
            self.depth = None
        self.metadata.update(status=status, state_samples=samples, camera_frames=self.writer.count)
        if hasattr(self.env, 'task_evaluation'):
            self.metadata['task_evaluation'] = self.env.task_evaluation()
            self.metadata['status_source'] = 'operator review; independent from automatic task score'
        self._metadata()
        self.writer = None
        print(f'Saved raw demonstration ({status}): {path}', flush=True)
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
