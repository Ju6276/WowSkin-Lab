"""MuJoCo acquisition transport for the unmodified Franka-Teach recorder.

In-process queues replace hardware subscribers only. Original save_rgb,
save_states and save_reskin methods do all AVI/metadata/PKL/H5 serialization.
No point tracks, training labels or normalization are generated here.
"""
from collections import deque
import json
from pathlib import Path
from queue import Queue
import re
import sys
import threading
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from wowskin_lab.data.cameras import CAMERAS
from wowskin_lab.data.recording import magnetic_baseline, magnetic_metadata
from wowskin_lab.learning.official import verify_reference


class _SimulationSubscriber:
    """Same receive API as hardware subscribers, with lossless backpressure."""
    def __init__(self):
        self.queue = Queue(maxsize=2)
        self.waiting = threading.Event()

    def _receive(self):
        self.waiting.set()
        value, received = self.queue.get()
        self.waiting.clear()
        received.set()
        if value is None:
            raise RuntimeError('Simulation acquisition aborted')
        return value

    recv_rgb_image = _receive
    recv_keypoints = _receive
    get_sensor_state = _receive

    def send(self, value):
        completed = threading.Event()
        self.queue.put((value, completed), timeout=10)
        return completed

    def stop(self):
        pass

    def abort(self):
        try:
            self.queue.put_nowait((None, threading.Event()))
        except Exception:
            pass  # Receiver already has buffered data and will observe stop.


class FTFSimRecorder:
    def __init__(self, env, directory, source='keyboard'):
        from omegaconf import OmegaConf
        reference, spec, _ = verify_reference('ftf-robot')
        self.teach_root = reference/'Franka-Teach'
        sys.path.insert(0, str(self.teach_root))
        from frankateach.data_collector import DataCollector
        from frankateach.messages import FrankaState, FrankaAction
        self.Collector, self.State, self.Action = DataCollector, FrankaState, FrankaAction
        self.camera_config = OmegaConf.create(dict(width=640, height=480, fps=30))
        self.env, self.directory, self.source = env, Path(directory), source
        self.revision = spec['revision']
        self.renderer = mujoco.Renderer(env.model, width=640, height=480)
        self.history = deque(maxlen=40)
        self.writer = None
        self.collector = None
        self.frames, self.samples = 0, 0
        self.errors, self.workers, self.subscribers = [], [], []

    @property
    def active(self):
        return self.collector is not None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _advance_hold(self):
        # Keep the viewer responsive during official pre/post-roll; rendering
        # and writers remain on their owning thread, and all samples are real.
        run = getattr(self.env, '_viewer_run', None)
        if run is None:
            return self.env.step(self.env.hold_action())
        return run(self.env.step, self.env.hold_action())

    def _field(self):
        # Restore the baseline to the measured (filtered/noisy/delayed) delta.
        # This is a simulated raw field in uT, not a force or ADC conversion.
        return (np.asarray(self.env.observation()['magnetic_ut'])+
                magnetic_baseline(self.env)).astype(np.float32)

    def _worker(self, function, subscribers, *args):
        try:
            function(*args)
        except BaseException as exc:
            self.errors.append(f'{function.__name__}: {type(exc).__name__}: {exc}')
        finally:
            for sub in subscribers:
                sub.stop()

    def start(self):
        if self.active:
            raise RuntimeError('Finish this demonstration before starting another')
        # Fill history with actual simulation samples, never duplicated padding.
        self.history.clear()
        for _ in range(40):
            self._advance_hold()
            self.history.append(self._field())
        self.directory.mkdir(parents=True, exist_ok=True)
        ids = [int(match.group(1)) for p in self.directory.glob('demonstration_*')
               if (match := re.match(r'^demonstration_(\d+)', p.name))]
        demo_num = max(ids, default=-1)+1
        path = self.directory/f'demonstration_{demo_num}'
        if path.exists():
            raise FileExistsError(path)
        # No hardware sockets/services are constructed. Only the data-source
        # API is supplied below; all official persistence methods are unchanged.
        c = self.Collector(str(self.directory), demo_num)
        c.image_subscribers = {i: _SimulationSubscriber() for i in (1, 2)}
        c.state_socket, c.commanded_state_socket = _SimulationSubscriber(), _SimulationSubscriber()
        c.reskin_subscriber = _SimulationSubscriber()
        self.collector = c
        self.subscribers = [*c.image_subscribers.values(), c.state_socket,
                            c.commanded_state_socket, c.reskin_subscriber]
        self.workers, self.errors = [], []
        jobs = [(c.save_rgb, [sub], (i, self.camera_config)) for i, sub in c.image_subscribers.items()]
        jobs += [(c.save_states, [c.state_socket, c.commanded_state_socket], ()),
                 (c.save_reskin, [c.reskin_subscriber], ())]
        for function, subscribers, args in jobs:
            thread = threading.Thread(target=self._worker, args=(function, subscribers, *args), daemon=True)
            self.workers.append(thread)
            thread.start()
        self.frames = self.samples = 0
        self.writer = SimpleNamespace(count=0, path=path)
        self.next_camera_time = self.env.data.time
        self.metadata = dict(backend='mujoco', source=self.source, task=getattr(self.env, 'task_name', 'pick_cube'),
            simulation_config=dict(self.env.cfg), simulation_episode_id=self.env.episode_id,
            format='official_frankateach_raw', status='in_progress',
            upstream_revision=self.revision, camera_size=[640, 480], nominal_camera_hz=30,
            state_hz=self.env.cfg['sample_hz'], tactile_hz=self.env.cfg['sample_hz'],
            timestamp_clock='simulation_seconds; camera metadata uses milliseconds',
            video_writer='unmodified Franka-Teach DataCollector.save_rgb; BGR XVID AVI',
            tactile='15 simulated raw magnetic channels in uT; baseline restored; NOT calibrated hardware ADC',
            pose_reference='tactile_tcp site in MuJoCo world; xyz metres and xyzw quaternion',
            tcp_position_in_hand_m=self.env.model.site_pos[self.env.tcp_site].tolist(),
            tcp_registration_to_original_franka='not yet validated',
            gripper='commanded closure mapped to -1=open,+1=closed, as original FrankaServer last_gripper_action',
            point_source='none; object tracking must be performed by official visual pipeline',
            sensor_baseline_subtracted=False, magnetic_baseline_ut=magnetic_baseline(self.env).tolist(),
            uncalibrated=True, training_ready=False)
        self.metadata.update(magnetic_metadata(self.env))
        self._write_metadata()
        self._idle_roll(20)  # Genuine preceding pause, tagged start_teleop=False.
        print(f'FTF official raw recording: {path}', flush=True)

    def _write_metadata(self):
        (self.writer.path/'simulation_metadata.json').write_text(json.dumps(self.metadata, indent=2)+'\n')

    def _idle_roll(self, count):
        for _ in range(count):
            self._advance_hold()
            self._capture(False)

    def _capture(self, start_teleop, force_camera=False):
        if self.errors:
            raise RuntimeError('; '.join(self.errors))
        e, c = self.env, self.collector
        t = float(e.data.time)
        if getattr(e, 'reskin_response', None):
            row = e.last_row or {}
            query = np.asarray(row.get('reskin_query_mm', [np.nan]*3))
            audit = dict(time_s=t, query_mm=query.tolist() if np.isfinite(query).all() else None,
                         status=row.get('reskin_status', 'unloaded_reference_zero'),
                         input_in_training_domain=bool(row.get('reskin_in_training_domain', False)),
                         transfer_validated=False)
            with (self.writer.path/'reskin_response.jsonl').open('a') as stream:
                stream.write(json.dumps(audit)+'\n')
        grip = float(2*e.gripper_fraction(target=True)-1)
        state = self.State(e.data.site_xpos[e.tcp_site].copy(),
                           Rotation.from_matrix(e.data.site_xmat[e.tcp_site].reshape(3, 3)).as_quat(),
                           grip, t, start_teleop)
        action = self.Action(e.target_xyz.copy(), Rotation.from_matrix(e.target_rotation).as_quat(), grip, False, t)
        field = self._field()
        self.history.append(field.copy())
        pending = [c.state_socket.send(state), c.commanded_state_socket.send(action),
                   c.reskin_subscriber.send(dict(timestamp=t, sensor_values=field,
                                                  sensor_history=np.asarray(self.history).copy()))]
        if force_camera or t+1e-9 >= self.next_camera_time:
            while self.next_camera_time <= t+1e-9:
                self.next_camera_time += 1/30
            calibration = {}
            for i, sub in c.image_subscribers.items():
                name = CAMERAS[f'pixels{i}']
                cid = e.model.camera(name).id
                self.renderer.update_scene(e.data, camera=name)
                bgr = self.renderer.render()[..., ::-1].copy()
                pending.append(sub.send((bgr, t*1000)))
                focal = 240/np.tan(np.deg2rad(e.model.cam_fovy[cid])/2)
                ext = np.eye(4)
                ext[:3, :3] = np.diag([1., -1., -1.]) @ e.data.cam_xmat[cid].reshape(3, 3).T
                ext[:3, 3] = -ext[:3, :3] @ e.data.cam_xpos[cid]
                calibration[f'cam_{i}'] = dict(ext=ext, int=np.array([[focal, 0, 319.5], [0, focal, 239.5], [0, 0, 1]]),
                                               dist_coeff=np.zeros(5))
            if self.frames == 0:
                np.save(self.writer.path/'sim_camera_calibration.npy', calibration)
            self.frames += 1
            self.writer.count = self.frames
        for done in pending:
            if not done.wait(10):
                raise RuntimeError('Official recorder did not consume a simulation sample within 10 s')
        self.samples += 1
        if self.errors:
            raise RuntimeError('; '.join(self.errors))

    def step(self, row):
        if self.active:
            self._capture(True)

    def finish(self, status='unreviewed'):
        if not self.active:
            return None
        if status not in ('success', 'failure', 'unreviewed', 'interrupted'):
            raise ValueError(status)
        path = self.writer.path
        try:
            # Official preprocessing crops from explicit pause boundaries.
            self._idle_roll(20)
            # Each writer must have completed its last iteration and entered
            # its next blocking receive before the stop flag changes. The
            # state writer reads state then command, so only its first receive
            # is a loop boundary.
            first_receivers = [*self.collector.image_subscribers.values(),
                               self.collector.state_socket, self.collector.reskin_subscriber]
            for sub in first_receivers:
                if not sub.waiting.wait(10):
                    raise RuntimeError('Official recorder did not reach its receive boundary')
            self.collector.run_event.clear()
            # A final actual sample unblocks the original blocking receivers.
            self._advance_hold()
            self._capture(False, force_camera=True)
            for thread in self.workers:
                thread.join(timeout=10)
            if self.errors or any(t.is_alive() for t in self.workers):
                raise RuntimeError('; '.join(self.errors) or 'Official recorder worker did not exit')
            self.metadata.update(status=status, camera_frames=self.frames, state_samples=self.samples)
            self._write_metadata()
            # The unmodified upstream processor filters by these name tokens.
            # Do not let failed/interrupted/unreviewed demonstrations silently
            # enter behavior cloning because it doesn't read our metadata.
            if status != 'success':
                suffix = '_fail' if status == 'failure' else '_ignore'
                target = path.with_name(path.name+suffix)
                if target.exists():
                    raise FileExistsError(target)
                path.rename(target)
                path = target
            return path
        except BaseException as exc:
            self.collector.run_event.clear()
            for sub in self.subscribers:
                sub.abort()
            for thread in self.workers:
                thread.join(timeout=1)
            self.metadata.update(status='failed', error=str(exc))
            self._write_metadata()
            ignored = path.with_name(path.name+'_ignore_failed')
            if path.exists() and not ignored.exists():
                path.rename(ignored)
            raise
        finally:
            self.collector = None
            self.writer = None

    def close(self):
        try:
            self.finish('interrupted')
        finally:
            self.renderer.close()

    def abort(self, reason):
        """Stop failed acquisition without advancing invalid simulation state.

        Upstream writers may have incomplete buffered output; the directory
        is explicitly excluded from preprocessing, never a training demo.
        """
        if not self.active:
            return None
        path = self.writer.path
        self.collector.run_event.clear()
        for subscriber in self.subscribers:
            subscriber.abort()
        for worker in self.workers:
            worker.join(timeout=1.)
        self.metadata.update(status='failed', error=reason, training_ready=False,
                             camera_frames=self.frames, state_samples=self.samples)
        self._write_metadata()
        ignored = path.with_name(path.name+'_ignore_failed')
        path.rename(ignored)
        self.collector = self.writer = None
        return ignored
