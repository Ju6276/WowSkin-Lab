"""Backend-independent episode storage. Each frame is durable before returning."""
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import numpy as np

SCHEMA_VERSION = 1


def magnetic_baseline(env):
    """Match the baseline to the selected observation backend, in its channel order."""
    response = getattr(env, 'reskin_response', None)
    return (np.asarray(response.model.meta['baseline_ut']) if response
            else env.baseline.reshape(15))


def magnetic_metadata(env):
    response = getattr(env, 'reskin_response', None)
    if not response:
        return {}
    return dict(magnetic_source='3dcal_local_mlp', channel_mapping=response.profile,
                contact_model='single-probe proxy of a cube patch; unvalidated',
                simulation_proxy_only=True, real_transfer_validated=False,
                magnetic_model_metadata=response.model.meta)


class EpisodeWriter:
    def __init__(self, directory, metadata):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        self.path = Path(directory)/f'episode_{stamp}_{uuid.uuid4().hex[:8]}'
        (self.path/'frames').mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata, schema_version=SCHEMA_VERSION, status='in_progress',
                             frame_count=0, created_utc=datetime.now(timezone.utc).isoformat())
        self.count, self.previous_time, self.controls = 0, None, []
        self.closed = False
        self._write_metadata()

    def _write_metadata(self):
        temp = self.path/'metadata.tmp'
        temp.write_text(json.dumps(self.metadata, indent=2, ensure_ascii=False)+'\n')
        temp.replace(self.path/'metadata.json')

    def append(self, sample):
        if self.closed:
            raise RuntimeError('Episode is closed')
        t = float(sample['time_s'])
        if self.previous_time is not None and t <= self.previous_time:
            raise ValueError('Episode timestamps must strictly increase; close before reset')
        values = {k: np.asarray(v) for k, v in sample.items()}
        if any(v.dtype.kind == 'O' or not np.isfinite(v).all() for v in values.values()):
            raise ValueError('Refusing nonfinite or object arrays in a training recording')
        if values['cartesian_states'].shape != (6,) or values['sensor_states'].shape != (15,):
            raise ValueError('Expected XYZ+rotvec and one 15-channel tactile sensor')
        target = self.path/'frames'/f'{self.count:06d}.npz'
        with target.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream, **values)
        target.with_suffix('.tmp').replace(target)
        self.count += 1
        self.previous_time = t

    def control(self, time_s, action, magnetic_ut, observation_time_s):
        self.controls.append((time_s, np.asarray(action).copy(),
                              np.asarray(magnetic_ut).copy(), observation_time_s))

    def finish(self, status='unreviewed'):
        if status not in ('success', 'failure', 'unreviewed', 'interrupted'):
            raise ValueError(status)
        if self.closed:
            return self.path
        if self.controls:
            np.savez_compressed(self.path/'controls.npz',
                                action_time_s=np.array([x[0] for x in self.controls]),
                                actions=np.array([x[1] for x in self.controls]),
                                magnetic_ut=np.array([x[2] for x in self.controls]),
                                observation_time_s=np.array([x[3] for x in self.controls]))
        self.metadata.update(status=status, frame_count=self.count,
                             finished_utc=datetime.now(timezone.utc).isoformat())
        self._write_metadata()
        self.closed = True
        return self.path


class MujocoRecorder:
    """Simulation adapter: cameras, poses and tactile refer to one sim timestamp."""
    def __init__(self, env, directory, hz=10, source='keyboard'):
        from .cameras import CameraRig
        ratio = env.cfg['sample_hz']/hz if hz > 0 else 0
        if ratio < 1 or not np.isclose(ratio, round(ratio)):
            raise ValueError('record-hz must be positive and divide the simulation sample rate')
        self.env, self.directory, self.hz = env, Path(directory), hz
        self.stride, self.source = round(ratio), source
        self.rig, self.writer = CameraRig(env), None
        self.steps = 0

    @property
    def active(self):
        return self.writer is not None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def start(self):
        if self.active:
            raise RuntimeError('Finish the active episode before starting another')
        from .cameras import CAMERAS
        metadata = dict(task='pick_cube', backend='mujoco', source=self.source,
                        record_hz=self.hz, control_hz=self.env.cfg['sample_hz'],
                        camera_keys=list(CAMERAS), color_order='RGB', image_size=[128, 128],
                        tactile_unit='uT', sensor_baseline_subtracted=True,
                        center_sensor_index=0, point_source='mujoco_ground_truth',
                        pose='world xyz metres + world orientation rotvec radians',
                        command='world dxyz metres + local drotvec radians + closure target 0..1',
                        gripper='actual closure fraction 0=open, 1=closed',
                        uncalibrated=True, config=self.env.cfg,
                        magnetic_baseline_ut=magnetic_baseline(self.env).tolist())
        metadata.update(magnetic_metadata(self.env))
        self.writer = EpisodeWriter(self.directory, metadata)
        self.steps = 0
        self.capture()
        print(f'Recording: {self.writer.path}', flush=True)

    def capture(self):
        from wowskin_lab.envs.pick_cube import rotation_vector
        e = self.env
        response = getattr(e, 'reskin_response', None)
        if response:
            raw_field = (e.last_row['raw_field_ut'] if e.last_row
                         else np.asarray(response.model.meta['baseline_ut']))
        else:
            raw_field = e.magnetic_field().reshape(15)
        sample = self.rig.capture()
        sample.update(time_s=e.data.time,
                      cartesian_states=np.r_[e.data.site_xpos[e.tcp_site],
                                             rotation_vector(e.data.site_xmat[e.tcp_site])],
                      gripper_states=e.gripper_fraction(),
                      gripper_target=e.gripper_fraction(target=True),
                      sensor_states=e.observation()['magnetic_ut'],
                      raw_magnetic_ut=raw_field,
                      qpos=e.data.qpos.copy(), qvel=e.data.qvel.copy(),
                      contact_force_n=(e.last_row['normal_force_n'] if e.last_row else 0.),
                      grasp_success=float(e.success))
        if response:
            row = e.last_row or {}
            query = np.asarray(row.get('reskin_query_mm', [np.nan]*3))
            sample.update(reskin_query_mm=np.nan_to_num(query),
                          reskin_query_present=bool(np.isfinite(query).all()),
                          reskin_input_in_training_domain=bool(row.get('reskin_in_training_domain', False)),
                          reskin_transfer_validated=False)
        self.writer.append(sample)

    def step(self, row):
        if not self.active:
            return
        self.writer.control(row['action_time_s'], row['action'], row['magnetic_ut'], row['time_s'])
        self.steps += 1
        if self.steps % self.stride == 0:
            self.capture()

    def finish(self, status='unreviewed'):
        if not self.active:
            return None
        path = self.writer.finish(status)
        print(f'Saved {self.writer.count} synchronized frames ({status}): {path}', flush=True)
        self.writer = None
        return path

    def close(self):
        try:
            self.finish('interrupted')
        finally:
            self.rig.close()
