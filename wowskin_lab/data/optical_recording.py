"""Lossless dual optical observations, separate from 100 Hz robot state.

One frame per actual RGB-D capture; no fabricated magnetic channels, marker
flow or optical frames. The source algorithm's native arrays are retained.
"""
from dataclasses import fields
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

OPTICAL_PATH = 'optical/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.h5'
SCHEMA = 'wowskin_dual_optical_v1'


def optical_path(source, meta):
    return (Path(source)/meta['optical']['path']).resolve()


def optical_metadata(env):
    from wowskin_lab.paths import ROOT
    backend = env.cfg['sensor']
    reference = {'tacto': 'tacto', 'digit_fots': 'fots', 'gelsight': 'gelsight_mujoco'}[backend]
    return dict(schema=SCHEMA, backend=backend, hardware='GelSight' if backend=='gelsight' else 'DIGIT',
        rgb_shape=[120,160,3] if backend=='gelsight' else [320,240,3],
        sides={'left': 'robot left_finger', 'right': 'robot right_finger'},
        clock='camera_timestamp; actual samples at nominal 30 Hz; not 100 Hz duplication',
        rgb='native uint8 RGB, lossless in this file; LeRobot also has encoded RGB video',
        depth='indentation_m, camera_depth_m, baseline_depth_m in metres; source render buffers stored separately with their native units',
        markers='FOTS native displacement in pixels and flow visualization when available; absent means not provided, never zero-filled',
        contact='All colliding scene bodies, not only the task target',
        source_manifest=json.loads((ROOT/f'assets/reference/{reference}/manifest.json').read_text()),
        optical_ideal=env.cfg.get('optical_ideal', False), physics_calibrated_to_hardware=False)


class OpticalWriter:
    def __init__(self, path, env, optics):
        self.file = h5py.File(path, 'w')
        self.file.attrs['metadata'] = json.dumps(optical_metadata(env))
        self.env, self.optics = env, optics
        self.count = 0
        self.file.create_dataset('timestamp', shape=(0,), maxshape=(None,), dtype='f8')

    def capture(self):
        frames = self.optics.render(self.env.data)
        if set(frames) != {'left', 'right'}:
            raise ValueError('Expected both robot fingers, without missing optical observations')
        group = self.file.create_group(f'frames/{self.count:06d}')
        for side, frame in frames.items():
            if not np.isclose(frame.time_s, self.env.data.time, atol=1e-9, rtol=0):
                raise ValueError('Stale optical frame')
            sensor = group.create_group(side)
            for field in fields(frame):
                value = getattr(frame, field.name)
                if value is None:
                    continue
                if isinstance(value, np.ndarray):
                    if not np.isfinite(value).all():
                        raise ValueError(f'Nonfinite optical array: {side}/{field.name}')
                    sensor.create_dataset(field.name, data=value, compression='lzf')
                else:
                    sensor.attrs[field.name] = value
            camera, _ = self.env.sensors[side]
            if mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera) >= 0:
                cid = self.env.model.camera(camera).id
                sensor.create_dataset('world_from_optical_camera', data=np.block([
                    [self.env.data.cam_xmat[cid].reshape(3,3), self.env.data.cam_xpos[cid,None].T],
                    [np.zeros((1,3)), np.ones((1,1))]]))
                sensor.attrs['mujoco_camera_fovy_deg'] = self.env.model.cam_fovy[cid]
            else:
                sid = self.env.model.site(camera).id
                sensor.create_dataset('world_from_optical_camera', data=np.block([
                    [self.env.data.site_xmat[sid].reshape(3,3), self.env.data.site_xpos[sid,None].T],
                    [np.zeros((1,3)), np.ones((1,1))]]))
            sensor.attrs['camera_axes'] = 'MuJoCo/OpenGL camera +X right, +Y up, -Z forward; source image readback orientation in source manifest'
        timestamps = self.file['timestamp']
        timestamps.resize(self.count+1, axis=0)
        timestamps[-1] = self.env.data.time
        self.count += 1
        self.file.flush()

    def close(self):
        self.file.close()


def validate_optical(path, times):
    with h5py.File(path) as f:
        if json.loads(f.attrs['metadata'])['schema'] != SCHEMA or not np.array_equal(f['timestamp'][:], times):
            raise ValueError('Optical schema or synchronized timestamps differ')
        if len(f['frames']) != len(times):
            raise ValueError('Missing optical frames')
        for i, t in enumerate(times):
            for side in ('left', 'right'):
                g = f[f'frames/{i:06d}/{side}']
                if not np.isclose(g.attrs['time_s'], t, atol=1e-9, rtol=0):
                    raise ValueError('Stale optical sample')
                rgb, depth = g['rgb'], g['indentation_m']
                if rgb.dtype != np.uint8 or rgb.shape != (*depth.shape, 3):
                    raise ValueError('Invalid optical RGB/depth dimensions')
                for value in g.values():
                    if not np.isfinite(value[:]).all():
                        raise ValueError('Nonfinite saved optical values')
