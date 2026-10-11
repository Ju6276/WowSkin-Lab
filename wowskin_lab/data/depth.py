"""Lossless metric RGB-aligned depth extension, independent of RGB codecs.

LeRobot v2.1 / original GR00T N1 do not load this extension automatically.
The raw camera index is preserved in Parquet so future fusion can read it.
"""
from pathlib import Path

import h5py
import numpy as np

DEPTH_PATH = 'depth/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.h5'


class DepthWriter:
    def __init__(self, path, model):
        self.file = h5py.File(path, 'w')
        self.near = float(model.vis.map.znear * model.stat.extent)
        self.far = float(model.vis.map.zfar * model.stat.extent)
        self.file.attrs.update(schema='wowskin_metric_depth_v1', units='metres',
            convention='OpenCV optical +Z, not Euclidean ray range',
            alignment='Same scene, camera, pixels and simulation time as RGB',
            source='MuJoCo renderer ground truth; no real depth-camera noise model',
            invalid='valid_cameraN excludes nonfinite/near/far-plane pixels; original depth values are preserved',
            near_m=self.near, far_m=self.far)

    def _append(self, key, value, image=False):
        value = np.asarray(value)
        if key not in self.file:
            self.file.create_dataset(key, shape=(0, *value.shape), maxshape=(None, *value.shape),
                dtype=value.dtype, chunks=(1, *value.shape) if image else True,
                compression='lzf' if image else None, shuffle=image)
        out = self.file[key]
        out.resize(len(out)+1, axis=0)
        out[-1] = value

    def camera(self, camera, depth, intrinsic):
        depth = np.asarray(depth, dtype=np.float32)
        if depth.shape != (480, 640) or not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ValueError('Invalid metric depth from renderer')
        valid = (depth >= self.near) & (depth < self.far * (1-1e-5))
        self._append(f'camera{camera}', depth, image=True)
        self._append(f'valid_camera{camera}', valid, image=True)
        if f'intrinsic_camera{camera}' not in self.file:
            self.file.create_dataset(f'intrinsic_camera{camera}', data=intrinsic)

    def finish_frame(self, timestamp, camera_base_to_cv, camera_world_to_cv):
        self._append('timestamp', timestamp)
        self._append('camera_base_to_cv', camera_base_to_cv)
        self._append('camera_world_to_cv', camera_world_to_cv)
        self.file.flush()

    def close(self):
        self.file.close()


def load_depth_frame(dataset_root, episode_index, raw_camera_index):
    """Use a LeRobot row's episode_index / aux.raw_camera_index to read RGB-D.

    The file includes genuine pre/post-roll frames. Do not use frame_index as
    a depth index: the LeRobot training episode excludes those boundaries.
    """
    import json
    root = Path(dataset_root)
    info = json.loads((root/'meta/info.json').read_text())
    path = root/DEPTH_PATH.format(episode_chunk=episode_index//info['chunks_size'],
                                 episode_index=episode_index)
    index = int(raw_camera_index)
    with h5py.File(path) as stream:
        result = dict(timestamp=float(stream['timestamp'][index]),
                      camera_base_to_cv=stream['camera_base_to_cv'][index],
                      camera_world_to_cv=stream['camera_world_to_cv'][index])
        for camera in sorted(int(key.removeprefix('camera')) for key in stream
                             if key.startswith('camera') and key.removeprefix('camera').isdigit()):
            result[f'camera{camera}'] = stream[f'camera{camera}'][index]
            result[f'valid_camera{camera}'] = stream[f'valid_camera{camera}'][index]
            result[f'intrinsic_camera{camera}'] = stream[f'intrinsic_camera{camera}'][:]
    return result
