"""RGB-D back-projection into robot-base coordinates; no simulated mesh points.

Input extrinsics map base -> OpenCV camera (+X right, +Y down, +Z forward).
Every bundle must describe one simulation instant, including the wrist pose.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RGBDView:
    camera_id: int
    timestamp: float
    rgb: np.ndarray
    depth: np.ndarray
    intrinsic: np.ndarray
    base_to_camera: np.ndarray
    near: float
    far: float


@dataclass(frozen=True)
class PointCloud:
    timestamp: float
    xyz: np.ndarray
    rgb: np.ndarray
    camera_ids: np.ndarray


def backproject(view, stride=1):
    if stride < 1:
        raise ValueError('Pixel stride must be positive')
    h, w = view.depth.shape
    if view.rgb.shape != (h, w, 3):
        raise ValueError('RGB and depth must be pixel aligned')
    k, ext = np.asarray(view.intrinsic), np.asarray(view.base_to_camera)
    if k.shape != (3, 3) or ext.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(ext).all():
        raise ValueError('Invalid camera calibration')
    v, u = np.mgrid[:h:stride, :w:stride]
    z = view.depth[::stride, ::stride]
    valid = np.isfinite(z) & (z > 0) & (z >= view.near) & (z < view.far*(1-1e-5))
    pixels = np.column_stack((u[valid], v[valid], np.ones(valid.sum())))
    camera = np.einsum('ij,nj->ni', np.linalg.inv(k), pixels)*z[valid, None]
    camera_to_base = np.linalg.inv(ext)
    xyz = np.einsum('ij,nj->ni', camera_to_base[:3, :3], camera)+camera_to_base[:3, 3]
    if not np.isfinite(xyz).all():
        raise ValueError('Non-finite reconstructed coordinates')
    return xyz, view.rgb[::stride, ::stride][valid]


def fuse_rgbd(views, *, stride=4, voxel_m=.002):
    views = tuple(views)
    if not views or len({v.camera_id for v in views}) != len(views):
        raise ValueError('Need distinct camera views')
    stamp = views[0].timestamp
    if not np.isfinite(stamp) or any(abs(v.timestamp-stamp) > 1e-9 or not np.isfinite(v.timestamp) for v in views):
        raise ValueError('Cannot fuse RGB-D from different simulation timestamps')
    if not np.isfinite(voxel_m) or voxel_m < 0:
        raise ValueError('Voxel size must be finite and nonnegative')
    points, colors, sources = [], [], []
    for view in views:
        xyz, rgb = backproject(view, stride)
        points.append(xyz); colors.append(rgb)
        sources.append(np.full(len(xyz), view.camera_id, dtype=np.int32))
    xyz, rgb, sources = np.concatenate(points), np.concatenate(colors), np.concatenate(sources)
    if voxel_m and len(xyz):
        # Preserve each view's samples/colors for the source-color diagnostic.
        # Spatial reduction never mixes different timestamps or invents surfaces.
        key = np.column_stack((sources, np.floor(xyz/voxel_m).astype(np.int64)))
        _, index = np.unique(key, axis=0, return_index=True)
        xyz, rgb, sources = xyz[index], rgb[index], sources[index]
    return PointCloud(stamp, xyz, rgb, sources)
