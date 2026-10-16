"""Live acquisition-camera previews, independent of demonstration recording."""
import time
from types import SimpleNamespace

import mujoco
import numpy as np

from wowskin_lab.data.cameras import CAMERAS, capture_options
from wowskin_lab.data.pointcloud import RGBDView
from wowskin_lab.data.robot_state import camera_from_world, world_from_base
from wowskin_lab.ui.pointcloud import OnlinePointCloud


class CameraPreview:
    def __init__(self, model, data, hud, guides=None, *, pointcloud=None, gpu_frames=None):
        self.model, self.data, self.hud = model, data, hud
        self.renderer = None
        self.row = None
        self.next_frame = 0.
        self.guides = guides
        self.cloud = OnlinePointCloud(**(pointcloud or {}))
        self.last_time = None
        self.gpu_frames = gpu_frames

    def __enter__(self):
        # Same cameras, aspect ratio and full RGB size used by acquisition.
        if self.gpu_frames is None:
            self.renderer = mujoco.Renderer(self.model, width=640, height=480)
        try:
            if hasattr(self.hud, "bind"):
                self.hud.bind(self.model, self.data)
        except BaseException:
            if self.renderer is not None:
                self.renderer.close()
            raise
        return self

    def __exit__(self, *exc):
        try:
            self.cloud.close()
        finally:
            if self.renderer is not None:
                self.renderer.close()
            if hasattr(self.hud, "close"):
                self.hud.close()

    def key(self, code):
        handled = self.cloud.key(code)
        if handled:
            self.next_frame = 0.
        return handled

    def show(self, handle, row=None):
        if row is not None:
            self.row = row
        if self.row is None or not handle.is_running() or time.monotonic() < self.next_frame:
            return
        frame = self.gpu_frames() if self.gpu_frames is not None else None
        if self.gpu_frames is not None and frame is None:
            return
        data = self.data if frame is None else frame[0]
        if self.last_time is not None and self.data.time < self.last_time:
            self.cloud.clear()
        self.last_time = float(self.data.time)
        self.cloud.poll()
        capture_cloud = self.cloud.ready()
        if capture_cloud:
            snapshot = SimpleNamespace(model=self.model, data=data)
            world_base = world_from_base(snapshot)
            near = self.model.vis.map.znear*self.model.stat.extent
            far = self.model.vis.map.zfar*self.model.stat.extent
        capture_started = time.perf_counter()
        if self.guides:
            self.guides.show(handle, self.model, self.data, self.row)
        frames, views = [], []
        for key, camera in CAMERAS.items():
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
                continue
            label = 'WRIST' if key == 'pixels52' else f'CAM {key.removeprefix("pixels")}'
            if frame is None:
                self.renderer.update_scene(self.data, camera=camera, scene_option=capture_options())
                rgb = self.renderer.render().copy()
            else:
                rgb = frame[1][key]
            frames.append((f'{label} | {camera} | t={data.time:.2f}s', rgb))
            if capture_cloud and (self.cloud.mode == 3 or key != 'pixels52'):
                # Same detached snapshot and same scene as RGB. Never step or
                # insert operator guides between modalities or camera views.
                if frame is None:
                    self.renderer.enable_depth_rendering()
                    try:
                        depth = self.renderer.render().copy()
                    finally:
                        self.renderer.disable_depth_rendering()
                else:
                    depth = frame[2][key]
                cid = self.model.camera(camera).id
                focal = 240/np.tan(np.deg2rad(self.model.cam_fovy[cid])/2)
                intrinsic = np.array([[focal,0,319.5],[0,focal,239.5],[0,0,1]])
                views.append(RGBDView(int(key.removeprefix('pixels')), float(data.time),
                    rgb, depth, intrinsic, camera_from_world(snapshot, cid) @ world_base, near, far))
        if capture_cloud:
            self.cloud.submit(views, capture_ms=(time.perf_counter()-capture_started)*1000)
        self.cloud.poll()
        self.hud.show(handle, self.row, cameras=frames, pointcloud=self.cloud.panel)
        self.next_frame = time.monotonic()+.1  # Preview only; recording clocks unchanged.
