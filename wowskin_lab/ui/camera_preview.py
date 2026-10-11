"""Live acquisition-camera previews, independent of demonstration recording."""
import time

import mujoco

from wowskin_lab.data.cameras import CAMERAS, capture_options


class CameraPreview:
    def __init__(self, model, data, hud, guides=None):
        self.model, self.data, self.hud = model, data, hud
        self.renderer = None
        self.row = None
        self.next_frame = 0.
        self.guides = guides

    def __enter__(self):
        # Same cameras, aspect ratio and full RGB size used by acquisition.
        self.renderer = mujoco.Renderer(self.model, width=640, height=480)
        return self

    def __exit__(self, *exc):
        self.renderer.close()

    def show(self, handle, row=None):
        if row is not None:
            self.row = row
        if self.row is None or not handle.is_running() or time.monotonic() < self.next_frame:
            return
        if self.guides:
            self.guides.show(handle, self.model, self.data, self.row)
        frames = []
        for key, camera in CAMERAS.items():
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
                continue
            label = 'WRIST' if key == 'pixels52' else f'CAM {key.removeprefix("pixels")}'
            self.renderer.update_scene(self.data, camera=camera, scene_option=capture_options())
            frames.append((f'{label} | {camera} | t={self.data.time:.2f}s',
                           self.renderer.render().copy()))
        self.hud.show(handle, self.row, cameras=frames)
        self.next_frame = time.monotonic()+.1  # Preview only; recording clocks unchanged.
