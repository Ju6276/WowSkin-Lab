"""Official MuJoCo Warp scene RGB-D renderer, directly from CUDA physics state.

An explicit renderer choice: ray-traced RGB is not pixel-identical to native
OpenGL. Source tactile optics use their own renderers and are not substituted.
"""
import numpy as np
import mujoco_warp as mjw
import warp as wp

from wowskin_lab.data.cameras import active_cameras


@wp.kernel
def _rgb(packed: wp.array2d[wp.uint32], rgb: wp.array2d[wp.uint8]):
    i = wp.tid()
    value = packed[0, i]
    rgb[i, 0] = wp.uint8((value >> wp.uint32(16)) & wp.uint32(255))
    rgb[i, 1] = wp.uint8((value >> wp.uint32(8)) & wp.uint32(255))
    rgb[i, 2] = wp.uint8(value & wp.uint32(255))


class WarpCameras:
    def __init__(self, env, width=640, height=480):
        self.env, self.width, self.height = env, width, height
        self.backend = None
        self.depth_mode = False
        self.camera = None
        self.stamp = None

    def _prepare(self):
        from wowskin_lab.core.stepping import prepare
        backend = prepare(self.env)
        if backend is None:
            raise ValueError('Warp scene cameras require --physics-backend warp')
        if self.backend is backend:
            return
        self.backend = backend
        sim = backend.sim
        self.context = mjw.create_render_context(self.env.model, cam_res=(self.width, self.height),
            cam_active=list(active_cameras(self.env).values()), render_rgb=True, render_depth=True,
            render_seg=False, use_shadows=True, render_skybox=True,
            enabled_geom_groups=[0, 1, 2, 4])
        self.ids = list(map(int, self.context.cam_id_map.numpy()))
        self.rgb_offsets = self.context.rgb_adr.numpy()
        self.depth_offsets = self.context.depth_adr.numpy()
        self.rgb = wp.empty((self.context.rgb_data.size, 3), dtype=wp.uint8)
        self._render()
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            self._render()
        self.graph = capture.graph
        self.stamp = None

    def _render(self):
        sim = self.backend.sim
        mjw.refit_bvh(sim.model, sim.data, self.context)
        mjw.render(sim.model, sim.data, self.context)
        wp.launch(_rgb, self.context.rgb_data.size, [self.context.rgb_data, self.rgb])

    def update_scene(self, data, *, camera, scene_option=None):
        self._prepare()
        if data is not self.env.data:
            raise ValueError('Warp cameras render only their live GPU environment')
        self.camera = self.env.model.camera(camera).id if isinstance(camera, str) else camera
        stamp = (self.backend.steps, float(data.time))
        if stamp != self.stamp:
            wp.capture_launch(self.graph)
            self.rgb_host = self.rgb.numpy()
            self.depth_host = self.context.depth_data.numpy()[0]
            self.stamp = stamp

    def render(self):
        index = self.ids.index(self.camera)
        size = self.width*self.height
        if self.depth_mode:
            start = int(self.depth_offsets[index])
            return self.depth_host[start:start+size].reshape(self.height, self.width).copy()
        start = int(self.rgb_offsets[index])
        return self.rgb_host[start:start+size].reshape(self.height, self.width, 3).copy()

    def enable_depth_rendering(self):
        self.depth_mode = True

    def disable_depth_rendering(self):
        self.depth_mode = False

    def close(self):
        self.backend = None
        self.context = None
        self.graph = None
