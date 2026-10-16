"""FOTS author's MuJoCo branch: original perspective depth, MLP and markers.

No orthographic proxy, fitted depth gain, marker saturation or multi-object
superposition. Compatibility changes are private imports, CPU weight loading,
NumPy's removed mat alias and OpenGL context/readback. Source files stay clean.
"""
import builtins
from functools import lru_cache
from types import ModuleType, SimpleNamespace

import cv2
import mujoco
import numpy as np
import torch

from wowskin_lab.paths import ROOT
from .optical_tactile import OpticalFrame, contact_forces

SOURCE = ROOT/'third_party/FOTS-mujoco'


@lru_cache(maxsize=2)
def upstream(device='cpu'):
    modules = {}

    def load(name, path, extra=None):
        imports = dict(modules, **(extra or {}))
        def private_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and name in imports:
                return imports[name]
            return builtins.__import__(name, globals, locals, fromlist, level)
        module = ModuleType('wowskin_native_fots_'+name)
        module.__dict__['__builtins__'] = dict(vars(builtins), __import__=private_import)
        file = SOURCE/path
        exec(compile(file.read_text(), str(file), 'exec'), module.__dict__)
        modules[name] = module
        return module

    mlp = load('src.train.mlp_model', 'src/train/mlp_model.py')
    preproc = load('utils.prepost_mlp', 'utils/prepost_mlp.py', {'torch': SimpleNamespace(**vars(torch))})
    preproc.device = torch.device(device)
    load('planar_shadow', 'planar_shadow.py', {'numpy': SimpleNamespace(**dict(vars(np), mat=np.asmatrix))})
    render = load('utils.mlp_render', 'utils/mlp_render.py')
    marker = load('marker_motion', 'marker_motion.py')
    return mlp.MLP, render.MLPRender, marker.MarkerMotion


class NativeFOTSRenderer:
    def __init__(self, device='cpu'):
        MLP, self.Render, self.Marker = upstream(device)
        self.model = MLP().to(device).eval()
        self.model.load_state_dict(torch.load(SOURCE/'models/mlp_n2c_r.pth', map_location='cpu', weights_only=True))
        self.background = np.load(SOURCE/'assets/gel/digit_bg.npy')
        self.baseline = np.load(SOURCE/'utils/utils_data/ini_depth_extent.npy')
        self.bg_render = np.load(SOURCE/'utils/utils_data/ini_bg_mlp.npy')

    def render(self, depth, *, shadow=True):
        # env_test.py creates MLPRender afresh for every frame. generate()
        # mutates bg_depth, so persisting that array would cause false drift.
        renderer = self.Render(background_img=self.background, bg_depth=self.baseline.copy(),
                               bg_render=self.bg_render, model=self.model)
        with torch.inference_mode():
            bgr = renderer.generate(depth.copy(), shadow=shadow)
        return bgr[..., ::-1].copy()

    def markers(self, rgb, depth_diff, trajectory):
        Original = self.Marker
        class Measured(Original):
            def _motion_callback(self, xx, yy):
                result = super()._motion_callback(xx, yy)
                self.displacement = np.stack([result[0]-xx, result[1]-yy], axis=-1)
                return result
        marker = Measured(rgb[..., ::-1], depth_diff, depth_diff > 0,
                          trajectory, [.00125, .00021, .00038])
        image = marker._marker_motion()
        return image[..., ::-1].copy(), marker.displacement


class NativeFOTSOptical:
    label = 'DIGIT / FOTS author MuJoCo branch'

    def __init__(self, model, sensors, *, ideal=False, shadow=True, device='cpu'):
        self.model, self.sensors, self.shadow = model, sensors, shadow
        self.optics = NativeFOTSRenderer(device)
        original = mujoco.MjModel.from_xml_path(str(SOURCE/'assets/gel/fots_mujoco.xml'))
        self.near, self.far = (float(np.float32(value*original.stat.extent))
                              for value in (original.vis.map.znear, original.vis.map.zfar))
        self.renderer = mujoco.Renderer(model, height=320, width=240)
        self.renderer._mjr_context.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZERONEAR
        self.history, self.last_time = {}, None

    def depth(self, data, camera):
        self.renderer.update_scene(data, camera=camera)
        for gl_camera in self.renderer.scene.camera:
            scale = self.near/gl_camera.frustum_near
            for field in ('frustum_bottom', 'frustum_top', 'frustum_center', 'frustum_width'):
                setattr(gl_camera, field, getattr(gl_camera, field)*scale)
            gl_camera.frustum_near, gl_camera.frustum_far = self.near, self.far
        self.renderer.render()
        raw = np.empty((320, 240), dtype=np.float32)
        mujoco.mjr_readPixels(None, raw, self.renderer._rect, self.renderer._mjr_context)
        # Exactly env_test.py: legacy raw readback then fliplr (not flipud).
        return np.fliplr(raw).copy()

    def metric(self, raw):
        return self.near*self.far/(self.far-raw.astype(np.float64)*(self.far-self.near))

    def render(self, data):
        if self.last_time is not None and data.time < self.last_time:
            self.reset()
        self.last_time = float(data.time)
        frames = {}
        for name, (camera, gel) in self.sensors.items():
            raw = self.depth(data, camera)
            difference = self.optics.baseline-raw
            rgb = self.optics.render(raw, shadow=self.shadow)
            forces = contact_forces(self.model, data, self.model.geom(gel).id)
            bodies = {int(self.model.geom_bodyid[g]) for g in forces}
            previous_body, trajectory = self.history.get(name, (None, []))
            status = 'single_contact' if len(bodies) == 1 else ('no_contact' if not bodies else 'multiple_bodies_unsupported')
            if len(bodies) == 1 and np.max(difference) > 0:
                body = next(iter(bodies))
                # Author's reference is digit:site, not its camera or a
                # contact point that slides around as the solver updates.
                site_name = ('digit:site' if camera == 'digit:camera' else
                             camera.removeprefix('optical_').removesuffix('_camera')+'_surface')
                site = self.model.site(site_name).id
                rotation = data.site_xmat[site].reshape(3, 3).T
                relative = rotation@(data.xpos[body]-data.site_xpos[site])
                angle = cv2.Rodrigues(rotation@data.xmat[body].reshape(3, 3))[0][2, 0]
                point = [-relative[1], relative[0], -angle]
                # The source formula uses only first and last poses. Keeping
                # these two endpoints is identical, without an unbounded log.
                trajectory = ([trajectory[0], point] if body == previous_body and trajectory else [point])
                self.history[name] = (body, trajectory)
            else:
                trajectory = []
                self.history.pop(name, None)
            # Original FOTS specifies one object trajectory. RGB/depth still
            # see all objects, but never fabricate a multi-body marker field.
            flow, displacement = (None, None) if len(bodies) > 1 else self.optics.markers(rgb, difference, trajectory)
            baseline, depth = self.metric(self.optics.baseline), self.metric(raw)
            valid = (raw > 0) & (raw < 1)
            frames[name] = OpticalFrame(rgb, baseline-depth, sum(forces.values()), tuple(forces), float(data.time),
                marker_flow_rgb=flow, marker_displacement_px=displacement,
                marker_status=status, contact_mask=difference > 0, valid_mask=valid,
                camera_depth_m=depth, baseline_depth_m=baseline.copy(), render_input_depth=raw,
                render_deformation=difference, render_depth_unit='legacy_opengl_buffer',
                depth_model='fots_author_perspective_depth')
        return frames

    def reset(self):
        self.history.clear()
        self.last_time = None

    def close(self):
        self.renderer.close()
