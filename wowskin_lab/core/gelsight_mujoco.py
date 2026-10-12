"""Pinned GelSightMujoco sensor and original Gomes/Phong image pipeline.

The author's depth input is the legacy OpenGL [0, 1] buffer, not metres.
Keep it separately from metric geometric indentation. No texture/intensity
gain is fitted here. The original source and CAD stay in third_party.
"""
from functools import lru_cache
from types import ModuleType

import mujoco
import numpy as np

from wowskin_lab.paths import ROOT
from .optical_tactile import OpticalFrame, contact_forces

SOURCE = ROOT/'third_party/GelSightMujoco/Test_bed/envs'


@lru_cache(maxsize=1)
def source_depth_planes():
    """Preserve the source camera's depth-buffer calibration across scenes.

    The original optical driver consumes nonlinear OpenGL depth, so copying
    only its constants while inheriting the host scene's clipping planes
    changes its input. Compile the pinned source to obtain its metric planes;
    do not make the sensor's response depend on the Panda room's extent.
    """
    model = mujoco.MjModel.from_xml_path(str(SOURCE/'models/gripper_testbed_PickandPlace.xml'))
    return tuple(float(np.float32(value*model.stat.extent))
                 for value in (model.vis.map.znear, model.vis.map.zfar))


@lru_cache(maxsize=1)
def source_config():
    """Read the original factory's literals, without importing old Gym/ROS."""
    import ast
    tree = ast.parse((SOURCE/'RobotArm__BaseEnv.py').read_text())
    factory = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_get_simapproach')
    values = {}
    for n in factory.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                values[n.targets[0].id] = ast.literal_eval(n.value)
            except (ValueError, TypeError):
                pass
    return {key: values[key] for key in ('ka', 'texture_sigma', 'px2m_ratio', 'elastomer_thickness', 'min_depth')}, values['light_sources_smartlab2014']


class GelSightRenderer:
    def __init__(self, width=160, height=120, *, ideal=False):
        import cv2
        path = SOURCE/'utils/gelsight_driver.py'
        if not path.is_file():
            raise FileNotFoundError('Run scripts/setup_upstream.py --only gelsight_mujoco')
        module = ModuleType('wowskin_gelsight_source')
        # Keep the pinned checkout clean, including no generated pycache.
        exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
        params, lights = source_config()
        bg = cv2.imread(str(SOURCE/'models/assets/background.png'))
        bg = cv2.resize(bg, (width, height), interpolation=cv2.INTER_AREA)
        self.original = module.SimulationApproach(light_sources=lights, background_img=bg, **params)
        if ideal:
            # Constructor uses `sigma or default`, so change the public field
            # afterwards to actually turn noise off. Default is unchanged.
            self.original.texture_sigma = 0.

    def render(self, raw_depth):
        rgb, deformed = self.original.generate(raw_depth.copy(), return_depth=True)
        return rgb[..., ::-1].copy(), deformed


class GelSightMujocoOptical:
    label = 'GelSight / upstream MuJoCo + Phong'

    def __init__(self, model, sensors, *, reference_data, ideal=False, width=160, height=120):
        self.model, self.sensors = model, sensors
        self.optics = GelSightRenderer(width, height, ideal=ideal)
        self.near, self.far = source_depth_planes()
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        # Modern Renderer depth mode makes transparent gel/glass opaque by
        # switching to segmentation. The source reads depth after normal RGB
        # rendering instead; preserve that visibility and the legacy map.
        self.renderer._mjr_context.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZERONEAR
        self.baselines = {}
        for name, (camera, _) in sensors.items():
            self.baselines[name] = self.depth(reference_data, camera)[0]

    def depth(self, data, camera):
        self.renderer.update_scene(data, camera=camera)
        # Change only this offscreen sensor camera. Preserve its field of view
        # while fixing the source metric near/far planes; task cameras and the
        # global model remain untouched.
        for gl_camera in self.renderer.scene.camera:
            if gl_camera.orthographic:
                raise ValueError('The source GelSight camera must be perspective')
            scale = self.near/gl_camera.frustum_near
            for field in ('frustum_bottom', 'frustum_top', 'frustum_center', 'frustum_width'):
                setattr(gl_camera, field, getattr(gl_camera, field)*scale)
            gl_camera.frustum_near, gl_camera.frustum_far = self.near, self.far
        self.renderer.render()
        raw = np.empty((self.renderer.height, self.renderer.width), dtype=np.float32)
        mujoco.mjr_readPixels(None, raw, self.renderer._rect, self.renderer._mjr_context)
        near, far = self.near, self.far
        depth = near*far/(far-raw.astype(np.float64)*(far-near))
        return depth, raw

    def render(self, data):
        frames = {}
        near, far = self.near, self.far
        for name, (camera, gel) in self.sensors.items():
            depth, raw = self.depth(data, camera)
            rgb, native_deformation = self.optics.render(raw)
            baseline = self.baselines[name]
            valid = (baseline > near) & (baseline < far) & (depth > 0)
            indentation = np.where(valid, baseline-depth, 0.)
            forces = contact_forces(self.model, data, self.model.geom(gel).id)
            frames[name] = OpticalFrame(rgb, indentation, sum(forces.values()), tuple(forces),
                float(data.time), contact_mask=valid & (indentation > 0), valid_mask=valid,
                camera_depth_m=depth, baseline_depth_m=baseline.copy(),
                render_input_depth=raw, render_deformation=native_deformation,
                render_depth_unit='legacy_opengl_buffer')
        return frames

    def reset(self):
        # Rigid sensor baseline is unchanged by global motion or world reset.
        pass

    def close(self):
        self.renderer.close()
