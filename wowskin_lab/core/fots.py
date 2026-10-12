"""Pinned FOTS GelSight optics with a MuJoCo geometric-depth adapter.

The upstream MLP, preprocessing and smoothing run unchanged. Depth is rigid
contact overlap, not a force-to-indentation law or an elastic gel solution.
"""
import builtins
from functools import lru_cache
from types import ModuleType, SimpleNamespace

import numpy as np
import torch

from wowskin_lab.paths import ROOT
from .optical_tactile import OpticalFrame, contact_forces, geom_mesh

FOTS_ROOT = ROOT/'third_party/FOTS'


@lru_cache(maxsize=1)
def upstream():
    """Load upstream's absolute imports privately, without altering sys.path.

    Old preproc assigns torch.seed = 42. A module-local facade prevents that
    assignment from replacing PyTorch's global function. NumPy 2's removed
    mat alias is provided only to the upstream shadow module.
    """
    if not (FOTS_ROOT/'utils/fots_render.py').exists():
        raise FileNotFoundError('Run .venv/bin/python scripts/setup_upstream.py --only fots')
    modules = {}

    def load(name, path, extra=None):
        module = ModuleType('wowskin_fots_'+name)
        imports = dict(modules)
        imports.update(extra or {})

        def private_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and name in imports:
                return imports[name]
            return builtins.__import__(name, globals, locals, fromlist, level)

        module.__dict__['__builtins__'] = dict(vars(builtins), __import__=private_import)
        file = FOTS_ROOT/path
        exec(compile(file.read_text(), str(file), 'exec'), module.__dict__)
        modules[name] = module
        return module

    params = load('params', 'params.py')
    mlp = load('mlp_calib.src.train.mlp_model', 'mlp_calib/src/train/mlp_model.py')
    preproc = load('utils.prepost_mlp', 'utils/prepost_mlp.py', {'torch': SimpleNamespace(**vars(torch))})
    preproc.device = torch.device('cpu')
    load('shadow_calib.planar_shadow', 'shadow_calib/planar_shadow.py',
         {'numpy': SimpleNamespace(**dict(vars(np), mat=np.asmatrix))})
    renderer = load('utils.fots_render', 'utils/fots_render.py')
    return params, mlp.MLP, renderer.FOTSRender


class FOTSRenderer:
    """Official GelSight calibration, native 240x320 BGR -> public RGB."""
    def __init__(self):
        self.params, MLP, Render = upstream()
        self.model = MLP().cpu().eval()
        weights = FOTS_ROOT/'mlp_calib/models/mlp_n2c_gelsight.pth'
        self.model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        self.renderer = Render(
            background_img=np.load(FOTS_ROOT/'assets/gel/gelsight_bg.npy'),
            bg_render=np.load(FOTS_ROOT/'utils/utils_data/ini_bg_fots_gelsight.npy'),
            model=self.model)

    def render(self, indentation_m):
        expected = (self.params.sensor_h, self.params.sensor_w)
        depth = np.asarray(indentation_m)
        if depth.shape != expected or not np.isfinite(depth).all() or (depth < 0).any():
            raise ValueError(f'FOTS requires finite nonnegative indentation in metres, shape {expected}')
        # Author's fots_test.py uses pixel-unit height and the same contact mask.
        with torch.inference_mode():
            bgr = self.renderer.generate(depth*1000*self.params.mm_to_pixel, depth > 0, shadow=False)
        return bgr[..., ::-1].copy()


class FOTSOptical:
    label = 'GelSight / FOTS (rigid geometry proxy)'

    def __init__(self, model, sensors, *, ideal=False):
        import pyrender
        from .optical_depth import depth_renderer
        self.model, self.optics = model, FOTSRenderer()
        p = self.optics.params
        self.sensors = {name: (model.site(site).id, model.geom(gel).id)
                        for name, (site, gel) in sensors.items()}
        self.r = depth_renderer(p.sensor_w, p.sensor_h)
        self.scene = pyrender.Scene()
        self.nodes = {}
        # Orthographic pixels match FOTS's calibrated mm_to_pixel grid.
        # Camera behind surface: +sensor X right, +sensor Y down, +Z outward.
        self.camera_distance_m = .02
        pose = np.diag([1., -1., -1., 1.])
        pose[2, 3] = -self.camera_distance_m
        self.scene.add(pyrender.OrthographicCamera(
            xmag=p.sensor_w/p.mm_to_pixel/2000,
            ymag=p.sensor_h/p.mm_to_pixel/2000, znear=.00001, zfar=.1), pose=pose)

    def render(self, data):
        import pyrender
        frames = {}
        for name, (site, gel) in self.sensors.items():
            forces = contact_forces(self.model, data, gel)
            for node in list(self.nodes.values()):
                if self.scene.has_node(node):
                    self.scene.remove_node(node)
            rotation = data.site_xmat[site].reshape(3, 3).T
            for geom in forces:
                if geom not in self.nodes:
                    mesh = pyrender.Mesh.from_trimesh(geom_mesh(self.model, geom), smooth=False)
                    self.nodes[geom] = pyrender.Node(mesh=mesh)
                node = self.nodes[geom]
                self.scene.add_node(node)
                pose = np.eye(4)
                pose[:3, :3] = rotation @ data.geom_xmat[geom].reshape(3, 3)
                pose[:3, 3] = rotation @ (data.geom_xpos[geom]-data.site_xpos[site])
                self.scene.set_pose(node, pose)
            depth = self.r.render(self.scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
            # Intersect object surface with the author's flat gel plane.
            # Background depth=0 is no geometry. No force scaling/whitelist.
            indentation = np.where(depth > 0, np.maximum(self.camera_distance_m-depth, 0), 0)
            frames[name] = OpticalFrame(self.optics.render(indentation), indentation,
                                       sum(forces.values()), tuple(forces), float(data.time))
        return frames

    def close(self):
        self.r.delete()
