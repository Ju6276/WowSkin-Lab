"""Pinned FOTS DIGIT/GelSight optics with a MuJoCo geometric-depth adapter.

The upstream MLP, preprocessing and smoothing run unchanged. Depth is rigid
contact overlap, not a force-to-indentation law or an elastic gel solution.
"""
import builtins
from functools import lru_cache
from types import ModuleType, SimpleNamespace

import numpy as np
import mujoco
import torch
from scipy.spatial.transform import Rotation

from wowskin_lab.paths import ROOT
from .optical_tactile import OpticalFrame, contact_forces, geom_mesh

FOTS_ROOT = ROOT/'third_party/FOTS'


@lru_cache(maxsize=2)
def upstream(hardware='gelsight'):
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

    if hardware not in ('gelsight', 'digit'):
        raise ValueError('FOTS hardware must be gelsight or digit')
    params = load('params', 'params.py')
    params.sensor_type = hardware
    if hardware == 'digit':
        # DIGIT marker grid published in params.py (the alternate block).
        params.N, params.M = 9, 8
        params.x0, params.y0, params.dx, params.dy = 40, 27, 27, 27
    mlp = load('mlp_calib.src.train.mlp_model', 'mlp_calib/src/train/mlp_model.py')
    preproc = load('utils.prepost_mlp', 'utils/prepost_mlp.py', {'torch': SimpleNamespace(**vars(torch))})
    preproc.device = torch.device('cpu')
    load('shadow_calib.planar_shadow', 'shadow_calib/planar_shadow.py',
         {'numpy': SimpleNamespace(**dict(vars(np), mat=np.asmatrix))})
    renderer = load('utils.fots_render', 'utils/fots_render.py')
    marker = load('utils.marker_motion', 'utils/marker_motion.py',
                  {'numpy': SimpleNamespace(**dict(vars(np), random=np.random.RandomState(42)))})
    return params, mlp.MLP, renderer.FOTSRender, marker.MarkerMotion


class FOTSRenderer:
    """Hardware-specific source calibration, native 240x320 BGR -> RGB."""
    def __init__(self, hardware='gelsight'):
        self.hardware = hardware
        self.params, MLP, Render, self.MarkerMotion = upstream(hardware)
        self.model = MLP().cpu().eval()
        weights = FOTS_ROOT/f'mlp_calib/models/mlp_n2c_{hardware}.pth'
        self.model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        self.renderer = Render(
            background_img=np.load(FOTS_ROOT/f'assets/gel/{hardware}_bg.npy'),
            bg_render=np.load(FOTS_ROOT/f'utils/utils_data/ini_bg_fots_{hardware}.npy'),
            model=self.model)

    def render(self, indentation_m, *, shadow=False):
        expected = (self.params.sensor_h, self.params.sensor_w)
        depth = np.asarray(indentation_m)
        if depth.shape != expected or not np.isfinite(depth).all() or (depth < 0).any():
            raise ValueError(f'FOTS requires finite nonnegative indentation in metres, shape {expected}')
        # Author's fots_test.py uses pixel-unit height and the same contact mask.
        with torch.inference_mode():
            bgr = self.renderer.generate(depth*1000*self.params.mm_to_pixel, depth > 0, shadow=shadow)
        return bgr[..., ::-1].copy()

    def markers(self, rgb, contacts):
        """Run original dilation/shear/twist, retaining numeric displacement.

        Each contact has its own relative trajectory. Displacements from
        simultaneous objects are summed on the common undeformed grid: this
        multi-object composition is local, not a coupled elasticity solution.
        Source coefficients, pixel quantization and +/-50 deg twist cap remain.
        """
        Original = self.MarkerMotion

        class MeasuredMarkers(Original):
            def _motion_callback(self, xx, yy):
                result = super()._motion_callback(xx, yy)
                self.displacement = np.stack([result[0]-xx, result[1]-yy], axis=-1)
                return result

        if not contacts:
            contacts = [(np.zeros(rgb.shape[:2]), [])]
        total = None
        for depth, trajectory in contacts:
            marker = MeasuredMarkers(frame0_blur=rgb[..., ::-1], depth=depth*1000,
                                     mask=depth > 0, traj=trajectory,
                                     lamb=[.00125, .00021, .00038])
            image = marker._marker_motion()
            total = marker.displacement.copy() if total is None else total+marker.displacement
        if len(contacts) > 1:
            image = marker._generate(marker.xx+total[..., 0], marker.yy+total[..., 1])
        return image[..., ::-1].copy(), total


class FOTSOptical:
    label = 'GelSight / FOTS (rigid geometry proxy)'

    def __init__(self, model, sensors, *, ideal=False, shadow=True, hardware='gelsight'):
        import pyrender
        from .optical_depth import depth_renderer
        self.model, self.optics = model, FOTSRenderer(hardware)
        self.label = 'DIGIT / FOTS' if hardware == 'digit' else self.label
        self.shadow, self.history, self.last_time = shadow, {}, None
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
        self.baselines = {}
        reference = mujoco.MjData(model)
        mujoco.mj_forward(model, reference)
        for name, (site, gel) in self.sensors.items():
            if model.geom_type[gel] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = geom_mesh(model, gel)
                transform = np.eye(4)
                rotation = reference.site_xmat[site].reshape(3, 3).T
                transform[:3, :3] = rotation @ reference.geom_xmat[gel].reshape(3, 3)
                transform[:3, 3] = rotation @ (reference.geom_xpos[gel]-reference.site_xpos[site])
                mesh.apply_transform(transform)
                mesh.update_faces(mesh.face_normals[:, 2] > 1e-6)
                mesh.faces = mesh.faces[:, ::-1]
                node = self.scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
                self.baselines[name] = self.r.render(self.scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
                self.scene.remove_node(node)
            else:
                self.baselines[name] = np.full((p.sensor_h, p.sensor_w), self.camera_distance_m)

    def render(self, data):
        import pyrender
        frames = {}
        if self.last_time is not None and data.time < self.last_time:
            self.reset()
        self.last_time = float(data.time)
        for name, (site, gel) in self.sensors.items():
            forces = contact_forces(self.model, data, gel)
            for node in list(self.nodes.values()):
                if self.scene.has_node(node):
                    self.scene.remove_node(node)
            rotation = data.site_xmat[site].reshape(3, 3).T
            history = self.history.setdefault(name, {})
            for geom in set(history)-set(forces):
                del history[geom]
            contacts = []
            indentation = np.zeros((self.optics.params.sensor_h, self.optics.params.sensor_w))
            baseline = self.baselines[name]
            valid = baseline > 0
            current = baseline.copy()
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
                self.scene.remove_node(node)
                # Per-object surface overlap, then frontmost union for RGB.
                overlap = np.where(valid & (depth > 0), np.maximum(baseline-depth, 0), 0)
                current = np.where(valid & (depth > 0), np.minimum(current, depth), current)
                indentation = np.maximum(indentation, overlap)
                trajectory = self.trajectory(history, geom, pose, overlap, baseline, depth)
                if trajectory is not None:
                    contacts.append((overlap, trajectory))
            rgb = self.optics.render(indentation, shadow=self.shadow)
            marker_image, displacement = self.optics.markers(rgb, contacts)
            frames[name] = OpticalFrame(rgb, indentation, sum(forces.values()), tuple(forces),
                                       float(data.time), marker_image, displacement,
                                       contact_mask=indentation > 0,
                                       valid_mask=valid, camera_depth_m=current,
                                       baseline_depth_m=baseline.copy(), depth_model='geometric_overlap')
        return frames

    def trajectory(self, history, geom, pose, depth, baseline, camera_depth):
        """Track a contacting material point, not the lifetime of a geom pair.

        Rolling can keep two geoms in contact after the original material
        point has left the gel. Carrying that old anchor forward would feed
        gross object travel to FOTS as elastic marker shear. Renew the anchor
        when it leaves the observed gel volume, using geometric depth only;
        there is no timer, flow cap, force threshold or frozen observation.
        """
        rows, cols = np.nonzero(depth > 0)
        if not len(rows):
            history.pop(geom, None)
            return None
        p = self.optics.params
        pixel_m = 1/(1000*p.mm_to_pixel)
        if geom in history:
            entry = history[geom]
            moved = pose[:3, :3] @ entry['local']+pose[:3, 3]
            # Pixel centres, not corners; interpolate the curved rest surface.
            col, row = moved[:2]/pixel_m+[p.sensor_w/2-.5, p.sensor_h/2-.5]
            touching = False
            if 0 <= col < p.sensor_w-1 and 0 <= row < p.sensor_h-1:
                x, y = int(np.floor(col)), int(np.floor(row))
                patch = baseline[y:y+2, x:x+2]
                if np.all(patch > 0):
                    wx, wy = col-x, row-y
                    surface = np.array([1-wy, wy]) @ patch @ np.array([1-wx, wx])
                    touching = moved[2] <= surface-self.camera_distance_m
            if not touching:
                del history[geom]
        if geom not in history:
            # Select an actually contacting pixel near the patch centroid.
            # z=0 is not the object surface, especially on curved DIGIT gel.
            weights = depth[rows, cols]
            centre = np.array([np.average(cols, weights=weights), np.average(rows, weights=weights)])
            index = np.argmin((cols-centre[0])**2+(rows-centre[1])**2)
            row, col = rows[index], cols[index]
            anchor = np.array([(col+.5-p.sensor_w/2)*pixel_m,
                               (row+.5-p.sensor_h/2)*pixel_m,
                               camera_depth[row, col]-self.camera_distance_m])
            history[geom] = dict(pose=pose.copy(), anchor=anchor,
                                 local=pose[:3, :3].T @ (anchor-pose[:3, 3]), angle=0.)
        entry = history[geom]
        moved = pose[:3, :3] @ entry['local']+pose[:3, 3]
        relative = pose[:3, :3] @ entry['pose'][:3, :3].T
        q = Rotation.from_matrix(relative).as_quat()
        angle = 2*np.arctan2(q[2], q[3])
        angle += 2*np.pi*round((entry['angle']-angle)/(2*np.pi))
        entry['angle'] = angle
        return [[*list(entry['anchor'][:2]*1000), 0.], [*list(moved[:2]*1000), angle]]

    def reset(self):
        self.history.clear()
        self.last_time = None

    def close(self):
        self.r.delete()
