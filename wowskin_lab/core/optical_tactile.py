"""MuJoCo rigid contacts -> pinned TACTO optics, in each sensor's frame.

This is an optical observation model, not a deformable gel solver. Upstream
Renderer is unchanged. This adapter replaces its MuJoCo pose/contact plumbing:
SE(3) composition, all contacting geoms, and independent per-sensor renderers.
"""
from dataclasses import dataclass
from pathlib import Path
import os
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class OpticalFrame:
    rgb: np.ndarray
    indentation_m: np.ndarray
    normal_force_n: float
    contacting_geoms: tuple
    time_s: float
    marker_flow_rgb: np.ndarray | None = None
    marker_displacement_px: np.ndarray | None = None
    contact_mask: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    camera_depth_m: np.ndarray | None = None
    baseline_depth_m: np.ndarray | None = None
    render_input_depth: np.ndarray | None = None
    render_deformation: np.ndarray | None = None
    render_depth_unit: str = 'metre'
    depth_model: str = 'unspecified'
    marker_status: str = 'not_provided'


def contact_forces(model, data, gel):
    gels = set(np.atleast_1d(gel).tolist())
    forces, force = {}, np.zeros(6)
    for index, contact in enumerate(data.contact):
        # Warp readback updates geom, not MuJoCo's deprecated geom1/geom2
        # storage. They are separate fields, so the legacy values can be stale.
        first, second = map(int, contact.geom)
        if not gels.intersection((first, second)):
            continue
        other = second if first in gels else first
        if other in gels:
            continue
        if other < 0:
            raise NotImplementedError('Optical flex contact surface export is not implemented')
        mujoco.mj_contactForce(model, data, index, force)
        if force[0] > 0:
            forces[other] = forces.get(other, 0.)+float(force[0])
    return forces


def geom_mesh(model, geom_id):
    """Compiled geom-local vertices: no extra Euler rotation or mirror."""
    import trimesh
    kind, size = int(model.geom_type[geom_id]), model.geom_size[geom_id]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom_id]
        v, f = model.mesh_vertadr[mesh], model.mesh_faceadr[mesh]
        return trimesh.Trimesh(model.mesh_vert[v:v+model.mesh_vertnum[mesh]].copy(),
                               model.mesh_face[f:f+model.mesh_facenum[mesh]].copy(), process=False)
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2*size)
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=3, radius=size[0])
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        mesh = trimesh.creation.icosphere(subdivisions=3)
        mesh.apply_scale(size)
        return mesh
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=size[0], height=2*size[1], sections=48)
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=size[0], height=2*size[1])
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        return mesh
    if kind == mujoco.mjtGeom.mjGEOM_PLANE:
        # Finite optical proxy only; the MuJoCo plane remains infinite.
        mesh = trimesh.creation.box(extents=[10., 10., .001])
        mesh.apply_translation([0, 0, -.0005])
        return mesh
    raise NotImplementedError(f'TACTO cannot render geom type {kind}; no silent zero output')


class OpticalTactile:
    label = 'DIGIT / TACTO'

    def __init__(self, model, sensors, *, ideal=False, width=240, height=320,
                 background='upstream', config_path=None):
        try:
            from tacto import Renderer, get_digit_config_path
            if sys.platform == 'darwin':
                from .optical_cgl import CGLTactoRenderer as Renderer
        except ImportError as exc:
            raise RuntimeError('Install the optional TACTO dependencies using README.md') from exc
        import cv2
        from wowskin_lab.paths import ROOT
        if background == 'upstream':
            background = ROOT/'third_party/mujoco-tacto/tacto/assets/bg_digit_240_320.jpg'
        self.background_path = str(Path(background).resolve()) if background is not None else None
        background_bgr = cv2.imread(self.background_path) if background is not None else None
        if background is not None and background_bgr is None:
            raise FileNotFoundError(f'Cannot read TACTO empty-contact background: {background}')
        self.config_path = str(config_path or get_digit_config_path())
        self.model, self.ideal = model, ideal
        self.sensors, self.renderers = {}, {}
        try:
            for name, (site, gel) in sensors.items():
                site_id = model.site(site).id
                gel_ids = tuple(model.geom(g).id for g in ((gel,) if isinstance(gel, str) else gel))
                self.sensors[name] = (site_id, gel_ids)
                self.renderers[name] = Renderer(width, height, background_bgr, self.config_path)
                if sys.platform != 'darwin' and os.environ.get('PYOPENGL_PLATFORM') == 'egl':
                    from .optical_egl import share_display
                    share_display(self.renderers[name].r)
                if self.renderers[name].nb_cam != 1:
                    raise ValueError('The DIGIT adapter requires one camera per module; it cannot discard extra cameras')
        except Exception:
            self.close()
            raise

    def contacts(self, data, gel):
        return contact_forces(self.model, data, gel)

    def render(self, data):
        frames = {}
        for name, (site, gel) in self.sensors.items():
            renderer = self.renderers[name]
            forces = self.contacts(data, gel)
            poses, normal = {}, {}
            origin = data.site_xpos[site]
            rotation = data.site_xmat[site].reshape(3, 3).T
            for geom, force in forces.items():
                key = f'geom_{geom}'
                if key not in renderer.object_nodes:
                    renderer.add_object(geom_mesh(self.model, geom), key)
                poses[key] = (rotation @ (data.geom_xpos[geom]-origin),
                              Rotation.from_matrix(rotation @ data.geom_xmat[geom].reshape(3, 3)).as_euler('xyz'))
                # Total normal contact force per geom, N; no sensor_id slicing
                # of sensordata, no pixel-grid layout assumption, no whitelist.
                normal[key] = force
            colors, depths = renderer.render(poses, normal, noise=not self.ideal)
            # A ray outside the configured optical gel has baseline depth=0
            # (renderer background sentinel), not a gel at the camera centre.
            # Subtract only where both surfaces exist; keep raw depths and
            # validity so outside-of-sensor pixels are never reported as data.
            valid = (renderer.depth0[0] > 0) & (depths[0] > 0)
            indentation = np.where(valid, renderer.depth0[0]-depths[0], 0.)
            frames[name] = OpticalFrame(colors[0], indentation,
                                         sum(forces.values()), tuple(forces), float(data.time),
                                         contact_mask=valid & (indentation > 0), valid_mask=valid,
                                         camera_depth_m=depths[0].copy(),
                                         baseline_depth_m=renderer.depth0[0].copy(),
                                         depth_model=('tacto_force_adjusted_optical' if renderer.force_enabled
                                                      else 'tacto_geometric_optical'))
        return frames

    def close(self):
        for renderer in self.renderers.values():
            renderer.r.delete()
        self.renderers.clear()
