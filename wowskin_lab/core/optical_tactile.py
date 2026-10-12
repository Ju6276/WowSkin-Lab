"""MuJoCo rigid contacts -> pinned TACTO optics, in each sensor's frame.

This is an optical observation model, not a deformable gel solver. Upstream
Renderer is unchanged. This adapter replaces its MuJoCo pose/contact plumbing:
SE(3) composition, all contacting geoms, and independent per-sensor renderers.
"""
from dataclasses import dataclass
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


def contact_forces(model, data, gel):
    forces, force = {}, np.zeros(6)
    for index, contact in enumerate(data.contact):
        if gel not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == gel else contact.geom1
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

    def __init__(self, model, sensors, *, ideal=False, width=120, height=160):
        try:
            from tacto import Renderer, get_digit_config_path
            if sys.platform == 'darwin':
                from .optical_cgl import CGLTactoRenderer as Renderer
        except ImportError as exc:
            raise RuntimeError('Install the optional TACTO dependencies using README.md') from exc
        self.model, self.ideal = model, ideal
        self.sensors, self.renderers = {}, {}
        try:
            for name, (site, gel) in sensors.items():
                self.sensors[name] = (model.site(site).id, model.geom(gel).id)
                self.renderers[name] = Renderer(width, height, None, get_digit_config_path())
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
            frames[name] = OpticalFrame(colors[0], renderer.depth0[0]-depths[0],
                                         sum(forces.values()), tuple(forces), float(data.time))
        return frames

    def close(self):
        for renderer in self.renderers.values():
            renderer.r.delete()
        self.renderers.clear()
