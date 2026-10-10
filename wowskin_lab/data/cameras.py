"""MuJoCo camera and point observations; all images are RGB, never viewer HUDs."""
import itertools
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

# Shared acquisition rig: exactly two fixed views of the same workspace.
CAMERAS = dict(pixels1='lab_front', pixels2='lab_side')


def add_camera(parent, name, position, target, fovy=55):
    z = np.asarray(position, float)-np.asarray(target, float)
    z /= np.linalg.norm(z)
    x = np.cross([0., 0., 1.], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([0., 1., 0.], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ET.SubElement(parent, 'camera', name=name, pos=' '.join(map(str, position)),
                  xyaxes=' '.join(map(str, np.r_[x, y])), fovy=str(fovy))


def add_recording_cameras(world, hand, tcp_z):
    add_camera(world, CAMERAS['pixels1'], [.65, -.40, .42], [.4, 0, .17])
    add_camera(world, CAMERAS['pixels2'], [.55, .40, .35], [.4, 0, .17])



def project(points, intrinsic, world_to_camera):
    camera = np.asarray(points) @ world_to_camera[:3, :3].T+world_to_camera[:3, 3]
    pixels = camera @ intrinsic.T
    pixels = pixels[:, :2]/np.maximum(camera[:, 2:3], 1e-8)
    return pixels.astype(np.float32), camera[:, 2] > 0


class CameraRig:
    def __init__(self, env, size=128):
        self.env, self.size = env, size
        self.renderer = mujoco.Renderer(env.model, height=size, width=size)

    def close(self):
        self.renderer.close()

    def points(self):
        e, d = self.env, self.env.data
        # Legacy simulation-adapter landmarks, NOT the official FTF geometry.
        # Retained for reproducibility of existing smoke-test data only.
        position = d.site_xpos[e.tcp_site]
        rotation = d.site_xmat[e.tcp_site].reshape(3, 3)
        offsets = np.array([[0, 0, 0], [0, 0, -.04], [0, .04, -.04],
                            [0, -.04, -.04], [0, 0, -.075],
                            [0, .04, -.075], [0, -.04, -.075]])
        robot = offsets @ rotation.T+position
        left = e.to_world(np.array([0., 0., e.cfg['total_height_m']]))
        right = e.to_world(np.array([0., 0., e.cfg['total_height_m']]), e.passive_mount)
        robot = np.concatenate([robot[:1], [left, right], robot[1:]], axis=0)
        half = e.model.geom_size[e.probe_geom, :3]
        corners = np.array(list(itertools.product([-1, 1], repeat=3)))*half
        obj = corners @ d.xmat[e.probe_body].reshape(3, 3).T+d.xpos[e.probe_body]
        return robot.astype(np.float32), obj.astype(np.float32)

    def capture(self):
        e, d, size = self.env, self.env.data, self.size
        robot, obj = self.points()
        result = dict(robot_points_world_m=robot, object_points_world_m=obj)
        for key, name in CAMERAS.items():
            cid = e.model.camera(name).id
            focal = .5*size/np.tan(np.deg2rad(e.model.cam_fovy[cid])/2)
            intrinsic = np.array([[focal, 0, (size-1)/2], [0, focal, (size-1)/2], [0, 0, 1.]])
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = np.diag([1., -1., -1.]) @ d.cam_xmat[cid].reshape(3, 3).T
            extrinsic[:3, 3] = -extrinsic[:3, :3] @ d.cam_xpos[cid]
            self.renderer.update_scene(d, camera=name)
            result[key] = self.renderer.render().copy()
            result[f'camera_intrinsic_{key}'] = intrinsic
            result[f'world_to_camera_{key}'] = extrinsic
            for label, points in [('robot', robot), ('object', obj)]:
                pixels, front = project(points, intrinsic, extrinsic)
                result[f'{label}_tracks_{key}'] = pixels
                result[f'{label}_in_frame_{key}'] = (front & (pixels >= 0).all(axis=1)
                                                     & (pixels < size).all(axis=1))
        return result
