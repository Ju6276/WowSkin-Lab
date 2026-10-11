"""MuJoCo camera and point observations; all images are RGB, never viewer HUDs."""
import json
from wowskin_lab.paths import ROOT
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

# Keep the historical wrist ID 52; never renumber old recordings.
CAMERAS = dict(pixels1='lab_front', pixels2='lab_side', pixels52='lab_wrist')

def active_cameras(env):
    return {key: name for key, name in CAMERAS.items()
            if mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0}


def camera_configuration(env):
    """Immutable optical/mount parameters for dataset append compatibility."""
    m = env.model
    return {key: dict(name=name, parent=m.body(int(m.cam_bodyid[m.camera(name).id])).name,
                      position_m=m.cam_pos[m.camera(name).id].tolist(),
                      quaternion_wxyz=m.cam_quat[m.camera(name).id].tolist(),
                      fovy_deg=float(m.cam_fovy[m.camera(name).id]))
            for key, name in active_cameras(env).items()}

def recorded_camera_ids(metadata):
    ids = tuple(int(i) for i in metadata.get('camera_ids', [1, 2]))
    if not ids or len(set(ids)) != len(ids) or any(f'pixels{i}' not in CAMERAS for i in ids):
        raise ValueError(f'Unsupported or duplicate recorded camera IDs: {ids}')
    return ids


def capture_options():
    """Include room backdrop group 4, independently of free-view cutaway."""
    options = mujoco.MjvOption()
    options.geomgroup[4] = 1
    return options


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


def camera_profile(profile=None, scene='pick_cube', tcp_z=0.):
    """Resolve the versioned local rig, including scene and wrist coordinates."""
    definition = json.loads((ROOT/'configs/cameras.json').read_text())
    profile = profile or definition['default_profile']
    ids = definition['profiles'][profile]['camera_ids']
    overrides = definition.get('scene_overrides', {}).get(scene, {})
    result = []
    for source in definition['cameras']:
        if source['id'] not in ids:
            continue
        camera = dict(source)
        camera.update(overrides.get(str(camera['id']), {}))
        if 'target_tcp_offset_m' in camera:
            camera['target_m'] = (np.asarray(camera['target_tcp_offset_m'])+[0., 0., tcp_z]).tolist()
        result.append(camera)
    return result


def add_recording_cameras(world, hand, tcp_z, profile=None, scene='pick_cube'):
    for camera in camera_profile(profile, scene, tcp_z):
        parent = world if camera['parent'] == 'world' else hand
        add_camera(parent, camera['name'], camera['position_m'], camera['target_m'], camera['fovy_deg'])
