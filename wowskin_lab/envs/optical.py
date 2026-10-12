"""Independent optical benches and Panda mounts, sharing the arm controller.

Original DIGIT STL / GelSightMujoco CAD, or the legacy flat FOTS proxy;
rigid contact and unchanged upstream optical algorithms. Frames are not
passed off as the existing ViSk AnySkin observations.
"""
from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from wowskin_lab.paths import ROOT
from wowskin_lab.envs.franka import HOME
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, pick_scene, response_configuration
from wowskin_lab.core.arm_control import configure_controller

OPTICAL_SENSORS = ('tacto', 'digit_fots', 'digit_compare', 'fots', 'gelsight')


def sensor_geometry(sensor):
    if sensor == 'gelsight':
        return source_gelsight_geometry()
    return gelsight_geometry() if sensor == 'fots' else digit_geometry()


def sensor_prefix(sensor):
    return 'source_gelsight' if sensor == 'gelsight' else ('gelsight' if sensor == 'fots' else 'digit')


def sensor_registration(sensor, names):
    result = {}
    for key, name in names.items():
        if sensor == 'gelsight':
            result[key] = (f'optical_{name}_camera', f'{name}_friction')
        else:
            result[key] = (f'{name}_{"surface" if sensor == "digit_fots" else "camera"}', f'{name}_gel')
    return result


def source_gelsight_geometry():
    # Source housing y bounds: +/-18 mm. Its glass front is source z=34.001 mm.
    # Rotate +source Z into the gap (-finger Y), and source Y down the finger.
    return dict(proximal_z=0., gel_center_z=.018)


def add_source_gelsight(asset, body, name, *, friction=.6):
    """Import the full source module; only names, pose and collision bits adapt."""
    import copy
    from wowskin_lab.core.gelsight_mujoco import SOURCE
    scene = ET.parse(SOURCE/'models/gripper_testbed_PickandPlace.xml').getroot()
    for kind, names in (('mesh', ('gelsight_glass', 'front_cover', 'back_cover')),
                        ('material', ('black_resin', 'gray_elastomer', 'transparent_glass'))):
        for old_name in names:
            new_name = f'source_gelsight_{old_name}'
            if asset.find(f"{kind}[@name='{new_name}']") is None:
                item = copy.deepcopy(scene.find(f"asset/{kind}[@name='{old_name}']"))
                item.set('name', new_name)
                if item.get('file'):
                    item.set('file', str(SOURCE/'models/meshes'/item.get('file')))
                asset.append(item)
    original = ET.parse(SOURCE/'models/assets/gelsight_mountable0.xml').getroot().find('body')
    module = copy.deepcopy(original)
    module.set('name', name+'_module')
    module.set('pos', '0 .034001 .018')
    module.set('quat', '.7071067811865476 .7071067811865476 0 0')
    for geom in module.findall('geom'):
        geom.set('name', name+'_'+geom.get('name').split(':')[-1])
        for attr in ('mesh', 'material'):
            if geom.get(attr):
                geom.set(attr, 'source_gelsight_'+geom.get(attr))
        # Source bit 32 -> local sensor bit 2. As in the source target-object
        # pair, the optical cover does not collide with task objects, but its
        # affinity still allows contact with the opposite sensor hardware.
        active = geom.get('contype') == '32'
        geom.set('contype', '2' if active else '0')
        geom.set('conaffinity', '7' if active else '2')
    camera = module.find('camera')
    camera.set('name', f'optical_{name}_camera')
    body.append(module)


def mount_optical(root, sensor, *, friction):
    add = add_source_gelsight if sensor == 'gelsight' else (add_gelsight if sensor == 'fots' else add_digit)
    mount_modules(root, sensor_prefix(sensor), add, sensor_geometry(sensor), friction=friction)


@lru_cache(maxsize=1)
def digit_geometry():
    import trimesh
    path = ROOT/'third_party/mujoco-tacto/meshes/digit.STL'
    if not path.is_file():
        raise FileNotFoundError('Run .venv/bin/python scripts/setup_upstream.py --only tacto')
    source = trimesh.load(path, force='mesh')
    parts = list(source.split(only_watertight=False))
    # The outermost +X connected component is the source CAD's curved gel.
    index = max(range(len(parts)), key=lambda i: parts[i].bounds[1, 0])
    gel = parts.pop(index)
    housing = trimesh.util.concatenate(parts)
    # Native +X normal -> finger -Y; native +Y -> finger +X.
    rotation = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    origin = [gel.bounds[1, 0], gel.bounds.mean(axis=0)[1], source.bounds[0, 2]]
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, -rotation @ origin
    for mesh in (housing, gel):
        mesh.apply_transform(transform)
    return dict(path=path, meshes=(housing, gel), proximal_z=0.,
                gel_center_z=float(gel.bounds.mean(axis=0)[2]))


def add_digit(asset, body, name, *, friction=.6):
    geometry = digit_geometry()
    for index, material in ((0, '.08 .09 .10 1'), (1, '.15 .17 .18 1')):
        mesh = f'digit_mesh_{index}'
        if asset.find(f"mesh[@name='{mesh}']") is None:
            source = geometry['meshes'][index]
            ET.SubElement(asset, 'mesh', name=mesh,
                          vertex=' '.join(map(str, source.vertices.ravel())),
                          face=' '.join(map(str, source.faces.ravel())))
        ET.SubElement(body, 'geom', name=f'{name}_{"housing" if index == 0 else "gel"}',
                      type='mesh', mesh=mesh, rgba=material, contype='2', conaffinity='7',
                      friction=f'{friction} .005 .0001', solref='.002 1', group='0')
    # Source URDF gives 0.1 kg but placeholder inertia=1. Use its mass and an
    # explicit uniform bounding-box inertia, not a fictitious calibrated tensor.
    bounds = geometry['meshes'][0].bounds.copy()
    bounds[0] = np.minimum(bounds[0], geometry['meshes'][1].bounds[0])
    bounds[1] = np.maximum(bounds[1], geometry['meshes'][1].bounds[1])
    size = bounds[1]-bounds[0]
    inertia = .1/12 * np.array([size[1]**2+size[2]**2, size[0]**2+size[2]**2, size[0]**2+size[1]**2])
    ET.SubElement(body, 'inertial', mass='.1', pos=' '.join(map(str, bounds.mean(axis=0))),
                  diaginertia=' '.join(map(str, inertia)))
    # Inward camera at Y=22 mm; its -Z axis looks at the gel front Y=0.
    ET.SubElement(body, 'site', name=f'{name}_camera',
                  pos=f'0 .022 {geometry["gel_center_z"]}',
                  quat='.7071067811865476 -.7071067811865476 0 0', size='.001', rgba='0 0 0 0')
    ET.SubElement(body, 'site', name=f'{name}_surface',
                  pos=f'0 0 {geometry["gel_center_z"]}',
                  quat='.7071067811865476 .7071067811865476 0 0', size='.001', rgba='0 0 0 0')


def mount_digits(root, *, friction):
    mount_modules(root, 'digit', add_digit, digit_geometry(), friction=friction)


def gelsight_geometry():
    from wowskin_lab.core.fots import upstream
    params, *_ = upstream()
    return dict(width=params.sensor_w/params.mm_to_pixel/1000,
                height=params.sensor_h/params.mm_to_pixel/1000,
                gel_center_z=.012,
                proximal_z=.012-params.sensor_h/params.mm_to_pixel/2000-.002)


def add_gelsight(asset, body, name, *, friction=.6):
    """Flat calibration-window proxy, not an official GelSight housing CAD.

    The source gelsight_pad.npy is flat. Active size uses source pixel/mm;
    2 mm backing and rectangular housing are explicit local mount geometry.
    """
    geometry = gelsight_geometry()
    x, z = geometry['width']/2, geometry['height']/2
    center = geometry['gel_center_z']
    ET.SubElement(body, 'geom', name=f'{name}_housing', type='box',
                  pos=f'0 .006 {center}', size=f'{x+.002} .004 {z+.002}',
                  rgba='.12 .13 .15 1', contype='2', conaffinity='7',
                  friction=f'{friction} .005 .0001', solref='.002 1')
    ET.SubElement(body, 'geom', name=f'{name}_gel', type='box',
                  pos=f'0 .001 {center}', size=f'{x} .001 {z}',
                  rgba='.65 .68 .72 1', contype='2', conaffinity='7',
                  friction=f'{friction} .005 .0001', solref='.002 1')
    # +X right, +Y image-down (finger +Z), +Z outward (finger -Y).
    ET.SubElement(body, 'site', name=f'{name}_camera', pos=f'0 0 {center}',
                  quat='.7071067811865476 .7071067811865476 0 0', size='.001', rgba='0 0 0 0')


def mount_gelsights(root, *, friction):
    mount_modules(root, 'gelsight', add_gelsight, gelsight_geometry(), friction=friction)


def mount_modules(root, prefix, add, geometry, *, friction):
    """Append a fixed module at the stock distal tip; preserve every finger geom.

    Sensing face stays on finger Y=0; housing extends away from the jaw gap.
    Mount pose is a local assembly, not a verified manufacturing adapter.
    """
    import trimesh
    paths = {item.get('name', Path(item.get('file', '')).stem): item.get('file')
             for item in root.findall('asset/mesh')}
    tip = max(trimesh.load(paths[name], force='mesh').bounds[1, 2]
              for name in ('finger_0', 'finger_1'))
    for side in ('left', 'right'):
        finger = root.find(f".//body[@name='{side}_finger']")
        mount = ET.SubElement(finger, 'body', name=f'{prefix}_{side}_mount',
                              pos=f'0 0 {tip-geometry["proximal_z"]}')
        add(root.find('asset'), mount, f'{prefix}_{side}', friction=friction)


class OpticalPandaEnv(PandaPickCubeTactileEnv):
    """Reuse Panda's command/IK/PD methods, with a separate optical lifecycle."""
    def __init__(self, args=None, *, sensor='tacto'):
        cfg = response_configuration(args or SimpleNamespace(task='pick_cube'))
        cfg.update(json.loads((ROOT/'configs/gripper_control.json').read_text()))
        cfg.update(sensor=sensor, skin_dof=None, camera_profile='three_view',
                   scene_design='industrial_workcell_v1')
        configure_controller(cfg)
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_string(pick_scene(cfg, []))
        self.data, self.ik_data = mujoco.MjData(self.model), mujoco.MjData(self.model)
        self.arm_joints = [self.model.joint(f'joint{i}').id for i in range(1, 8)]
        self.arm_q, self.arm_v = self.model.jnt_qposadr[self.arm_joints], self.model.jnt_dofadr[self.arm_joints]
        self.arm_controls = [self.model.actuator(f'actuator{i}').id for i in range(1, 8)]
        self.finger_joints = [self.model.joint(f'finger_joint{i}').id for i in (1, 2)]
        self.finger_q, self.finger_v = self.model.jnt_qposadr[self.finger_joints], self.model.jnt_dofadr[self.finger_joints]
        self.gripper_control = self.model.actuator('fingers_actuator').id
        self.gripper_controls = [self.gripper_control]
        self.tcp_site, self.hand = self.model.site('tactile_tcp').id, self.model.body('hand').id
        self.probe_body = self.model.body('cube').id
        self.cube_q = self.model.jnt_qposadr[self.model.joint('cube_joint').id]
        prefix = sensor_prefix(sensor)
        self.sensors = sensor_registration(sensor, {side: f'{prefix}_{side}' for side in ('left', 'right')})
        self.reset()

    def reset(self, seed=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_q] = HOME
        mujoco.mj_forward(self.model, self.data)
        self.joint_target = HOME.copy()
        self.target_xyz = self.data.site_xpos[self.tcp_site].copy()
        self.servo_xyz = self.target_xyz.copy()
        self.target_rotation = self.data.site_xmat[self.tcp_site].reshape(3, 3).copy()
        self.servo_rotation = self.target_rotation.copy()
        self.grip_target, self.tug_until, self.mouse_wrench = 0., 0., np.zeros(6)
        for _ in range(round(.25/self.cfg['timestep_s'])):
            self.control()
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.data.time = 0.

    def step(self, action):
        self.apply_action_target(action)
        self.solve_ik()
        for _ in range(round(1/self.cfg['sample_hz']/self.cfg['timestep_s'])):
            self.control()
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)


class OpticalBench:
    def __init__(self, *, sensor='tacto', probe_shape='sphere'):
        is_fots = sensor == 'fots'
        name = sensor_prefix(sensor)
        root = ET.Element('mujoco', model='WowSkin-Lab | standalone '+sensor)
        ET.SubElement(root, 'compiler', angle='radian')
        ET.SubElement(root, 'option', timestep='.0005', integrator='implicitfast')
        visual = ET.SubElement(root, 'visual')
        ET.SubElement(visual, 'headlight', ambient='.7 .7 .7', diffuse='.2 .2 .2', specular='.1 .1 .1')
        asset, world = ET.SubElement(root, 'asset'), ET.SubElement(root, 'worldbody')
        ET.SubElement(world, 'light', pos='0 -.3 .5')
        ET.SubElement(world, 'geom', type='plane', size='.3 .3 .01', rgba='.3 .35 .4 1')
        body = ET.SubElement(world, 'body', pos='0 0 .05', quat='.7071067811865476 -.7071067811865476 0 0')
        add = add_source_gelsight if sensor == 'gelsight' else (add_gelsight if is_fots else add_digit)
        add(asset, body, name)
        geometry = sensor_geometry(sensor)
        probe = ET.SubElement(world, 'body', name='probe', pos=f'0 {geometry["gel_center_z"]} .065')
        for axis, vector in zip('xyz', ['1 0 0', '0 1 0', '0 0 1']):
            ET.SubElement(probe, 'joint', name=f'probe_{axis}', type='slide', axis=vector, damping='1')
        for axis, vector in zip('xyz', ['1 0 0', '0 1 0', '0 0 1']):
            ET.SubElement(probe, 'joint', name=f'probe_r{axis}', type='hinge', axis=vector, damping='.01')
        sizes = dict(sphere='.005', box='.004 .005 .005', cylinder='.004 .005')
        if probe_shape not in sizes:
            raise ValueError(f'Unsupported probe shape: {probe_shape}')
        ET.SubElement(probe, 'geom', name='probe', type=probe_shape, size=sizes[probe_shape], mass='.02', rgba='.2 .6 .9 1')
        actuator = ET.SubElement(root, 'actuator')
        for axis in 'xyz':
            ET.SubElement(actuator, 'position', joint=f'probe_{axis}', kp='3000', kv='5')
        for axis in 'xyz':
            ET.SubElement(actuator, 'position', joint=f'probe_r{axis}', kp='1', kv='.02')
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        self.sensors = sensor_registration(sensor, {name.upper(): name})
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, velocity):
        velocity = np.asarray(velocity)
        self.data.ctrl[:len(velocity)] += velocity*.01
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
