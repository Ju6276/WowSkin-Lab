"""Independent optical benches and Panda mounts, sharing the arm controller.

Separate source DIGIT models / GelSightMujoco CAD;
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
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv


def configure_optical_solver(option, sensor):
    """Replace the host options, including engine defaults, with sensor source.

    Integration step and gravity belong to the Panda/bench runner. Do not
    leave CG, cone, Jacobian, tolerance or flags inherited from an AnySkin task.
    """
    import copy
    timestep = option.get('timestep')
    gravity = option.get('gravity', '0 0 -9.81')
    option.clear()
    if sensor == 'tacto':
        source = ET.parse(ROOT/'third_party/mujoco-tacto/examples/xml/tacto_common.xml').getroot()
    elif sensor == 'digit_fots':
        source = ET.parse(ROOT/'third_party/FOTS-mujoco/assets/gel/fots_mujoco.xml').getroot()
    elif sensor == 'gelsight':
        from wowskin_lab.core.gelsight_mujoco import SOURCE
        source = ET.parse(SOURCE/'models/gripper_testbed_PickandPlace.xml').getroot()
    else:
        raise ValueError(f'Unknown optical backend: {sensor}')
    for item in source.findall('option'):
        option.attrib.update(item.attrib)
        option.extend(copy.deepcopy(list(item)))
    if sensor == 'tacto':
        scene = ET.parse(ROOT/'third_party/mujoco-tacto/examples/xml/touch_playground_rot.xml').getroot()
        for item in scene.findall('option'):
            option.attrib.update(item.attrib)
    option.set('timestep', timestep)
    # The source zero-gravity press fixture is not the robot task.
    option.set('gravity', gravity)


def optical_configuration(args, sensor):
    """Robot/runner settings only: never load the magnetic skin profiles."""
    cfg = json.loads((ROOT/'configs/optical.json').read_text())
    cfg.update(sensor=sensor, skin_dof=None, gripper='panda', seed=getattr(args, 'seed', 0),
               disable_cartesian_limits=bool(getattr(args, 'no_cartesian_limits', False)),
               disable_workspace_limits=True, teleop_frame='base', teleop_base_lateral_sign=1.,
               translation_target_limiter='direction_preserving_v1',
               camera_profile='three_view', scene_design='industrial_workcell_v1')
    return cfg


def sensor_geometry(sensor):
    if sensor == 'gelsight':
        return source_gelsight_geometry()
    return digit_geometry(sensor)


def sensor_prefix(sensor):
    return 'source_gelsight' if sensor == 'gelsight' else 'digit'


def sensor_registration(sensor, names):
    result = {}
    for key, name in names.items():
        if sensor == 'gelsight':
            result[key] = (f'optical_{name}_camera', f'{name}_friction')
        elif sensor in ('tacto', 'digit_fots'):
            from .digit_sources import registration
            result[key] = registration(sensor, name)
        else:
            result[key] = (f'{name}_camera', f'{name}_gel')
    return result


@lru_cache(maxsize=1)
def source_gelsight_geometry():
    """Align the long housing axis to the finger, with its tail proximal."""
    import trimesh
    from wowskin_lab.core.gelsight_mujoco import SOURCE
    scene = ET.parse(SOURCE/'models/gripper_testbed_PickandPlace.xml').getroot()
    mesh = scene.find("asset/mesh[@name='front_cover']")
    vertices = trimesh.load(SOURCE/'models/meshes'/mesh.get('file'), force='mesh').vertices
    vertices = vertices*np.fromstring(mesh.get('scale'), sep=' ')
    source = ET.parse(SOURCE/'models/assets/gelsight_mountable0.xml').getroot()
    contact = source.find(".//geom[@name='gelsight0:friction']")
    surface = np.fromstring(contact.get('pos'), sep=' ')
    surface[2] += np.fromstring(contact.get('size'), sep=' ')[2]
    # +source X (the 50 mm tail) -> -finger Z, +source Y -> finger X,
    # +source Z (outward sensing normal) -> -finger Y, into the jaw gap.
    rotation = np.array([[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]])
    position = np.array([0., surface[2], vertices[:, 0].max()])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    return dict(proximal_z=0., gel_center_z=float((rotation@surface+position)[2]),
                position=position, quaternion=quat, source_surface=surface)


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
    geometry = source_gelsight_geometry()
    module.set('pos', ' '.join(map(str, geometry['position'])))
    module.set('quat', ' '.join(map(str, geometry['quaternion'])))
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
    add = add_source_gelsight if sensor == 'gelsight' else (add_fots_digit if sensor == 'digit_fots' else add_digit)
    mount_modules(root, sensor_prefix(sensor), add, sensor_geometry(sensor), friction=friction)


def digit_geometry(sensor='tacto'):
    from .digit_sources import geometry
    return geometry(sensor)


def add_digit(asset, body, name, *, friction=None):
    from .digit_sources import add
    return add(asset, body, name, sensor='tacto')


def add_fots_digit(asset, body, name, *, friction=None):
    from .digit_sources import add
    return add(asset, body, name, sensor='digit_fots')


def mount_modules(root, prefix, add, geometry, *, friction):
    """Replace stock geometry for a complete finger; tip-mount bare modules.

    Sensing faces remain at finger Y=0, preserving the 80 mm jaw opening.
    Source CAD is not cut, rescaled or replaced by a geometric proxy.
    """
    import trimesh
    paths = {item.get('name', Path(item.get('file', '')).stem): item.get('file')
             for item in root.findall('asset/mesh')}
    tip = max(trimesh.load(paths[name], force='mesh').bounds[1, 2]
              for name in ('finger_0', 'finger_1'))
    for side in ('left', 'right'):
        finger = root.find(f".//body[@name='{side}_finger']")
        # Complete source long assemblies replace the stock finger geometry:
        # TACTO has its finger stub; GelSight its mountable housing tail.
        # FOTS is a bare module, retained on the original tip instead.
        replacement = add in (add_digit, add_source_gelsight)
        if replacement:
            for item in list(finger):
                if item.tag in ('geom', 'inertial'):
                    finger.remove(item)
        mount = ET.SubElement(finger, 'body', name=f'{prefix}_{side}_mount',
                              pos=f'0 0 {(0. if replacement else tip)-geometry["proximal_z"]}')
        add(root.find('asset'), mount, f'{prefix}_{side}', friction=friction)


class OpticalPandaEnv(PandaPickCubeTactileEnv):
    """Standalone compatibility entry; same lifecycle as all task environments."""
    def __init__(self, args=None, *, sensor='tacto'):
        super().__init__(optical_configuration(args or SimpleNamespace(), sensor), pollen=False)


class OpticalBench:
    def __init__(self, *, sensor='tacto', probe_shape='sphere'):
        name = sensor_prefix(sensor)
        root = ET.Element('mujoco', model='WowSkin-Lab | standalone '+sensor)
        ET.SubElement(root, 'compiler', angle='radian')
        option = ET.SubElement(root, 'option', timestep='.0005', integrator='implicitfast')
        configure_optical_solver(option, sensor)
        # This six-axis diagnostic position servo is local, unlike the source
        # motor-driven arm. Integrate its velocity damping implicitly: the
        # source Euler step is unstable for this small probe's rotational DOF.
        # Contact stiffness, friction and sensor geometry remain source values.
        option.set('integrator', 'implicitfast')
        visual = ET.SubElement(root, 'visual')
        ET.SubElement(visual, 'headlight', ambient='.7 .7 .7', diffuse='.2 .2 .2', specular='.1 .1 .1')
        asset, world = ET.SubElement(root, 'asset'), ET.SubElement(root, 'worldbody')
        ET.SubElement(world, 'light', pos='0 -.3 .5')
        ET.SubElement(world, 'geom', type='plane', size='.3 .3 .01', rgba='.3 .35 .4 1')
        body = ET.SubElement(world, 'body', pos='0 0 .05', quat='.7071067811865476 -.7071067811865476 0 0')
        add = add_source_gelsight if sensor == 'gelsight' else (add_fots_digit if sensor == 'digit_fots' else add_digit)
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
        # Diagnostic fixture, not a sensor material parameter. The former
        # underdamped 20 g servo overshot a 12 mm target step through the thin
        # source contact plane. Critical damping includes the joint's 1 Ns/m.
        translation_kv = 2*np.sqrt(3000*.02)-1.
        for axis in 'xyz':
            ET.SubElement(actuator, 'position', joint=f'probe_{axis}', kp='3000', kv=str(translation_kv))
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
