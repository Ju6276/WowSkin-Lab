"""Independent DIGIT bench and Panda mount, sharing the existing arm controller.

Original mujoco-tacto housing/gel OBJ assets, rigid MuJoCo contact and upstream
optics. No flex/magnetic model is constructed, and no optical frames are passed
off as the existing ViSk AnySkin observations.
"""
from functools import lru_cache
import json
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from wowskin_lab.paths import ROOT
from wowskin_lab.envs.franka import HOME
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, pick_scene, response_configuration
from wowskin_lab.core.arm_control import configure_controller


@lru_cache(maxsize=1)
def digit_geometry():
    directory = ROOT/'third_party/mujoco-tacto/examples/xml/assets/digit_gripper'
    paths = [directory/f'digit_stub_full_{i}.obj' for i in (0, 1)]
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError('Run .venv/bin/python scripts/setup_upstream.py --only tacto')
    vertices = [np.array([[float(x) for x in row.split()[1:4]]
                          for row in p.read_text().splitlines() if row.startswith('v ')]) for p in paths]
    housing, gel = vertices
    # Source has distal direction -Z, sensing normal +Y. Rotate Rx(pi).
    # Register source proximal tip to finger Z=0 and gel front to Y=0.
    origin = np.array([(gel[:, 0].min()+gel[:, 0].max())/2,
                       gel[:, 1].max(), housing[:, 2].max()])
    position = -np.diag([1., -1., -1.]) @ origin
    return dict(paths=paths, position=position,
                gel_center_z=origin[2]-(gel[:, 2].min()+gel[:, 2].max())/2)


def add_digit(asset, body, name, *, friction=.6):
    geometry = digit_geometry()
    for index, material in ((0, '.08 .09 .10 1'), (1, '.15 .17 .18 1')):
        mesh = f'digit_mesh_{index}'
        if asset.find(f"mesh[@name='{mesh}']") is None:
            ET.SubElement(asset, 'mesh', name=mesh, file=str(geometry['paths'][index]))
        ET.SubElement(body, 'geom', name=f'{name}_{"housing" if index == 0 else "gel"}',
                      type='mesh', mesh=mesh, pos=' '.join(map(str, geometry['position'])),
                      quat='0 1 0 0', rgba=material, contype='2', conaffinity='7',
                      friction=f'{friction} .005 .0001', solref='.002 1', group='0')
    # Inward camera at Y=22 mm; its -Z axis looks at the gel front Y=0.
    ET.SubElement(body, 'site', name=f'{name}_camera',
                  pos=f'0 .022 {geometry["gel_center_z"]}',
                  quat='.7071067811865476 -.7071067811865476 0 0', size='.001', rgba='0 0 0 0')


def mount_digits(root, *, friction):
    for side in ('left', 'right'):
        finger = root.find(f".//body[@name='{side}_finger']")
        for geom in list(finger.findall('geom')):
            finger.remove(geom)
        add_digit(root.find('asset'), finger, f'digit_{side}', friction=friction)


class OpticalPandaEnv(PandaPickCubeTactileEnv):
    """Reuse Panda's command/IK/PD methods, with a separate optical lifecycle."""
    def __init__(self, args=None):
        cfg = response_configuration(args or SimpleNamespace(task='pick_cube'))
        cfg.update(json.loads((ROOT/'configs/gripper_control.json').read_text()))
        cfg.update(sensor='tacto', skin_dof=None, camera_profile='three_view',
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
        self.sensors = {side: (f'digit_{side}_camera', f'digit_{side}_gel') for side in ('left', 'right')}
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
    def __init__(self):
        root = ET.Element('mujoco', model='WowSkin-Lab | standalone DIGIT / TACTO')
        ET.SubElement(root, 'option', timestep='.0005', integrator='implicitfast')
        visual = ET.SubElement(root, 'visual')
        ET.SubElement(visual, 'headlight', ambient='.7 .7 .7', diffuse='.2 .2 .2', specular='.1 .1 .1')
        asset, world = ET.SubElement(root, 'asset'), ET.SubElement(root, 'worldbody')
        ET.SubElement(world, 'light', pos='0 -.3 .5')
        ET.SubElement(world, 'geom', type='plane', size='.3 .3 .01', rgba='.3 .35 .4 1')
        sensor = ET.SubElement(world, 'body', pos='0 0 .05', quat='.7071067811865476 -.7071067811865476 0 0')
        add_digit(asset, sensor, 'digit')
        probe = ET.SubElement(world, 'body', name='probe', pos=f'0 {digit_geometry()["gel_center_z"]} .065')
        for axis, vector in zip('xyz', ['1 0 0', '0 1 0', '0 0 1']):
            ET.SubElement(probe, 'joint', name=f'probe_{axis}', type='slide', axis=vector, damping='1')
        ET.SubElement(probe, 'geom', name='probe', type='sphere', size='.005', mass='.02', rgba='.2 .6 .9 1')
        actuator = ET.SubElement(root, 'actuator')
        for axis in 'xyz':
            ET.SubElement(actuator, 'position', joint=f'probe_{axis}', kp='3000', kv='5')
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        self.sensors = {'DIGIT': ('digit_camera', 'digit_gel')}
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, velocity):
        self.data.ctrl[:] += np.asarray(velocity)*.01
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
