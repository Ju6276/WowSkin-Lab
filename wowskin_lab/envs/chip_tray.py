"""Franka + AnySkin grasping an experimental saddle chip on the EBiM tray.

The original feeding task is unchanged. No human, spoon, beans or bench probe.
Fracture uses the standalone chip's uncalibrated force/moment seam law.
"""
import copy
import json
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from wowskin_lab.paths import ROOT
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, pick_scene, response_configuration
from wowskin_lab.envs.feeding import ASSETS, TABLE_TOP, ROBOT_POSITION, selected_world, asset_subset
from wowskin_lab.envs.feeding_room import add_care_room
from wowskin_lab.envs.brittle_chip import configuration, validate_configuration, add_chip, chip_mesh, BreakableChip
from wowskin_lab.data.cameras import add_recording_cameras
from wowskin_lab.core.grip_force import GripForceFeedback, pad_normal_forces


def scene_config():
    return json.loads((ROOT/'configs/chip_tray.json').read_text())


def chip_config():
    c = configuration()
    c.update(scene_config()['chip_overrides'])
    return validate_configuration(c)


def tray_scene(cfg, layout):
    source = ET.parse(ASSETS/'scene_100.xml').getroot()
    root = ET.fromstring(pick_scene(cfg, layout))
    root.set('model', 'WowSkin-Lab | PandaChip: saddle chip on EBiM tray')
    world, asset = root.find('worldbody'), root.find('asset')
    robot = world.find("body[@name='link0']")
    for e in list(world):
        if e is not robot:
            world.remove(e)
    robot.set('pos', ' '.join(map(str, ROBOT_POSITION)))
    robot.set('quat', '.7071067811865476 0 0 .7071067811865476')
    for geom in [root.find(".//default[@class='panda/collision']/geom"),
                 *[root.find(f".//geom[@name='{n}']") for n in ('standard_shell', 'silicone_backing')]]:
        geom.set('contype', '32'); geom.set('conaffinity', '1')
    for flex in robot.iter('flexcomp'):
        flex.find('contact').set('contype', '16')
        flex.find('contact').set('conaffinity', '17')
    default = ET.SubElement(root.find('default'), 'default', {'class': 'ebim_task3'})
    for e in source.find('default'):
        default.append(copy.deepcopy(e))
    elements = [e for e in selected_world(source)
                if e.tag == 'geom' or e.get('name') == 'dynamic_simple_tray']
    for e in asset_subset(source, elements):
        e = copy.deepcopy(e)
        if e.get('file'):
            e.set('file', str((ASSETS/'assets' if e.tag == 'mesh' else ASSETS)/e.get('file')))
        asset.append(e)
    for original in elements:
        e = copy.deepcopy(original)
        e.set('childclass' if e.tag == 'body' else 'class', 'ebim_task3')
        world.append(e)
    tray = world.find("body[@name='dynamic_simple_tray']")
    # Reuse the source bowl's colour; tray geometry and contact remain original.
    material_name = cfg['chip_tray']['tray_visual_material_source']
    if asset.find(f"material[@name='{material_name}']") is None:
        asset.append(copy.deepcopy(source.find(f"asset/material[@name='{material_name}']")))
    tray.find("geom[@name='visual_dynamic_simple_tray']").set('material', material_name)
    floor = tray.find(".//geom[@name='collision_simple_tray_floor']")
    frame = tray.find("body[@name='dynamic_simple_tray_collision_frame']")
    tray_pos = np.fromstring(tray.get('pos'), sep=' ')
    floor_z = float(floor.get('pos').split()[2])
    floor_half = float(floor.get('size').split()[2])
    # Initial placement only: the source tray floor rests on the source table.
    tray_pos[2] = TABLE_TOP-floor_z+floor_half
    tray.set('pos', ' '.join(map(str, tray_pos)))
    c = cfg['brittle_chip']
    chip_pos = tray_pos+np.fromstring(frame.get('pos'), sep=' ')
    pieces, _ = chip_mesh(c)
    lowest = min((vertices+center)[:, 2].min() for center, vertices, _ in pieces)
    chip_pos[2] += floor_z+floor_half-lowest
    add_chip(root, c, chip_pos, prefix='chip_')
    geoms = {e.get('name') for e in root.iter('geom')}
    contact = root.find('contact')
    if contact is None:
        contact = ET.SubElement(root, 'contact')
    for e in source.find('contact'):
        if e.tag == 'pair' and e.get('geom1') in geoms and e.get('geom2') in geoms:
            contact.append(copy.deepcopy(e))
    for e in source.findall('worldbody/light'):
        world.append(copy.deepcopy(e))
    add_care_room(root)
    for parent in root.iter():
        for child in list(parent):
            if child.tag == 'camera':
                parent.remove(child)
    hand = root.find(".//body[@name='hand']")
    tcp_z = float(hand.find("site[@name='tactile_tcp']").get('pos').split()[2])
    add_recording_cameras(world, hand, tcp_z, profile=cfg.get('camera_profile'), scene='chip_tray')
    root.find('statistic').set('center', '-1.73 .02 1.15')
    root.find('statistic').set('extent', '1.2')
    root.find('option').set('jacobian', 'sparse')
    for key, value in cfg['chip_tray']['solver'].items():
        root.find('option').set(key, str(value))
    return ET.tostring(root, encoding='unicode')


class PandaChipEnv(PandaPickCubeTactileEnv):
    task_name = 'chip_tray'

    def __init__(self, cfg=None, pollen=False):
        cfg = response_configuration(SimpleNamespace(task=self.task_name)) if cfg is None else dict(cfg)
        cfg.update(brittle_chip=chip_config(), chip_tray=scene_config(), scene_design='care_room_chip_tray_v1')
        if cfg.get('gripper', 'panda') != 'panda' or cfg.get('gripper_control_mode', 'panda_menagerie_position') != 'panda_menagerie_position':
            raise ValueError('Chip force feedback currently uses the Panda Menagerie inner servo')
        force_config = cfg['chip_tray']['grip_force_feedback']
        if cfg.get('chip_grip_force_n') is not None:
            force_config['target_n'] = cfg['chip_grip_force_n']
        self.force_grip = GripForceFeedback(force_config)
        super().__init__(cfg, pollen, xml_factory=tray_scene,
                         contact_body='chip_piece_0', contact_geom='chip_piece_0', object_joint=None)
        self.contact_geoms = self.fracture.piece_geoms

    def reset(self, seed=None):
        self.force_grip.reset()
        if not hasattr(self, 'fracture'):
            pieces, bonds = chip_mesh(self.cfg['brittle_chip'])
            self.fracture = BreakableChip(self.model, self.data, self.cfg['brittle_chip'],
                                         pieces, bonds, prefix='chip_', external_bits=6)
        self.fracture.reset_fracture()
        result = super().reset(seed)
        self.fracture.events = [dict(e, time_s=e['time_s']-.25) for e in self.fracture.events]
        return result

    def command_gripper(self, close):
        if close:
            if not self.force_grip.enabled:
                self.grip_target = float(self.data.qpos[self.finger_q].mean())
            self.force_grip.enabled = True
            self.force_grip.input_fraction = 1.
        else:
            self.force_grip.enabled = False
            super().command_gripper(False)

    def gripper_key(self, code):
        if code in (ord('-'), ord('=')):
            step = self.force_grip.config['key_step_n']
            self.force_grip.selected_target_n = max(0., self.force_grip.selected_target_n + (step if code == ord('=') else -step))
            return True
        # C holds the selected force, rather than latching the current width.
        return code == ord('C') and self.force_grip.enabled

    def teleop_gripper(self, fraction):
        """Pico trigger requests 0..selected force; releasing it opens."""
        self.command_gripper(fraction > 0.)
        self.force_grip.input_fraction = float(fraction)
        return self.gripper_fraction(target=True)

    def control(self, mode=None):
        self.grip_target = self.force_grip.update(
            self.grip_target,
            pad_normal_forces(self.model, self.data, (self.skin_flex, self.passive_flex)),
            self.model.opt.timestep, self.model.jnt_range[self.finger_joints[0]])
        super().control(mode)

    def gripper_feedback_sample(self):
        return dict(enabled=bool(self.force_grip.enabled), target_n=self.force_grip.target_n,
                    selected_target_n=self.force_grip.selected_target_n,
                    pad_normal_force_n=pad_normal_forces(self.model, self.data, (self.skin_flex, self.passive_flex)),
                    servo_target_closure=self.gripper_fraction(target=True))

    def after_physics_step(self):
        self.fracture.break_overloaded_bonds()

    after_reset_physics_step = after_physics_step

    def observation(self):
        obs = super().observation()
        obs['chip_position_m'] = obs.pop('cube_position_m')
        return obs

    def evaluation_metrics(self, row=None):
        f = self.fracture
        return dict(chip_broken_bonds=int(np.count_nonzero(~self.data.eq_active[f.eq_ids].astype(bool))),
                    chip_fragment_groups=f.components(), chip_peak_load_ratio=f.max_load_ratio)

    def task_evaluation(self):
        return dict(available=False, reason='manual outcome; no automatic success criterion',
                    metrics=self.evaluation_metrics(), fracture_events=self.fracture.events,
                    config=self.cfg['chip_tray'])

    def step(self, action):
        _, _, terminated, truncated, row = super().step(action)
        row['chip_xyz_m'] = row.pop('cube_xyz_m')
        row.update(self.evaluation_metrics())
        row.update({'grip_force_'+key: value for key, value in self.gripper_feedback_sample().items()})
        row['success'] = self.success = False  # Do not inherit the cube's lift evaluator.
        self.last_row = row
        return self.observation(), 0., terminated, truncated, row

    def viewer_camera(self, close=False, operator=False):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [-1.78, .13, 1.13] if close else [-1.70, .02, 1.20]
        cam.distance, cam.azimuth, cam.elevation = (.43, 125, -22) if close else (1.55, 125, -28)
        return cam
