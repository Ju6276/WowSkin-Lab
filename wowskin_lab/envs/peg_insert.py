"""Physically grasped SERL++ peg in the local Franka tactile workcell.

Source meshes and convex decompositions are unchanged. No source grasp weld is
imported. Robot/controller, initial grasp and geometric score are local.
"""
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from wowskin_lab.paths import ROOT
from wowskin_lab.core.physics import configuration
from wowskin_lab.envs.franka import mount_reference
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, pick_scene, rotation_matrix
from wowskin_lab.envs.task_evaluation import task_config

ASSETS = ROOT/'assets/serl_peg'


def peg_scene(cfg, layout):
    source = ET.parse(ASSETS/'source_peg_hole.xml').getroot()
    root = ET.fromstring(pick_scene(cfg, layout))
    root.set('model', 'WowSkin-Lab | PandaPegInsert: SERL++ assets, physical grasp')
    world, asset = root.find('worldbody'), root.find('asset')
    world.remove(world.find("body[@name='cube']"))
    for geom in list(world.findall('geom')):
        if geom.get('name', '').startswith(('cell_pick_', 'cell_place_')):
            world.remove(geom)
    # Namespace source defaults to avoid inheriting Panda's collision class.
    for default in source.find('default'):
        default = copy.deepcopy(default)
        default.set('class', 'serl_'+default.get('class'))
        root.find('default').append(default)
    for item in source.find('asset'):
        item = copy.deepcopy(item)
        if item.tag == 'mesh':
            name = item.get('name') or Path(item.get('file')).stem
            item.set('name', 'serl_'+name)
            item.set('file', str(ASSETS/item.get('file')))
        else:
            item.set('name', 'serl_'+item.get('name'))
        asset.append(item)
    for original in source.find('worldbody').findall('body'):
        body = copy.deepcopy(original)
        name = body.get('name')
        if name not in ('peg', 'hole'):
            continue
        body.attrib.pop('euler', None)
        body.set('quat', '0 1 0 0')
        if name == 'hole':
            body.set('pos', ' '.join(map(str, cfg['peg_insert']['socket_origin_world_m'])))
        else:
            body.set('pos', '.40 0 .35')
            body.find('freejoint').set('name', 'peg_joint')
        for index, geom in enumerate(body.iter('geom')):
            cls = geom.get('class')
            geom.set('class', 'serl_'+cls)
            geom.set('name', name+'_visual' if cls == 'visual' else f'{name}_collision_{index}')
            for key in ('mesh', 'material'):
                if geom.get(key):
                    geom.set(key, 'serl_'+geom.get(key))
            if cls == 'collision':
                geom.set('contype', '4' if name == 'peg' else '1')
                geom.set('conaffinity', '3' if name == 'peg' else '6')
        world.append(body)
    return ET.tostring(root, encoding='unicode')


class PegInsertEvaluator:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.success, self.since, self.metrics = False, None, {}

    def update(self, time, peg_position, peg_rotation, socket_position, socket_rotation):
        c = self.config
        # Local +Z points down into the source socket after its Rx(pi) pose.
        rotation = socket_rotation.T@peg_rotation
        tip = socket_rotation.T@(peg_position+peg_rotation@np.asarray(c['tip_local_m'])-socket_position)
        depth = float(tip[2]-c['socket_entry_local_z_m'])
        lateral = float(np.max(np.abs(tip[:2]-np.asarray(c['tip_local_m'])[:2])))
        angle = float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2, -1., 1.))))
        aligned = lateral <= c['max_lateral_error_m'] and angle <= c['max_orientation_error_deg']
        inserted = bool(aligned and c['required_insertion_m'] <= depth <= c['maximum_insertion_m'])
        self.since = (time if self.since is None else self.since) if inserted else None
        dwell = 0. if self.since is None else max(0., time-self.since)
        self.success |= bool(inserted and dwell+1e-9 >= c['continuous_dwell_s'])
        self.metrics = dict(peg_success=self.success, peg_inserted=inserted, peg_aligned=bool(aligned),
                            peg_depth_m=depth, peg_lateral_error_m=lateral,
                            peg_orientation_error_deg=angle, peg_dwell_s=float(dwell))
        return dict(self.metrics)

    def summary(self):
        return dict(criterion=self.config['schema'], config=self.config,
                    success=self.success, metrics=self.metrics)


class PandaPegInsertEnv(PandaPickCubeTactileEnv):
    task_name = 'peg_insert'

    def __init__(self, cfg=None, pollen=False):
        cfg = dict(configuration() if cfg is None else cfg)
        if 'peg_insert' not in cfg:
            cfg['peg_insert'] = task_config('peg_insert')
            if cfg.get('gripper') == 'robotiq_2f85':
                grasp = cfg['peg_insert']['robotiq_initial_grasp']
                for key in ('initial_pinch_width_m', 'tool_yaw_about_peg_rad'):
                    cfg['peg_insert'][key] = grasp[key]
        # Same pinned robosuite finger servos as cable. Robotiq replacement
        # subsequently selects its own EBiM hinge servo.
        cfg.setdefault('gripper_control_mode', 'panda_robosuite_position')
        cfg.setdefault('scene_design', 'serl_peg_physical_grasp_v1')
        self.peg_score = PegInsertEvaluator(cfg['peg_insert'])
        super().__init__(cfg, pollen=pollen, xml_factory=peg_scene,
                         contact_body='peg', contact_geom='peg_collision_1', object_joint='peg_joint')
        self.contact_geoms = {i for i in range(self.model.ngeom) if
                              self.model.geom_bodyid[i] == self.probe_body and self.model.geom_contype[i]}
        self.cfg['peg_asset_source'] = json.loads((ASSETS/'manifest.json').read_text())['revision']
        self.cfg['grasp_assist'] = False
        self.cfg['arm_control'] = 'Local Cartesian IK + joint PD torque (600/45) + bias compensation; not SERL++ Cartesian impedance'

    def reset(self, seed=None):
        super().reset(seed)
        c, m, d = self.cfg['peg_insert'], self.model, self.data
        self.peg_score.reset()
        self.target_xyz = np.asarray(c['initial_tcp_world_m'], float).copy()
        self.servo_xyz = self.target_xyz.copy()
        peg_rotation = np.diag([1., -1., -1.])
        self.target_rotation = peg_rotation@rotation_matrix([0., 0., c['tool_yaw_about_peg_rad']])
        self.servo_rotation = self.target_rotation.copy()
        for _ in range(80):
            self.solve_ik()
        d.qpos[self.arm_q] = self.joint_target
        d.qvel[:] = 0.
        mujoco.mj_forward(m, d)
        self.baseline = self.magnetic_field().copy()
        width = c['initial_pinch_width_m']
        if self.cfg.get('gripper') == 'robotiq_2f85':
            from .robotiq import initialize_gap
            initialize_gap(m, d, self.cfg['total_height_m'], width)
        else:
            gap = 2*(.04+mount_reference(self.cfg)['position_m'][1]-self.cfg['total_height_m'])
            d.qpos[self.finger_q] = -(gap-width)/2
            mujoco.mj_forward(m, d)
        surfaces = []
        for name in ('wowskin_mount', 'silicone_mount'):
            bid = m.body(name).id
            surfaces.append(d.xpos[bid]+d.xmat[bid].reshape(3,3)@np.array([0.,0.,self.cfg['total_height_m']]))
        pinch = np.mean(surfaces, axis=0)
        rotation = peg_rotation
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotation.ravel())
        d.qpos[self.cube_q:self.cube_q+7] = np.r_[pinch-rotation@np.asarray(c['grasp_local_m']), quat]
        self.command_gripper(True)
        mujoco.mj_forward(m, d)
        for _ in range(round(c['initial_settle_s']/self.cfg['timestep_s'])):
            self.control()
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        if np.any(d.warning.number):
            raise RuntimeError('Peg grasp initialization produced a MuJoCo warning')
        self.initial_cube_z = float(d.xpos[self.probe_body, 2])
        d.time = self.zero_time = 0.
        self.last_row = None
        self.evaluate_peg()
        return self.observation(), {'uncalibrated': True}

    def evaluate_peg(self):
        p, h = self.probe_body, self.model.body('hole').id
        return self.peg_score.update(float(self.data.time), self.data.xpos[p],
            self.data.xmat[p].reshape(3,3), self.data.xpos[h], self.data.xmat[h].reshape(3,3))

    def observation(self):
        obs = super().observation()
        obs['peg_position_m'] = obs.pop('cube_position_m')
        return obs

    def evaluation_metrics(self, row=None):
        return dict(self.peg_score.metrics)

    def task_evaluation(self):
        return self.peg_score.summary()

    def step(self, action):
        row = super().step(action)[-1]
        row['peg_position_m'] = row.pop('cube_xyz_m')
        row['peg_lift_m'] = row.pop('lift_m')
        row.update(self.evaluate_peg())
        row['success'] = self.success = self.peg_score.success
        self.last_row = row
        return self.observation(), float(self.success), False, False, row
