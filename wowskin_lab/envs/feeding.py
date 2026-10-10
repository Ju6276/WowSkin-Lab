"""Fixed single Panda using EBiM task-3 feeding station source assets.

Local delivery-only variant, not EBiM's mobile dual-arm evaluator. Original
meshes, spoon and bean mechanics are retained. Reset physically grips a loaded
spoon. The user-requested head completion/enlargement is local to this scene.
A local rigid convex head collision proxy reports simulation loads;
it does not model a mouth, swallowing, tissue compliance or ISO safety limits.
"""
import copy
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from wowskin_lab.paths import ROOT
from wowskin_lab.core.physics import configuration
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, pick_scene, rotation_matrix
from wowskin_lab.envs.franka import mount_reference
from wowskin_lab.data.cameras import add_camera

ASSETS = ROOT/'assets/ebim_feeding'
TABLE_TOP = 1.08485019
ROBOT_POSITION = [-1.63, -.38, TABLE_TOP]
# User-requested visual enlargement of the complete source head + neck.
# Scale eyes, posterior completion and collision together; keep its base on table.
HEAD_SCALE = 1.5
HEAD_POSITION = [-1.45, .25, TABLE_TOP+.1*HEAD_SCALE]
BEAN_COUNT = 6
HEAD_COMPLETION = ROOT/'assets/generated/feeding'
# Source head eyes are on local -Y. Face the robot, not away from it.
HEAD_YAW = np.arctan2(ROBOT_POSITION[0]-HEAD_POSITION[0],
                     HEAD_POSITION[1]-ROBOT_POSITION[1])
HEAD_QUAT = [np.cos(HEAD_YAW/2), 0., 0., np.sin(HEAD_YAW/2)]
BODIES = ('dynamic_bowl2', 'dynamic_plate2', 'dynamic_simple_tray', 'dynamic_spoon2',
          'dynamic_cup', 'configurable_visual_head')


def selected_world(source):
    return [e for e in source.find('worldbody') if (
        e.tag == 'body' and (e.get('name') in BODIES or e.get('name', '').startswith('dynamic_coffee_bean_'))
        or e.tag == 'geom' and (e.get('name') == 'reference_floor' or any(
            e.get('name') == prefix+str(i) for prefix in ('visual_box_Rectangle', 'collision_Rectangle')
            for i in range(127, 136))))]


def asset_subset(source, elements):
    needed = {e.get(k) for parent in elements for e in parent.iter() for k in ('mesh', 'material', 'texture') if e.get(k)}
    assets = {e.get('name'): e for e in source.find('asset')}
    pending = list(needed)
    while pending:
        for e in assets[pending.pop()].iter():
            for key in ('mesh', 'material', 'texture'):
                value = e.get(key)
                if value and value not in needed:
                    needed.add(value)
                    pending.append(value)
    return [e for e in source.find('asset') if e.get('name') in needed]


def feeding_scene(cfg, layout):
    if not (ASSETS/'manifest.json').exists():
        raise FileNotFoundError('Run .venv/bin/python scripts/fetch_ebim_feeding.py first')
    source = ET.parse(ASSETS/'scene_100.xml').getroot()
    official_cfg = json.loads((ASSETS/'config.json').read_text())
    root = ET.fromstring(pick_scene(cfg, layout))
    root.set('model', 'WowSkin-Lab | PandaFeeding: EBiM task 3 assets')
    world, asset = root.find('worldbody'), root.find('asset')
    cap = json.loads((HEAD_COMPLETION/'head_back.json').read_text())
    head_scale = ' '.join([str(HEAD_SCALE)]*3)
    ET.SubElement(asset, 'mesh', name='local_head_back_mesh', file=str(HEAD_COMPLETION/'head_back.obj'),
                  inertia='shell', scale=head_scale)
    ET.SubElement(asset, 'material', name='local_head_back_skin',
                  rgba=' '.join(map(str, cap['skin_rgba'])), specular='.08', shininess='.10')
    robot = world.find("body[@name='link0']")
    # Retain only the existing robot. Industrial cube decorations are not part
    # of EBiM and must not leak into the extracted upstream station.
    for e in list(world):
        if e is not robot:
            world.remove(e)
    robot.set('pos', ' '.join(map(str, ROBOT_POSITION)))
    robot.set('quat', '.7071067811865476 0 0 .7071067811865476')
    # Map our robot into upstream's collision layers; do not modify props.
    # Rigid housing avoids its own silicone, while both interact with type 1
    # EBiM objects. Type 16 is the upstream spoon's gripper contact layer.
    root.find(".//default[@class='panda/collision']/geom").set('contype', '32')
    root.find(".//default[@class='panda/collision']/geom").set('conaffinity', '1')
    for name in ('standard_shell', 'silicone_backing'):
        geom = root.find(f".//geom[@name='{name}']")
        geom.set('contype', '32')
        geom.set('conaffinity', '1')
    for flex in robot.iter('flexcomp'):
        flex.find('contact').set('contype', '16')
        flex.find('contact').set('conaffinity', '17')
    gap = 2*(.04+mount_reference(cfg)['position_m'][1]-cfg['total_height_m'])
    minimum_gap = 2*cfg.get('fem_contact_radius_m', .0002)
    root.find(".//default[@class='finger']/joint").set('range', f'{-(gap-minimum_gap)/2} 0')
    # The global Panda defaults are empty. Give imported source elements their
    # own child class so EBiM's default contact values remain exactly intact.
    default = ET.SubElement(root.find('default'), 'default', {'class': 'ebim_task3'})
    for e in source.find('default'):
        default.append(copy.deepcopy(e))
    elements = [e for e in selected_world(source) if e.get('name') not in
                ('dynamic_bowl2', 'dynamic_plate2', 'dynamic_simple_tray', 'dynamic_cup') and not (
                    e.get('name', '').startswith('dynamic_coffee_bean_') and
                    int(e.get('name').rsplit('_', 1)[1]) >= BEAN_COUNT)]
    for e in asset_subset(source, elements):
        e = copy.deepcopy(e)
        if e.tag == 'mesh' and e.get('name', '').startswith('visual_head_'):
            e.set('scale', head_scale)
        if e.get('file'):
            directory = ASSETS/'assets' if e.tag == 'mesh' else ASSETS
            e.set('file', str(directory/e.get('file')))
        asset.append(e)
    for e in elements:
        e = copy.deepcopy(e)
        e.set('childclass' if e.tag == 'body' else 'class', 'ebim_task3')
        if e.get('name') == 'configurable_visual_head':
            e.set('pos', ' '.join(map(str, HEAD_POSITION)))
            e.set('quat', ' '.join(map(str, HEAD_QUAT)))
            skin = next(g for g in e.findall('geom') if 'skin' in g.get('mesh', ''))
            collision = copy.deepcopy(skin)
            collision.set('name', 'local_head_collision')
            collision.set('contype', '1')
            collision.set('conaffinity', '49')
            collision.set('group', '3')
            collision.set('rgba', '0 0 0 0')
            collision.attrib.pop('material', None)
            e.append(collision)
            ET.SubElement(e, 'geom', name='local_head_back_visual', type='mesh', mesh='local_head_back_mesh',
                          material='local_head_back_skin', group='1', contype='0', conaffinity='0', density='0')
            ET.SubElement(e, 'geom', name='local_head_back_collision', type='mesh', mesh='local_head_back_mesh',
                          group='3', contype='1', conaffinity='49', density='0', rgba='0 0 0 0')
        if e.get('name') == 'dynamic_cup':
            # These are official teleop.py runtime settings, not local tuning.
            for g in e.iter('geom'):
                if int(g.get('contype', '1')):
                    g.set('friction', ' '.join(map(str, official_cfg['contact']['cup_friction'])))
                    g.set('priority', str(official_cfg['contact']['cup_contact_priority']))
        world.append(e)
    geoms = {e.get('name') for e in world.iter('geom')}
    contact = root.find('contact')
    if contact is None:
        contact = ET.SubElement(root, 'contact')
    for e in source.find('contact'):
        if e.tag == 'pair' and e.get('geom1') in geoms and e.get('geom2') in geoms:
            contact.append(copy.deepcopy(e))
    # Preserve original lights and floor. Cameras alone are local acquisition
    # views and are saved with intrinsics/extrinsics by the existing recorders.
    for e in source.findall('worldbody/light'):
        world.append(copy.deepcopy(e))
    add_camera(world, 'lab_front', [-2.45, -.75, 1.90], [-1.73, .12, 1.19])
    add_camera(world, 'lab_side', [-.80, -.05, 1.70], [-1.74, .12, 1.18])
    root.find('statistic').set('center', '-1.65 .0 1.25')
    root.find('statistic').set('extent', '1.2')
    # The thin local sensor requires a smaller timestep than EBiM's 1 ms.
    # Use sparse Jacobians for its 100 free beans plus the elastic fingertips.
    option = root.find('option')
    for key, value in source.find('option').attrib.items():
        if key != 'timestep':
            option.set(key, value)
    option.set('jacobian', 'sparse')
    option.set('noslip_iterations', '0')
    return ET.tostring(root, encoding='unicode')


class PandaFeedingEnv(PandaPickCubeTactileEnv):
    task_name = 'feeding'
    workspace = np.array([[-2.01, -.36, TABLE_TOP+.026], [-1.26, .52, 1.68]])
    contact_geometry = 'spoon'

    def __init__(self, cfg=None, pollen=False):
        cfg = dict(configuration() if cfg is None else cfg)
        for key, value in dict(cap_top_inner_fraction=.85, fem_internal_contact=True,
                               fem_contact_radius_m=.0002, fem_contact_timeconst_s=.0002).items():
            cfg.setdefault(key, value)
        cfg.setdefault('scene_design', 'ebim_task3_fixed_panda')
        cfg.setdefault('gripper_control_mode', 'panda_menagerie_position')
        cfg['silicone_numerics'] = 'edge-contact mesh + internal constraints; material uncalibrated'
        super().__init__(cfg, pollen=pollen, xml_factory=feeding_scene,
                         contact_body='dynamic_spoon2', contact_geom='spoon_collision_original_stem_mesh_00',
                         object_joint=None)
        # All spoon parts, not only a single collision segment, count for load.
        body_ids = {self.model.body(b.get('name')).id for b in
                    ET.fromstring(self.xml).find(".//body[@name='dynamic_spoon2']").iter('body')}
        self.contact_geoms = {i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] in body_ids}
        self.cfg.update(task='single_arm_feeding_scene',
                        ebim_revision=json.loads((ASSETS/'manifest.json').read_text())['revision'],
                        bean_count=BEAN_COUNT, robot_base_world_m=ROBOT_POSITION, head_world_m=HEAD_POSITION,
                        head_uniform_scale=HEAD_SCALE, head_height_with_neck_m=.2*HEAD_SCALE,
                        gripper_empty_gap_m=2*cfg['fem_contact_radius_m'], upstream_timestep_s=.001,
                        head_collision='local convex hulls of source skin and posterior completion; rigid, unvalidated',
                        head_completion_manifest=str(HEAD_COMPLETION/'head_back.json'),
                        feeding_start='physically gripped spoon with six free beans',
                        grasp_assist=False, task_success_evaluator=False,
                        source_asset_manifest=str(ASSETS/'manifest.json'))

    def reset(self, seed=None):
        super().reset(seed=seed)  # Unloaded magnetic baseline before closure.
        # Pinch the broad top/bottom of the handle, not its thin side edges.
        self.target_rotation = self.target_rotation@rotation_matrix([np.pi/2, 0., 0.])
        self.servo_rotation = self.target_rotation.copy()
        for _ in range(30):
            self.solve_ik()
        self.data.qpos[self.arm_q] = self.joint_target
        self.data.qvel[:] = 0.
        mujoco.mj_forward(self.model, self.data)
        self.baseline = self.magnetic_field().copy()
        self.bean_bodies = [self.model.body(f'dynamic_coffee_bean_{i:04d}').id for i in range(BEAN_COUNT)]
        self.head_geom = self.model.geom('local_head_collision').id
        self.head_geoms = {self.head_geom, self.model.geom('local_head_back_collision').id}
        self.head_force_n = self.head_peak_n = self.head_interval_peak_n = 0.
        self.head_contact_impulse_ns = 0.
        self.head_net_force = np.zeros(3)
        self._contact_force = np.zeros(6)
        # Horizontal spoon: handle at local -X, concave bowl at +X, +Z up.
        # Long axis points forward from the gripper. Only freejoint initial
        # conditions change; no weld, mocap, gravity cancellation or pose copy.
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotation.ravel())
        origin = self.data.site_xpos[self.tcp_site]-rotation@np.array([-.025, 0., .0059])
        adr = self.model.jnt_qposadr[self.model.joint('dynamic_spoon2_freejoint').id]
        self.data.qpos[adr:adr+7] = np.r_[origin, quat]
        self.data.qvel[:] = 0.
        # Start at handle contact, rather than waiting for the jaws to close
        # while an unsupported spoon falls. The pinch width comes from the
        # original stem mesh at the grasp section (not a force setting).
        empty_gap = 2*(.04+mount_reference(self.cfg)['position_m'][1]-self.cfg['total_height_m'])
        self.data.qpos[self.finger_q] = -(empty_gap-.0025)/2
        self.command_gripper(True)
        for i, bid in enumerate(self.bean_bodies):
            jadr = self.model.jnt_qposadr[self.model.body_jntadr[bid]]
            point = [.031+.009*(i//2), -.0035+.007*(i%2), .0005]
            self.data.qpos[jadr:jadr+3] = origin+rotation@point
        mujoco.mj_forward(self.model, self.data)
        for _ in range(round(self.cfg.get('feeding_settle_s', .30)/self.cfg['timestep_s'])):
            self.control()
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if np.any(self.data.warning.number):
            raise RuntimeError('Preloaded spoon reset produced a MuJoCo warning')
        self.initial_cube_z = float(self.data.xpos[self.probe_body, 2])
        self.data.time = self.zero_time = 0.
        self.last_row = None
        return self.observation(), {'uncalibrated': True}

    def after_physics_step(self):
        """Sum contact force magnitudes on head, peak over every physics step.

        Sum of magnitudes avoids cancellation from opposite contacts. Also
        record the net world force separately. This is a local metric, not the
        unpublished competition scorer or a validated human safety measure.
        """
        total = 0.
        net = np.zeros(3)
        # Filter the array view in NumPy before creating Python contact
        # objects. Spoon/FEM contacts dominate ncon; visiting each of them at
        # 10 kHz cost millions of Python iterations even with no head contact.
        # Still evaluate every head contact on every physics step, in source
        # order: no sampling, force averaging or missed peak to gain speed.
        geoms = self.data.contact.geom
        hit = np.zeros(self.data.ncon, dtype=bool)
        for geom in self.head_geoms:
            hit |= (geoms[:, 0] == geom) | (geoms[:, 1] == geom)
        for i in np.flatnonzero(hit):
            contact = self.data.contact[i]
            mujoco.mj_contactForce(self.model, self.data, int(i), self._contact_force)
            total += float(np.linalg.norm(self._contact_force[:3]))
            sign = 1. if contact.geom[1] in self.head_geoms else -1.
            net += sign*contact.frame.reshape(3, 3).T@self._contact_force[:3]
        self.head_force_n, self.head_net_force = total, net
        self.head_interval_peak_n = max(self.head_interval_peak_n, total)
        self.head_peak_n = max(self.head_peak_n, total)
        self.head_contact_impulse_ns += total*self.cfg['timestep_s']

    def observation(self):
        obs = super().observation()
        obs['spoon_position_m'] = obs.pop('cube_position_m')
        return obs

    def viewer_camera(self, close=False, operator=False):
        cam = mujoco.MjvCamera()
        # Side view keeps the enlarged head from hiding spoon/pad contact.
        cam.lookat[:] = [-1.55, .08, 1.28] if close else [-1.60, .04, 1.28]
        cam.distance, cam.azimuth, cam.elevation = (1.05, 0, -22) if close else (2.05, 0, -28)
        return cam

    def step(self, action):
        self.head_interval_peak_n = 0.
        _, _, _, _, row = super().step(action)
        row['spoon_xyz_m'] = row.pop('cube_xyz_m')
        row['spoon_lift_m'] = row.pop('lift_m')
        row['bean_positions_m'] = self.data.xpos[self.bean_bodies].copy()
        row['success'] = self.success = False
        row.update(head_contact_force_n=self.head_force_n, head_interval_peak_n=self.head_interval_peak_n,
                   head_peak_force_n=self.head_peak_n, head_net_force_n=self.head_net_force.copy(),
                   head_contact_impulse_ns=self.head_contact_impulse_ns)
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jp, jr, self.tcp_site)
        singular = np.linalg.svd(np.vstack([jp[:, self.arm_v], .2*jr[:, self.arm_v]]), compute_uv=False)
        q, bounds = self.data.qpos[self.arm_q], self.model.jnt_range[self.arm_joints]
        row['ik_weighted_condition'] = float(singular[0]/max(singular[-1], 1e-15))
        row['ik_min_singular_value'] = float(singular[-1])
        row['joint_limit_margin_deg'] = float(np.rad2deg(np.minimum(q-bounds[:, 0], bounds[:, 1]-q)).min())
        row['task_status'] = 'delivery only; local rigid head contact metric; outcome manually labelled'
        self.last_row = row
        return self.observation(), 0., False, False, row
