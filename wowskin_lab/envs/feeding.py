"""Fixed single Panda using EBiM task-3 feeding station source assets.

Local delivery-only variant, not EBiM's mobile dual-arm evaluator. Original
meshes, spoon and bean mechanics are retained. Reset physically grips a loaded
spoon. The head completion/enlargement and seated care room are local additions.
A local rigid convex head collision proxy reports simulation loads;
it does not model a mouth, swallowing, tissue compliance or ISO safety limits.
"""
import copy
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from wowskin_lab.core.arm_control import joint_pd_reset
from wowskin_lab.paths import ROOT
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, pick_scene, rotation_matrix
from wowskin_lab.data.cameras import add_recording_cameras
from wowskin_lab.envs.feeding_room import add_care_room, add_seated_adult, update_care_room_view
from wowskin_lab.envs.feeding_evaluation import DeliveryEvaluator, delivery_config

ASSETS = ROOT/'assets/ebim_feeding'
TABLE_TOP = 1.08485019
ROBOT_POSITION = [-1.63, -.38, TABLE_TOP]
# User-requested visual enlargement of the complete source head + neck.
# Scale eyes, posterior completion and collision together.
HEAD_SCALE = 1.5
HEAD_POSITION = [-1.09, .04, 1.36]
LEGACY_HEAD_POSITION = [-1.45, .25, TABLE_TOP+.1*HEAD_SCALE]
BEAN_COUNT = 6
HEAD_COMPLETION = ROOT/'assets/generated/feeding'
# Source head eyes are on local -Y. Face the robot, not away from it.
HEAD_YAW = np.arctan2(ROBOT_POSITION[0]-HEAD_POSITION[0],
                     HEAD_POSITION[1]-ROBOT_POSITION[1])
HEAD_QUAT = [np.cos(HEAD_YAW/2), 0., 0., np.sin(HEAD_YAW/2)]
BODIES = ('dynamic_bowl2', 'dynamic_plate2', 'dynamic_simple_tray', 'dynamic_spoon2',
          'dynamic_cup', 'configurable_visual_head')


def head_placement(setting):
    if setting not in ('care_room', 'ebim_station'):
        raise ValueError('feeding_setting must be care_room or ebim_station')
    position = HEAD_POSITION if setting == 'care_room' else LEGACY_HEAD_POSITION
    yaw = np.arctan2(ROBOT_POSITION[0]-position[0], position[1]-ROBOT_POSITION[1])
    return position, [np.cos(yaw/2), 0., 0., np.sin(yaw/2)]


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
    care_room = cfg.get('feeding_setting', 'care_room') == 'care_room'
    head_position, head_quat = head_placement(cfg.get('feeding_setting', 'care_room'))
    root = ET.fromstring(pick_scene(cfg, layout))
    root.set('model', 'WowSkin-Lab | PandaFeeding: Care room' if care_room
             else 'WowSkin-Lab | PandaFeeding: Tabletop')
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
        if geom is not None:
            geom.set('contype', '32')
            geom.set('conaffinity', '1')
    for flex in robot.iter('flexcomp'):
        flex.find('contact').set('contype', '16')
        flex.find('contact').set('conaffinity', '17')
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
    # A derived neck fit; original source mesh remains the collision
    # reference and the legacy station remains visually unchanged.
    if care_room:
        if not (HEAD_COMPLETION/'head_short_hair.obj').exists():
            from scripts.build_feeding_head import build_hair
            build_hair()
        ET.SubElement(asset, 'mesh', name='local_short_hair_mesh',
                      file=str(HEAD_COMPLETION/'head_short_hair.obj'), scale=head_scale)
        ET.SubElement(asset, 'texture', name='local_short_hair_texture', type='2d',
                      file=str(HEAD_COMPLETION/'head_short_hair.png'))
        ET.SubElement(asset, 'material', name='local_short_hair_material',
                      texture='local_short_hair_texture', specular='.05', shininess='.08')
        if not (HEAD_COMPLETION/'head_seated_visual.obj').exists():
            from scripts.build_feeding_head import build_seated_face
            build_seated_face()
        ET.SubElement(asset, 'mesh', name='local_seated_face_mesh',
                      file=str(HEAD_COMPLETION/'head_seated_visual.obj'), scale=head_scale)
    for e in elements:
        e = copy.deepcopy(e)
        e.set('childclass' if e.tag == 'body' else 'class', 'ebim_task3')
        if e.get('name') == 'configurable_visual_head':
            e.set('pos', ' '.join(map(str, head_position)))
            e.set('quat', ' '.join(map(str, head_quat)))
            skin = next(g for g in e.findall('geom') if 'skin' in g.get('mesh', ''))
            collision = copy.deepcopy(skin)
            collision.set('name', 'local_head_collision')
            collision.set('contype', '1')
            collision.set('conaffinity', '49')
            collision.set('group', '3')
            collision.set('rgba', '0 0 0 0')
            collision.attrib.pop('material', None)
            e.append(collision)
            if care_room:
                skin.set('mesh', 'local_seated_face_mesh')
                ET.SubElement(e, 'geom', name='local_short_hair_visual', type='mesh',
                              mesh='local_short_hair_mesh', material='local_short_hair_material',
                              contype='0', conaffinity='0', density='0', group='1')
            # Goal regions are drawn only in the operator's user_scn. There
            # is no target site/geom in the camera or physics model.
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
    # Keep the original station available for old recordings; care-room assets
    # are local and never modify the vendored EBiM files.
    for e in source.findall('worldbody/light'):
        world.append(copy.deepcopy(e))
    if care_room:
        add_seated_adult(root, head_position, head_quat, cap['skin_rgba'])
        add_care_room(root)
    for parent in root.iter():
        for child in list(parent):
            if child.tag == 'camera' and not child.get('name', '').startswith('optical_'):
                parent.remove(child)
    hand = root.find(".//body[@name='hand']")
    tcp_z = float(hand.find("site[@name='tactile_tcp']").get('pos').split()[2])
    add_recording_cameras(world, hand, tcp_z, profile=cfg.get('camera_profile'),
                          scene='feeding_care_room' if care_room else 'feeding')
    root.find('statistic').set('center', '-1.38 .0 1.15' if care_room else '-1.65 .0 1.25')
    root.find('statistic').set('extent', '1.2')
    # The thin local sensor requires a smaller timestep than EBiM's 1 ms.
    # Use sparse Jacobians for the loaded spoon, six beans and elastic fingertips.
    option = root.find('option')
    for key, value in source.find('option').attrib.items():
        if key != 'timestep':
            option.set(key, value)
    # User-selected local solver override. Keep source assets/materials intact.
    option.set('solver', 'CG')
    option.set('jacobian', 'sparse')
    option.set('noslip_iterations', '0')
    if cfg.get('skin_dof') == 'trilinear':
        option.set('integrator', 'discrete')
    return ET.tostring(root, encoding='unicode')


class PandaFeedingEnv(PandaPickCubeTactileEnv):
    task_name = 'feeding'
    workspace = np.array([[-2.01, -.36, -np.inf], [-1.26, .52, 1.68]])
    contact_geometry = 'spoon'

    def __init__(self, cfg=None, pollen=False):
        if cfg is None:
            from types import SimpleNamespace
            from .pick_cube import response_configuration
            cfg = response_configuration(SimpleNamespace(task='feeding'))
        cfg = dict(cfg)
        if not cfg.get('sensor'):
            for key, value in dict(cap_top_inner_fraction=.85,
                                   fem_contact_radius_m=.0002, fem_contact_timeconst_s=.0002).items():
                cfg.setdefault(key, value)
        cfg.setdefault('feeding_setting', 'care_room')
        cfg.setdefault('feeding_delivery', delivery_config())
        if cfg.get('gripper') == 'robotiq_2f85':
            upstream = json.loads((ASSETS/'config.json').read_text())
            cfg.setdefault('feeding_preload', dict(
                source='EBiM task3 teleop.py:update_gripper, revision 161db49e9eafd34be31e5bdada72d8e2f796c16a',
                scope='reset only; flex external normal forces replace source rigid-pad force sums',
                update_period_s=upstream['scene']['timestep_s'],
                **{k: v for k, v in upstream['controller'].items() if k.startswith('gripper_')}))
        self.delivery = DeliveryEvaluator(cfg['feeding_delivery'])
        head_position, _ = head_placement(cfg['feeding_setting'])
        cfg['scene_design'] = ('local_care_room_v2_delivery' if cfg['feeding_setting'] == 'care_room'
                               else 'ebim_task3_fixed_panda')
        self.workspace = type(self).workspace.copy()
        if cfg['feeding_setting'] == 'care_room':
            self.workspace[1, 0] = -1.0  # Allow approaching the seated person's face.
        cfg.setdefault('gripper_control_mode', 'panda_menagerie_position')
        if not cfg.get('sensor'):
            cfg['silicone_numerics'] = 'Trilinear reduced skin, bottom nodes pinned; material uncalibrated'
        super().__init__(cfg, pollen=pollen, xml_factory=feeding_scene,
                         contact_body='dynamic_spoon2', contact_geom='spoon_collision_original_stem_mesh_00',
                         object_joint=None)
        # All spoon parts, not only a single collision segment, count for load.
        body_ids = {self.model.body(b.get('name')).id for b in
                    ET.fromstring(self.xml).find(".//body[@name='dynamic_spoon2']").iter('body')}
        self.contact_geoms = {i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] in body_ids}
        self.cfg.update(task='single_arm_feeding_scene',
                        constraint_solver='CG', reset_constraint_solver='CG', constraint_jacobian='sparse',
                        constraint_solver_source='Local user-requested override of EBiM Newton; source iteration limit retained',
                        ebim_revision=json.loads((ASSETS/'manifest.json').read_text())['revision'],
                        bean_count=BEAN_COUNT, robot_base_world_m=ROBOT_POSITION, head_world_m=head_position,
                        head_uniform_scale=HEAD_SCALE, head_height_with_neck_m=.2*HEAD_SCALE,
                        gripper_empty_gap_m=(float(np.linalg.norm(np.diff(self.sensor_surfaces(), axis=0)))-2*float(self.data.qpos[self.finger_q].mean()) if self.is_optical else 2*cfg['fem_contact_radius_m']), upstream_timestep_s=.001,
                        head_collision='local convex hulls of source skin and posterior completion; rigid, unvalidated',
                        head_completion_manifest=str(HEAD_COMPLETION/'head_back.json'),
                        seated_human=('local static adult with rigid torso/limb/chair collision; no soft tissue or animation'
                                      if cfg['feeding_setting'] == 'care_room' else 'absent'),
                        room_decoration=('local visual-only background; original table collision retained'
                                         if cfg['feeding_setting'] == 'care_room' else 'original station, no care-room decoration'),
                        camera_scene_profile='feeding_care_room' if cfg['feeding_setting'] == 'care_room' else 'feeding',
                        feeding_start='physically gripped spoon with six free beans',
                        grasp_assist=False, task_success_evaluator=self.delivery.config['schema'],
                        source_asset_manifest=str(ASSETS/'manifest.json'))
        if self.is_optical:
            self.cfg.pop('silicone_numerics', None)
            self.cfg.update(constraint_solver='source optical', reset_constraint_solver='source optical', constraint_solver_source=self.cfg['sensor'], constraint_jacobian='source optical')

    @joint_pd_reset
    def reset(self, seed=None):
        if not self.is_optical:
            self.model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        return self._reset_loaded_spoon(seed)

    def _reset_loaded_spoon(self, seed=None):
        super().reset(seed=seed)  # Unloaded magnetic baseline before closure.
        self.delivery.reset()
        # Pinch the broad top/bottom of the handle, not its thin side edges.
        self.target_rotation = self.target_rotation@rotation_matrix([np.pi/2, 0., 0.])
        spoon_yaw = 0.
        if self.cfg['feeding_setting'] == 'care_room':
            self.target_xyz = np.asarray(self.cfg['feeding_delivery']['initial_tcp_world_m']).copy()
            self.servo_xyz = self.target_xyz.copy()
            head_position, _ = head_placement(self.cfg['feeding_setting'])
            toward = np.asarray(head_position)-self.target_xyz
            spoon_yaw = np.arctan2(toward[1], toward[0])-np.pi/2
            self.target_rotation = rotation_matrix([0., 0., spoon_yaw])@self.target_rotation
            long_axis = np.array([np.cos(spoon_yaw+np.pi/2), np.sin(spoon_yaw+np.pi/2), 0.])
            self.target_rotation = rotation_matrix(np.pi*long_axis)@self.target_rotation
            self.joint_target = np.asarray(self.cfg['feeding_delivery']['initial_arm_joint_seed_rad']).copy()
        self.servo_rotation = self.target_rotation.copy()
        for _ in range(80):
            self.solve_ik()
        self.data.qpos[self.arm_q] = self.joint_target
        self.data.qvel[:] = 0.
        mujoco.mj_forward(self.model, self.data)
        if not self.is_optical:
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
        rotation = rotation_matrix([0., 0., spoon_yaw])@rotation
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotation.ravel())
        # Initial placement only. Historical configurations retain the old
        # grasp; the current user-selected point is nearer the bowl.
        grasp = np.asarray(self.delivery.config.get('initial_grasp_spoon_m', [-.025, 0., .0059]))
        self.data.qvel[:] = 0.
        # Start at handle contact, rather than waiting for the jaws to close
        # while an unsupported spoon falls. The pinch width comes from the
        # original stem mesh at the grasp section (not a force setting).
        pinch = self.data.site_xpos[self.tcp_site].copy()
        if self.cfg.get('gripper') == 'robotiq_2f85':
            from .robotiq import initialize_gap
            initial_angle = initialize_gap(self.model, self.data, self.cfg['total_height_m'], .0025)
            # Unlike Panda's slide fingers, the source parallel linkage also
            # moves the pads along the tool axis as it closes. Place the
            # preloaded spoon at the actual initial pad midpoint.
            surfaces = []
            for name in ('wowskin_mount', 'silicone_mount'):
                bid = self.model.body(name).id
                surfaces.append(self.data.xpos[bid]+self.data.xmat[bid].reshape(3, 3) @
                                np.array([0., 0., self.cfg['total_height_m']]))
            pinch = np.mean(surfaces, axis=0)
        else:
            empty_gap = np.linalg.norm(np.diff(self.sensor_surfaces(), axis=0))
            self.data.qpos[self.finger_q] = -(empty_gap-.0025)/2
        origin = pinch-rotation@grasp
        adr = self.model.jnt_qposadr[self.model.joint('dynamic_spoon2_freejoint').id]
        self.data.qpos[adr:adr+7] = np.r_[origin, quat]
        if self.cfg.get('gripper') == 'robotiq_2f85':
            self.grip_target = initial_angle
        else:
            self.command_gripper(True)
        for i, bid in enumerate(self.bean_bodies):
            jadr = self.model.jnt_qposadr[self.model.body_jntadr[bid]]
            point = [.031+.009*(i//2), -.0035+.007*(i%2), .0005]
            self.data.qpos[jadr:jadr+3] = origin+rotation@point
        mujoco.mj_forward(self.model, self.data)
        preload = self.cfg.get('feeding_preload') if self.cfg.get('gripper') == 'robotiq_2f85' else None
        filtered = np.zeros(2)
        confirmations, preloaded = 0, False
        update_steps = 1 if preload is None else round(preload['update_period_s']/self.cfg['timestep_s'])
        if preload and (update_steps < 1 or not np.isclose(
                update_steps*self.cfg['timestep_s'], preload['update_period_s'])):
            raise ValueError('Robotiq preload requires a physics step dividing the source 1 ms controller period')
        for step in range(round(self.cfg.get('feeding_settle_s', .30)/self.cfg['timestep_s'])):
            if preload and not preloaded and step % update_steps == 0:
                # Port of EBiM's close ramp + confirmed force preload. Only
                # pad force extraction changes from rigid geoms to flexes.
                alpha = preload['gripper_force_filter_alpha']
                filtered = alpha*self._external_pad_forces()+(1-alpha)*filtered
                confirmations = (confirmations+1 if filtered.sum() >=
                    preload['gripper_contact_force_threshold_n'] else 0)
                endpoint = min(preload['gripper_fixed_close_target_rad'],
                    self.model.actuator_ctrlrange[self.gripper_control, 1],
                    self.model.jnt_range[self.finger_joints[0], 1])
                if confirmations >= preload['gripper_contact_confirmation_steps']:
                    self.grip_target = min(endpoint, float(self.data.qpos[self.finger_q[0]])+
                        preload['gripper_contact_hold_increment_rad'])
                    preloaded = True
                else:
                    self.grip_target = min(endpoint, self.grip_target+
                        preload['gripper_close_rate_rad_s']*preload['update_period_s'])
                    preloaded = self.grip_target >= endpoint-1e-12
            self.control()
            mujoco.mj_step(self.model, self.data)
        if preload and not preloaded:
            raise RuntimeError('Source Robotiq preload did not finish within feeding_settle_s')
        mujoco.mj_forward(self.model, self.data)
        if np.any(self.data.warning.number):
            raise RuntimeError('Preloaded spoon reset produced a MuJoCo warning')
        self.initial_cube_z = float(self.data.xpos[self.probe_body, 2])
        self.data.time = self.zero_time = 0.
        self.last_row = None
        return self.observation(), {'uncalibrated': True}

    def _external_pad_forces(self):
        """Sum external normal contact forces for each flex, without cancellation."""
        if not hasattr(self, '_robot_geom_ids'):
            root = self.model.body('link0').id
            bodies = {root}
            for bid in range(root+1, self.model.nbody):
                if self.model.body_parentid[bid] in bodies:
                    bodies.add(bid)
            self._robot_geom_ids = {i for i, bid in enumerate(self.model.geom_bodyid) if bid in bodies}
        forces = np.zeros(2)
        flexes = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_FLEX, name)
                  for name in ('skin', 'silicone')]
        wrench = np.zeros(6)
        for index, contact in enumerate(self.data.contact):
            for side, fid in enumerate(flexes):
                if fid not in contact.flex:
                    continue
                other = 1-int(np.flatnonzero(contact.flex == fid)[0])
                if contact.flex[other] in flexes or contact.geom[other] in self._robot_geom_ids:
                    continue
                mujoco.mj_contactForce(self.model, self.data, index, wrench)
                forces[side] += abs(float(wrench[0]))
        return forces

    def sample(self, controller=None):
        if self.is_optical or self.model.opt.solver != mujoco.mjtSolver.mjSOL_CG:
            return super().sample(controller)
        if not hasattr(self, '_sample_backup'):
            self._sample_backup = mujoco.MjData(self.model)
        mujoco.mj_copyData(self._sample_backup, self.model, self.data)
        names = ('head_force_n', 'head_net_force', 'head_interval_peak_n',
                 'head_peak_n', 'head_contact_impulse_ns', 'filtered', 'delay')
        saved = {name: copy.deepcopy(getattr(self, name)) for name in names}
        rng = copy.deepcopy(self.rng.bit_generator.state)
        try:
            return super().sample(controller)
        except RuntimeError as error:
            if not any(message in str(error) for message in ('MuJoCo instability/warning', 'FEM invalid', 'Silicone mesh inverted/nonfinite')):
                raise
            # Restore the sample boundary before propagating the failure.
            # Discard failed force peaks, filtering and random draws;
            # never retry with another solver or apply the action twice.
            mujoco.mj_copyData(self.data, self.model, self._sample_backup)
            for name, value in saved.items():
                setattr(self, name, value)
            self.rng.bit_generator.state = rng
            raise

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
        if self.cfg.get('feeding_setting') == 'care_room':
            cam.lookat[:] = [-1.33, .01, 1.31] if close else [-1.39, .02, 1.07]
            cam.distance, cam.azimuth, cam.elevation = (1.22, 95, -12) if close else (2.65, 95, -17)
            return cam
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
        row.update(self.evaluate_delivery(row))
        row['success'] = self.success = self.delivery.success
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
        row['constraint_solver'] = ('source optical' if self.is_optical else mujoco.mjtSolver(self.model.opt.solver).name.removeprefix('mjSOL_'))
        row['task_status'] = 'delivered to mouth approach region' if self.success else 'delivery in progress'
        self.last_row = row
        return self.observation(), float(self.success), False, False, row

    def evaluate_delivery(self, row):
        head = self.model.body('configurable_visual_head').id
        rotation = self.data.xmat[self.probe_body].reshape(3, 3)
        point = self.data.xpos[self.probe_body]+rotation@np.asarray(self.delivery.config['bowl_center_spoon_m'])
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jac, None, point, self.probe_body)
        contacts = sum(any(g in self.head_geoms for g in c.geom) and c.dist <= 0
                       for c in self.data.contact)
        return self.delivery.update(float(self.data.time), head_pos=self.data.xpos[head],
            head_rotation=self.data.xmat[head].reshape(3, 3), spoon_pos=self.data.xpos[self.probe_body],
            spoon_rotation=rotation, bean_positions=self.data.xpos[self.bean_bodies],
            bowl_speed=np.linalg.norm(jac@self.data.qvel), grasped=row['grasped'], head_contacts=contacts)

    def evaluation_metrics(self, row=None):
        metrics = self.evaluate_delivery(row or dict(grasped=False)) if row is None else row
        keys = ('delivery_success', 'delivery_eligible', 'delivery_dwell_s', 'delivery_distance_m',
                'delivery_inside_target', 'beans_in_spoon', 'spoon_grasped', 'spoon_tilt_deg',
                'spoon_approach_angle_deg', 'spoon_bowl_speed_m_s', 'head_contact_count')
        return dict({key: metrics[key] for key in keys}, head_peak_force_n=self.head_peak_n,
                    head_contact_force_n=self.head_force_n,
                    head_interval_peak_n=self.head_interval_peak_n,
                    head_net_force_n=self.head_net_force.copy(),
                    head_contact_impulse_ns=self.head_contact_impulse_ns)

    def task_evaluation(self):
        return self.delivery.summary()

    def update_viewer_options(self, handle):
        if self.cfg['feeding_setting'] == 'care_room':
            update_care_room_view(handle)
