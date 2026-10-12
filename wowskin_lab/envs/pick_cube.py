"""PandaPickCube tactile teleoperation and unchanged Pollen real-to-sim test.

Seven-action reset/step interface (XYZ, local rotvec increments, gripper target).
The cube is a free rigid body. Only contact/friction can lift it; no weld,
attachment, pose copying, or object gravity compensation is used.
"""
import argparse
from types import SimpleNamespace
from contextlib import nullcontext
from collections import deque
import csv
import json
from pathlib import Path
import textwrap
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from wowskin_lab.envs.franka import FrankaTactileSim, HOME, ViewerStyleGuard, apply_free_camera, build_scene, mount_reference
from wowskin_lab.core.physics import CHANNELS, TactileSim, configuration
from wowskin_lab.ui.keyboard import BACKSPACE, ENTER, ESC, MOVE_AXES, ViewerKeyboard, direction
from wowskin_lab.data.cameras import add_recording_cameras
from wowskin_lab.teleop.command import CartesianCommand
from wowskin_lab.teleop.target_limits import straight_step, workspace_bounds
from wowskin_lab.envs.workcell import industrial_workcell
from wowskin_lab.core.arm_control import configure_controller, joint_pd_reset, cartesian_active, arm_torque
from wowskin_lab.paths import ROOT
from wowskin_lab.envs.task_evaluation import CubePickPlaceEvaluator, cube_corners, gripper_contacts, task_config

TABLE_Z = .10
CUBE_HALF = .020
CUBE_MASS = .020
WORKSPACE = np.array([[.24, -.18, -np.inf], [.58, .18, .40]])
ROTATE_AXES = {
    ord('I'): (1, 0, 0), ord('K'): (-1, 0, 0),
    ord('L'): (0, 1, 0), ord('J'): (0, -1, 0),
    ord('U'): (0, 0, 1), ord('O'): (0, 0, -1),
}
MOTION_KEYS = set(MOVE_AXES) | set(ROTATE_AXES)


def rotation_vector(matrix):
    """SO(3) logarithm, radians; MuJoCo uses scalar-first quaternions."""
    quat, vector = np.zeros(4), np.zeros(3)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix).reshape(9))
    mujoco.mju_quat2Vel(vector, quat, 1.)
    return vector


def rotation_matrix(vector):
    """SO(3) exponential for a rotation vector, not Euler-angle addition."""
    vector = np.asarray(vector, dtype=float)
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    quat, matrix = np.zeros(4), np.zeros(9)
    mujoco.mju_axisAngle2Quat(quat, vector/angle, angle)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


def limit_vector(vector, limit):
    return vector*min(1., limit/max(np.linalg.norm(vector), 1e-12))


def pick_scene(cfg, layout):
    root = ET.fromstring(build_scene(cfg, layout))
    # The upstream Panda includes two wrist cameras; acquisition uses only
    # the shared pair of fixed tabletop cameras requested for this project.
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "camera":
                parent.remove(child)
    root.set("model", "WowSkin-Lab | PandaPickCube: WowSkin + plain silicone")
    # Broad box/flex contacts create hundreds of constraints. CG avoids the
    # costly Newton factorization while preserving the same contact geometry.
    option = root.find("option")
    option.set("solver", "CG")
    option.set("jacobian", cfg.get("constraint_jacobian", "dense"))
    option.set("iterations", "30")
    option.set("tolerance", "1e-8")
    option.set("noslip_iterations", "1")
    if cfg.get('skin_dof') == 'trilinear':
        # Required by MuJoCo 3.15's implicit flex elasticity integration.
        option.set('integrator', 'discrete')
        option.set('noslip_iterations', '0')
    world = root.find("worldbody")
    world.remove(world.find("body[@name='probe']"))
    floor = world.find("geom[@name='floor']")
    floor.set("conaffinity", "7")
    ET.SubElement(world, "geom", name="table", type="box", pos=".43 0 .05",
                  size=".22 .24 .05", rgba=".28 .34 .40 1", contype="1", conaffinity="7",
                  friction=".6 .005 .0001", solref=".002 1")
    cube = ET.SubElement(world, "body", name="cube", pos=f".40 0 {TABLE_Z+CUBE_HALF+.0002}")
    ET.SubElement(cube, "freejoint", name="cube_joint")
    ET.SubElement(cube, "geom", name="cube", type="box", size=f"{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}",
                  mass=str(CUBE_MASS), rgba=".16 .55 .82 1", contype="4", conaffinity="3",
                  friction=f"{cfg['friction']} .005 .0001", solref=".002 1")
    for name in ("standard_shell", "silicone_backing"):
        shell = root.find(f".//geom[@name='{name}']")
        shell.set("contype", "1")
        shell.set("conaffinity", "5")
    # New TCP follows the shifted pads, not the upstream stock fingertips.
    hand = root.find(".//body[@name='hand']")
    z = .0584+mount_reference(cfg)["position_m"][2]
    ET.SubElement(hand, "site", name="tactile_tcp", pos=f"0 0 {z}", size=".001",
                  rgba=".1 .9 .7 .4")
    add_recording_cameras(world, hand, z, profile=cfg.get('camera_profile'))
    # The split tendon applies half its force to each finger. Retain the
    # sourced position servo unless the legacy force mode is explicit.
    act = root.find(".//general[@name='fingers_actuator']")
    if cfg.get('gripper_control_mode') == 'panda_menagerie_position':
        source = ET.parse(ROOT/'assets/reference/panda_menagerie/panda.xml').getroot()
        servo = source.find(".//general[@tendon='split']")
        defaults = source.find(".//default[@class='panda']/general")
        for key in ('dyntype', 'biastype'):
            act.set(key, defaults.get(key))
        for key in ('ctrlrange', 'forcerange', 'biasprm'):
            act.set(key, servo.get(key))
        gain = servo.get('gainprm').split()
        gain[0] = str(-float(gain[0]))
        act.set('gainprm', ' '.join(gain))
    elif cfg.get('gripper_control_mode') == 'panda_robosuite_position':
        # Read pinned benchmark values rather than inventing servo gains.
        source = ET.parse(ROOT/'assets/reference/robosuite_panda/panda_gripper.xml').getroot()
        actuator = root.find('actuator')
        actuator.remove(act)
        for i in (1, 2):
            original = source.find(f"actuator/position[@joint='finger_joint{i}']")
            kp = float(original.get('kp'))
            ET.SubElement(actuator, 'general', name='fingers_actuator' if i == 1 else 'finger2_actuator',
                          joint=f'finger_joint{i}', dyntype='none', biastype='affine',
                          ctrllimited='true', ctrlrange='0 255', forcelimited='true',
                          forcerange=original.get('forcerange'),
                          gainprm=f'{-kp*.04/255} 0 0', biasprm=f'0 {-kp} 0')
            # Both local joints have positive opening and negative closure.
            joint = root.find(f".//joint[@name='finger_joint{i}']")
            source_joint = source.find(f".//joint[@name='finger_joint{i}']")
            for key in ('damping', 'armature', 'frictionloss'):
                joint.set(key, source_joint.get(key))
    else:
        act.set("gainprm", "1 0 0")
        act.set("biasprm", "0 0 0")
        actuator_limit = 2*cfg.get('gripper_force_max_n', 1.)
        act.set("ctrlrange", f"{-actuator_limit} {actuator_limit}")
        act.set("forcerange", f"{-actuator_limit} {actuator_limit}")
    # Original Panda 40 mm finger stroke, expressed in local q_source - .04.
    # Added pads stop through physical contact, not an artificial empty gap.
    root.find(".//default[@class='finger']/joint").set("range", "-.04 0")
    industrial_workcell(root, pick_place=cfg.get('task_success'))
    return ET.tostring(root, encoding="unicode")


class PandaPickCubeTactileEnv(FrankaTactileSim):
    """step([dx, dy, dz, drx, dry, drz, grip]) -> obs, reward, done, trunc, info.

    XYZ: world metres, clipped to +/-5 mm. Rotation: local rotvec radians,
    norm limited to .03 rad per command, composed on the right. Grip is an
    absolute closure target in [0, 1]: 0 opens, 1 closes (force limited).
    Sampling is 100 Hz sim time. Use hold_action() to retain the grip target.
    """
    task_name = 'pick_cube'

    def __init__(self, cfg=None, pollen=True, *, xml_factory=pick_scene,
                 contact_body="cube", contact_geom="cube", object_joint="cube_joint"):
        cfg = response_configuration(SimpleNamespace(task='pick_cube')) if cfg is None else dict(cfg)
        configure_controller(cfg)
        self.pick_place = None
        if self.task_name == 'pick_cube':
            cfg.setdefault('task_success', task_config('pick_cube'))
            self.pick_place = CubePickPlaceEvaluator(cfg['task_success'])
        gripper_cfg = json.loads((ROOT/'configs/gripper_control.json').read_text())
        for key, value in gripper_cfg.items():
            cfg.setdefault(key, value)
        if cfg['gripper_control_mode'] == 'legacy_force_limited':
            for key, value in gripper_cfg['legacy_force_control'].items():
                cfg.setdefault(key, value)
            limits = [cfg['gripper_force_min_n'], cfg['gripper_force_per_finger_n'], cfg['gripper_force_max_n']]
            if not np.isfinite(limits).all() or not 0 < limits[0] <= limits[1] <= limits[2]:
                raise ValueError('Gripper limits must be finite, positive and ordered min <= selected <= max')
        elif cfg['gripper_control_mode'] not in ('panda_menagerie_position', 'panda_robosuite_position', 'ebim_robotiq_position'):
            raise ValueError('Unknown gripper controller')
        # The reduced displacement field is inexpensive, but still needs a
        # resolved collision/integration mesh. Twelve outline samples create
        # long, thin skirt tetrahedra: a recorded cable motion gives V/V0=.148
        # on that mesh, versus .763/.761 with 48/96 samples under the SAME
        # trilinear deformation. Keep the material and volume check unchanged.
        cfg.setdefault("mesh_outline_samples", 48 if cfg.get('skin_dof') == 'trilinear' else 12)
        cfg.setdefault("scene_design", "industrial_workcell_v1")
        cfg.setdefault("camera_profile", "three_view")
        if cfg.get('gripper') == 'robotiq_2f85':
            from .robotiq import replace_gripper
            original_factory = xml_factory
            xml_factory = lambda c, l: replace_gripper(original_factory(c, l), c)
        TactileSim.__init__(self, cfg, xml_factory=xml_factory, mount_body="wowskin_mount",
                           contact_body=contact_body, contact_geom=contact_geom)
        self.cfg.update(cfg)
        self.initialize_mounts_and_arm()
        self.tcp_site = self.model.site("tactile_tcp").id
        self.hand = self.model.body("hand").id
        if cfg.get('gripper') == 'robotiq_2f85':
            from .robotiq import DRIVER
            self.finger_joints = [self.model.joint(DRIVER).id]
        else:
            self.finger_joints = [self.model.joint(f"finger_joint{i}").id for i in (1, 2)]
        self.finger_q = self.model.jnt_qposadr[self.finger_joints]
        self.finger_v = self.model.jnt_dofadr[self.finger_joints]
        self.gripper_control = self.model.actuator("fingers_actuator").id
        self.gripper_controls = ([self.gripper_control, self.model.actuator('finger2_actuator').id]
                                 if cfg['gripper_control_mode'] == 'panda_robosuite_position'
                                 else [self.gripper_control])
        if cfg['gripper_control_mode'] == 'panda_robosuite_position':
            self.cfg['transmission'] = 'Independent joint actuators, one per finger; no split-tendon force factor'
            self.cfg['gripper_source'] = json.loads((ROOT/'assets/reference/robosuite_panda/manifest.json').read_text())
            self.cfg['gripper_force_status'] = 'Pinned robosuite Panda per-finger position actuator and joint dynamics; native 40 mm stroke in translated coordinates. 20 N actuator cap per finger, not calibrated contact force.'
        self.cube_q = self.model.jnt_qposadr[self.model.joint(object_joint).id] if object_joint else None
        self.ik_data = mujoco.MjData(self.model)
        self.predictor = None
        if pollen:
            from wowskin_lab.baselines.pollen import PollenPredictor
            self.predictor = PollenPredictor()
        self.reskin_response = None
        self.reset()
        if cfg.get('reskin_model'):
            from wowskin_lab.baselines.reskin_contact import ReskinContactResponse
            self.reskin_response = ReskinContactResponse(cfg['reskin_model'], cfg.get('reskin_extrapolate', False))
            profile = json.loads((ROOT/'configs/magnetic_fit_pollen_nonslip.json').read_text())
            if not np.isclose(profile['fit_magnetization_a_per_m'], cfg['magnetization_a_per_m']):
                raise ValueError('Comparison fit uses a different magnetization reference')
            self.comparison_gains = np.asarray(profile['gains'])
            self.cfg['comparison_magnetic_fit_profile'] = profile

    @joint_pd_reset
    def reset(self, seed=None):
        self.episode_id = getattr(self, "episode_id", -1)+1
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_q] = HOME
        if self.cube_q is not None and self.cfg.get('randomize_cube', False):
            self.data.qpos[self.cube_q:self.cube_q+2] += self.rng.uniform(-.025, .025, 2)
        mujoco.mj_forward(self.model, self.data)
        self.joint_target = HOME.copy()
        self.target_xyz = self.data.site_xpos[self.tcp_site].copy()
        self.servo_xyz = self.target_xyz.copy()
        self.target_rotation = self.data.xmat[self.hand].reshape(3, 3).copy()
        self.servo_rotation = self.target_rotation.copy()
        self.grip_target = 0.
        self.grip_force = self.cfg.get('gripper_force_per_finger_n', float('nan'))
        self.tug_until = 0.
        self.mouse_wrench = np.zeros(6)
        self.success = False
        self.phase = "teleop"
        for _ in range(round(.25/self.cfg["timestep_s"])):
            self.control()
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.initial_cube_z = float(self.data.xpos[self.probe_body, 2])
        if self.pick_place is not None:
            self.pick_place.reset(cube_corners(self), TABLE_Z)
        self.baseline = self.magnetic_field().copy()
        self.filtered[:] = 0.
        self.delay.clear()
        self.delay.extend(np.zeros((5, 3)) for _ in range(self.cfg["delay_samples"]))
        self.drift_rate = self.rng.normal(0, self.cfg["drift_ut_per_s"], (5, 3))
        self.zero_time = self.data.time = 0.
        self.last_row = None
        return self.observation(), {"uncalibrated": True}

    def observation(self):
        return dict(tcp_position_m=self.data.site_xpos[self.tcp_site].copy(),
                    tcp_rotvec_rad=rotation_vector(self.data.site_xmat[self.tcp_site]),
                    gripper_closed_fraction=self.gripper_fraction(),
                    cube_position_m=self.data.xpos[self.probe_body].copy(),
                    magnetic_ut=(self.last_row["magnetic_ut"].copy() if self.last_row else np.full(15, np.nan) if getattr(self, "reskin_response", None) else np.zeros(15)),
                    joint_positions=self.data.qpos[self.arm_q].copy())

    def gripper_closed_target(self):
        if self.cfg.get('gripper') == 'robotiq_2f85':
            return self.cfg['robotiq_closed_angle_rad']
        endpoint = 0
        return float(self.model.jnt_range[self.finger_joints[0], endpoint])

    def command_gripper(self, close):
        self.grip_target = float(self.gripper_closed_target()) if close else 0.

    def gripper_fraction(self, target=False):
        lower = self.gripper_closed_target()
        value = self.grip_target if target else self.data.qpos[self.finger_q].mean()
        return float(np.clip(value/lower, 0., 1.))

    def hold_action(self):
        return np.r_[np.zeros(6), self.gripper_fraction(target=True)]

    def tug(self):
        self.tug_until = self.data.time+.6

    def update_mouse_perturbation(self, perturb):
        """Read a snapshot of the viewer's selected-cube drag, once per sample."""
        self.mouse_wrench[:] = 0.
        self.data.xfrc_applied[:] = 0.
        if perturb.active and perturb.select == self.probe_body:
            mujoco.mjv_applyPerturbForce(self.model, self.data, perturb)
            self.mouse_wrench[:] = self.data.xfrc_applied[self.probe_body]
            for sl, limit in ((slice(0, 3), 1.), (slice(3, 6), .01)):
                norm = np.linalg.norm(self.mouse_wrench[sl])
                self.mouse_wrench[sl] *= min(1., limit/max(norm, 1e-12))

    def solve_ik(self):
        # Optional local Cartesian slew limits; IK and physical actuator
        # bounds remain active in both modes. No joint teleportation.
        delta = self.target_xyz-self.servo_xyz
        length = np.linalg.norm(delta)
        linear_step = self.cfg.get('tcp_speed_m_s', .08)/self.cfg['sample_hz']
        angular_step = self.cfg.get('tcp_angular_speed_rad_s', .8)/self.cfg['sample_hz']
        unrestricted = self.cfg.get('disable_cartesian_limits', False)
        self.servo_xyz += delta if unrestricted else delta*min(1., linear_step/max(length, 1e-12))
        delta_rotation = rotation_vector(self.servo_rotation.T @ self.target_rotation)
        self.servo_rotation = self.servo_rotation @ rotation_matrix(
            delta_rotation if unrestricted else limit_vector(delta_rotation, angular_step))
        if cartesian_active(self):
            return  # Same target slew; task-space torque needs no runtime IK.
        m, d = self.model, self.ik_data
        d.qpos[:] = self.data.qpos
        d.qpos[self.arm_q] = self.joint_target
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(8):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            rotation = d.xmat[self.hand].reshape(3, 3)
            rot_error = rotation_vector(self.servo_rotation @ rotation.T)
            error = np.r_[self.servo_xyz-d.site_xpos[self.tcp_site], .2*rot_error]
            if np.linalg.norm(error) < 1e-5:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.tcp_site)
            jac = np.vstack([jacp[:, self.arm_v], .2*jacr[:, self.arm_v]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T+.0001*np.eye(6), error)
            d.qpos[self.arm_q] += np.clip(dq, -.04, .04)
            bounds = m.jnt_range[self.arm_joints]
            d.qpos[self.arm_q] = np.clip(d.qpos[self.arm_q], bounds[:, 0]+.02, bounds[:, 1]-.02)
        self.joint_target = d.qpos[self.arm_q].copy()

    def control(self, mode=None):
        progress = getattr(self, '_viewer_progress', None)
        if progress is not None:
            progress()
        d, m = self.data, self.model
        torque = arm_torque(self)
        limits = m.actuator_ctrlrange[self.arm_controls]
        d.ctrl[self.arm_controls] = np.clip(torque, limits[:, 0], limits[:, 1])
        if self.cfg.get('gripper') == 'robotiq_2f85':
            d.ctrl[self.gripper_control] = self.grip_target
        elif self.cfg.get('gripper_control_mode') in ('panda_menagerie_position', 'panda_robosuite_position', 'ebim_robotiq_position'):
            # Both sourced controllers use opening targets in [0,.04].
            # Local joints use q=q_src-.04; command is the closure fraction.
            closure = (self.grip_target / -.04 if self.cfg['gripper_control_mode'] == 'panda_robosuite_position'
                       else self.gripper_fraction(target=True))
            d.ctrl[self.gripper_controls] = 255*closure
        else:
            tendon_force = 100*(self.grip_target-d.qpos[self.finger_q].mean())-2*d.qvel[self.finger_v].mean()
            d.ctrl[self.gripper_control] = np.clip(tendon_force, -2*self.grip_force, 2*self.grip_force)
        d.xfrc_applied[:] = 0.
        d.xfrc_applied[self.probe_body] = self.mouse_wrench
        if d.time < self.tug_until:
            d.xfrc_applied[self.probe_body, 2] -= .5

    def apply_action_target(self, action):
        action = CartesianCommand.from_array(action).as_array()
        unrestricted = self.cfg.get('disable_cartesian_limits', False)
        applied = (action.copy() if unrestricted else
                   np.r_[np.clip(action[:3], -.005, .005), limit_vector(action[3:6], .03), action[6]])
        applied[6] = np.clip(action[6], 0., 1.)
        target = self.target_xyz+applied[:3]
        if not unrestricted:
            target = np.clip(target, *workspace_bounds(self, WORKSPACE))
        applied[:3] = target-self.target_xyz
        self.target_xyz = target
        self.target_rotation = self.target_rotation @ rotation_matrix(applied[3:6])
        self.grip_target = float(applied[6]*self.gripper_closed_target())
        return applied

    def step(self, action):
        before_pose = np.r_[self.data.site_xpos[self.tcp_site],
                            rotation_vector(self.data.site_xmat[self.tcp_site])]
        action_time = float(self.data.time)
        applied = self.apply_action_target(action)
        self.solve_ik()
        row = self.reading()
        if self.reskin_response:
            row['dipole_magnetic_ut'] = row['magnetic_ut'].copy()
            row['dipole_ideal_delta_ut'] = row['ideal_delta_ut'].copy()
            row['dipole_raw_field_ut'] = row['raw_field_ut'].copy()
            # Side-by-side on this exact mechanical state. Keep native U order;
            # analysis reorders to the network's candidate position grouping.
            row['fitted_dipole_ideal_delta_ut'] = (row['ideal_delta_ut'].reshape(5, 3)
                                                  * self.comparison_gains[:, None]).reshape(15)
            row.update(self.reskin_response.evaluate(self))
            row['tactile_valid'] = bool(np.isfinite(row['magnetic_ut']).all())
            if not row['tactile_valid']:
                row['tactile_status'] = row.get('reskin_status', 'network observation unavailable')
        row["action"] = applied
        row["action_time_s"] = action_time
        row["tcp_pose_before_m_rad"] = before_pose
        row["tcp_rotvec_rad"] = rotation_vector(self.data.site_xmat[self.tcp_site])
        row["gripper_closed_fraction"] = self.gripper_fraction()
        row['gripper_force_limit_per_finger_n'] = (20. if self.cfg.get('gripper_control_mode') == 'panda_robosuite_position' else float('nan') if self.cfg.get('gripper_control_mode') in ('panda_menagerie_position', 'panda_robosuite_position', 'ebim_robotiq_position') else self.grip_force)
        row['finger_actuator_force_n'] = (np.full(len(self.finger_v), np.nan) if self.cfg.get('gripper') == 'robotiq_2f85' else self.data.qfrc_actuator[self.finger_v].copy())
        if self.cfg.get('gripper') == 'robotiq_2f85':
            row['gripper_hinge_torque_nm'] = self.data.qfrc_actuator[self.finger_v].copy()
        row["cube_xyz_m"] = self.data.xpos[self.probe_body].copy()
        row["tcp_xyz_m"] = self.data.site_xpos[self.tcp_site].copy()
        row["pollen_probability"] = self.predictor.update(row["magnetic_ut"]) if self.predictor and np.isfinite(row['magnetic_ut']).all() else float("nan")
        row["tug_force_n"] = .5 if self.data.time < self.tug_until else 0.
        row["lift_m"] = max(0., float(row["cube_xyz_m"][2])-self.initial_cube_z)
        row["gripper_gap_m"] = float(np.linalg.norm(self.to_world(np.array([0., 0., self.cfg["total_height_m"]]))
                                - self.to_world(np.array([0., 0., self.cfg["total_height_m"]]), self.passive_mount)))
        row["grasped"] = bool(row["normal_force_n"] > .01 and row["passive_normal_force_n"] > .01)
        row["phase"] = self.phase
        row["episode_id"] = self.episode_id
        row["success"] = bool(row["grasped"] and row["lift_m"] > .06)
        if self.pick_place is not None:
            row.update(self.evaluate_pick_place(row))
            row['success'] = self.success = self.pick_place.success
        else:
            self.success |= row['success']
        self.last_row = row
        reward = .3*np.exp(-20*np.linalg.norm(row["cube_xyz_m"]-row["tcp_xyz_m"]))+.7*min(row["lift_m"]/.06, 1.)
        # Continue teleoperation after success so the user can lower/open/tug.
        return self.observation(), float(reward), False, False, row


    def evaluate_pick_place(self, row):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                self.probe_body, velocity, 0)
        return self.pick_place.update(float(self.data.time), corners=cube_corners(self),
            grasped=row.get('grasped', False),
            gripper_contact_count=gripper_contacts(self, {self.probe_geom}),
            speed=np.linalg.norm(velocity[3:]), angular_speed=np.linalg.norm(velocity[:3]))

    def evaluation_metrics(self, row=None):
        return dict(self.pick_place.metrics)

    def task_evaluation(self):
        return self.pick_place.summary()


def task_camera(env, close=False, operator=False):
    if hasattr(env, 'viewer_camera'):
        return env.viewer_camera(close=close, operator=operator)
    cam = mujoco.MjvCamera()
    if close:
        cam.lookat[:] = env.data.site_xpos[env.tcp_site]
        cam.distance, cam.azimuth, cam.elevation = .24, 225, -12
    else:
        cam.lookat[:] = [.28, 0., .18]
        cam.distance, cam.azimuth, cam.elevation = 1.6, 135, -25
    if operator:
        # Preserve visibility of the cube; Pico maps motion into this view's
        # ground-plane axes, rather than mirroring images or reversing world axes.
        cam.azimuth = 135
    return cam


def key_command(env, code):
    if code == ord("X"):
        env.command_gripper(True)
    elif code == ord("Z"):
        env.command_gripper(False)
    elif code == ord("C"):
        actual = float(env.data.qpos[env.finger_q].mean())
        env.grip_target = (actual/-.04*env.model.jnt_range[env.finger_joints[0], 0]
                           if env.cfg.get('gripper_control_mode') == 'panda_menagerie_position' else actual)
    elif code in (ord("-"), ord("=")):
        if env.cfg.get('gripper_control_mode') in ('panda_menagerie_position', 'panda_robosuite_position', 'ebim_robotiq_position'):
            return  # Source position actuator; no invented per-finger limit.
        env.grip_force = float(np.clip(env.grip_force+(.1 if code == ord("=") else -.1),
                                      env.cfg['gripper_force_min_n'], env.cfg['gripper_force_max_n']))
    elif code == ord("T"):
        env.tug()
    elif code in (ord("R"), 259):
        env.reset()


def hold_current_pose(env):
    """Cancel an outstanding Cartesian target when manual input ends/starts."""
    env.target_xyz = env.data.site_xpos[env.tcp_site].copy()
    if not env.cfg.get('disable_cartesian_limits', False):
        env.target_xyz = np.clip(env.target_xyz, *workspace_bounds(env, WORKSPACE))
    env.servo_xyz = env.target_xyz.copy()
    env.target_rotation = env.data.site_xmat[env.tcp_site].reshape(3, 3).copy()
    env.servo_rotation = env.target_rotation.copy()
    env.joint_target = env.data.qpos[env.arm_q].copy()


def view_translation(vector, azimuth):
    """Map screen right / ground-plane forward / up into world XYZ."""
    angle = np.deg2rad(azimuth)
    right = np.array([np.sin(angle), -np.cos(angle), 0.])
    forward = np.array([np.cos(angle), np.sin(angle), 0.])
    return vector[0]*right + vector[1]*forward + np.array([0., 0., vector[2]])


def manual_action(env, held, was_moving, camera_azimuth=None):
    translation = keyboard_translation(env, direction(held), camera_azimuth)
    velocity = translation*env.cfg.get('keyboard_speed_m_s', .04)
    angular_velocity = direction(held, ROTATE_AXES)*.4
    moving = bool(np.any(velocity) or np.any(angular_velocity))
    limited = not env.cfg.get('disable_cartesian_limits', False)
    if limited and moving != was_moving:
        hold_current_pose(env)
    if limited and moving:
        actual = env.data.site_xpos[env.tcp_site]
        requested_step = velocity/env.cfg['sample_hz']
        bounds = workspace_bounds(env, WORKSPACE)
        workspace_step = straight_step(env.target_xyz, requested_step, actual, bounds, lead_m=np.inf)
        env.keyboard_workspace_limited = not np.allclose(workspace_step, requested_step, atol=1e-12)
        velocity = straight_step(env.target_xyz, requested_step, actual, bounds)*env.cfg['sample_hz']
        actual_rotation = env.data.site_xmat[env.tcp_site].reshape(3, 3)
        lead = rotation_vector(actual_rotation.T @ env.target_rotation)
        env.target_rotation = actual_rotation @ rotation_matrix(limit_vector(lead, .03))
    else:
        env.keyboard_workspace_limited = False
    return np.r_[velocity/env.cfg['sample_hz'], angular_velocity/env.cfg['sample_hz'],
                 env.gripper_fraction(target=True)], moving


def keyboard_translation(env, vector, camera_azimuth=None, *, rotation=False):
    if env.cfg.get('teleop_frame', env.cfg.get('keyboard_frame', 'view')) == 'base':
        from wowskin_lab.data.robot_state import world_from_base
        # Fixed front-facing translation convention: forward +X, right +Y.
        # Angular input retains the proper right-handed basis; reflecting a
        # translation axis must not silently reverse the calibrated cap tilt.
        # Missing sign preserves the preceding base-frame acquisition profile.
        lateral = -1. if rotation else env.cfg.get('teleop_base_lateral_sign', -1.)
        return world_from_base(env)[:3, :3] @ np.array([vector[1], lateral*vector[0], vector[2]])
    return view_translation(vector, camera_azimuth) if camera_azimuth is not None else np.asarray(vector)


def save_records(rows, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/"episode.npz", **{key: np.array([r[key] for r in rows]) for key in rows[0]})
    with (output/"inference.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        lift_header = ['lift_m'] if 'lift_m' in rows[0] else []
        writer.writerow(["episode_id", "sim_time_s", "phase", *CHANNELS, "left_slip", "left_force_n", "right_force_n", "pollen_probability", *lift_header, "tug_force_n"])
        writer.writerows([r["episode_id"], r["time_s"], r["phase"], *r["magnetic_ut"], r["slip"], r["normal_force_n"],
                         r["passive_normal_force_n"], r["pollen_probability"], *[r[k] for k in lift_header], r["tug_force_n"]] for r in rows)


def fault_overlay(fault, row, magnetic_response):
    """Keep the failure and recovery keys inside the native text buffer."""
    reason = fault.split('. Physics stopped', 1)[0]
    if len(reason) > 400:
        reason = reason[:360] + '... (full error in terminal)'
    reason_lines = textwrap.wrap(reason, width=64) or ['Unknown error']
    labels = ['Status', 'Reason', *['']*(len(reason_lines)-1),
              'Recovery', 'Displayed data', 'Magnetic response', 'Last valid step']
    values = ['PAUSED: local physics check', *reason_lines,
              'R reset | P save diagnostics and exit',
              f"Last valid sample at {row['time_s']:.2f} s; not live data",
              magnetic_response,
              f"{row.get('physics_and_sensor_wall_s', 0.)*1000:.1f} ms / 10 ms sim"]
    return labels, values


def scripted_action(env):
    t = env.data.time
    if t < 1.:
        target, phase = [.40, 0., .185], "align"
    elif t < 2.:
        target, phase = [.40, 0., .129], "approach"
    elif t < 3.:
        target, phase = [.40, 0., .129], "close"
        env.command_gripper(True)
    elif t < 4.5:
        target, phase = [.40, 0., .229], "lift"
    elif t < 5.5:
        target, phase = [.40, 0., .229], "hold"
    elif t < 6.1:
        target, phase = [.40, 0., .229], "tug"
        env.tug_until = 6.1
    elif t < 7.:
        target, phase = [.40, 0., .229], "after tug"
    else:
        target, phase = [.40, 0., .229], "release"
        env.command_gripper(False)
    env.target_xyz[:] = target
    env.phase = phase
    return env.hold_action()


def response_configuration(args):
    cfg = configuration(seed=getattr(args, 'seed', 0))
    cfg['disable_cartesian_limits'] = bool(getattr(args, 'no_cartesian_limits', False))
    cfg['disable_workspace_limits'] = not bool(getattr(args, 'workspace_limits', False))
    if getattr(args, 'no_workspace_limits', False):
        cfg['disable_workspace_limits'] = True
    cfg['teleop_frame'] = getattr(args, 'control_frame', 'base')
    cfg['teleop_base_lateral_sign'] = 1.
    cfg['translation_target_limiter'] = 'direction_preserving_v1'
    if getattr(args, 'arm_controller', 'ik_pd') != 'ik_pd':
        cfg['arm_controller'] = args.arm_controller
    cfg['gripper'] = getattr(args, 'gripper', 'panda')
    if cfg['gripper'] == 'robotiq_2f85' and getattr(args, 'grip_force', None) is not None:
        raise ValueError('Robotiq uses the sourced hinge servo; --grip-force is a Panda legacy option')
    cfg['skin_dof'] = getattr(args, 'skin_dof', 'trilinear')
    if cfg['skin_dof'] == 'trilinear':
        cfg.update(skin_model_status='Experimental trilinear interpolation; bounding-box bottom attachment; no internal contact',
                   skin_model_source='https://github.com/google-deepmind/mujoco/blob/3.5.0/model/flex/gripper_trilinear.xml',
                   integrator='discrete', noslip_iterations=0, fem_node_joint_damping=0.,
                   engine_version=mujoco.__version__)
    if getattr(args, 'task', 'pick_cube') == 'cable':
        cfg['cable_layout'] = getattr(args, 'cable_layout', None) or 'routing'
        if getattr(args, 'clip_model', 'rigid') != 'rigid':
            cfg['clip_model'] = args.clip_model
    physics_step_us = getattr(args, 'physics_step_us', None)
    task = getattr(args, 'task', 'pick_cube')
    if task == 'feeding':
        cfg['feeding_setting'] = getattr(args, 'feeding_setting', 'care_room')
    if physics_step_us is None and task in ('pick_cube', 'cable', 'feeding', 'peg_insert'):
        # Feeding's recorded 6D spoon motion becomes unstable at 100 us and
        # passes at 50 us with this thin-silicone/contact configuration.
        # This costs throughput; the sensor/control clock remains 100 Hz.
        physics_step_us = 500 if cfg['skin_dof'] == 'trilinear' else (50 if task == 'feeding' else 100)
    if physics_step_us is not None and physics_step_us > 100 and cfg['skin_dof'] != 'trilinear':
        raise ValueError('Steps above 100 us are tested only with the reduced skin')
    if task in ('pick_cube', 'peg_insert'):
        # Same full FEM/contact model; sparse CG avoids dense Jacobian work.
        # Keep the selected integration step in acquisition metadata.
        cfg['constraint_jacobian'] = 'sparse'
    if physics_step_us is not None:
        cfg['timestep_s'] = physics_step_us*1e-6
        cfg['timestep_source'] = 'Teleoperation numerical integration setting; 100 Hz sensor clock retained'
    if getattr(args, 'ideal', False):
        cfg.update(noise_ut=0., drift_ut_per_s=0., filter_tau_s=0., delay_samples=0)
    if getattr(args, 'grip_force', None) is not None:
        cfg['gripper_control_mode'] = 'legacy_force_limited'
        cfg['gripper_force_per_finger_n'] = args.grip_force
    if getattr(args, 'reskin_model', None):
        cfg['reskin_model'] = str(args.reskin_model.resolve())
        cfg['reskin_extrapolate'] = args.reskin_extrapolate
        cfg['magnetic_model_status'] = 'Experimental ReSkin single-probe surrogate; mapping/axes/cube transfer unvalidated'
        print(cfg['magnetic_model_status'], flush=True)
    if getattr(args, 'magnetic_fit', None):
        if cfg.get('reskin_model'):
            raise ValueError('Choose --magnetic-fit or --reskin-model, not both')
        cfg['magnetic_fit'] = str(args.magnetic_fit.resolve())
        cfg['magnetic_model_status'] = 'Dipole with Pollen/network simulation-amplitude fit; no real sensor calibration'
        print(cfg['magnetic_model_status'], flush=True)
    return cfg


def demo(args):
    cfg = response_configuration(args)
    env = PandaPickCubeTactileEnv(cfg, pollen=not args.no_pollen)
    from wowskin_lab.ui.tactile import TactileHUD
    hud = TactileHUD(env)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(env.model, height=600, width=800) if args.render else None
    rows, frames = [], []
    try:
        for i in range(850):
            _, _, _, _, row = env.step(scripted_action(env))
            rows.append(row)
            if i % 100 == 0:
                print(f"{env.phase}: t={row['time_s']:.2f} lift={row['lift_m']:.3f} "
                      f"F={row['normal_force_n']:.3f}/{row['passive_normal_force_n']:.3f} p={row['pollen_probability']:.3f}", flush=True)
            if renderer and i % 10 == 0:
                import imageio.v3 as iio
                from PIL import Image, ImageDraw
                renderer.update_scene(env.data, camera=task_camera(env, True))
                picture = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(picture)
                draw.rectangle((0, 0, 800, 52), fill=(20, 25, 32))
                draw.text((12, 7), "PandaPickCube | actual contact grasp | UNCALIBRATED real-to-sim", fill="white")
                draw.text((12, 29), f"{env.phase} | lift={row['lift_m']*1000:.1f} mm | slip={row['slip']} | Pollen={row['pollen_probability']:.3f}", fill="white")
                picture = hud.composite(np.asarray(picture), row)
                frames.append(picture)
                if i in (0, 450, 570, 800):
                    iio.imwrite(out/f"{i:03d}_{env.phase.replace(' ', '_')}.png", picture)
    finally:
        if renderer:
            renderer.close()
    save_records(rows, out)
    from wowskin_lab.envs.franka import plot_response
    values = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    plot_response(values, out/"real2sim.png")
    labels = values["slip"].astype(bool)
    scores = values["pollen_probability"]
    valid = np.isfinite(scores)
    predictions = scores >= .5
    report = dict(status="Uncalibrated zero-shot real-to-sim test, not force calibration",
                  model="unchanged published Pollen weights and scaler" if env.predictor else None,
                  cube_mass_kg=CUBE_MASS, cube_edge_m=2*CUBE_HALF,
                  force_limit_per_finger_n=float(env.grip_force) if np.isfinite(env.grip_force) else None,
                  mount=mount_reference(env.cfg), samples=len(rows), picked_up=bool(np.any(values["grasped"] & (values["lift_m"] > .06))),
                  task_evaluation=env.task_evaluation(),
                  config=env.cfg,
                  max_lift_m=float(values["lift_m"].max()),
                  slip_frames=int(labels.sum()), non_slip_frames=int((~labels).sum()),
                  min_skin_volume_ratio=float(values["min_volume_ratio"].min()),
                  min_silicone_volume_ratio=float(values["passive_min_volume_ratio"].min()),
                  warnings=env.data.warning.number.tolist())
    if env.reskin_response:
        contact_rows = [r for r in rows if r['reskin_contact_count']]
        report.update(magnetic_model=env.reskin_response.model.meta,
                      magnetic_model_status='Experimental cube proxy, not a validated WowSkin response',
                      response_contact_frames=len(contact_rows),
                      response_input_in_training_domain_frames=sum(r['reskin_in_training_domain'] for r in contact_rows),
                      response_transfer_validated=False)
    if valid.any():
        report.update(pollen_score_range=[float(scores[valid].min()), float(scores[valid].max())],
                      confusion=dict(TP=int((labels & predictions & valid).sum()),
                                     FP=int((~labels & predictions & valid).sum()),
                                     FN=int((labels & ~predictions & valid).sum()),
                                     TN=int((~labels & ~predictions & valid).sum())))
    (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    (out/"scene.xml").write_text(env.xml)
    if frames:
        import imageio.v3 as iio
        iio.imwrite(out/"pick_cube.gif", np.stack(frames), duration=100, loop=0)
    print(json.dumps(report, indent=2), flush=True)


def viewer(args):
    import copy
    import mujoco.viewer
    from wowskin_lab.ui.physics_worker import PhysicsWorker
    from wowskin_lab.ui.camera_preview import CameraPreview
    from wowskin_lab.ui.teleop_hud import operator_overlay, render_operator_panel, saved_recording_notice
    from wowskin_lab.ui.task_guides import TaskGuides
    cfg = response_configuration(args)
    cfg['randomize_cube'] = args.randomize
    cfg['teleop_frame'] = args.control_frame
    cfg['teleop_base_lateral_sign'] = 1.
    cfg['translation_target_limiter'] = 'direction_preserving_v1'
    # New sessions have no hand-written tabletop target box. Older recordings
    # retain their saved configuration when reconstructed for evaluation.
    cfg['disable_workspace_limits'] = not args.workspace_limits
    cable_task = getattr(args, 'task', 'pick_cube') == 'cable'
    feeding_task = getattr(args, 'task', 'pick_cube') == 'feeding'
    peg_task = getattr(args, 'task', 'pick_cube') == 'peg_insert'
    other_task = cable_task or feeding_task or peg_task
    keyboard_speed = getattr(args, 'keyboard_speed', None)
    keyboard_speed = (.12 if other_task else .04) if keyboard_speed is None else keyboard_speed
    if not np.isfinite(keyboard_speed) or not (0 < keyboard_speed <= .25):
        raise ValueError('Keyboard speed must be in (0, .25] m/s')
    cfg['keyboard_speed_m_s'] = keyboard_speed
    if args.input == 'keyboard':
        cfg['tcp_speed_m_s'] = max(cfg.get('tcp_speed_m_s', .08), keyboard_speed)
        cfg['keyboard_target_limiter'] = 'direction_preserving_v1; 6 mm per-axis lead, stop along requested direction at workspace boundary'
    if args.input == 'pico':
        if not np.isfinite([args.pico_speed, args.pico_angular_speed]).all() or not (0 < args.pico_speed <= .5 and 0 < args.pico_angular_speed <= 3.):
            raise ValueError('Pico speed must be in (0, .5] m/s and angular speed in (0, 3] rad/s')
        cfg['tcp_speed_m_s'] = args.pico_speed
        cfg['tcp_angular_speed_rad_s'] = args.pico_angular_speed
    mouse, mouse_controller = None, None
    if args.input == 'spacemouse':
        import sys
        if sys.platform != 'darwin':
            raise RuntimeError('SpaceMouse viewer focus handling currently validated only on macOS')
        from wowskin_lab.teleop.spacemouse import SpaceMouse, SpaceMouseController
        mouse = SpaceMouse()
        mouse_controller = SpaceMouseController(args.spacemouse_speed, args.spacemouse_angular_speed,
                                               args.spacemouse_deadzone)
        cfg['tcp_speed_m_s'] = args.spacemouse_speed
        cfg['tcp_angular_speed_rad_s'] = args.spacemouse_angular_speed
        cfg['spacemouse'] = mouse_controller.settings(cfg['teleop_frame'])
    if cable_task:
        from wowskin_lab.envs.cable import PandaCableEnv
        env = PandaCableEnv(cfg, pollen=not args.no_pollen)
    elif feeding_task:
        from wowskin_lab.envs.feeding import PandaFeedingEnv
        env = PandaFeedingEnv(cfg, pollen=not args.no_pollen)
    elif peg_task:
        from wowskin_lab.envs.peg_insert import PandaPegInsertEnv
        env = PandaPegInsertEnv(cfg, pollen=not args.no_pollen)
    else:
        env = PandaPickCubeTactileEnv(cfg, pollen=not args.no_pollen)
    title = ('PandaCable / single clip' if cable_task and env.cable_layout == 'single_clip'
             else 'PandaCable' if cable_task else 'PandaFeeding' if feeding_task else 'PandaPegInsert' if peg_task else 'PandaPickCube')
    print('Arm controller: '+env.cfg.get('arm_controller', 'ik_pd'), flush=True)
    if cfg['disable_cartesian_limits']:
        print('Cartesian limits OFF: no workspace, action increment, servo slew or target-lead limits. '
              'Centered input retains the target; focus loss still cancels outstanding motion. '
              'IK, joint ranges and actuator limits are unchanged.', flush=True)
    source_gripper = env.cfg.get('gripper_control_mode') in ('panda_menagerie_position', 'panda_robosuite_position', 'ebim_robotiq_position')
    if env.cfg.get('gripper') == 'robotiq_2f85':
        print('Gripper: EBiM Robotiq 2F-85 hinge position servo; source torque limits retained', flush=True)
    elif source_gripper:
        print('Gripper: robosuite Panda independent finger servos, 20 N actuator cap each' if env.cfg['gripper_control_mode'] == 'panda_robosuite_position' else 'Gripper: MuJoCo Menagerie Panda position servo; 100 N tendon actuator cap, NOT fingertip contact force; no 0.35 N limiter', flush=True)
    else:
        print(f"Gripper: local, uncalibrated per-finger drive limit {env.grip_force:.2f} N; "
              "configs/gripper_control.json (--grip-force overrides the initial value)", flush=True)
    from wowskin_lab.ui.tactile import TactileHUD
    hud = TactileHUD(env)
    if cable_task and hud.reskin_mode:
        hud.subtitle = 'Experimental cable proxy [uT]'
        hud.footer = 'NOT calibrated | cable/probe mismatch'
    keyboard = ViewerKeyboard(set(map(ord, "WASDEQXZCTFRVBNPHMY09-=,.")) | MOTION_KEYS | {ESC, ENTER, BACKSPACE},
                              title_contains='PandaCable:' if cable_task else title+':')
    recorder = None
    from wowskin_lab.data.recording import MissingTactileError
    recording_notice = ''
    if args.record_dir and cfg.get('reskin_model') and not cfg.get('reskin_extrapolate'):
        print('Strict ReSkin mode: unsupported predictions are blank; synchronized training recording is disabled. Diagnostic rows are saved on exit.', flush=True)
    if args.record_dir and not (cfg.get('reskin_model') and not cfg.get('reskin_extrapolate')):
        if args.record_format == 'lerobot':
            from wowskin_lab.data.lerobot_recording import LeRobotSimRecorder
            recorder = LeRobotSimRecorder(env, args.record_dir, source=args.input,
                instruction=args.instruction, record_depth=args.record_depth)
        else:
            from wowskin_lab.data.learning_recording import LearningSimRecorder
            recorder = LearningSimRecorder(env, args.record_dir, source=args.input,
                instruction=args.instruction, record_depth=args.record_depth)
    pico = None
    if args.input == 'pico':
        from wowskin_lab.teleop.pico import PicoBridge
        pico = PicoBridge(args.pico_port, args.pico_scale, args.pico_speed, args.pico_angular_speed)
        if cfg['teleop_frame'] == 'base':
            from wowskin_lab.data.robot_state import world_from_base
            pico.controller.robot_rotation = world_from_base(env)[:3, :3].copy()
            pico.controller.base_lateral_sign = cfg['teleop_base_lateral_sign']
    rows = [] if args.diagnostic_log else deque(maxlen=200)
    sampled_count = 0
    close, fault, automatic = cable_task, "", False
    faults = []
    was_moving = False
    next_hud = 0.
    show_diagnostics = False
    realtime_factor, loop_hz = 0., 0.
    deadline = time.monotonic()+args.seconds if args.seconds else float("inf")
    grip_help = env.cfg['gripper_control_mode'] if source_gripper else '-/= legacy force limit'
    print(f"{title} 7D: input frame {cfg['teleop_frame']} | workspace box {'ON' if args.workspace_limits else 'OFF'} | HOLD WASD/arrows, E/Q Z | I/K local Rx, L/J Ry, U/O Rz | release to stop | X close Z open C hold | {grip_help} | R reset | V view | P exit", flush=True)
    if recorder:
        print('Recording: B new/reset or stop (unreviewed) | Enter save SUCCESS | N save FAILURE | R interrupts | P save interrupted and exit', flush=True)
    display_model, display_data = PhysicsWorker.display(env)
    from wowskin_lab.data.cameras import camera_profile
    from wowskin_lab.data.robot_state import world_from_base
    # Frame existing camera targets, without cropping points or changing physics.
    targets = [c['target_m'] for c in camera_profile(scene=args.task) if c['parent'] == 'world']
    center_world = np.mean(targets, axis=0)
    base_world = np.linalg.inv(world_from_base(env))
    cloud_options = dict(enabled=args.pointcloud, hz=args.pointcloud_hz,
        stride=args.pointcloud_stride, voxel_m=args.pointcloud_voxel,
        center=base_world[:3,:3] @ center_world+base_world[:3,3])
    with (mouse or nullcontext()), (pico or nullcontext()), (recorder or nullcontext()), mujoco.viewer.launch_passive(display_model, display_data, key_callback=keyboard.native_callback,
                                    show_left_ui=False, show_right_ui=False) as handle, keyboard, PhysicsWorker(
                                        env, handle, keyboard, display_model, display_data,
                                        on_abort=getattr(recorder, 'abort', None)) as physics, CameraPreview(display_model, display_data, hud, TaskGuides(env), pointcloud=cloud_options) as previews:
        recording_busy, busy_started = False, 0.
        def present(handle):
            if recording_busy:
                stage, detail = recorder.progress
                elapsed = time.monotonic()-busy_started
                quitting = ord('P') in keyboard.peek().commands or not handle.is_running()
                hud.operator_panel = render_operator_panel(
                    [stage, 'Elapsed', 'Controls'],
                    [detail, f'{elapsed:.1f} s',
                     'Exit requested; finishing current file write' if quitting else
                     'Please wait | B/Enter/motion resume after completion'])
            previews.show(handle)
        physics.on_present = present
        physics.on_status = lambda labels, values: setattr(hud, 'operator_panel', render_operator_panel(labels, values))
        def recording_call(stage, detail, function, *args):
            nonlocal recording_busy, busy_started, next_hud
            recording_busy, busy_started = True, time.monotonic()
            recorder.progress = (stage, detail)
            previews.next_frame = 0.
            physics.present()  # Publish acknowledgement before reset/encoding.
            try:
                return function(*args)
            finally:
                recording_busy = False
                next_hud = 0.
                keyboard.discard_busy_input()
                if mouse:
                    mouse.state.sample()  # Discard button presses made during saving.
                    mouse_controller.reset()
                if pico:
                    pico.controller.reset()
                    _, _, _, pending = pico.mailbox.sample()
                    if ord('P') in pending:
                        keyboard.event(ord('P'), True)
                        keyboard.event(ord('P'), False)
        if mouse:
            env.cfg['spacemouse']['device'] = mouse.info
            print(f"SpaceMouse: {mouse.info['product_string']} | exclusive HID={mouse.info.get('exclusive_hid', False)} | push/tilt in camera frame | left close, right open | B/Enter/N record | click MuJoCo window to control", flush=True)
        def apply_camera():
            cam = task_camera(env, close, operator=bool(pico))
            apply_free_camera(handle, cam)
        apply_camera()
        if pico:
            pico.open_usb()
        style = None if keyboard.continuous else ViewerStyleGuard(handle, "WASDEQXZCTFRVHIJKLUOMY09,.")
        print("Keyboard: window-local press/release; native shortcuts consumed" if keyboard.continuous
              else "Keyboard: non-macOS single-step fallback (no held-key support)", flush=True)
        print('Point cloud: M two/three/off | Y view | 0 RGB/source color | 9 zoom; preview only, no temporal accumulation', flush=True)
        while handle.is_running() and time.monotonic() < deadline:
            start = time.monotonic()
            sample_start = float(env.data.time)
            snapshot = keyboard.sample()
            commands = list(snapshot.commands)
            if pico:
                _, _, _, pico_commands = pico.mailbox.sample()
                commands.extend(pico_commands)
            if ord('P') in commands:
                break  # Finish the recorder before returning from the UI loop.
            held = snapshot.held
            fallback_motion = env.hold_action()
            for code in commands:
                if previews.key(code):
                    next_hud = 0.
                elif code == ord('H'):
                    show_diagnostics = not show_diagnostics
                    next_hud = 0.
                elif pico and code == ord('G'):
                    frame, received, _, _ = pico.mailbox.sample(drain=False)
                    if time.monotonic()-received <= .25:
                        hold_current_pose(env)
                        pico.controller.calibrate(frame)
                elif recorder and code in (ord('B'), ENTER, ord('N')):
                    if mouse_controller:
                        hold_current_pose(env)
                        mouse_controller.reset()
                    if pico:
                        hold_current_pose(env)
                        pico.controller.reset()
                    if recorder.active:
                        try:
                            outcome = 'success' if code == ENTER else ('failure' if code == ord('N') else 'unreviewed')
                            saved = recording_call('SAVING', 'Finishing recording; preparing file write', recorder.finish, outcome)
                            recording_notice = saved_recording_notice(recorder, saved, outcome)
                        except (OSError, ValueError, RuntimeError) as error:
                            recording_notice = 'Recording save failed; see terminal; capture files retained'
                            print(f'Recording save failed: {error}', flush=True)
                    elif code == ord('B'):
                        automatic, fault, was_moving = False, '', False
                        keyboard.clear()
                        held = frozenset()
                        try:
                            def start_recording():
                                physics.run(env.reset)
                                recorder.start()
                            recording_call('PREPARING', 'Resetting scene before new recording', start_recording)
                            recording_notice = f'STARTED {recorder.writer.path.name} | recording now'
                        except MissingTactileError as error:
                            if hasattr(recorder, 'abort'):
                                recorder.abort(str(error))
                            else:
                                recorder.finish('interrupted')
                            recording_notice = 'Unavailable tactile: recording not started'
                            print(f'Recording unavailable: {error}', flush=True)
                        except (OSError, ValueError, RuntimeError) as error:
                            recording_notice = 'Recording not started; see terminal'
                            print(f'Recording not started: {error}', flush=True)
                        if pico:
                            pico.controller.reset()
                    else:
                        recording_notice = 'No active recording | B starts recording'
                    # Do not replay stale held keys or a second queued Enter/B
                    # after an operation that took seconds to finish.
                    held = frozenset()
                    break
                elif code == 256:
                    close = False
                    apply_camera()
                elif code == ord("V"):
                    close = not close
                    apply_camera()
                elif code == ord("F") and not pico and not mouse:
                    if other_task:
                        print('F cube routine is disabled for this task; use manual controls.', flush=True)
                        continue
                    if recorder:
                        recorder.finish('interrupted')
                    automatic = not automatic
                    if automatic:
                        physics.run(env.reset)
                        fault = ""
                        keyboard.clear()
                        held, was_moving = frozenset(), False
                else:
                    if recorder and code in (ord('R'), BACKSPACE):
                        recorder.finish('interrupted')
                    hud.key(code)
                    physics.run(key_command, env, code)
                    if code in MOTION_KEYS or code in map(ord, "XZCR") or code == BACKSPACE:
                        if automatic:
                            hold_current_pose(env)
                        automatic = False
                        env.phase = "teleop"
                    if code in MOVE_AXES and not keyboard.continuous:
                        fallback_motion[:3] += np.array(MOVE_AXES[code])*.005
                    if code in ROTATE_AXES and not keyboard.continuous:
                        fallback_motion[3:6] += np.array(ROTATE_AXES[code])*.03
                    if code in (ord("R"), 259):
                        fault = ""
                        if mouse_controller:
                            mouse_controller.reset()
                        if pico:
                            pico.controller.reset()
                        close = False
                        apply_camera()
                        keyboard.clear()
                        held, was_moving = frozenset(), False
            if not fault:
                try:
                    with handle.lock():
                        perturb = copy.copy(handle.perturb)
                        camera_azimuth = handle.cam.azimuth
                    env.update_mouse_perturbation(perturb)
                    if automatic:
                        action = scripted_action(env)
                    elif pico:
                        frame, received, epoch, _ = pico.mailbox.sample(drain=False)
                        action = pico.controller.action(env, frame, received, epoch)
                        was_moving = pico.controller.engaged
                    elif mouse:
                        mouse_snapshot = mouse.state.sample()
                        action = mouse_controller.action(env, mouse_snapshot, camera_azimuth, keyboard.focused)
                        was_moving = mouse_controller.moving
                    elif keyboard.continuous:
                        action, was_moving = manual_action(env, held, was_moving, camera_azimuth)
                    else:
                        fallback_motion[:3] = keyboard_translation(env, fallback_motion[:3], camera_azimuth)
                        fallback_motion[6] = env.gripper_fraction(target=True)
                        action = fallback_motion
                    row = physics.run(env.step, action)[-1]
                    row['physics_and_sensor_wall_s'] = physics.last_operation_wall_s
                    row['total_contacts'] = int(env.data.ncon)
                    row['constraint_rows'] = int(env.data.nefc)
                    if mouse:
                        row['spacemouse_axes'] = mouse_snapshot.axes.copy()
                        row['spacemouse_buttons'] = mouse_snapshot.buttons
                    row['held_motion_keys'] = ' '.join(str(k) for k in sorted(held & MOTION_KEYS))
                    row['viewer_loop_wall_s'] = time.monotonic()-start
                    row['viewer_realtime_factor'] = max(0., float(env.data.time)-sample_start)/max(row['viewer_loop_wall_s'], 1e-9)
                    if automatic and env.data.time >= 8.5:
                        automatic = False
                    rows.append(row)
                    sampled_count += 1
                    # Keep invalid intervals in diagnostics (NaN measurements
                    # plus validity/time fields), never in training episodes.
                    if recorder:
                        recorder.step(row)
                except MissingTactileError as error:
                    # An unsupported observation is not a failed physics step.
                    # End this recording without FTF advancing a post-roll or
                    # filling missing tactile samples; keep teleoperation live.
                    if hasattr(recorder, 'abort'):
                        recorder.abort(str(error))
                    elif recorder:
                        recorder.finish('interrupted')
                    recording_notice = 'Unavailable tactile: recording interrupted'
                    print(f'Recording interrupted: {error}', flush=True)
                except RuntimeError as error:
                    if recorder:
                        # Invalid physics must not be stepped again for FTF's
                        # normal post-roll. Keep the failed episode excluded.
                        if hasattr(recorder, 'abort'):
                            recorder.abort(str(error))
                        else:
                            recorder.finish('interrupted')
                    if mouse_controller:
                        mouse_controller.reset()
                    fault = str(error)
                    faults.append(fault)
                    print(f"Paused: {fault}. R resets the simulation.", flush=True)
            if pico:
                pico.update_preview(env, task_camera(env, close, operator=True), bool(recorder and recorder.active))
            # Text and glyph image submission can wait for the render thread.
            # Refresh the panel at 10 Hz wall time; physics, input, viewer sync
            # and dataset samples continue independently on every control tick.
            if env.last_row and time.monotonic() >= next_hud:
                next_hud = time.monotonic()+.1
                row = env.last_row
                input_kind = 'SpaceMouse' if mouse else 'Pico' if pico else 'Keyboard'
                input_status = (mouse_controller.status if mouse else pico.controller.status if pico else
                                env.phase if automatic else 'Moving' if was_moving else 'Ready')
                labels, text = operator_overlay(env, row, title=title, input_kind=input_kind,
                    input_status=input_status, recorder=recorder, recording_notice=recording_notice,
                    diagnostics=show_diagnostics, realtime_factor=realtime_factor, loop_hz=loop_hz)
                if fault:
                    magnetic_response = ('Forward MLP (experimental)' if env.cfg.get('reskin_model') else
                        'Dipole + fitted gains' if env.cfg.get('magnetic_fit') else 'Dipole (unfitted)')
                    labels, text = fault_overlay(fault, row, magnetic_response)
                if not handle.is_running():
                    break
                hud.operator_panel = render_operator_panel(labels, text, success=bool(row.get('success')), fault=bool(fault))
                # MuJoCo mjr_drawPixels assumes 2-D GL state. An empty LIST
                # skips mjr_overlay/init2D, leaving the 3-D depth test active
                # and hiding image panels. One empty overlay initializes 2-D
                # state without drawing duplicate white text over our bitmap.
                handle.set_texts([(mujoco.mjtFont.mjFONT_NORMAL,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT, '', '')])
                previews.show(handle, row)
            before_sync = display_data.time
            if style:
                style.sync(handle)
            else:
                handle.sync()
            # MuJoCo's built-in Backspace resets to qpos0, bypassing our
            # controller and baseline. Restore the task reset before stepping.
            if display_data.time < before_sync-1e-9:
                physics.run(env.reset)
                automatic, fault = False, ""
                if mouse_controller:
                    mouse_controller.reset()
            time.sleep(max(0., .01-(time.monotonic()-start)))
            elapsed = time.monotonic()-start
            advance = max(0., float(env.data.time)-sample_start)
            realtime_factor = .8*realtime_factor+.2*advance/max(elapsed, 1e-9)
            loop_hz = .8*loop_hz+.2/max(elapsed, 1e-9)
            if rows and rows[-1] is env.last_row:
                rows[-1]['viewer_loop_wall_s'] = elapsed
                rows[-1]['viewer_realtime_factor'] = advance/max(elapsed, 1e-9)
    if rows and (args.diagnostic_log or faults):
        destination = Path(args.output)/time.strftime("teleop_%Y%m%d_%H%M%S")
        save_records(rows, destination)
        if faults:
            (destination/'faults.json').write_text(json.dumps({
                'faults': faults,
                'last_valid_sample_time_s': float(rows[-1]['time_s']),
            }, indent=2)+'\n')
        print(f"Saved {len(rows)} simulated samples to {destination}", flush=True)
    return dict(samples=sampled_count, faults=faults)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["viewer", "demo"], nargs="?", default="viewer")
    parser.add_argument("--no-pollen", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seconds", type=float, default=0.,
                        help='Optional wall-clock limit; 0 runs until P/window close. Exiting interrupts an active recording.')
    parser.add_argument('--input', choices=['keyboard', 'pico', 'spacemouse'], default='keyboard')
    parser.add_argument('--control-frame', choices=['base', 'view'], default='base',
                        help='All inputs: fixed robot base (default), or legacy view-relative convention')
    parser.add_argument('--arm-controller', choices=['ik_pd', 'cartesian_impedance'], default='ik_pd',
                        help='Default IK/joint PD, or pinned SERL++ task-space torque with upstream default gains; reset still uses joint PD')
    parser.add_argument('--pointcloud', action='store_true', help='Show live three-view RGB-D fusion in robot base coordinates; M toggles even without this flag')
    parser.add_argument('--pointcloud-hz', type=float, default=5., help='Maximum point-cloud preview capture Hz (wall time); does not change recording clocks')
    parser.add_argument('--pointcloud-stride', type=int, default=4, help='Preview pixel sampling stride, 1 for every depth pixel; recording remains full resolution')
    parser.add_argument('--pointcloud-voxel', type=float, default=.002, help='Preview voxel size [m], 0 to disable; recording is unchanged')
    parser.add_argument('--no-cartesian-limits', action='store_true',
                        help='Simulation experiment: disable workspace, per-action, servo-speed and target-lead limits; retain input scale, IK, joint and actuator bounds')
    parser.add_argument('--no-workspace-limits', action='store_true',
                        help='Disable only the local TCP target bounding box; keep target lead, speed, joint and actuator limits')
    parser.add_argument('--workspace-limits', action='store_true',
                        help='Opt in to the legacy, locally chosen tabletop TCP target box; OFF by default')
    parser.add_argument('--task', choices=['pick_cube', 'cable', 'feeding', 'peg_insert'], default='pick_cube')
    parser.add_argument('--cable-layout', choices=['routing', 'single_clip'],
                        help='Cable fixture layout: local two-post routing (default), or one reversed C-clip')
    parser.add_argument('--clip-model', choices=['rigid', 'elastic_beam'], default='rigid',
                        help='Rigid original proxy (default), or experimental uncalibrated passive elastic clip')
    parser.add_argument('--feeding-setting', choices=['care_room', 'ebim_station'], default='care_room',
                        help='Feeding setting: seated adult in a local care room (default), or previous isolated EBiM station')
    parser.add_argument('--grip-force', type=float, help='Explicitly enable the OLD local force-limited controller [N/finger]; default uses sourced Menagerie Panda position servo')
    parser.add_argument('--ideal', action='store_true', help='Disable synthetic magnetic noise, drift, readout filtering and sample delay; retain contact mechanics and deformation response')
    parser.add_argument('--physics-step-us', type=int, choices=[20, 50, 100, 500, 1000], help='Integration step in microseconds; 500/1000 require reduced skin; recorded in metadata')
    parser.add_argument('--gripper', choices=['panda', 'robotiq_2f85'], default='panda', help='Panda or EBiM Robotiq 2F-85 with local tactile mounts')
    parser.add_argument('--skin-dof', choices=['full', 'trilinear'], default='trilinear', help='Reduced skin for all tasks (MuJoCo 3.15); full keeps the reference FEM')
    parser.add_argument('--keyboard-speed', type=float, help='Translation speed in m/s of simulation time; cable .12, cube .04; maximum .25')
    parser.add_argument('--spacemouse-speed', type=float, default=.12, help='Maximum translation m/s of simulation time')
    parser.add_argument('--spacemouse-angular-speed', type=float, default=.5, help='Maximum rotation rad/s of simulation time')
    parser.add_argument('--spacemouse-deadzone', type=float, default=.08, help='Normalized knob deadzone')
    parser.add_argument('--pico-port', type=int, default=8088)
    parser.add_argument('--pico-scale', type=float, default=1., help='Robot translation / controller translation')
    parser.add_argument('--pico-speed', type=float, default=.25, help='TCP speed limit in m per simulation second')
    parser.add_argument('--pico-angular-speed', type=float, default=1.5, help='TCP rotation limit in radians per simulation second')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--randomize', action='store_true', help='Randomize cube XY at reset')
    parser.add_argument('--reskin-model', type=Path, help='Opt-in mapped 3D Cal response checkpoint; strict mode blocks unsupported contacts')
    parser.add_argument('--magnetic-fit', type=Path, help='Simulation-only fitted dipole gain profile; alternative to --reskin-model')
    parser.add_argument('--reskin-extrapolate', action='store_true', help='EXPERIMENTAL preview: allow unvalidated cube-patch and out-of-domain predictions')
    parser.add_argument('--record-dir', type=Path, help='Enable synchronized episode recording with B/Enter/N')
    parser.add_argument('--instruction', help='Episode language instruction, stored verbatim; otherwise use the task template')
    parser.add_argument('--record-depth', action=argparse.BooleanOptionalAction, default=True,
                        help='Save synchronized lossless metric depth from all configured cameras (default: enabled)')
    parser.add_argument('--record-format', choices=['lerobot', 'dataset'], default='lerobot',
                        help='lerobot: official v2.1 plus full-rate raw attachment; dataset: raw only')
    parser.add_argument("--diagnostic-log", action="store_true",
                        help="Also save full-session NPZ/CSV diagnostics; default keeps only a 200-sample fault buffer")
    parser.add_argument("--output", default="outputs/diagnostics/pick_cube")
    args = parser.parse_args()
    if args.cable_layout is not None and args.task != 'cable':
        parser.error('--cable-layout requires --task cable')
    if args.clip_model != 'rigid' and args.task != 'cable':
        parser.error('--clip-model requires --task cable')
    if args.workspace_limits and args.no_workspace_limits:
        parser.error('Choose --workspace-limits or --no-workspace-limits')
    if args.reskin_extrapolate and not args.reskin_model:
        parser.error('--reskin-extrapolate requires --reskin-model')
    if args.task in ('cable', 'feeding', 'peg_insert'):
        if args.command != 'viewer' or args.randomize:
            parser.error('Cable/feeding/peg support manual viewer; cube demo/randomization do not apply')
        if args.output == 'outputs/diagnostics/pick_cube':
            args.output = 'outputs/diagnostics/cable_insert' if args.task == 'cable' and args.cable_layout == 'single_clip' else 'outputs/diagnostics/'+args.task
    return (demo if args.command == "demo" else viewer)(args)


if __name__ == "__main__":
    main()
