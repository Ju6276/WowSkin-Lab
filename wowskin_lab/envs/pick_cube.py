"""PandaPickCube tactile teleoperation and unchanged Pollen real-to-sim test.

Seven-action reset/step interface (XYZ, local rotvec increments, gripper target).
The cube is a free rigid body. Only contact/friction can lift it; no weld,
attachment, pose copying, or object gravity compensation is used.
"""
import argparse
from contextlib import nullcontext
import csv
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from wowskin_lab.envs.franka import FrankaTactileSim, HOME, ViewerStyleGuard, apply_free_camera, build_scene, mount_reference
from wowskin_lab.core.physics import CHANNELS, TactileSim, configuration
from wowskin_lab.ui.keyboard import BACKSPACE, ENTER, ESC, MOVE_AXES, ViewerKeyboard, direction
from wowskin_lab.data.cameras import add_recording_cameras
from wowskin_lab.teleop.command import CartesianCommand
from wowskin_lab.envs.workcell import industrial_workcell
from wowskin_lab.paths import ROOT

TABLE_Z = .10
CUBE_HALF = .020
CUBE_MASS = .020
WORKSPACE = np.array([[.24, -.18, .127], [.58, .18, .40]])
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
    root.set("model", "WowSkin-Lab | PandaPickCube: WowSkin + plain silicone")
    # Broad box/flex contacts create hundreds of constraints. CG avoids the
    # costly Newton factorization while preserving the same contact geometry.
    option = root.find("option")
    option.set("solver", "CG")
    option.set("jacobian", "dense")
    option.set("iterations", "30")
    option.set("tolerance", "1e-8")
    option.set("noslip_iterations", "1")
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
    add_recording_cameras(world, hand, z)
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
    # Prevent overlapping pads on empty closure, accounting for added thickness.
    gap = 2*(.04+mount_reference(cfg)["position_m"][1]-cfg["total_height_m"])
    lower = -(gap-.006)/2
    root.find(".//default[@class='finger']/joint").set("range", f"{lower} 0")
    industrial_workcell(root)
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
        cfg = configuration() if cfg is None else dict(cfg)
        gripper_cfg = json.loads((ROOT/'configs/gripper_control.json').read_text())
        for key, value in gripper_cfg.items():
            cfg.setdefault(key, value)
        if cfg['gripper_control_mode'] == 'legacy_force_limited':
            for key, value in gripper_cfg['legacy_force_control'].items():
                cfg.setdefault(key, value)
            limits = [cfg['gripper_force_min_n'], cfg['gripper_force_per_finger_n'], cfg['gripper_force_max_n']]
            if not np.isfinite(limits).all() or not 0 < limits[0] <= limits[1] <= limits[2]:
                raise ValueError('Gripper limits must be finite, positive and ordered min <= selected <= max')
        elif cfg['gripper_control_mode'] not in ('panda_menagerie_position', 'panda_robosuite_position'):
            raise ValueError('Unknown gripper controller')
        # Task-specific resolution; bench resolution/parameters stay unchanged.
        cfg.setdefault("mesh_outline_samples", 12)
        cfg.setdefault("scene_design", "industrial_workcell_v1")
        TactileSim.__init__(self, cfg, xml_factory=xml_factory, mount_body="wowskin_mount",
                           contact_body=contact_body, contact_geom=contact_geom)
        self.initialize_mounts_and_arm()
        self.tcp_site = self.model.site("tactile_tcp").id
        self.hand = self.model.body("hand").id
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
            self.cfg['gripper_force_status'] = 'Pinned robosuite Panda per-finger position actuator and joint dynamics; target uses local fingertip travel bounds. 20 N actuator cap per finger, not calibrated contact force.'
        self.cube_q = self.model.jnt_qposadr[self.model.joint(object_joint).id] if object_joint else None
        self.ik_data = mujoco.MjData(self.model)
        self.predictor = None
        if pollen:
            from wowskin_lab.baselines.pollen import PollenPredictor
            self.predictor = PollenPredictor()
        self.reset()
        self.reskin_response = None
        if cfg.get('reskin_model'):
            from wowskin_lab.baselines.reskin_contact import ReskinContactResponse
            self.reskin_response = ReskinContactResponse(cfg['reskin_model'], cfg.get('reskin_extrapolate', False))

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
                    magnetic_ut=(self.last_row["magnetic_ut"].copy() if self.last_row else np.zeros(15)),
                    joint_positions=self.data.qpos[self.arm_q].copy())

    def command_gripper(self, close):
        self.grip_target = float(self.model.jnt_range[self.finger_joints[0], 0]) if close else 0.

    def gripper_fraction(self, target=False):
        lower = self.model.jnt_range[self.finger_joints[0], 0]
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
        # Rate-limit the Cartesian target before solving; key repeats cannot
        # teleport joints or command unbounded velocities.
        delta = self.target_xyz-self.servo_xyz
        length = np.linalg.norm(delta)
        linear_step = self.cfg.get('tcp_speed_m_s', .08)/self.cfg['sample_hz']
        angular_step = self.cfg.get('tcp_angular_speed_rad_s', .8)/self.cfg['sample_hz']
        self.servo_xyz += delta*min(1., linear_step/max(length, 1e-12))
        delta_rotation = rotation_vector(self.servo_rotation.T @ self.target_rotation)
        self.servo_rotation = self.servo_rotation @ rotation_matrix(limit_vector(delta_rotation, angular_step))
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
        d, m = self.data, self.model
        torque = 600*(self.joint_target-d.qpos[self.arm_q])-45*d.qvel[self.arm_v]+d.qfrc_bias[self.arm_v]
        limits = m.actuator_ctrlrange[self.arm_controls]
        d.ctrl[self.arm_controls] = np.clip(torque, limits[:, 0], limits[:, 1])
        if self.cfg.get('gripper_control_mode') in ('panda_menagerie_position', 'panda_robosuite_position'):
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

    def step(self, action):
        action = CartesianCommand.from_array(action).as_array()
        applied = np.r_[np.clip(action[:3], -.005, .005), limit_vector(action[3:6], .03),
                        np.clip(action[6], 0., 1.)]
        before_pose = np.r_[self.data.site_xpos[self.tcp_site],
                            rotation_vector(self.data.site_xmat[self.tcp_site])]
        action_time = float(self.data.time)
        target = np.clip(self.target_xyz+applied[:3], *getattr(self, 'workspace', WORKSPACE))
        applied[:3] = target-self.target_xyz
        self.target_xyz = target
        self.target_rotation = self.target_rotation @ rotation_matrix(applied[3:6])
        self.grip_target = float(applied[6]*self.model.jnt_range[self.finger_joints[0], 0])
        self.solve_ik()
        row = self.reading()
        if self.reskin_response:
            row['dipole_magnetic_ut'] = row['magnetic_ut'].copy()
            row['dipole_ideal_delta_ut'] = row['ideal_delta_ut'].copy()
            row['dipole_raw_field_ut'] = row['raw_field_ut'].copy()
            row.update(self.reskin_response.evaluate(self))
        row["action"] = applied
        row["action_time_s"] = action_time
        row["tcp_pose_before_m_rad"] = before_pose
        row["tcp_rotvec_rad"] = rotation_vector(self.data.site_xmat[self.tcp_site])
        row["gripper_closed_fraction"] = self.gripper_fraction()
        row['gripper_force_limit_per_finger_n'] = (20. if self.cfg.get('gripper_control_mode') == 'panda_robosuite_position' else float('nan') if self.cfg.get('gripper_control_mode') in ('panda_menagerie_position', 'panda_robosuite_position') else self.grip_force)
        row['finger_actuator_force_n'] = self.data.qfrc_actuator[self.finger_v].copy()
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
        self.success |= row["success"]
        self.last_row = row
        reward = .3*np.exp(-20*np.linalg.norm(row["cube_xyz_m"]-row["tcp_xyz_m"]))+.7*min(row["lift_m"]/.06, 1.)
        # Continue teleoperation after success so the user can lower/open/tug.
        return self.observation(), float(reward), False, False, row


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
        if env.cfg.get('gripper_control_mode') in ('panda_menagerie_position', 'panda_robosuite_position'):
            return  # Source position actuator; no invented per-finger limit.
        env.grip_force = float(np.clip(env.grip_force+(.1 if code == ord("=") else -.1),
                                      env.cfg['gripper_force_min_n'], env.cfg['gripper_force_max_n']))
    elif code == ord("T"):
        env.tug()
    elif code in (ord("R"), 259):
        env.reset()


def hold_current_pose(env):
    """Cancel an outstanding Cartesian target when manual input ends/starts."""
    env.target_xyz = np.clip(env.data.site_xpos[env.tcp_site].copy(), *getattr(env, 'workspace', WORKSPACE))
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
    translation = direction(held)
    if camera_azimuth is not None:
        translation = view_translation(translation, camera_azimuth)
    velocity = translation*env.cfg.get('keyboard_speed_m_s', .04)
    angular_velocity = direction(held, ROTATE_AXES)*.4
    moving = bool(np.any(velocity) or np.any(angular_velocity))
    if moving != was_moving:
        hold_current_pose(env)
    if moving:
        actual = env.data.site_xpos[env.tcp_site]
        env.target_xyz = np.clip(env.target_xyz, actual-.006, actual+.006)
        actual_rotation = env.data.site_xmat[env.tcp_site].reshape(3, 3)
        lead = rotation_vector(actual_rotation.T @ env.target_rotation)
        env.target_rotation = actual_rotation @ rotation_matrix(limit_vector(lead, .03))
    return np.r_[velocity/env.cfg['sample_hz'], angular_velocity/env.cfg['sample_hz'],
                 env.gripper_fraction(target=True)], moving


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
    if getattr(args, 'task', 'pick_cube') == 'cable':
        cfg['cable_layout'] = getattr(args, 'cable_layout', None) or 'routing'
    physics_step_us = getattr(args, 'physics_step_us', None)
    if physics_step_us is None and getattr(args, 'task', 'pick_cube') in ('cable', 'feeding'):
        # Feeding's recorded 6D spoon motion becomes unstable at 100 us and
        # passes at 50 us with this thin-silicone/contact configuration.
        # This costs throughput; the sensor/control clock remains 100 Hz.
        physics_step_us = 50 if args.task == 'feeding' else 100
    if physics_step_us is not None:
        cfg['timestep_s'] = physics_step_us*1e-6
        cfg['timestep_source'] = 'Teleoperation numerical integration setting; 100 Hz sensor clock retained'
    if getattr(args, 'ideal', False):
        cfg.update(noise_ut=0., drift_ut_per_s=0.)
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
                  mount=mount_reference(env.cfg), samples=len(rows), picked_up=bool(env.success),
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
    cfg = response_configuration(args)
    cfg['randomize_cube'] = args.randomize
    cable_task = getattr(args, 'task', 'pick_cube') == 'cable'
    feeding_task = getattr(args, 'task', 'pick_cube') == 'feeding'
    other_task = cable_task or feeding_task
    keyboard_speed = getattr(args, 'keyboard_speed', None)
    keyboard_speed = (.12 if other_task else .04) if keyboard_speed is None else keyboard_speed
    if not np.isfinite(keyboard_speed) or not (0 < keyboard_speed <= .25):
        raise ValueError('Keyboard speed must be in (0, .25] m/s')
    cfg['keyboard_speed_m_s'] = keyboard_speed
    if args.input == 'keyboard':
        cfg['tcp_speed_m_s'] = max(cfg.get('tcp_speed_m_s', .08), keyboard_speed)
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
        cfg['spacemouse'] = mouse_controller.settings()
    if cable_task:
        from wowskin_lab.envs.cable import PandaCableEnv
        env = PandaCableEnv(cfg, pollen=not args.no_pollen)
    elif feeding_task:
        from wowskin_lab.envs.feeding import PandaFeedingEnv
        env = PandaFeedingEnv(cfg, pollen=not args.no_pollen)
    else:
        env = PandaPickCubeTactileEnv(cfg, pollen=not args.no_pollen)
    title = ('PandaCable / single clip' if cable_task and env.cable_layout == 'single_clip'
             else 'PandaCable' if cable_task else 'PandaFeeding' if feeding_task else 'PandaPickCube')
    source_gripper = env.cfg.get('gripper_control_mode') in ('panda_menagerie_position', 'panda_robosuite_position')
    if source_gripper:
        print('Gripper: robosuite Panda independent finger servos, 20 N actuator cap each' if env.cfg['gripper_control_mode'] == 'panda_robosuite_position' else 'Gripper: MuJoCo Menagerie Panda position servo; 100 N tendon actuator cap, NOT fingertip contact force; no 0.35 N limiter', flush=True)
    else:
        print(f"Gripper: local, uncalibrated per-finger drive limit {env.grip_force:.2f} N; "
              "configs/gripper_control.json (--grip-force overrides the initial value)", flush=True)
    from wowskin_lab.ui.tactile import TactileHUD
    hud = TactileHUD(env)
    if cable_task and hud.reskin_mode:
        hud.subtitle = 'Experimental cable proxy [uT]'
        hud.footer = 'NOT calibrated | cable/probe mismatch'
    keyboard = ViewerKeyboard(set(map(ord, "WASDEQXZCTFRVBNP-=,.")) | MOTION_KEYS | {ESC, ENTER, BACKSPACE},
                              title_contains='PandaCable:' if cable_task else title+':')
    recorder = None
    if args.record_dir and cfg.get('reskin_model') and not cfg.get('reskin_extrapolate'):
        print('Strict ReSkin mode: unsupported predictions are blank; synchronized training recording is disabled. Diagnostic rows are saved on exit.', flush=True)
    if args.record_dir and not (cfg.get('reskin_model') and not cfg.get('reskin_extrapolate')):
        if args.record_format == 'ftf-raw':
            from wowskin_lab.data.ftf_recording import FTFSimRecorder
            recorder = FTFSimRecorder(env, args.record_dir, source=args.input)
        elif args.record_format == 'visk-raw':
            from wowskin_lab.data.visk_recording import ViSkSimRecorder
            recorder = ViSkSimRecorder(env, args.record_dir, source=args.input)
        elif args.record_format == 'both-raw':
            from wowskin_lab.data.learning_recording import LearningSimRecorder
            recorder = LearningSimRecorder(env, args.record_dir, source=args.input)
        else:
            from wowskin_lab.data.recording import MujocoRecorder
            recorder = MujocoRecorder(env, args.record_dir, args.record_hz, source=args.input)
    pico = None
    if args.input == 'pico':
        from wowskin_lab.teleop.pico import PicoBridge
        pico = PicoBridge(args.pico_port, args.pico_scale, args.pico_speed, args.pico_angular_speed)
    rows, close, fault, automatic = [], cable_task, "", False
    faults = []
    was_moving = False
    next_hud = 0.
    realtime_factor, loop_hz = 0., 0.
    deadline = time.monotonic()+args.seconds if args.seconds else float("inf")
    grip_help = env.cfg['gripper_control_mode'] if source_gripper else '-/= legacy force limit'
    print(f"{title} 7D: HOLD WASD/arrows view-relative XY, E/Q Z | I/K local Rx, L/J Ry, U/O Rz | release to stop | X close Z open C hold | {grip_help} | T tug | R reset | V view | ,/. display scale | Esc overview | P exit", flush=True)
    if recorder:
        print('Recording: B new/reset or stop (unreviewed) | Enter save SUCCESS | N save FAILURE | R interrupts | P save interrupted and exit', flush=True)
    with (mouse or nullcontext()), (pico or nullcontext()), (recorder or nullcontext()), mujoco.viewer.launch_passive(env.model, env.data, key_callback=keyboard.native_callback,
                                    show_left_ui=False, show_right_ui=False) as handle, keyboard:
        if mouse:
            env.cfg['spacemouse']['device'] = mouse.info
            print(f"SpaceMouse: {mouse.info['product_string']} | push/tilt in camera frame | left close, right open | B/Enter/N record | click MuJoCo window to control", flush=True)
        def apply_camera():
            cam = task_camera(env, close, operator=bool(pico))
            apply_free_camera(handle, cam)
        apply_camera()
        if pico:
            pico.open_usb()
        style = None if keyboard.continuous else ViewerStyleGuard(handle, "WASDEQXZCTFRVIJKLUO,.")
        print("Keyboard: window-local press/release; native shortcuts consumed" if keyboard.continuous
              else "Keyboard: non-macOS single-step fallback (no held-key support)", flush=True)
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
                if pico and code == ord('G'):
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
                        recorder.finish('success' if code == ENTER else ('failure' if code == ord('N') else 'unreviewed'))
                    elif code == ord('B'):
                        env.reset()
                        automatic, fault, was_moving = False, '', False
                        keyboard.clear()
                        held = frozenset()
                        recorder.start()
                        if pico:
                            pico.controller.reset()
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
                        env.reset()
                        fault = ""
                        keyboard.clear()
                        held, was_moving = frozenset(), False
                else:
                    if recorder and code in (ord('R'), BACKSPACE):
                        recorder.finish('interrupted')
                    hud.key(code)
                    key_command(env, code)
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
                        fallback_motion[:3] = view_translation(fallback_motion[:3], camera_azimuth)
                        fallback_motion[6] = env.gripper_fraction(target=True)
                        action = fallback_motion
                    physics_started = time.perf_counter()
                    row = env.step(action)[-1]
                    row['physics_and_sensor_wall_s'] = time.perf_counter()-physics_started
                    row['total_contacts'] = int(env.data.ncon)
                    row['constraint_rows'] = int(env.data.nefc)
                    if mouse:
                        row['spacemouse_axes'] = mouse_snapshot.axes.copy()
                        row['spacemouse_buttons'] = mouse_snapshot.buttons
                    if recorder:
                        recorder.step(row)
                    row['held_motion_keys'] = ' '.join(str(k) for k in sorted(held & MOTION_KEYS))
                    row['viewer_loop_wall_s'] = time.monotonic()-start
                    row['viewer_realtime_factor'] = max(0., float(env.data.time)-sample_start)/max(row['viewer_loop_wall_s'], 1e-9)
                    if automatic and env.data.time >= 8.5:
                        automatic = False
                    rows.append(row)
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
                labels = [title, "Move (camera plane + Z)", "Rotate (local XYZ)", "Gripper", "Test / view", "Left / right load [N]",
                          "Pollen: LEFT 15 channels", "Physics: left slip", "Task" if other_task else "Lift / picked", "Target XYZ [m]", "Status"]
                text = ["7D | WowSkin + plain silicone", "W/S forward/back | A/D left/right | E/Q up/down", "Hold I/K Rx | L/J Ry | U/O Rz (0.4 rad/s)", f"X close | Z open | C hold | -/= {env.grip_force:.2f} N limit",
                        ("T tug | R reset | V view | Esc" if other_task else "F test | T tug | R reset | V view | Esc"), f"{row['normal_force_n']:.3f} / {row['passive_normal_force_n']:.3f}",
                        (f"{row['pollen_probability']:.3f} (transfer unvalidated)" if env.predictor else 'disabled'), str(row['slip']),
                        ("Loaded spoon | 6 beans | manual outcome" if feeding_task else
                         ("Single clip | manual outcome" if env.cable_layout == 'single_clip' else "Manual outcome; routing not scored")
                         if cable_task else f"{row['lift_m']*1000:.1f} mm / {env.success}"), " ".join(f"{v:.3f}" for v in env.target_xyz),
                        ("PAUSED: physics invalid | R reset | P save/exit" if fault else
                         (env.phase if automatic else ("Moving / rotating" if was_moving else "Manual hold"))+" | uncalibrated")]
                labels += ['Simulation speed']
                text += [f'{realtime_factor:.2f}x real time | {loop_hz:.1f} updates/s | keyboard {keyboard_speed*1000:.0f} mm/s sim']
                if mouse:
                    labels += ['SpaceMouse USB']
                    text += [mouse_controller.status]
                    text[1:5] = ['Push knob: camera XY | lift: world Z', 'Tilt / twist: camera-frame rotation',
                                 f'Left close | right open | -/= {env.grip_force:.2f} N limit',
                                 'B record | Enter success | N failure | R reset']
                    text[-2] = f'{realtime_factor:.2f}x real time | {loop_hz:.1f} updates/s | max {mouse_controller.speed*1000:.0f} mm/s sim'
                if pico:
                    labels += ['Pico USB', 'Pico placement']
                    text += [pico.controller.status, f'{pico.controller.mode or "waiting"} | right B switches mode | right A optional align']
                    text[1:5] = ['Right GRIP + move | release stops', 'Right GRIP + rotate',
                                 'Right TRIGGER closes | release GRIP holds closure', 'Left X record | Y success | stick failure | V/Esc view']
                if recorder:
                    labels += ['Dataset recording']
                    text += [f'REC {recorder.writer.count} frames | Enter success / N failure' if recorder.active
                             else 'B starts new episode | P saves and exits']
                if source_gripper:
                    text[3] = ('Left close | right open' if mouse else 'X close | Z open | C hold')+(' | robosuite Panda servo' if env.cfg['gripper_control_mode'] == 'panda_robosuite_position' else ' | Menagerie Panda servo')
                if feeding_task:
                    labels += ['Head load / episode peak [N]', 'IK condition / joint margin']
                    text += [f"{row['head_contact_force_n']:.3f} / {row['head_peak_force_n']:.3f} | local rigid proxy",
                             f"{row['ik_weighted_condition']:.2f} / {row['joint_limit_margin_deg']:.1f} deg | rotation weight 0.2 m"]
                labels += ['Physics + sensor cost']
                text += [f"{row.get('physics_and_sensor_wall_s', 0.)*1000:.1f} ms / 10 ms sim | {row.get('total_contacts', 0)} contacts, {row.get('constraint_rows', 0)} constraints"]
                if fault:
                    # Keep recovery keys visible; the full exception can be
                    # wider than the viewer and previously hid the reset hint.
                    labels += ['Physics fault', 'Displayed data']
                    text += [fault.split('. Physics stopped', 1)[0], 'Last valid sample; recording stopped']
                if not handle.is_running():
                    break
                handle.set_texts([(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                   "\n".join(labels), "\n".join(text))])
                hud.show(handle, row)
            before_sync = env.data.time
            if style:
                style.sync(handle)
            else:
                handle.sync()
            # MuJoCo's built-in Backspace resets to qpos0, bypassing our
            # controller and baseline. Restore the task reset before stepping.
            if env.data.time < before_sync-1e-9:
                env.reset()
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
    if rows:
        destination = Path(args.output)/time.strftime("teleop_%Y%m%d_%H%M%S")
        save_records(rows, destination)
        print(f"Saved {len(rows)} simulated samples to {destination}", flush=True)
    return dict(samples=len(rows), faults=faults)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["viewer", "demo"], nargs="?", default="viewer")
    parser.add_argument("--no-pollen", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seconds", type=float, default=0.)
    parser.add_argument('--input', choices=['keyboard', 'pico', 'spacemouse'], default='keyboard')
    parser.add_argument('--task', choices=['pick_cube', 'cable', 'feeding'], default='pick_cube')
    parser.add_argument('--cable-layout', choices=['routing', 'single_clip'],
                        help='Cable fixture layout: original routing (default), or one reversed C-clip')
    parser.add_argument('--grip-force', type=float, help='Explicitly enable the OLD local force-limited controller [N/finger]; default uses sourced Menagerie Panda position servo')
    parser.add_argument('--ideal', action='store_true', help='Disable synthetic magnetic noise/drift; retain deformation response and filter/delay')
    parser.add_argument('--physics-step-us', type=int, choices=[20, 50, 100], help='Integration step in microseconds; recorded in dataset metadata')
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
    parser.add_argument('--record-hz', type=int, default=10)
    parser.add_argument('--record-format', choices=['local', 'ftf-raw', 'visk-raw', 'both-raw'], default='local',
                        help='visk-raw: paper single-sensor/four cameras; ftf-raw: original Franka-Teach writer; training is separate')
    parser.add_argument("--output", default="outputs/pick_cube")
    args = parser.parse_args()
    if args.cable_layout is not None and args.task != 'cable':
        parser.error('--cable-layout requires --task cable')
    if args.reskin_extrapolate and not args.reskin_model:
        parser.error('--reskin-extrapolate requires --reskin-model')
    if args.task in ('cable', 'feeding'):
        if args.command != 'viewer' or args.randomize:
            parser.error('Cable/feeding support manual viewer; cube demo/randomization do not apply')
        if args.record_dir and args.record_format == 'local':
            parser.error('Cable/feeding recording requires both-raw, visk-raw or ftf-raw; legacy cube-point format is not supported')
        if args.output == 'outputs/pick_cube':
            args.output = 'outputs/cable_insert' if args.task == 'cable' and args.cable_layout == 'single_clip' else 'outputs/'+args.task
    return (demo if args.command == "demo" else viewer)(args)


if __name__ == "__main__":
    main()
