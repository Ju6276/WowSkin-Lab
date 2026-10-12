"""Measured robot state in the named base frame, independent of scene placement.

T_world_base maps base coordinates into world coordinates. Camera extrinsics
map base coordinates into OpenCV camera coordinates. This does not change the
controller's world-frame API or the installed tactile TCP reference point.
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

BASE_BODY = 'link0'


def world_from_base(env):
    bid = env.model.body(BASE_BODY).id
    transform = np.eye(4)
    transform[:3, :3] = env.data.xmat[bid].reshape(3, 3)
    transform[:3, 3] = env.data.xpos[bid]
    return transform


def pose_in_base(position, rotation, transform):
    base_rotation = transform[:3, :3].T
    return np.r_[base_rotation @ (position-transform[:3, 3]),
                 Rotation.from_matrix(base_rotation @ np.asarray(rotation).reshape(3, 3)).as_rotvec()]


def command_in_base(action, transform):
    """Translation is a vector; local TCP rotation and gripper are unchanged."""
    result = np.asarray(action, dtype=float).copy()
    result[:3] = transform[:3, :3].T @ result[:3]
    return result


def camera_from_world(env, camera_id):
    transform = np.eye(4)
    transform[:3, :3] = np.diag([1., -1., -1.]) @ env.data.cam_xmat[camera_id].reshape(3, 3).T
    transform[:3, 3] = -transform[:3, :3] @ env.data.cam_xpos[camera_id]
    return transform


def joint_sample(env):
    return dict(arm_qpos=env.data.qpos[env.arm_q].copy(),
                arm_qvel=env.data.qvel[env.arm_v].copy(),
                gripper_driver_qpos=env.data.qpos[env.finger_q].copy(),
                gripper_driver_qvel=env.data.qvel[env.finger_v].copy())


def robot_sample(env):
    transform = world_from_base(env)
    position = env.data.site_xpos[env.tcp_site]
    rotation = env.data.site_xmat[env.tcp_site].reshape(3, 3)
    # Spatial velocity from actual MuJoCo qvel, not a derivative of rotvec.
    jac_pos, jac_rot = np.zeros((3, env.model.nv)), np.zeros((3, env.model.nv))
    mujoco.mj_jacSite(env.model, env.data, jac_pos, jac_rot, env.tcp_site)
    linear_world, angular_world = jac_pos @ env.data.qvel, jac_rot @ env.data.qvel
    jac_base_pos, jac_base_rot = np.zeros_like(jac_pos), np.zeros_like(jac_rot)
    mujoco.mj_jacBody(env.model, env.data, jac_base_pos, jac_base_rot, env.model.body(BASE_BODY).id)
    base_linear, base_angular = jac_base_pos @ env.data.qvel, jac_base_rot @ env.data.qvel
    linear_base = transform[:3, :3].T @ (linear_world-base_linear-np.cross(base_angular, position-transform[:3, 3]))
    angular_base = transform[:3, :3].T @ (angular_world-base_angular)
    return dict(cartesian_states=pose_in_base(position, rotation, transform),
                commanded_pose=pose_in_base(env.target_xyz, env.target_rotation, transform),
                cartesian_states_world=np.r_[position, Rotation.from_matrix(rotation).as_rotvec()],
                commanded_pose_world=np.r_[env.target_xyz, Rotation.from_matrix(env.target_rotation).as_rotvec()],
                tcp_velocity=np.r_[linear_base, angular_base],
                tcp_velocity_world=np.r_[linear_world, angular_world],
                world_from_base=transform, **joint_sample(env))


def robot_metadata(env):
    from wowskin_lab.envs.pick_cube import WORKSPACE
    parent = env.model.site_bodyid[env.tcp_site]
    bounds = np.asarray(getattr(env, 'workspace', WORKSPACE))
    limited = not env.cfg.get('disable_cartesian_limits', False)
    workspace_limited = limited and not env.cfg.get('disable_workspace_limits', False)
    cartesian = env.cfg.get('arm_controller') == 'cartesian_impedance'
    control = dict(source='Simulation control and local input shaping; source and gains recorded separately',
        arm_controller=env.cfg.get('arm_controller', 'ik_pd'),
        arm_control_source=env.cfg.get('arm_control_source'),
        cartesian_impedance=env.cfg.get('cartesian_impedance') if cartesian else None,
        teleop_frame=env.cfg.get('teleop_frame', 'legacy_unspecified'),
        teleop_base_lateral_sign=env.cfg.get('teleop_base_lateral_sign', -1.),
        translation_target_limiter=env.cfg.get('translation_target_limiter', 'legacy_component_clip'),
        cartesian_limits_enabled=limited,
        workspace_limits_enabled=workspace_limited,
        workspace_world_m=[[float(v) if np.isfinite(v) else None for v in row] for row in bounds] if workspace_limited else None,
        unbounded_axis_value=None, max_translation_component_per_tick_m=.005 if limited else None,
        max_rotation_norm_per_tick_rad=.03 if limited else None,
        tcp_speed_m_s=env.cfg.get('tcp_speed_m_s', .08) if limited else None,
        tcp_angular_speed_rad_s=env.cfg.get('tcp_angular_speed_rad_s', .8) if limited else None,
        ik_iterations=8, ik_damping_squared=.0001, ik_max_joint_update_rad=.04,
        ik_rotation_error_weight=.2, ik_error_tolerance=1e-5,
        joint_limit_margin_rad=.02, arm_pd_position_gain=600., arm_pd_velocity_gain=45.,
        ik_pd_usage='Reset initialization only' if cartesian else 'Reset and runtime',
        keyboard_spacemouse_target_lead_component_m=.006 if limited else None,
        keyboard_spacemouse_rotation_lead_rad=.03 if limited else None,
        pico_target_lead_component_m=.03 if limited else None, pico_rotation_lead_rad=.10 if limited else None,
        keyboard_spacemouse_recenter_on_release=limited,
        input_scale='Keyboard/SpaceMouse velocity gain and deadzone are retained; not a second target slew limit',
        input_loss='Focus loss or stale tracking cancels target; Pico grip release remains a clutch',
        pico_tracking_jump_rejection_m_rad=[.15, .8],
        policy_target='Accumulate predicted increments; no teleoperator target-lead clamp',
        actuator_limits='Source MuJoCo model actuator_ctrlrange / forcerange; actual contact force is solved',
        automatic_task_score='Scoring only; no snapping, grasp weld or commanded force overrides')
    return dict(robot_state_schema='robot_base_v1', pose_frame='robot_base', base_body=BASE_BODY,
                controller_contract=control,
                tcp_site='tactile_tcp', tcp_parent_body=env.model.body(parent).name,
                tcp_position_in_parent_m=env.model.site_pos[env.tcp_site].tolist(),
                tcp_quaternion_in_parent_wxyz=env.model.site_quat[env.tcp_site].tolist(),
                world_from_base_at_start=world_from_base(env).tolist(),
                transform_convention='T_A_B maps coordinates from B to A; column vectors',
                camera_extrinsic_frame='robot_base_to_opencv_camera',
                arm_joint_names=[env.model.joint(j).name for j in env.arm_joints],
                arm_qpos_unit='rad', arm_qvel_unit='rad/s',
                tcp_velocity='base-frame relative linear vx/vy/vz (m/s), angular wx/wy/wz (rad/s); actual MuJoCo Jacobian @ qvel',
                joint_state_source='measured simulation qpos/qvel, not joint command targets',
                gripper_driver_joint_names=[env.model.joint(j).name for j in env.finger_joints],
                gripper_driver_position_units=['m' if int(env.model.jnt_type[j]) == 2 else 'rad'
                                               for j in env.finger_joints],
                gripper_driver_velocity_units=['m/s' if int(env.model.jnt_type[j]) == 2 else 'rad/s'
                                               for j in env.finger_joints])


def append_h5(stream, key, value):
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise ValueError(f'Nonfinite raw recording: {key}')
    if key not in stream:
        stream.create_dataset(key, shape=(0, *value.shape), maxshape=(None, *value.shape),
                              dtype=value.dtype, chunks=True)
    dataset = stream[key]
    dataset.resize(dataset.shape[0]+1, axis=0)
    dataset[-1] = value
