"""The existing eight-iteration damped IK controller, captured on CUDA.

Forward kinematics and Jacobians use MuJoCo Warp. A kinematics-only model
contains the unchanged base-to-hand chain, avoiding work on scene objects and
soft bodies during each IK iteration. Joint limits, damping and weights match
PandaPickCubeTactileEnv.solve_ik; initialization still uses native MuJoCo.
"""
import xml.etree.ElementTree as ET

import weakref

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

Mat6 = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
Vec6 = wp.types.vector(length=6, dtype=wp.float64)


@wp.func
def rotation_vector(r: wp.mat33d):
    q = wp.normalize(wp.quat_from_matrix(r))
    if q[3] < wp.float64(0.0):
        q = -q
    v = wp.vec3d(q[0], q[1], q[2])
    length = wp.length(v)
    scale = wp.float64(2.0)
    if length > wp.float64(1.e-12):
        scale = wp.float64(2.0)*wp.atan2(length, q[3])/length
    return v*scale


@wp.kernel
def _start(target: wp.array[wp.float64], servo: wp.array[wp.float64],
           linear_step: wp.float64, angular_step: wp.float64, unlimited: bool,
           joint_target: wp.array[float], qpos: wp.array2d[float], active: wp.array[int]):
    delta = wp.vec3d(target[0]-servo[0], target[1]-servo[1], target[2]-servo[2])
    scale = wp.float64(1.0)
    if not unlimited:
        scale = wp.min(scale, linear_step/wp.max(wp.length(delta), wp.float64(1.e-12)))
    for i in range(3):
        servo[i] += scale*delta[i]
    a = wp.mat33d()
    b = wp.mat33d()
    for i in range(3):
        for j in range(3):
            a[i, j] = servo[3+3*i+j]
            b[i, j] = target[3+3*i+j]
    angle = rotation_vector(wp.transpose(a)*b)
    norm = wp.length(angle)
    if not unlimited:
        angle *= wp.min(wp.float64(1.0), angular_step/wp.max(norm, wp.float64(1.e-12)))
    length = wp.length(angle)
    inc = wp.identity(n=3, dtype=wp.float64)
    if length > wp.float64(1.e-15):
        inc = wp.quat_to_matrix(wp.quat_from_axis_angle(angle/length, length))
    a = a*inc
    for i in range(3):
        for j in range(3):
            servo[3+3*i+j] = a[i, j]
    for i in range(7):
        qpos[0, i] = joint_target[i]
    active[0] = 1


@wp.kernel
def _update(servo: wp.array[wp.float64], position: wp.array2d[wp.vec3],
            rotation: wp.array2d[wp.mat33], hand: int,
            jp: wp.array3d[float], jr: wp.array3d[float],
            bounds: wp.array2d[float], qpos: wp.array2d[float],
            joint_target: wp.array[float], active: wp.array[int]):
    if active[0] == 0:
        return
    desired = wp.mat33d()
    for i in range(3):
        for j in range(3):
            desired[i, j] = servo[3+3*i+j]
    angle = rotation_vector(desired*wp.transpose(wp.mat33d(rotation[0, hand])))
    error = Vec6()
    for i in range(3):
        error[i] = servo[i]-wp.float64(position[0, 0][i])
        error[i+3] = wp.float64(.2)*angle[i]
    if wp.length(error) < wp.float64(1.e-5):
        active[0] = 0
        return
    # Cholesky of J J^T + 1e-4 I (positive definite even at singular poses).
    matrix = Mat6()
    for i in range(6):
        for j in range(6):
            value = wp.float64(0.0)
            for k in range(7):
                vi = wp.float64(0.0)
                vj = wp.float64(0.0)
                if i < 3:
                    vi = wp.float64(jp[0, i, k])
                else:
                    vi = wp.float64(.2)*wp.float64(jr[0, i-3, k])
                if j < 3:
                    vj = wp.float64(jp[0, j, k])
                else:
                    vj = wp.float64(.2)*wp.float64(jr[0, j-3, k])
                value += vi*vj
            if i == j:
                value += wp.float64(.0001)
            matrix[i, j] = value
    lower = Mat6()
    for i in range(6):
        for j in range(i+1):
            value = matrix[i, j]
            for k in range(j):
                value -= lower[i, k]*lower[j, k]
            if i == j:
                lower[i, j] = wp.sqrt(value)
            else:
                lower[i, j] = value/lower[j, j]
    result = Vec6()
    for i in range(6):
        value = error[i]
        for j in range(i):
            value -= lower[i, j]*result[j]
        result[i] = value/lower[i, i]
    for k in range(6):
        i = 5-k
        value = result[i]
        for j in range(i+1, 6):
            value -= lower[j, i]*result[j]
        result[i] = value/lower[i, i]
    for k in range(7):
        dq = wp.float64(0.0)
        for i in range(3):
            dq += wp.float64(jp[0, i, k])*result[i] + wp.float64(.2)*wp.float64(jr[0, i, k])*result[i+3]
        q = qpos[0, k]+float(wp.clamp(dq, wp.float64(-.04), wp.float64(.04)))
        qpos[0, k] = wp.clamp(q, bounds[k, 0]+.02, bounds[k, 1]-.02)
        joint_target[k] = qpos[0, k]


def arm_model(env):
    m = env.model
    chain = []
    body = env.hand
    while body:
        chain.append(body)
        body = int(m.body_parentid[body])
    chain.reverse()
    joints = [int(m.body_jntadr[b])+i for b in chain for i in range(m.body_jntnum[b])]
    if joints != list(env.arm_joints) or len(joints) != 7 or any(m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in joints):
        raise ValueError('GPU IK requires the sourced seven-hinge arm with a fixed base')
    if m.site_bodyid[env.tcp_site] != env.hand or any(m.body_mocapid[b] >= 0 for b in chain):
        raise ValueError('GPU IK requires a fixed-base hand-mounted TCP')
    def vec(value):
        return ' '.join(map(str, np.asarray(value)))
    root = ET.Element('mujoco')
    ET.SubElement(root, 'compiler', angle='radian')
    parent = ET.SubElement(root, 'worldbody')
    for b in chain:
        parent = ET.SubElement(parent, 'body', name=m.body(b).name,
            pos=vec(m.body_pos[b]), quat=vec(m.body_quat[b]))
        if m.body_mass[b] > 0:
            ET.SubElement(parent, 'inertial', pos=vec(m.body_ipos[b]), quat=vec(m.body_iquat[b]),
                mass=str(m.body_mass[b]), diaginertia=vec(m.body_inertia[b]))
        for j in range(m.body_jntadr[b], m.body_jntadr[b]+m.body_jntnum[b]):
            ET.SubElement(parent, 'joint', type='hinge', name=m.joint(j).name,
                pos=vec(m.jnt_pos[j]), axis=vec(m.jnt_axis[j]), ref=str(m.qpos0[m.jnt_qposadr[j]]))
    ET.SubElement(parent, 'site', name='tcp', pos=vec(m.site_pos[env.tcp_site]),
                  quat=vec(m.site_quat[env.tcp_site]))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


class GpuIK:
    def __init__(self, env, joint_target):
        self.env = weakref.proxy(env)
        native = arm_model(env)
        self.model = mjw.put_model(native)
        self.data = mjw.put_data(native, mujoco.MjData(native), nconmax=1, njmax=1)
        self.hand = native.body('hand').id
        self.joint_target = joint_target
        self.target = wp.array(np.r_[env.target_xyz, env.target_rotation.ravel()], dtype=wp.float64)
        self.servo = wp.array(np.r_[env.servo_xyz, env.servo_rotation.ravel()], dtype=wp.float64)
        self.bounds = wp.array(env.model.jnt_range[env.arm_joints], dtype=float)
        self.active = wp.ones(1, dtype=int)
        self.jp = wp.zeros((1, 3, 7), dtype=float)
        self.jr = wp.zeros_like(self.jp)
        self.body = wp.array([self.hand], dtype=int)
        self.run()
        wp.synchronize()
        # Warmup must not advance the controller or change its initial target.
        joint_target.assign(env.joint_target)
        self.servo.assign(np.r_[env.servo_xyz, env.servo_rotation.ravel()])
        with wp.ScopedCapture() as captured:
            self.run()
        self.graph = captured.graph

    def run(self):
        cfg = self.env.cfg
        wp.launch(_start, 1, [self.target, self.servo,
            cfg.get('tcp_speed_m_s', .08)/cfg['sample_hz'],
            cfg.get('tcp_angular_speed_rad_s', .8)/cfg['sample_hz'],
            cfg.get('disable_cartesian_limits', False), self.joint_target, self.data.qpos, self.active])
        for _ in range(8):
            mjw.kinematics(self.model, self.data)
            mjw.com_pos(self.model, self.data)
            mjw.jac(self.model, self.data, self.jp, self.jr, self.data.site_xpos.flatten(), self.body)
            wp.launch(_update, 1, [self.servo, self.data.site_xpos, self.data.xmat, self.hand,
                self.jp, self.jr, self.bounds, self.data.qpos, self.joint_target, self.active])

    def solve(self):
        self.target.assign(np.r_[self.env.target_xyz, self.env.target_rotation.ravel()])
        wp.capture_launch(self.graph)

    def read_target(self):
        self.env.joint_target = self.joint_target.numpy().astype(float)
        servo = self.servo.numpy()
        self.env.servo_xyz = servo[:3].copy()
        self.env.servo_rotation = servo[3:].reshape(3, 3).copy()
