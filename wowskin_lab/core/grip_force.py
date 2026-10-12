"""Experimental force-feedback outer loop for a coupled parallel gripper.

The inner loop remains the sourced position servo. There is no object-width
lookup: the commanded finger position integrates the measured force error.
Forces are MuJoCo contact-normal sums, not magnetometer-derived estimates.
"""
import numpy as np
import mujoco


def pad_normal_forces(model, data, flex_ids):
    """Each pad's normal load from ALL counterpart objects, in newtons."""
    forces = np.zeros(len(flex_ids))
    wrench = np.zeros(6)
    for i in range(data.ncon):
        contact = data.contact[i]
        sides = [j for j, flex in enumerate(flex_ids) if flex in contact.flex]
        if not sides:
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        if not np.isfinite(wrench[0]):
            raise RuntimeError('Nonfinite contact force feedback')
        forces[sides] += max(0., float(wrench[0]))
    if not np.isfinite(forces).all():
        raise RuntimeError('Nonfinite contact force feedback')
    return forces


class GripForceFeedback:
    def __init__(self, config):
        self.config = dict(config)
        values = [config[k] for k in ('target_n', 'admittance_m_per_ns', 'max_finger_speed_m_s', 'key_step_n')]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Force target, gain, speed and key increment must be finite and positive')
        self.reset()

    def reset(self):
        self.enabled = False
        self.selected_target_n = float(self.config['target_n'])
        self.input_fraction = 1.
        self.measured_n = np.zeros(2)
        self.command_velocity_m_s = 0.

    @property
    def target_n(self):
        return self.selected_target_n*self.input_fraction if self.enabled else 0.

    def update(self, target_position, forces, dt, joint_bounds):
        self.measured_n = np.asarray(forces, dtype=float).copy()
        if not np.isfinite(self.measured_n).all():
            raise RuntimeError('Nonfinite contact force feedback')
        if not np.isfinite(self.target_n) or self.target_n < 0:
            raise ValueError('Force target must be finite and nonnegative')
        self.command_velocity_m_s = 0.
        if not self.enabled:
            return target_position
        # A single coupled actuator cannot regulate the two sides separately.
        # Use the more loaded pad; averaging would double its load in a
        # one-sided contact. Unequal readings remain visible to the operator.
        error = self.target_n-float(np.max(self.measured_n))
        speed = self.config['max_finger_speed_m_s']
        self.command_velocity_m_s = float(np.clip(-self.config['admittance_m_per_ns']*error, -speed, speed))
        return float(np.clip(target_position+self.command_velocity_m_s*dt, *joint_bounds))
