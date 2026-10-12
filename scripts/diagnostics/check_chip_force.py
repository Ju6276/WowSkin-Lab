"""Physical chip grasp/lift check, with force feedback and no width lookup.

Run from the repository: python -m scripts.diagnostics.check_chip_force
Grasp pose and timing are explicit test inputs, not feedback controller inputs.
The controller has no access to chip position, geometry, mass or fracture state.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import mujoco

from wowskin_lab.envs.chip_tray import PandaChipEnv
from wowskin_lab.envs.pick_cube import response_configuration, rotation_matrix, rotation_vector


def chip_contact_loads(env):
    """Resolve pad forces ON the chip; diagnostics only, never controller input.

    Scalar normal loads alone do not say whether a curved contact supports
    the chip upward. Exclude internal fragment contacts from the balance.
    """
    pads = (env.skin_flex, env.passive_flex)
    pieces = set(env.fracture.piece_geoms)
    normal, tangent, other = np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(3)
    wrench = np.zeros(6)
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        sides = [side for side in (0, 1) if c.geom[side] in pieces]
        if len(sides) != 1:
            continue
        side = sides[0]
        mujoco.mj_contactForce(env.model, env.data, i, wrench)
        # Contact frame points from side 0 to side 1.
        basis = c.frame.reshape(3, 3)*(1 if side == 1 else -1)
        n, t = basis[0]*wrench[0], basis[1:].T@wrench[1:3]
        opposite_flex = c.flex[1-side]
        if opposite_flex in pads:
            j = pads.index(opposite_flex)
            normal[j] += n
            tangent[j] += t
        else:
            other += n+t
    return dict(pad_normal_world_n=normal.tolist(), pad_tangent_world_n=tangent.tolist(),
                other_force_world_n=other.tolist())


def run(target_n, grasp_z=1.123, closing_seconds=32., yaw_deg=0., admittance=None, finger_speed=None):
    cfg = response_configuration(SimpleNamespace(task='chip_tray', ideal=True, chip_grip_force=target_n))
    env = PandaChipEnv(cfg)
    if admittance is not None:
        env.force_grip.config['admittance_m_per_ns'] = admittance
    if finger_speed is not None:
        env.force_grip.config['max_finger_speed_m_s'] = finger_speed
    trace, peak = [], np.zeros(2)
    stage = 'approach'
    first_break = None
    original = env.after_physics_step
    grasp_rotation = rotation_matrix([0., 0., np.deg2rad(yaw_deg)]) @ env.target_rotation

    def after():
        nonlocal peak, first_break
        force = env.gripper_feedback_sample()['pad_normal_force_n']
        peak = np.maximum(peak, force)
        previous = len(env.fracture.events)
        original()
        if len(env.fracture.events) > previous and first_break is None:
            first_break = dict(stage=stage, force_n=force.tolist(), event=env.fracture.events[previous])

    env.after_physics_step = after

    def tick(delta=np.zeros(3)):
        action = env.hold_action()
        action[:3] = delta
        turn = rotation_vector(env.target_rotation.T @ grasp_rotation)
        action[3:6] = turn*min(1., .01/max(np.linalg.norm(turn), 1e-12))
        _, _, _, _, row = env.step(action)
        sample = env.gripper_feedback_sample()
        trace.append(dict(time_s=float(env.data.time), stage=stage,
                          force_n=sample['pad_normal_force_n'].tolist(),
                          target_n=sample['target_n'], servo_closure=sample['servo_target_closure'],
                          measured_closure=env.gripper_fraction(),
                          chip_contact_n=[row['normal_force_n'], row['passive_normal_force_n']],
                          chip_contact_loads=chip_contact_loads(env),
                          chip_center_m=center().tolist(), tcp_m=env.data.site_xpos[env.tcp_site].tolist(),
                          broken_bonds=row['chip_broken_bonds']))
        return first_break is None

    def center():
        return np.average(env.data.xipos[env.fracture.body_ids], axis=0,
                          weights=env.model.body_mass[env.fracture.body_ids])

    for target in ([-1.7822603, .13180272, 1.27065], [-1.7822603, .13180272, grasp_z]):
        for _ in range(450):
            delta = np.asarray(target)-env.target_xyz
            delta *= min(1., .0005/max(np.linalg.norm(delta), 1e-12))
            if not tick(delta):
                break
        if first_break:
            break

    start_z = center()[2]
    if not first_break:
        stage = 'close'
        env.command_gripper(True)
        for _ in range(round(closing_seconds*env.cfg['sample_hz'])):
            if not tick():
                break
    grasp_sample = env.gripper_feedback_sample()
    if not first_break:
        stage = 'lift'
        for _ in range(200):
            if not tick(np.array([0., 0., .00025])):
                break
    if not first_break:
        stage = 'hold'
        for _ in range(200):
            if not tick():
                break
    result = dict(target_n=target_n, grasp_z_m=grasp_z, yaw_deg=yaw_deg, closing_seconds=closing_seconds,
                  first_break=first_break, peak_contact_n=peak.tolist(),
                  grasp_force_n=grasp_sample['pad_normal_force_n'].tolist(),
                  final_force_n=env.gripper_feedback_sample()['pad_normal_force_n'].tolist(),
                  final_servo_closure=env.gripper_fraction(target=True),
                  center_lift_m=float(center()[2]-start_z), metrics=env.evaluation_metrics(),
                  warnings=env.data.warning.number.tolist(), controller=env.force_grip.config,
                  chip_mass_kg=float(env.model.body_mass[env.fracture.body_ids].sum()),
                  gravity_m_s2=env.model.opt.gravity.tolist(),
                  strength=env.cfg['brittle_chip'], trace=trace)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forces', type=float, nargs='+', default=[.04, .06, .08])
    parser.add_argument('--grasp-z', type=float, default=1.123)
    parser.add_argument('--closing-seconds', type=float, default=32.)
    parser.add_argument('--yaw-deg', type=float, default=0.)
    parser.add_argument('--admittance', type=float, help='Explicit experimental gain override [m/(N s)]')
    parser.add_argument('--finger-speed', type=float, help='Explicit target slew override [m/s per finger]')
    parser.add_argument('--output', type=Path, default=Path('outputs/diagnostics/chip_force_feedback'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for force in args.forces:
        result = run(force, args.grasp_z, args.closing_seconds, args.yaw_deg, args.admittance, args.finger_speed)
        (args.output/f'force_{force:g}.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k: v for k, v in result.items() if k not in ('trace', 'strength', 'warnings')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
