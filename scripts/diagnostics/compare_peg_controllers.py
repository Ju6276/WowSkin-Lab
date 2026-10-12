"""Same-scene controller comparison; does not change teleoperation defaults.

Runs local IK/joint PD and the pinned SERL++ Cartesian controller, with its
default gains, on the same physically grasped peg. Shared target slew, actuator
limits, gripper, geometry, contact solver and success test remain unchanged.
Only the upstream mj_fullM call is adapted for the installed MuJoCo API.
This is a default-gain comparison, not a tuned controller benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wowskin_lab.paths import ROOT
from wowskin_lab.envs.pick_cube import response_configuration, rotation_matrix, rotation_vector
from wowskin_lab.envs.peg_insert import PandaPegInsertEnv
from wowskin_lab.teleop.target_limits import straight_step


def run(kind, scenario, output, target_lead_m):
    cfg = response_configuration(SimpleNamespace(task='peg_insert', skin_dof='trilinear',
        ideal=True, seed=0, gripper='panda',
        arm_controller='cartesian_impedance' if kind == 'serl_cartesian' else 'ik_pd',
        magnetic_fit=ROOT/'configs/magnetic_fit_pollen_nonslip.json'))
    env = PandaPegInsertEnv(cfg)
    m, d = env.model, env.data
    # Upstream uses a shared qpos/qvel index array; verify this model permits it.
    np.testing.assert_array_equal(env.arm_q, env.arm_v)
    np.testing.assert_allclose(d.site_xmat[env.tcp_site].reshape(3, 3),
                               d.xmat[env.hand].reshape(3, 3), atol=1e-9)
    initial_position = env.target_xyz.copy()
    initial_rotation = env.target_rotation.copy()
    initial_state_sha256 = hashlib.sha256(d.qpos.tobytes()+d.qvel.tobytes()).hexdigest()
    original_control = env.control
    controller_ms, samples, error = [], [], None
    peak_normal, peak_net, peak_torque_fraction = 0., 0., 0.
    hole_geoms = set(np.flatnonzero(m.geom_bodyid == m.body('hole').id))

    def control(mode=None):
        # Keep the original gripper and all force application logic intact.
        start = time.perf_counter()
        original_control(mode)
        controller_ms.append(1000*(time.perf_counter()-start))

    def after_physics():
        nonlocal peak_normal, peak_net, peak_torque_fraction
        normal, net = 0., np.zeros(3)
        wrench = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if not ((int(c.geom[0]) in env.contact_geoms and int(c.geom[1]) in hole_geoms)
                    or (int(c.geom[1]) in env.contact_geoms and int(c.geom[0]) in hole_geoms)):
                continue
            mujoco.mj_contactForce(m, d, i, wrench)
            normal += max(0., wrench[0])
            net += (1 if int(c.geom[1]) in env.contact_geoms else -1)*c.frame.reshape(3, 3).T@wrench[:3]
        peak_normal = max(peak_normal, float(normal))
        peak_net = max(peak_net, float(np.linalg.norm(net)))
        limits = np.max(np.abs(m.actuator_ctrlrange[env.arm_controls]), axis=1)
        peak_torque_fraction = max(peak_torque_fraction,
            float(np.max(np.abs(d.ctrl[env.arm_controls])/limits)))

    env.control = control
    env.after_physics_step = after_physics
    trace = []
    start = time.perf_counter()
    try:
        for tick in range(700 if scenario == 'free_motion' else 600):
            t = tick/100.
            hold = False
            if scenario == 'free_motion':
                # Three 20 mm translations and a 5 degree pitch, with holds.
                times = np.array([0., .5, 1.5, 2., 3., 3.5, 4.5, 5., 6., 7.])
                offsets = np.array([[0,0,0], [0,0,0], [.02,0,0], [.02,0,0],
                    [.02,.02,0], [.02,.02,0], [.02,.02,.02], [.02,.02,.02],
                    [0,0,0], [0,0,0]])
                env.target_xyz = initial_position+np.array([np.interp(t,times,offsets[:,i]) for i in range(3)])
                angle = np.interp(t, times, [0,0,0,0,0,0,5,5,0,0])
                env.target_rotation = initial_rotation@rotation_matrix([0., np.deg2rad(angle), 0.])
                hold = any(lo <= t < hi for lo,hi in [(1.7,2.), (3.2,3.5), (4.7,5.), (6.7,7.)])
            else:
                # Identical object-feedback waypoint law for both controllers.
                # This scripted operator uses simulation object pose, not a policy.
                socket = d.xpos[m.body('hole').id].copy()
                target = socket.copy()
                if scenario == 'offset_3mm':
                    target[0] += .003
                target[2] = (.22 if tick < 180 else socket[2]+
                    env.peg_score.config['tip_local_m'][2]-
                    env.peg_score.config['socket_entry_local_z_m']-.025)
                increment = (target-d.xpos[env.probe_body])*.1
                increment *= min(1., .0005/max(np.linalg.norm(increment), 1e-12))
                if target_lead_m > 0:
                    increment = straight_step(env.target_xyz, increment,
                        d.site_xpos[env.tcp_site], np.array([[-np.inf]*3,[np.inf]*3]), target_lead_m)
                env.target_xyz += increment

            row = env.step(env.hold_action())[-1]
            position_error = float(np.linalg.norm(env.servo_xyz-d.site_xpos[env.tcp_site]))
            orientation_error = float(np.linalg.norm(rotation_vector(
                env.servo_rotation@d.site_xmat[env.tcp_site].reshape(3,3).T)))
            trace.append(np.r_[d.time, env.servo_xyz, d.site_xpos[env.tcp_site],
                position_error, orientation_error, row['peg_depth_m'],
                row['peg_lateral_error_m'], row['peg_orientation_error_deg'],
                row['normal_force_n'], row['passive_normal_force_n'],
                row['min_volume_ratio'], row['passive_min_volume_ratio'],
                row['success'], hold, bool(row.get('tactile_valid', True))])
            samples.append(row)
            if tick % 100 == 99:
                print(kind, scenario, tick+1, 'error_mm', round(position_error*1000,3),
                      'depth_mm', round(row['peg_depth_m']*1000,2), flush=True)
            if scenario != 'free_motion' and env.success:
                break
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    elapsed = time.perf_counter()-start
    a = np.asarray(trace)
    rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else None
    hold_mask = a[:,17].astype(bool) if len(a) else np.zeros(0, bool)
    report = dict(controller=kind, scenario=scenario, steps=len(a), success=bool(env.success),
        error=error, simulated_s=float(d.time), wall_s=elapsed,
        realtime_factor=float(d.time/elapsed), timestep_s=float(m.opt.timestep),
        initial_state_sha256=initial_state_sha256,
        insertion_target_lead_m=target_lead_m,
        controller_update_hz=1/m.opt.timestep, target_update_hz=cfg['sample_hz'],
        position_tracking_rmse_mm=rms(a[:,7]*1000) if len(a) else None,
        orientation_tracking_rmse_deg=rms(np.rad2deg(a[:,8])) if len(a) else None,
        settled_position_rmse_mm=rms(a[hold_mask,7]*1000) if len(a) else None,
        settled_orientation_rmse_deg=rms(np.rad2deg(a[hold_mask,8])) if len(a) else None,
        final_depth_mm=float(a[-1,9]*1000) if len(a) else None,
        peak_peg_socket_normal_sum_n=peak_normal, peak_peg_socket_net_force_n=peak_net,
        peak_actuator_limit_fraction=peak_torque_fraction,
        controller_call_median_ms=float(np.median(controller_ms)) if controller_ms else None,
        min_skin_volume_ratio=float(a[:,14:16].min()) if len(a) else None,
        invalid_tactile_samples=int((a[:,18] == 0).sum()) if len(a) else None,
        warnings=d.warning.number.tolist(), evaluation=env.task_evaluation(),
        notes=['Same initial physical grasp settled with local PD in both runs.',
               'Cartesian gains are upstream defaults; nullspace reference is the initial joint posture.',
               'Both controllers use the shared target slew; Cartesian torque does not run runtime IK.',
               'Insertion uses the same scripted object-feedback operator; free motion uses identical timed targets.',
               'Insertion uses the direction-preserving target lead rule; 0 disables it.',
               'Free-motion timed targets have no lead clamp so the requested path is identical for both controllers.',
               'No force, friction, grasp weld, geometry or success threshold changes.',
               'Controller timing excludes target solving and includes gripper bookkeeping in both modes.',
               'Single deterministic run per case, no claim of benchmark success rate or real force calibration.'])
    np.savez_compressed(output/f'{kind}_{scenario}.npz', values=a, columns=np.array([
        'time_s','target_x_m','target_y_m','target_z_m','tcp_x_m','tcp_y_m','tcp_z_m',
        'position_error_m','orientation_error_rad','depth_m','lateral_m','angle_deg',
        'left_normal_n','right_normal_n','left_volume_ratio','right_volume_ratio',
        'success','settled_hold','tactile_valid']))
    (output/f'{kind}_{scenario}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('evaluation','notes')}), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scenarios', nargs='+', choices=['free_motion','insert','offset_3mm'],
                   default=['free_motion','insert','offset_3mm'])
    p.add_argument('--target-lead-m', type=float, default=.006,
                   help='Insertion-only target lead per axis; existing keyboard/SpaceMouse default is .006, 0 for stress test')
    args = p.parse_args()
    if args.target_lead_m < 0:
        p.error('--target-lead-m must be nonnegative')
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = json.loads((ROOT/'assets/reference/serl_controller/manifest.json').read_text())
    report = dict(engine=mujoco.__version__, upstream=provenance, runs=[])
    for scenario in args.scenarios:
        for kind in ('local_ik_pd','serl_cartesian'):
            report['runs'].append(run(kind, scenario, args.output, args.target_lead_m))
            (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if any(run['error'] is not None for run in report['runs']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
