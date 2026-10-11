"""Compare the actual SpaceMouse control path, without a HID or learned policy.

Matched waypoints and input law, not identical axis samples: the scripted
operator feeds back the commanded target. A separate identical-input contact
test measures target accumulation and motion after centering the knob.
No object pose/force edits, actuator changes or success-rule overrides occur.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from wowskin_lab.paths import ROOT
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, response_configuration, view_translation
from wowskin_lab.teleop.spacemouse import SpaceMouseController, MouseSnapshot, HID_AXIS_SIGNS
from wowskin_lab.data.lerobot_recording import LeRobotSimRecorder, validate_dataset


def snapshot(controller, velocity, tick, presses=()):
    # Invert the documented camera/HID transform and deadzone for a scripted
    # operator. Controller.action still performs every normal input check.
    basis = np.column_stack([view_translation(v, 0.) for v in np.eye(3)])
    axes = np.r_[basis.T @ velocity/controller.speed, np.zeros(3)] * HID_AXIS_SIGNS
    raw = np.sign(axes)*(controller.deadzone+(1-controller.deadzone)*np.abs(axes))
    return MouseSnapshot(raw, np.full(2, tick), 0, presses)


def run(directory, limited, seed, speed, record=True, stress=False):
    directory.mkdir(parents=True, exist_ok=False)
    cfg = response_configuration(SimpleNamespace(task='pick_cube', skin_dof='trilinear',
        ideal=True, seed=seed, no_cartesian_limits=not limited,
        magnetic_fit=ROOT/'configs/magnetic_fit_pollen_nonslip.json'))
    cfg['randomize_cube'] = seed != 0
    env = PandaPickCubeTactileEnv(cfg, pollen=False)
    env.reset(seed=seed)
    ctl = SpaceMouseController()
    ctl.action(env, snapshot(ctl, np.zeros(3), 0.), 0., now=0.)
    cube = env.data.xpos[env.probe_body].copy()
    dest = env.pick_place.config['target_center_xy_m']
    phases = [(1., 'baseline', env.target_xyz.copy(), 0),
              (3., 'align', [*cube[:2], .185], 0),
              (5., 'approach', [*cube[:2], .129], 0),
              (6.5, 'close', [*cube[:2], .129], 1),
              (9., 'lift', [*cube[:2], .229], 1),
              (10., 'hold', [*cube[:2], .229], 1),
              (14., 'transfer', [*dest, .229], 1),
              (16.5, 'lower', [*dest, .129], 1),
              (18., 'release', [*dest, .129], 0),
              (20., 'retreat', [*dest, .185], 0)]
    recorder = LeRobotSimRecorder(env, directory, source='scripted_spacemouse_control_comparison') if record else None
    rows, error, saved, trace = [], None, None, []
    start = time.monotonic()
    last_grip, previous = 0, None
    try:
        if recorder:
            recorder.start()
        for tick in range(2000 if not stress else 500):
            t = tick/100.
            if stress:
                # Empty open jaws pushed into the table; same raw axes/time
                # in both profiles, then centered. No disabled contacts.
                phase = 'push_table' if t < 3. else 'centered' if t < 3.5 else 'retreat'
                velocity = np.array([0., 0., -.12 if t < 3. else 0. if t < 3.5 else .12])
                target, grip = env.target_xyz.copy(), 0
            else:
                _, phase, target, grip = next(p for p in phases if t < p[0])
                delta = np.asarray(target)-env.target_xyz
                velocity = 8*delta
                velocity *= min(1., speed/max(np.linalg.norm(velocity), 1e-12))
                if np.linalg.norm(delta) < .00015:
                    velocity[:] = 0.
            presses = (() if grip == last_grip else ('close' if grip else 'open',))
            action = ctl.action(env, snapshot(ctl, velocity, t, presses), 0., now=t)
            last_grip = grip
            row = env.step(action)[-1]
            if recorder:
                recorder.step(row)
            rows.append(row)
            trace.append(np.r_[t, velocity, env.target_xyz, row['tcp_xyz_m'],
                target, row['normal_force_n'], row['passive_normal_force_n']])
            if phase != previous:
                print(directory.name, phase, 'lift', round(row['lift_m'], 4), flush=True)
                previous = phase
    except Exception as exc:
        error = str(exc)
        if recorder:
            recorder.abort(error)
    finally:
        if recorder:
            if recorder.active:
                saved = recorder.finish('success' if env.success and not error else 'failure')
            recorder.close()
    a = np.asarray(trace)
    result = dict(limits_enabled=limited, seed=seed, speed_m_s=speed, stress=stress,
        success=bool(env.success), error=error, samples=len(rows), wall_s=time.monotonic()-start,
        max_lift_m=max((r['lift_m'] for r in rows), default=0.),
        min_volume_ratio=min((min(r['min_volume_ratio'], r['passive_min_volume_ratio']) for r in rows), default=1.),
        final_cube_m=env.data.xpos[env.probe_body].tolist(),
        max_target_error_m=float(np.linalg.norm(a[:, 4:7]-a[:, 7:10], axis=1).max()) if len(a) else None,
        warnings=env.data.warning.number.tolist(), saved=str(saved) if saved else None,
        task_evaluation=env.task_evaluation())
    if stress and len(a) > 300:
        result['motion_after_center_m'] = float(np.linalg.norm(a[300:350, 7:10]-a[299, 7:10], axis=1).max())
    if stress and len(a) > 450:
        result['upward_retreat_in_1s_m'] = float(a[449, 9]-a[349, 9])
    if saved and env.success:
        result['validation'] = validate_dataset(directory/'pick_cube/lerobot')
    np.savez_compressed(directory/'control_trace.npz', values=a,
        columns=np.array(['time_s','vx','vy','vz','target_x','target_y','target_z',
                          'actual_x','actual_y','actual_z','waypoint_x','waypoint_y','waypoint_z',
                          'force_left_n','force_right_n']))
    (directory/'report.json').write_text(json.dumps(result, indent=2, default=str)+'\n')
    print(json.dumps(result, default=str), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--speed', type=float, default=.08)
    p.add_argument('--no-record', action='store_true')
    p.add_argument('--stress', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for seed in args.seeds:
        for limited in (True, False):
            results.append(run(args.output/f'{"limits_on" if limited else "limits_off"}_seed{seed}',
                               limited, seed, args.speed, not args.no_record, args.stress))
            (args.output/'report.json').write_text(json.dumps(results, indent=2, default=str)+'\n')


if __name__ == '__main__':
    main()
