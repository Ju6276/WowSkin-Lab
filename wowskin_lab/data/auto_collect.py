"""Scripted pick/lower/release collection for testing the official data routes.

The controller knows the initial simulation object position; learned datasets
contain only rendered cameras, measured robot state and tactile measurements.
This is scripted simulation data, not human teleoperation or paper results.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from wowskin_lab.core.physics import configuration
from wowskin_lab.paths import ROOT
from wowskin_lab.data.learning_recording import LearningSimRecorder
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv


def collect(output, episodes=2, seed=0, randomize=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new collection directory: {output}')
    if episodes < 1:
        raise ValueError('episodes must be positive')
    output.mkdir(parents=True)
    cfg = configuration(seed=seed)
    cfg["magnetic_fit"] = str(ROOT/"configs/magnetic_fit_pollen_nonslip.json")
    cfg['randomize_cube'] = bool(randomize)
    env = PandaPickCubeTactileEnv(cfg, pollen=False)
    result = dict(source='scripted_simulation_acquisition', human_demonstrations=False,
                  controller_uses_initial_simulator_object_position=True,
                  object_ground_truth_exported_to_policy=False,
                  calibrated=False, episodes=[])
    report = output/'collection_report.json'
    with LearningSimRecorder(env, output, source='scripted_pick_lower_release') as recorder:
        for episode in range(episodes):
            env.reset(seed=seed+episode)
            cube = env.data.xpos[env.probe_body].copy()
            recorder.start()
            start = float(env.data.time)
            original_pose = env.target_xyz.copy()
            phases = [
                (1., 'baseline', original_pose, 0.),
                (2., 'align', [cube[0], cube[1], .185], 0.),
                (3., 'approach', [cube[0], cube[1], .129], 0.),
                (4., 'close', [cube[0], cube[1], .129], 1.),
                (5.5, 'lift', [cube[0], cube[1], .229], 1.),
                (6.5, 'hold', [cube[0], cube[1], .229], 1.),
                (8., 'lower', [cube[0], cube[1], .129], 1.),
                (9., 'release', [cube[0], cube[1], .129], 0.),
                (10., 'retreat', [cube[0], cube[1], .185], 0.),
            ]
            max_lift, min_volume, grasped_hold = 0., 1., 0
            previous_phase = None
            for step in range(1000):
                elapsed = step/100.
                _, phase, target, grip = next(p for p in phases if elapsed < p[0])
                env.phase = phase
                command = np.r_[np.clip(np.asarray(target)-env.target_xyz, -.005, .005), np.zeros(3), grip]
                row = env.step(command)[-1]
                recorder.step(row)
                max_lift = max(max_lift, float(row['lift_m']))
                min_volume = min(min_volume, float(row['min_volume_ratio']), float(row['passive_min_volume_ratio']))
                if phase == 'hold' and row['grasped'] and row['lift_m'] > .08:
                    grasped_hold += 1
                if phase != previous_phase:
                    print(f'episode {episode+1}/{episodes}: {phase}, lift={row["lift_m"]:.3f} m', flush=True)
                    previous_phase = phase
            success = bool(max_lift > .08 and grasped_hold >= 80 and
                           row['lift_m'] < .005 and min_volume > .5 and
                           not np.any(env.data.warning.number))
            saved = recorder.finish('success' if success else 'failure')
            result['episodes'].append(dict(seed=seed+episode, success=success,
                duration_s=float(row['time_s']-start), max_lift_m=max_lift,
                held_grasp_samples=grasped_hold, final_lift_m=float(row['lift_m']),
                min_volume_ratio=min_volume, paths={k:str(v) for k,v in saved.items()}))
            result['successes'] = sum(e['success'] for e in result['episodes'])
            report.write_text(json.dumps(result, indent=2)+'\n')
            print(f'Saved episode {episode+1}: success={success}; report={report}', flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--randomize', action='store_true')
    args = parser.parse_args(argv)
    result = collect(args.output, args.episodes, args.seed, args.randomize)
    print(json.dumps(result, indent=2))
    if result['successes'] == 0:
        raise SystemExit('No successful trajectories; failure recordings were preserved and excluded')


if __name__ == '__main__':
    main()
