"""Scripted pick/move/place collection for testing the official data routes.

The controller knows the initial simulation object position; learned datasets
contain only rendered cameras, measured robot state and tactile measurements.
This is scripted simulation data, not human teleoperation or paper results.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from wowskin_lab.data.learning_recording import LearningSimRecorder
from wowskin_lab.data.recording import require_training_tactile
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, response_configuration


def collect(output, episodes=2, seed=0, randomize=False, *, magnetic_fit=None,
            reskin_model=None, reskin_extrapolate=False, skin_dof='trilinear', ideal=False,
            record_format='lerobot', instruction=None, record_depth=True):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new collection directory: {output}')
    if episodes < 1:
        raise ValueError('episodes must be positive')
    if (magnetic_fit is None) == (reskin_model is None):
        raise ValueError('Select exactly one --magnetic-fit or --reskin-model; no implicit backend or extrapolation')
    cfg = response_configuration(SimpleNamespace(task='pick_cube', seed=seed,
        skin_dof=skin_dof, ideal=ideal,
        magnetic_fit=Path(magnetic_fit) if magnetic_fit else None,
        reskin_model=Path(reskin_model) if reskin_model else None,
        reskin_extrapolate=reskin_extrapolate))
    cfg['randomize_cube'] = bool(randomize)
    env = PandaPickCubeTactileEnv(cfg, pollen=False)
    # Validate the first observation before creating output files or cameras.
    # A complete pick/lower/release includes unloaded states; never fill them.
    require_training_tactile(env)
    output.mkdir(parents=True)
    result = dict(source='scripted_simulation_acquisition', human_demonstrations=False,
                  controller_uses_initial_simulator_object_position=True,
                  object_ground_truth_exported_to_policy=False,
                  calibrated=False, episodes=[])
    report = output/'collection_report.json'
    if record_format not in ('dataset', 'lerobot'):
        raise ValueError(record_format)
    recorder_type = LearningSimRecorder
    if record_format == 'lerobot':
        from .lerobot_recording import LeRobotSimRecorder
        recorder_type = LeRobotSimRecorder
    with recorder_type(env, output, source='scripted_pick_move_place',
                       instruction=instruction, record_depth=record_depth) as recorder:
        for episode in range(episodes):
            env.reset(seed=seed+episode)
            cube = env.data.xpos[env.probe_body].copy()
            destination = np.asarray(env.pick_place.config['target_center_xy_m'])
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
                (8., 'transfer', [*destination, .229], 1.),
                (9.5, 'lower', [*destination, .129], 1.),
                (10.5, 'release', [*destination, .129], 0.),
                (11.5, 'retreat', [*destination, .185], 0.),
            ]
            max_lift, min_volume, grasped_hold = 0., 1., 0
            previous_phase = None
            for step in range(1150):
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
            placement_error = float(np.linalg.norm(env.data.xpos[env.probe_body][:2]-destination))
            success = bool(env.success and not np.any(env.data.warning.number))
            saved = recorder.finish('success' if success else 'failure')
            result['episodes'].append(dict(seed=seed+episode, success=success,
                duration_s=float(row['time_s']-start), max_lift_m=max_lift,
                held_grasp_samples=grasped_hold, final_lift_m=float(row['lift_m']),
                min_volume_ratio=min_volume, placement_error_m=placement_error,
                initial_cube_position_m=cube.tolist(), target_xy_m=destination.tolist(),
                final_cube_position_m=env.data.xpos[env.probe_body].tolist(),
                dataset_episode=str(saved), record_format=record_format,
                lerobot_episode=getattr(recorder, 'last_lerobot_episode', None)))
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
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument('--magnetic-fit', type=Path)
    backend.add_argument('--reskin-model', type=Path)
    parser.add_argument('--reskin-extrapolate', action='store_true',
                        help='Explicit single-probe preview; does not supply missing unloaded responses')
    parser.add_argument('--skin-dof', choices=('full', 'trilinear'), default='trilinear')
    parser.add_argument('--ideal', action='store_true')
    parser.add_argument('--record-format', choices=('lerobot', 'dataset'), default='lerobot')
    parser.add_argument('--instruction', help='Language instruction to store verbatim for this task')
    parser.add_argument('--record-depth', action=argparse.BooleanOptionalAction, default=True,
                        help='Record lossless RGB-aligned metric depth from all configured cameras')
    args = parser.parse_args(argv)
    result = collect(args.output, args.episodes, args.seed, args.randomize,
        magnetic_fit=args.magnetic_fit, reskin_model=args.reskin_model,
        reskin_extrapolate=args.reskin_extrapolate, skin_dof=args.skin_dof, ideal=args.ideal,
        record_format=args.record_format, instruction=args.instruction, record_depth=args.record_depth)
    print(json.dumps(result, indent=2))
    if result['successes'] == 0:
        raise SystemExit('No successful trajectories; failure recordings were preserved and excluded')


if __name__ == '__main__':
    main()
