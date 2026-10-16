"""Scripted cube, feeding and single-clip collection for the data routes.

The controllers use simulator object poses to plan demonstration actions;
policy inputs contain rendered cameras, measured robot state and tactile measurements.
This is scripted simulation data, not human teleoperation or paper results.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from wowskin_lab.data.learning_recording import LearningSimRecorder
from wowskin_lab.data.recording import require_training_tactile
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, response_configuration


def collect(output, episodes=2, seed=0, randomize=False, *, magnetic_fit=None,
            reskin_model=None, reskin_extrapolate=False, ideal=False,
            record_format='lerobot', instruction=None, record_depth=True, sensor_backend=None,
            task='pick_cube', cable_layout=None, clip_model='rigid',
            physics_backend='cpu', gpu_disable_multiccd=False, gpu_disable_nativeccd=False,
            gpu_disable_noslip=False, gpu_solver_iterations=None, gpu_solver_tolerance=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new collection directory: {output}')
    if episodes < 1:
        raise ValueError('episodes must be positive')
    if clip_model not in ('rigid', 'elastic_beam'):
        raise ValueError('Unknown clip model')
    if clip_model != 'rigid' and task != 'cable':
        raise ValueError('--clip-model requires --task cable')
    if sum(value is not None for value in (magnetic_fit, reskin_model, sensor_backend)) != 1:
        raise ValueError('Select exactly one --magnetic-fit, --reskin-model or --sensor-backend')
    if sensor_backend is not None and sensor_backend not in ('tacto', 'digit_fots', 'gelsight'):
        raise ValueError('Unknown optical sensor backend')
    if sensor_backend and reskin_extrapolate:
        raise ValueError('--reskin-extrapolate requires the ReSkin backend')
    if task != 'pick_cube':
        if randomize:
            raise ValueError('--randomize currently only applies to pick_cube')
        from .scripted_tasks import collect_task
        return collect_task(output, episodes, seed, task=task, cable_layout=cable_layout,
            sensor_backend=sensor_backend, ideal=ideal, record_format=record_format,
            instruction=instruction, record_depth=record_depth, clip_model=clip_model,
            physics_backend=physics_backend, gpu_disable_multiccd=gpu_disable_multiccd,
            gpu_disable_nativeccd=gpu_disable_nativeccd, gpu_disable_noslip=gpu_disable_noslip,
            gpu_solver_iterations=gpu_solver_iterations, gpu_solver_tolerance=gpu_solver_tolerance)
    if cable_layout is not None:
        raise ValueError('--cable-layout requires --task cable')
    cfg = response_configuration(SimpleNamespace(task='pick_cube', seed=seed,
        ideal=ideal, sensor_backend=sensor_backend,
        magnetic_fit=Path(magnetic_fit) if magnetic_fit else None,
        reskin_model=Path(reskin_model) if reskin_model else None,
        reskin_extrapolate=reskin_extrapolate, physics_backend=physics_backend,
        gpu_disable_multiccd=gpu_disable_multiccd,
        gpu_disable_nativeccd=gpu_disable_nativeccd, gpu_disable_noslip=gpu_disable_noslip,
        gpu_solver_iterations=gpu_solver_iterations, gpu_solver_tolerance=gpu_solver_tolerance))
    cfg['randomize_cube'] = bool(randomize)
    env = PandaPickCubeTactileEnv(cfg, pollen=False)
    # Validate the first observation before creating output files or cameras.
    # A complete pick/lower/release includes unloaded states; never fill them.
    require_training_tactile(env)
    output.mkdir(parents=True)
    controller_source = Path(__file__).read_bytes()
    (output/'controller_source.py').write_bytes(controller_source)
    result = dict(source='scripted_simulation_acquisition', human_demonstrations=False,
                  controller_sha256=hashlib.sha256(controller_source).hexdigest(),
                  controller_source_file='controller_source.py',
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
            env.cfg['seed'] = seed+episode
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
            max_lift, min_volume, grasped_hold = 0., None if env.is_optical else 1., 0
            previous_phase = None
            for step in range(1150):
                elapsed = step/100.
                _, phase, target, grip = next(p for p in phases if elapsed < p[0])
                env.phase = phase
                command = np.r_[np.clip(np.asarray(target)-env.target_xyz, -.005, .005), np.zeros(3), grip]
                row = env.step(command)[-1]
                recorder.step(row)
                max_lift = max(max_lift, float(row['lift_m']))
                if not env.is_optical:
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
    from wowskin_lab.core.stepping import add_arguments
    add_arguments(parser)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--randomize', action='store_true')
    parser.add_argument('--task', choices=('pick_cube', 'feeding', 'cable'), default='pick_cube')
    parser.add_argument('--cable-layout', choices=('single_clip',), help='Required for --task cable')
    parser.add_argument('--clip-model', choices=('rigid', 'elastic_beam'), default='rigid',
                        help='Cable clip physics: rigid (default) or passive elastic upper and lower lips')
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument('--magnetic-fit', type=Path)
    backend.add_argument('--reskin-model', type=Path)
    backend.add_argument('--sensor-backend', choices=('tacto', 'digit_fots', 'gelsight'))
    parser.add_argument('--reskin-extrapolate', action='store_true',
                        help='Explicit single-probe preview; does not supply missing unloaded responses')
    parser.add_argument('--ideal', action='store_true')
    parser.add_argument('--record-format', choices=('lerobot', 'dataset'), default='lerobot')
    parser.add_argument('--instruction', help='Language instruction to store verbatim for this task')
    parser.add_argument('--record-depth', action=argparse.BooleanOptionalAction, default=True,
                        help='Record lossless RGB-aligned metric depth from all configured cameras')
    args = parser.parse_args(argv)
    result = collect(args.output, args.episodes, args.seed, args.randomize,
        magnetic_fit=args.magnetic_fit, reskin_model=args.reskin_model,
        reskin_extrapolate=args.reskin_extrapolate, ideal=args.ideal,
        record_format=args.record_format, instruction=args.instruction, record_depth=args.record_depth,
        sensor_backend=args.sensor_backend, task=args.task, cable_layout=args.cable_layout,
        clip_model=args.clip_model, physics_backend=args.physics_backend,
        gpu_disable_multiccd=args.gpu_disable_multiccd, gpu_disable_nativeccd=args.gpu_disable_nativeccd,
        gpu_disable_noslip=args.gpu_disable_noslip, gpu_solver_iterations=args.gpu_solver_iterations,
        gpu_solver_tolerance=args.gpu_solver_tolerance)
    print(json.dumps(result, indent=2))
    if result['successes'] == 0:
        raise SystemExit('No successful trajectories; failure recordings were preserved and excluded')


if __name__ == '__main__':
    main()
