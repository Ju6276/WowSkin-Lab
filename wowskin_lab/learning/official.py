"""Prepare upstream workspaces without replacing preprocessing or training.

The ViSk paper profile changes only the published two-sensor defaults to one.
Every change is recorded; policies, losses, normalization and action chunks
are preserved. Reference checkouts always remain untouched.
Preparation is not a successful reproduction or a working MuJoCo deployment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys

from wowskin_lab.paths import ROOT

ROUTES = ('visk', 'ftf-human', 'ftf-robot')
FOLDERS = {'visk': 'visuoskin', 'ftf': 'feel-the-force'}


def git(directory, *args):
    return subprocess.run(['git', '-C', str(directory), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def source(route):
    method = 'visk' if route == 'visk' else 'ftf'
    spec = json.loads((ROOT/'configs/upstream.json').read_text())[method]
    return ROOT/'third_party/official'/FOLDERS[method], spec


def verify_reference(route):
    directory, spec = source(route)
    if not directory.is_dir():
        raise FileNotFoundError('Run scripts/setup_upstream.py first')
    if git(directory, 'rev-parse', 'HEAD') != spec['revision']:
        raise ValueError(f'{directory}: revision differs from the pinned source')
    if git(directory, 'status', '--porcelain'):
        raise ValueError(f'{directory}: official source has local modifications')
    submodules = git(directory, 'submodule', 'status', '--recursive')
    for line in submodules.splitlines():
        if line.startswith(('+', '-', 'U')):
            raise ValueError(f'Uninitialized or changed submodule: {line}')
    return directory, spec, submodules


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracked_hashes(directory, prefix=''):
    result = {}
    for entry in git(directory, 'ls-files', '--stage', '-z').split('\0'):
        if not entry:
            continue
        mode, _, rest = entry.split(' ', 2)
        _, name = rest.split('\t', 1)
        relative = f'{prefix}{name}'
        if mode == '160000':
            result.update(tracked_hashes(directory/name, relative+'/'))
        else:
            result[relative] = digest(directory/name)
    return result


def prepare(route, destination, data_root, task, task_config=None, visk_profile='paper-single'):
    import yaml
    reference, spec, submodules = verify_reference(route)
    destination = Path(destination).resolve()
    data_root = Path(data_root).resolve()
    if destination.exists():
        raise FileExistsError(f'Use a new workspace; will not overwrite {destination}')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', task):
        raise ValueError('Task must be a simple directory/config name')
    task_text = None
    dataset_task = task
    if route != 'visk':
        task_path = Path(task_config) if task_config else reference/f'point_policy/cfgs/suite/task/franka_env/{task}.yaml'
        if not task_path.is_file():
            raise FileNotFoundError('FTF needs an existing official task config or --task-config; '
                                    'object point count must come from semantic annotations, not cube corners')
        task_text = task_path.read_text()
        task_data = yaml.safe_load(task_text)
        dataset_task = task_data['task_name']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', dataset_task):
            raise ValueError('Task config task_name must be a simple dataset directory name')
        if not isinstance(task_data['num_object_points'], int) or task_data['num_object_points'] < 1:
            raise ValueError('FTF task config needs a positive num_object_points')
        if task_data['object_labels'] != ['objects']:
            raise ValueError('The pinned converters use object_labels=[objects]; align annotations/config explicitly')
    hashes = tracked_hashes(reference)
    destination.mkdir(parents=True)
    code = destination/'source'
    # Copy exactly the tracked files, including upstream's committed bytecode;
    # do not include local caches or omit files when checking provenance.
    for name in hashes:
        target = code/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reference/name, target)
    configured = {}

    def write(relative, content, reason):
        path = code/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        configured[relative] = {'reason': reason, 'sha256': digest(path)}

    commands = []

    def command(stage, cwd, argv, requires):
        commands.append(dict(stage=stage, cwd=str(code/cwd), argv=argv, requires=requires))

    if route == 'visk':
        if visk_profile not in ('paper-single', 'published-default'):
            raise ValueError('Unknown ViSk profile')
        utils = (code/'utils.py').read_text()
        if utils.count('DATA_DIR = ""') != 1:
            raise ValueError('Unexpected upstream DATA_DIR definition')
        write('utils.py', utils.replace('DATA_DIR = ""', f'DATA_DIR = {str(data_root)!r}'),
              'README installation step 3: raw-data path only')
        write('cfgs/local_config.yaml', yaml.safe_dump({'root_dir': str(data_root)}),
              'README installation step 3: required local path config, absent upstream')
        if visk_profile == 'paper-single':
            reason = ('User requested paper IV-A: left AnySkin, right plain silicone; '
                      'one 15-channel sensor. No changes to encoders/losses/normalization/action chunks.')
            replacements = {
                'cfgs/suite/xarm_env_aa.yaml': ('aux_keys: ["sensor0","sensor1"]', 'aux_keys: ["sensor0"]'),
                'read_data/xarm_env_aa.py': ('self._num_anyskin_sensors = 2', 'self._num_anyskin_sensors = 1'),
                'envs/xarm-env/xarm_env/envs/robot_env.py': ('self.n_sensors = 2', 'self.n_sensors = 1'),
            }
            import difflib
            changes = []
            for name, (old, new) in replacements.items():
                before = (code/name).read_text()
                if before.count(old) != 1:
                    raise ValueError(f'Unexpected upstream sensor count in {name}')
                after = before.replace(old, new)
                write(name, after, reason)
                changes.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                               fromfile='a/'+name, tofile='b/'+name))
            (destination/'PAPER_SINGLE_SENSOR.patch').write_text(''.join(changes))
        task_override = 'suite.task.tasks=['+task+']'
        command('process', '', ['python', 'process_xarm_data.py', '-t', task],
                ['original XArm raw format only; MuJoCo recordings use `official process-visk-sim`',
                 '15-channel tactile (paper-single)' if visk_profile == 'paper-single' else '30-channel tactile (published-default)'])
        command('convert', '', ['python', 'convert_to_pkl.py', '-t', task], ['process output'])
        command('train', '', ['python', 'train_bc.py', task_override],
                ['official environment/dependencies and converted dataset'])
        # This is the original hardware evaluation, not a functioning sim adapter.
        command('evaluate', '', ['python', 'eval.py', task_override, 'bc_weight=CHECKPOINT_PATH'],
                ['checkpoint', 'configured original XArm services; Franka/MuJoCo adapter is not connected'])
        training_config = 'cfgs/config.yaml'
        dataset = data_root/'processed_data_pkl_aa'/f'{task}.pkl'
    else:
        points_path = 'point_policy/cfgs/suite/points_cfg.yaml'
        points = yaml.safe_load((code/points_path).read_text())
        points['root_dir'] = str(code)
        write(points_path, yaml.safe_dump(points, sort_keys=False), 'Fill only the upstream root_dir placeholder')
        # Training and evaluation otherwise retain the complete upstream configuration.
        for config in ('config.yaml', 'config_eval.yaml'):
            relative = 'point_policy/cfgs/'+config
            cfg = yaml.safe_load((code/relative).read_text())
            cfg['root_dir'] = str(code)
            cfg['data_dir'] = str(data_root/'processed_data_pkl/expert_demos')
            write(relative, yaml.safe_dump(cfg, sort_keys=False), 'Fill only root_dir and data_dir placeholders')
        relative = f'point_policy/cfgs/suite/task/franka_env/{task}.yaml'
        if not (code/relative).exists() or (code/relative).read_text() != task_text:
            write(relative, task_text, 'User-supplied task config; semantic point count is not inferred')
        branch = 'human' if route == 'ftf-human' else 'robot'
        suite = 'point_policy' if branch == 'human' else 'p3po'
        raw = data_root/'extracted_data'/dataset_task
        command('teleop', 'Franka-Teach', ['python', 'teleop.py', f'teleop_mode={branch}'],
                ['configured Franka-Teach cameras, tactile and VR/state services'])
        command('collect', 'Franka-Teach', ['python', 'collect_data.py', f'storage_path={raw}',
                'collect_img=true', 'collect_state=true', 'collect_reskin=true', 'demo_num=0'],
                ['teleop/services running; a new demo_num for each recording; camera/USB settings must be configured'])
        processing = 'point_policy/robot_utils/franka'
        command('process', processing, ['python', f'process_data_{branch}.py', '--data_dir', str(data_root),
                '--task_names', dataset_task], ['actual synchronized raw demonstrations'])
        base = ['python', f'convert_to_pkl_{branch}.py', '--data_dir', str(data_root),
                '--calib_path', str(code/'calib/calib.npy'), '--task_names', dataset_task]
        command('convert', processing, base, ['processed data', 'own camera calibration at source/calib/calib.npy'])
        command('annotate', processing, ['jupyter', 'notebook', 'label_points.ipynb'],
                ['set notebook paths to this task; manually label corresponding semantic object points in both views'])
        command('track', processing, base+['--process_points']+(['--smooth_tracks'] if branch == 'human' else []),
                ['CUDA, DIFT weights, CoTracker checkpoint', 'reference images/annotations under source/coordinates/TASK/',
                 'calibration; no MuJoCo object coordinates used'])
        if branch == 'human':
            command('retarget', processing, ['python', 'convert_pkl_human_to_robot.py', '--data_dir', str(data_root),
                    '--calib_path', str(code/'calib/calib.npy'), '--task_name', dataset_task, '--smooth_robot_tracks'],
                    ['tracked human-hand/object points and tactile from human demonstrations'])
        selection = [f'agent={suite}', f'suite={suite}', f'dataloader={suite}',
                     f'suite/task/franka_env={task}', 'suite.predict_force=true', 'suite.mask_force=false']
        selection += ['suite.subsample=3'] if branch == 'human' else ['suite.subsample=15', 'suite.binarize_gripper_data=true']
        command('train', 'point_policy', ['python', 'train.py', *selection, 'eval=false'],
                ['full official dependencies', 'tracked/retargeted dataset; upstream training defaults unchanged'])
        command('evaluate', 'point_policy', ['python', 'eval_point_track.py', *selection, 'eval=true',
                'suite.variable_desired_force=true', 'suite.force_controller=true',
                'suite.continuous_gripper=false', 'bc_weight=CHECKPOINT_PATH'],
                ['trained checkpoint', 'camera/tactile/robot services and verified force_match_tolerance',
                 'original hardware evaluation; MuJoCo interface is not connected'])
        training_config = 'point_policy/cfgs/config.yaml'
        dataset = data_root/'processed_data_pkl/expert_demos/franka_env'/f'{dataset_task}.pkl'
    manifest = dict(route=route, status='prepared_only_not_validated', upstream=spec,
                    submodules=submodules, task_config=task, dataset_task=dataset_task,
                    data_root=str(data_root), dataset=str(dataset), training_config=training_config,
                    original_sha256=hashes, configured_files=configured, commands=commands)
    if route == 'visk':
        manifest['visk_profile'] = visk_profile
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    lines = ['# Prepared official commands — not an end-to-end validation', '',
             'Use the corresponding upstream environment. Commands are listed separately; do not run hardware stages before configuring services.', '']
    for c in commands:
        lines.extend([f'## {c["stage"]}', '', 'Requires: '+'; '.join(c['requires']), '',
                      '```bash', 'cd '+shlex.quote(c['cwd']), shlex.join(c['argv']), '```', ''])
    (destination/'COMMANDS.md').write_text('\n'.join(lines))
    return dict(workspace=str(destination), status=manifest['status'], dataset=str(dataset),
                configured_files=configured, commands_file=str(destination/'COMMANDS.md'))


def verify_workspace(path):
    path = Path(path).resolve()
    manifest = json.loads((path/'manifest.json').read_text())
    _, spec, _ = verify_reference(manifest['route'])
    if manifest['upstream']['revision'] != spec['revision']:
        raise ValueError('Workspace refers to a different upstream revision')
    modified = []
    for name, expected in manifest['original_sha256'].items():
        expected = manifest['configured_files'].get(name, {}).get('sha256', expected)
        file = path/'source'/name
        if not file.is_file() or digest(file) != expected:
            modified.append(name)
    for name, configured in manifest['configured_files'].items():
        file = path/'source'/name
        if not file.is_file() or digest(file) != configured['sha256']:
            modified.append(name)
    return dict(workspace=str(path), source_matches_prepared_manifest=not modified,
                unexpected_changes=sorted(set(modified)), status='source_check_only_not_pipeline_validation')


def status():
    result = dict(platform=f'{platform.system()} {platform.machine()}', routes={},
                  reproduction_complete=False, simulation_policy_rollout='not_connected',
                  ftf_human_data='not_collected; user requires acquisition',
                  default_local_recorder='single sensor / 10 Hz / 128px / GT diagnostics; not official reproduction input')
    result['ftf_raw_acquisition'] = 'MuJoCo input -> original Franka-Teach writers -> original process_data_robot verified; visual tracking/training not completed'
    result['visk_profile'] = 'paper-single: one AnySkin, 15 channels; explicit three-line sensor-count patch in prepared workspace'
    result['visk_acquisition'] = 'visk-raw records four cameras at nominal 30 Hz and state/tactile at 100 Hz; process-visk-sim runs original PKL converter and loader validation'
    try:
        import torch
        result['compute'] = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.cuda.is_available())
    except ImportError:
        result['compute'] = dict(python=platform.python_version(), torch=None, cuda=False)
    for route in ROUTES:
        try:
            directory, spec, modules = verify_reference(route)
            result['routes'][route] = dict(source=str(directory), revision=spec['revision'],
                                           source_clean=True, submodules=modules, full_pipeline_validated=False)
        except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
            result['routes'][route] = dict(source_clean=False, error=str(exc), full_pipeline_validated=False)
    return result


def process_ftf_robot(data_root, task):
    """Run the original first preprocessing stage with local tool paths only."""
    reference, _, _ = verify_reference('ftf-robot')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', task):
        raise ValueError('Task must be a simple directory name')
    data_root = Path(data_root).resolve()
    if not (data_root/'extracted_data'/task).is_dir():
        raise FileNotFoundError(data_root/'extracted_data'/task)
    eligible = [p for p in (data_root/'extracted_data'/task).iterdir()
                if p.is_dir() and 'fail' not in p.name and 'ignore' not in p.name]
    if not eligible:
        raise ValueError('No eligible demonstrations: review recordings before preprocessing')
    if (data_root/'processed_data'/task).exists():
        raise FileExistsError('Processed task already exists; use the original interactive script to review append/overwrite')
    environment = os.environ.copy()
    teach = str(reference/'Franka-Teach')
    environment['PYTHONPATH'] = os.pathsep.join(filter(None, [teach, environment.get('PYTHONPATH')]))
    if shutil.which('ffmpeg') is None:
        import imageio_ffmpeg
        binary = ROOT/'.tools/bin/ffmpeg'
        binary.parent.mkdir(parents=True, exist_ok=True)
        if not binary.exists():
            binary.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
        environment['PATH'] = str(binary.parent)+os.pathsep+environment.get('PATH', '')
    command = [sys.executable, 'process_data_robot.py', '--data_dir', str(data_root), '--task_names', task]
    subprocess.run(command, cwd=reference/'point_policy/robot_utils/franka', env=environment,
                   stdin=subprocess.DEVNULL, check=True)
    return {'stage': 'official process_data_robot only', 'data_root': str(data_root),
            'task': task, 'visual_tracking_complete': False, 'training_ready': False}


def convert_ftf_robot_initial(data_root, task, calibration):
    """Run upstream conversion without visual tracking; this is NOT training data."""
    reference, spec, _ = verify_reference('ftf-robot')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', task):
        raise ValueError('Task must be a simple directory name')
    data_root, calibration = Path(data_root).resolve(), Path(calibration).resolve()
    if not (data_root/'processed_data'/task).is_dir():
        raise FileNotFoundError(data_root/'processed_data'/task)
    if not calibration.is_file():
        raise FileNotFoundError(calibration)
    output = data_root/'processed_data_pkl/expert_demos/franka_env'/f'{task}.pkl'
    if output.exists():
        raise FileExistsError(f'Will not append untracked observations to {output}')
    # The original converter writes here even when --process_points is absent,
    # but does not create the directory in that branch. No source patch needed.
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, 'convert_to_pkl_robot.py', '--data_dir', str(data_root),
               '--task_names', task, '--calib_path', str(calibration)]
    subprocess.run(command, cwd=reference/'point_policy/robot_utils/franka',
                   stdin=subprocess.DEVNULL, check=True)
    import pickle
    import numpy as np
    with output.open('rb') as stream:
        data = pickle.load(stream)  # Locally produced by the verified converter.
    episodes = []
    for observation in data['observations']:
        keys = ('pixels1', 'pixels2', 'cartesian_states', 'gripper_states', 'sensor_states')
        shapes = {key: list(observation[key].shape) for key in keys}
        if len({shape[0] for shape in shapes.values()}) != 1:
            raise ValueError(f'FTF converted lengths differ: {shapes}')
        if any(not np.isfinite(observation[key]).all() for key in keys[2:]):
            raise ValueError('FTF converted state contains nonfinite values')
        if any('tracks' in key for key in observation):
            raise ValueError('Unexpected tracks in initial conversion')
        episodes.append(shapes)
    if not episodes:
        raise ValueError('Original converter produced no observations')
    report = dict(stage='original converter without --process_points',
                  upstream_revision=spec['revision'], command=command,
                  output=str(output), calibration=str(calibration), episodes=episodes,
                  visual_tracking_complete=False, training_ready=False,
                  next_step='Semantic annotation, then original DIFT/CoTracker on CUDA. '
                            'Archive this initial PKL before tracked conversion: upstream appends to existing files.',
                  upstream_note='The unused sensor_history parser hardcodes reshape(-1,30). '
                                'The original sensor_states output uses the first magnetometer only; neither was changed.')
    output.with_suffix('.initial_conversion.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def prepare_ftf_annotation(data_root, task, output):
    """Copy the original notebook, configuring only task and local file paths."""
    reference, spec, _ = verify_reference('ftf-robot')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', task):
        raise ValueError('Task must be a simple directory name')
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    dataset = data_root/'processed_data_pkl/expert_demos/franka_env'/f'{task}.pkl'
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if output.exists():
        raise FileExistsError(f'Will not overwrite annotation work: {output}')
    source = reference/'point_policy/robot_utils/franka/label_points.ipynb'
    notebook = json.loads(source.read_text())
    coordinates = data_root/'coordinates'/task
    replacements = {
        'task_name = "0325_pick_bread_on_plate"': f'task_name = {task!r}',
        'pickle_path = f"path/to/baton-force/Franka-Teach/data/processed_data_pkl/{task_name}.pkl"':
            f'pickle_path = {str(dataset)!r}',
        'coordinates_path = f"../../../coordinates/{task_name}"':
            f'coordinates_path = {str(coordinates)!r}',
    }
    counts = {old: 0 for old in replacements}
    for cell in notebook['cells']:
        text = ''.join(cell.get('source', []))
        for old, new in replacements.items():
            counts[old] += text.count(old)
            text = text.replace(old, new)
        cell['source'] = text.splitlines(True)
    if any(count != 1 for count in counts.values()):
        raise ValueError('Upstream notebook configuration changed; inspect before preparing')
    output.mkdir(parents=True)
    destination = output/'label_points.ipynb'
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n')
    report = dict(upstream_revision=spec['revision'], source=str(source), source_sha256=digest(source),
                  notebook=str(destination), notebook_sha256=digest(destination),
                  changed_settings=list(replacements.values()),
                  coordinates=str(coordinates), points_annotated=False,
                  visual_tracking_complete=False, training_ready=False)
    (output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('status')
    make = sub.add_parser('prepare', help='Copy official code and fill path/task config; never train a substitute model')
    make.add_argument('route', choices=ROUTES)
    make.add_argument('--workspace', type=Path, required=True)
    make.add_argument('--data-root', type=Path, required=True)
    make.add_argument('--task', required=True, help='ViSk task name / FTF task-config stem (dataset name read from YAML)')
    make.add_argument('--task-config', type=Path)
    make.add_argument('--visk-profile', choices=['paper-single', 'published-default'], default='paper-single')
    verify = sub.add_parser('verify')
    verify.add_argument('workspace', type=Path)
    process = sub.add_parser('process-ftf-robot', help='Run original raw -> video/CSV preprocessing; no point/label substitute')
    process.add_argument('--data-root', type=Path, required=True)
    process.add_argument('--task', required=True)
    initial = sub.add_parser('convert-ftf-robot-initial', help='Original untracked image/state PKL for annotation; NOT ready for training')
    initial.add_argument('--data-root', type=Path, required=True)
    initial.add_argument('--task', required=True)
    initial.add_argument('--calib-path', type=Path, required=True)
    annotate = sub.add_parser('prepare-ftf-annotation', help='Configure original notebook paths for manual semantic point annotation')
    annotate.add_argument('--data-root', type=Path, required=True)
    annotate.add_argument('--task', required=True)
    annotate.add_argument('--output', type=Path, required=True)
    visk = sub.add_parser('process-visk-sim', help='Align MuJoCo recordings, run original ViSk converter and validate its paper-profile loader')
    visk.add_argument('--workspace', type=Path, required=True)
    visk.add_argument('--raw-root', type=Path)
    args = p.parse_args(argv)
    try:
        if args.command == 'status':
            result = status()
        elif args.command == 'prepare':
            result = prepare(args.route, args.workspace, args.data_root, args.task, args.task_config, args.visk_profile)
        elif args.command == 'process-ftf-robot':
            result = process_ftf_robot(args.data_root, args.task)
        elif args.command == 'convert-ftf-robot-initial':
            result = convert_ftf_robot_initial(args.data_root, args.task, args.calib_path)
        elif args.command == 'prepare-ftf-annotation':
            result = prepare_ftf_annotation(args.data_root, args.task, args.output)
        elif args.command == 'process-visk-sim':
            from wowskin_lab.data.visk_processing import process_visk_sim
            result = process_visk_sim(args.workspace, args.raw_root)
        else:
            result = verify_workspace(args.workspace)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if result.get('unexpected_changes'):
            raise SystemExit(1)
        return result
    except (ValueError, FileNotFoundError, FileExistsError, subprocess.CalledProcessError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()
