"""One command menu for acquisition, simulation tools and policy adapters."""
import argparse
import importlib
import os
import sys

from .paths import ROOT

COMMANDS = {
    'teleop': 'Franka in MuJoCo: keyboard/Pico/SpaceMouse teleoperation and episode recording',
    'record-auto': 'Record scripted cube, optical feeding or single-clip demonstrations',
    'dataset-export': 'Select shared episodes for LeRobot v2.1, ViSk or official FTF preprocessing',
    'dataset-check': 'Validate LeRobot v2.1 using its original loader and video decoder',
    'dataset-review': 'Build an offline synchronized optical episode replay',
    'optical-act': 'Train/evaluate a local optical baseline using upstream LeRobot ACT',
    'gelsight-ddqn': 'Run pinned GelSightMujoco DDQN on Panda pick/place with recorded adapter patches',
    'official': 'Pinned ViSk / FTF human / FTF robot source, configuration and readiness',
    'pick-demo': 'Run the existing physical grasp/tug regression demo',
    'cable': 'Single Franka cable scene using pinned EBiM fixtures (local variant)',
    'whip': 'Published 25-link whip: independent bending/swinging dynamics comparator',
    'brittle-chip': 'Experimental potato chip: prefragmented bodies with load-triggered weld failure',
    'sensor': 'Standalone sensor viewer/demo',
    'digit-compare': 'Compare native TACTO and FOTS DIGIT models on matched probe commands',
    'franka-bench': 'Mounted sensor probe viewer/demo',
    'pollen': 'Published Pollen checkpoint download/evaluation',
    'reskin-response': 'Local position/depth to magnetic response model using 3D Cal data',
    'reskin-inverse': 'Official MagNet architecture: candidate-ordered magnetic input to position/depth',
    'cad': 'Reference CAD viewer/render',
}


def configure_offscreen_gl():
    """Use EGL when Linux has no display. A set MUJOCO_GL value is left alone."""
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY') and not os.environ.get('MUJOCO_GL'):
        os.environ['MUJOCO_GL'] = 'egl'
        os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(prog='python -m wowskin_lab', description='WowSkin-Lab: tactile data and robot learning',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('command', choices=COMMANDS, help='\n'.join(f'{k}: {v}' for k, v in COMMANDS.items()))
    if not argv or argv[0] in ('-h', '--help'):
        parser.print_help()
        return
    if argv[0] in ('export', 'check-visk', 'check-ftf', 'train-visk', 'train-ftf'):
        parser.error('This simplified pipeline has been removed. '
                     'Use `official status` and docs/LEARNING_WORKFLOWS_ZH.md for the official pipelines.')
    command = parser.parse_args(argv[:1]).command
    rest = argv[1:]
    optical_backend = None
    if command in ('sensor', 'teleop'):
        sensor_parser = argparse.ArgumentParser(prog=f'python -m wowskin_lab {command}', add_help=False,
                                                description='Sensor selection; task options follow below.')
        sensor_parser.add_argument('--sensor', choices=('anyskin', 'digit', 'gelsight'), default='anyskin',
                                   help='One hardware type per run (default: anyskin)')
        sensor_parser.add_argument('--optics', choices=('tacto', 'fots'), default=None,
                                   help='DIGIT renderer (default: tacto); not used by other sensors')
        sensor_args, rest = sensor_parser.parse_known_args(rest)
        if any(arg in ('-h', '--help') for arg in rest):
            sensor_parser.print_help()
        if sensor_args.optics and sensor_args.sensor != 'digit':
            parser.error('--optics selects the DIGIT algorithm; use --sensor digit')
        if sensor_args.sensor in ('digit', 'gelsight'):
            from .tools.optical_sensor import main as optical
            backend = ('digit_fots' if sensor_args.optics == 'fots' else 'tacto') if sensor_args.sensor == 'digit' else 'gelsight'
            if command == 'sensor' or '--demo' in rest:
                return optical(rest, mounted=command == 'teleop', sensor=backend)
            optical_backend = backend
            rest = ['--sensor-backend', backend, *rest]
    if command == 'digit-compare':
        from .tools.optical_compare import main as compare
        return compare(rest)
    if command == 'dataset-export':
        from .data.dataset_export import main as export_data
        return export_data(rest)
    if command == 'dataset-check':
        from .data.lerobot_recording import main as check_data
        return check_data(rest)
    if command == 'dataset-review':
        from .data.episode_review import main as review_data
        return review_data(rest)
    if command == 'optical-act':
        from .learning.optical_act import main as optical_act
        return optical_act(rest)
    if command == 'gelsight-ddqn':
        from .learning.gelsight_ddqn import main as gelsight_ddqn
        return gelsight_ddqn(rest)
    if command == 'record-auto':
        # MuJoCo reads MUJOCO_GL when auto_collect imports it.
        configure_offscreen_gl()
        from .data.auto_collect import main as collect_auto
        return collect_auto(rest)
    if command == 'official':
        from .learning.official import main as official
        return official(rest)
    if command in ('teleop', 'pick-demo'):
        if not optical_backend and not any(a.split('=')[0] in ('--magnetic-fit', '--reskin-model') for a in rest):
            rest = ['--magnetic-fit', str(ROOT/'configs/magnetic_fit_pollen_nonslip.json'), *rest]
        task = 'pick_cube'
        for i, value in enumerate(rest):
            if value.startswith('--task='):
                task = value.split('=', 1)[1]
            elif value == '--task' and i+1 < len(rest):
                task = rest[i+1]
        if task not in ('pick_cube', 'cable', 'feeding', 'peg_insert', 'chip_tray'):
            parser.error('--task must be pick_cube, cable, feeding, peg_insert or chip_tray')
        if not any(a.split('=')[0] == '--record-format' for a in rest):
            rest = ['--record-format', 'lerobot', *rest]
        if '--pollen' in rest:
            rest.remove('--pollen')
        else:
            rest = ['--no-pollen', *rest]
        if command == 'teleop' and not any(a.split('=')[0] == '--record-dir' for a in rest):
            directory = 'data/episodes'+('_'+optical_backend if optical_backend else '')
            rest = ['--record-dir', str(ROOT/directory), *rest]
        rest = ['viewer' if command == 'teleop' else 'demo', *rest]
        module = 'envs.pick_cube'
    elif command == 'sensor':
        module = 'tools.sensor'
        rest = rest or ['viewer']
    else:
        module = {'franka-bench': 'envs.franka', 'pollen': 'baselines.pollen', 'cad': 'tools.render_cad',
                  'reskin-response': 'baselines.reskin_forward', 'reskin-inverse': 'baselines.reskin_inverse',
                  'cable': 'envs.cable', 'whip': 'envs.whip', 'brittle-chip': 'envs.brittle_chip'}[command]
    old = sys.argv
    try:
        sys.argv = [module, *rest]
        return importlib.import_module('wowskin_lab.'+module).main()
    finally:
        sys.argv = old


if __name__ == '__main__':
    main()
