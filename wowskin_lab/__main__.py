"""One command menu for acquisition, simulation tools and policy adapters."""
import argparse
import importlib
import sys

from .paths import ROOT

COMMANDS = {
    'teleop': 'Franka in MuJoCo: keyboard/Pico/SpaceMouse teleoperation and episode recording',
    'record-auto': 'Automatically pick/lower/release into the shared raw dataset',
    'dataset-export': 'Select shared episodes for LeRobot v2.1, ViSk or official FTF preprocessing',
    'dataset-check': 'Validate LeRobot v2.1 using its original loader and video decoder',
    'official': 'Pinned ViSk / FTF human / FTF robot source, configuration and readiness',
    'pick-demo': 'Run the existing physical grasp/tug regression demo',
    'cable': 'Single Franka cable scene using pinned EBiM fixtures (local variant)',
    'whip': 'Published 25-link whip: independent bending/swinging dynamics comparator',
    'brittle-chip': 'Experimental potato chip: prefragmented bodies with load-triggered weld failure',
    'sensor': 'Standalone sensor viewer/demo',
    'franka-bench': 'Mounted sensor probe viewer/demo',
    'pollen': 'Published Pollen checkpoint download/evaluation',
    'reskin-response': 'Local position/depth to magnetic response model using 3D Cal data',
    'reskin-inverse': 'Official MagNet architecture: candidate-ordered magnetic input to position/depth',
    'cad': 'Reference CAD viewer/render',
}


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
    if command == 'dataset-export':
        from .data.dataset_export import main as export_data
        return export_data(rest)
    if command == 'dataset-check':
        from .data.lerobot_recording import main as check_data
        return check_data(rest)
    if command == 'record-auto':
        from .data.auto_collect import main as collect_auto
        return collect_auto(rest)
    if command == 'official':
        from .learning.official import main as official
        return official(rest)
    if command in ('teleop', 'pick-demo'):
        if not any(a.split('=')[0] == '--skin-dof' for a in rest):
            rest = ['--skin-dof', 'trilinear', *rest]
        if not any(a.split('=')[0] in ('--magnetic-fit', '--reskin-model') for a in rest):
            rest = ['--magnetic-fit', str(ROOT/'configs/magnetic_fit_pollen_nonslip.json'), *rest]
        task = 'pick_cube'
        cable_layout = 'routing'
        for i, value in enumerate(rest):
            if value.startswith('--task='):
                task = value.split('=', 1)[1]
            elif value == '--task' and i+1 < len(rest):
                task = rest[i+1]
            elif value.startswith('--cable-layout='):
                cable_layout = value.split('=', 1)[1]
            elif value == '--cable-layout' and i+1 < len(rest):
                cable_layout = rest[i+1]
        if task not in ('pick_cube', 'cable', 'feeding', 'peg_insert'):
            parser.error('--task must be pick_cube, cable, feeding or peg_insert')
        if not any(a.split('=')[0] == '--record-format' for a in rest):
            rest = ['--record-format', 'lerobot', *rest]
        if '--pollen' in rest:
            rest.remove('--pollen')
        else:
            rest = ['--no-pollen', *rest]
        if command == 'teleop' and not any(a.split('=')[0] == '--record-dir' for a in rest):
            directory = 'data/episodes'
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
