"""One command menu for acquisition, simulation tools and policy adapters."""
import argparse
import importlib
import sys

from .paths import ROOT

COMMANDS = {
    'teleop': 'Franka in MuJoCo: keyboard/Pico/SpaceMouse teleoperation and episode recording',
    'record-demo': 'Record a scripted acquisition fixture (not human demonstrations)',
    'record-auto': 'Automatically pick/lower/release and record both ViSk and FTF raw data',
    'official': 'Pinned ViSk / FTF human / FTF robot source, configuration and readiness',
    'pick-demo': 'Run the existing physical grasp/tug regression demo',
    'cable': 'Single Franka cable scene using pinned EBiM fixtures (local variant)',
    'whip': 'Published 25-link whip: independent bending/swinging dynamics comparator',
    'sensor': 'Standalone sensor viewer/demo',
    'franka-bench': 'Mounted sensor probe viewer/demo',
    'pollen': 'Published Pollen checkpoint download/evaluation',
    'reskin-response': 'Local position/depth to magnetic response model using 3D Cal data',
    'reskin-inverse': 'Official MagNet architecture: candidate-ordered magnetic input to position/depth',
    'cad': 'Reference CAD viewer/render',
    'legacy-slip': 'Earlier synthetic slip experiments',
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
        parser.error('This simplified pipeline has been retired; its data/checkpoints are preserved. '
                     'Use `official status` and docs/LEARNING_WORKFLOWS_ZH.md for the official pipelines.')
    command = parser.parse_args(argv[:1]).command
    rest = argv[1:]
    if command == 'record-auto':
        from .data.auto_collect import main as collect_auto
        return collect_auto(rest)
    if command == 'record-demo':
        from .data.demo import main as collect
        return collect(rest)
    if command == 'official':
        from .learning.official import main as official
        return official(rest)
    if command in ('teleop', 'pick-demo'):
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
        if task not in ('pick_cube', 'cable', 'feeding'):
            parser.error('--task must be pick_cube, cable or feeding')
        if task in ('cable', 'feeding') and not any(a.split('=')[0] == '--record-format' for a in rest):
            rest = ['--record-format', 'both-raw', *rest]
        if '--pollen' in rest:
            rest.remove('--pollen')
        else:
            rest = ['--no-pollen', *rest]
        if command == 'teleop' and not any(a.split('=')[0] == '--record-dir' for a in rest):
            ftf_raw = '--record-format=ftf-raw' in rest or any(
                a == '--record-format' and i+1 < len(rest) and rest[i+1] == 'ftf-raw'
                for i, a in enumerate(rest))
            visk_raw = '--record-format=visk-raw' in rest or any(
                a == '--record-format' and i+1 < len(rest) and rest[i+1] == 'visk-raw'
                for i, a in enumerate(rest))
            both_raw = '--record-format=both-raw' in rest or any(
                a == '--record-format' and i+1 < len(rest) and rest[i+1] == 'both-raw'
                for i, a in enumerate(rest))
            record_task = 'cable_insert' if task == 'cable' and cable_layout == 'single_clip' else task
            directory = ('data/official_two_camera' if both_raw else
                         f'data/official_two_camera/ftf-robot/extracted_data/{record_task}' if ftf_raw else
                         f'data/official_two_camera/visk/raw/{record_task}' if visk_raw else f'data/raw/{record_task}')
            if any(a.split('=')[0] == '--reskin-model' for a in rest):
                directory = 'outputs/diagnostics/reskin_preview/two_camera/episodes'
            if any(a.split('=')[0] == '--magnetic-fit' for a in rest):
                directory = 'outputs/diagnostics/fitted_dipole_preview/two_camera/episodes'
            rest = ['--record-dir', str(ROOT/directory), *rest]
        rest = ['viewer' if command == 'teleop' else 'demo', *rest]
        module = 'envs.pick_cube'
    elif command in ('sensor', 'legacy-slip'):
        module = 'legacy.cli'
        rest = rest or ['viewer' if command == 'sensor' else 'slip-viewer']
    else:
        module = {'franka-bench': 'envs.franka', 'pollen': 'baselines.pollen', 'cad': 'tools.render_cad',
                  'reskin-response': 'baselines.reskin_forward', 'reskin-inverse': 'baselines.reskin_inverse',
                  'cable': 'envs.cable', 'whip': 'envs.whip'}[command]
    old = sys.argv
    try:
        sys.argv = [module, *rest]
        return importlib.import_module('wowskin_lab.'+module).main()
    finally:
        sys.argv = old


if __name__ == '__main__':
    main()
