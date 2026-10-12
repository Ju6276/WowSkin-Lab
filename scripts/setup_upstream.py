"""Fetch clean, pinned upstream repositories. Never apply experiment patches."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()


def main():
    manifest = json.loads((ROOT/'configs/upstream.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', choices=list(manifest), help='Fetch only selected repositories')
    args = parser.parse_args()
    folders = dict(visk='visuoskin', ftf='feel-the-force', franka='franka_xbox', openteach='Open-Teach', tacto='mujoco-tacto')
    selected = set(args.only or manifest)
    if 'visk' in selected:
        selected.add('openteach')  # The original XArm environment imports its network helpers even offline.
    for name, spec in manifest.items():
        if name not in selected:
            continue
        parent = ROOT/'third_party'/('dependencies' if name == 'openteach' else 'official')
        if name in ('franka', 'tacto'):
            parent = ROOT/'third_party'
        directory = parent/folders[name]
        if not directory.exists():
            parent.mkdir(parents=True, exist_ok=True)
            run('git', 'clone', '--no-checkout', spec['url'], str(directory))
            run('git', 'checkout', '--detach', spec['revision'], cwd=directory)
        if run('git', 'rev-parse', 'HEAD', cwd=directory) != spec['revision']:
            raise RuntimeError(f'{directory}: unexpected revision; existing work left untouched')
        if name != 'franka' and run('git', 'status', '--porcelain', cwd=directory):
            raise RuntimeError(f'{directory}: official source was modified; existing work left untouched')
        if name == 'ftf':
            run('git', 'submodule', 'update', '--init', '--recursive', cwd=directory)
            statuses = run('git', 'submodule', 'status', '--recursive', cwd=directory)
            # strip() removes the first leading space, so check each remaining marker.
            if any(line.startswith(('+', '-', 'U')) for line in statuses.splitlines()):
                raise RuntimeError(f'{directory}: a submodule is not at its recorded revision')
        print(f'{name}: {spec["revision"]} ready (no experiment patches applied)')


if __name__ == '__main__':
    main()
