"""One entry point for the local simulation / RGB-D / LeRobot environment."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if sys.version_info[:2] != (3, 12):
    raise SystemExit('Use Python 3.12: python3.12 scripts/setup_capture.py')
target = ROOT/'.venv-reduced'
python = target/'bin/python'
if not python.exists():
    subprocess.run([sys.executable, '-m', 'venv', str(target)], check=True)
subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(ROOT/'requirements.txt')], check=True)
subprocess.run([str(python), '-m', 'pip', 'install', '--no-deps', '-e', str(ROOT), 'lerobot==0.3.3'], check=True)
subprocess.run([str(python), '-c',
    'import mujoco; from wowskin_lab.data.lerobot_recording import upstream; upstream(); assert mujoco.__version__ == "3.15.0"; print("Ready: simulation + RGB-D + original LeRobot v2.1 data API")'],
    cwd=ROOT, check=True)
