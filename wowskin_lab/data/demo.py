"""Scripted acquisition fixture, labelled as scripted rather than human demos."""
import argparse
from pathlib import Path

from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, scripted_action
from .recording import MujocoRecorder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seconds', type=float, default=8.5)
    args = parser.parse_args(argv)
    if args.seconds < 1.2 or args.episodes < 1:
        parser.error('Use at least 1 episode and 1.2 seconds')
    env = PandaPickCubeTactileEnv(pollen=False)
    with MujocoRecorder(env, args.output, source='scripted_fixture') as recorder:
        for i in range(args.episodes):
            env.reset(seed=i)
            recorder.start()
            for _ in range(round(args.seconds*env.cfg['sample_hz'])):
                row = env.step(scripted_action(env))[-1]
                recorder.step(row)
            recorder.finish('success' if env.success else 'unreviewed')
