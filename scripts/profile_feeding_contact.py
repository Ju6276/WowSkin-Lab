"""Compare closed/open feeding costs without changing physics or control clocks."""
import argparse
import cProfile
import json
from pathlib import Path
import pstats
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from wowskin_lab.envs.feeding import PandaFeedingEnv
from wowskin_lab.envs.pick_cube import response_configuration, save_records
from wowskin_lab.baselines.reskin_forward import MAPPED_OUTPUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cfg = response_configuration(SimpleNamespace(task='feeding', ideal=True, reskin_model=MAPPED_OUTPUT, reskin_extrapolate=True))
    env = PandaFeedingEnv(cfg, pollen=False)
    profile = cProfile.Profile()
    rows, positions = [], []
    phases = [('grasp_hold', 60), ('grasp_move', 60), ('release', 50), ('open_move', 60)]
    for phase, samples in phases:
        if phase == 'release':
            env.command_gripper(False)
        for i in range(samples):
            action = env.hold_action()
            if phase.endswith('move'):
                action[2] = .0001  # Same requested speed with and without spoon.
            env.phase = phase
            start = time.perf_counter()
            profile.enable()
            row = env.step(action)[-1]
            profile.disable()
            row['step_wall_s'] = time.perf_counter()-start
            row['total_contacts'] = env.data.ncon
            row['constraint_rows'] = env.data.nefc
            row['solver_iterations'] = env.data.solver_niter.copy()
            rows.append(row)
            positions.append(env.data.qpos.copy())
        print(phase, 'median step ms', 1000*np.median([r['step_wall_s'] for r in rows[-samples:]]), flush=True)
    save_records(rows, args.output)
    np.savez_compressed(args.output/'qpos.npz', qpos=np.array(positions))
    profile.dump_stats(str(args.output/'profile.pstats'))
    stats = pstats.Stats(profile)
    functions = [dict(file=f, line=l, name=n, calls=v[1], own_s=v[2], cumulative_s=v[3])
                 for (f,l,n),v in stats.stats.items()]
    report = dict(model_nv=env.model.nv, warnings=env.data.warning.number.tolist(),
                  note='Headless, cProfile enabled for both comparisons. No renderer or recording video costs.',
                  stages={}, top_functions=sorted(functions,key=lambda r:r['own_s'],reverse=True)[:25],
                  config=env.cfg)
    for phase, _ in phases:
        rr = [r for r in rows if r['phase']==phase]
        report['stages'][phase] = dict(samples=len(rr),
            median_step_ms=1000*float(np.median([r['step_wall_s'] for r in rr])),
            median_contacts=float(np.median([r['total_contacts'] for r in rr])),
            median_constraint_rows=float(np.median([r['constraint_rows'] for r in rr])),
            mean_left_force_n=float(np.mean([r['normal_force_n'] for r in rr])),
            all_grasped=all(r['grasped'] for r in rr))
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['top_functions'][:12],indent=2),flush=True)


if __name__=='__main__':
    main()
