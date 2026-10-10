"""Replay saved 7D commands with the SpaceMouse target clipping convention.

Does not recreate unrecorded pointer perturbations or lost failed commands.
Checks pose error against the original trace and saves even on physics faults.
"""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import imageio.v3 as iio
import mujoco
import numpy as np

from wowskin_lab.envs.pick_cube import hold_current_pose, rotation_vector, rotation_matrix, limit_vector, save_records, response_configuration
from wowskin_lab.envs.cable import PandaCableEnv
from wowskin_lab.envs.feeding import PandaFeedingEnv
from wowskin_lab.ui.tactile import TactileHUD


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['cable','feeding'], required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=550)
    p.add_argument('--magnetic-fit', type=Path)
    p.add_argument('--physics-step-us', type=float)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    a = np.load(args.source)
    cfg = response_configuration(SimpleNamespace(task=args.task, ideal=True,
                                  physics_step_us=args.physics_step_us, magnetic_fit=args.magnetic_fit))
    cfg.update(tcp_speed_m_s=.12, tcp_angular_speed_rad_s=.5)
    env = (PandaCableEnv if args.task == 'cable' else PandaFeedingEnv)(cfg, pollen=False)
    rows, moving, error = [], False, None
    start = time.monotonic()
    try:
        for i, command in enumerate(a['action'][:args.steps]):
            active = bool(np.any(command[:6]))
            if active != moving:
                hold_current_pose(env)
            if active:
                actual = env.data.site_xpos[env.tcp_site]
                rotation = env.data.site_xmat[env.tcp_site].reshape(3, 3)
                env.target_xyz = np.clip(env.target_xyz, actual-.006, actual+.006)
                env.target_rotation = rotation@rotation_matrix(limit_vector(rotation_vector(rotation.T@env.target_rotation), .03))
            moving = active
            row = env.step(command)[-1]
            row['constraint_finger_force_n'] = env.data.qfrc_constraint[env.finger_v].copy()
            row['reference_tcp_error_mm'] = np.linalg.norm(row['tcp_xyz_m']-a['tcp_xyz_m'][i])*1000
            rows.append(row)
            if i%100 == 0:
                print(i, 'gap mm', row['gripper_gap_m']*1000, 'reference error mm', row['reference_tcp_error_mm'], flush=True)
    except RuntimeError as exc:
        error = str(exc)
    finally:
        if rows:
            save_records(rows, args.output)
        report = dict(task=args.task, source=str(args.source), frames=len(rows), error=error,
                      wall_s=time.monotonic()-start, warnings=env.data.warning.number.tolist(),
                      max_reference_tcp_error_mm=max((r['reference_tcp_error_mm'] for r in rows), default=None),
                      config=env.cfg,
                      limitation='Replays recorded commands, not unrecorded viewer mouse forces or failed frames.')
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if error is None and rows:
            renderer = mujoco.Renderer(env.model, width=1000, height=750)
            renderer.update_scene(env.data,camera=env.viewer_camera(True))
            iio.imwrite(args.output/'last.png',TactileHUD(env).composite(renderer.render(),rows[-1]))
            renderer.close()
        print(json.dumps({k:v for k,v in report.items() if k!='config'},indent=2),flush=True)


if __name__=='__main__':
    main()
