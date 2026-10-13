"""Run real MuJoCo contact grasp with the explicitly experimental ReSkin proxy."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, response_configuration, scripted_action, task_camera, save_records
from wowskin_lab.baselines.reskin_forward import MAPPED_OUTPUT
from wowskin_lab.ui.tactile import TactileHUD
from wowskin_lab.paths import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/diagnostics/reskin_cube')
    parser.add_argument('--samples', type=int, default=850)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = response_configuration(SimpleNamespace(task='pick_cube', ideal=True,
        reskin_model=MAPPED_OUTPUT, reskin_extrapolate=True))
    env = PandaPickCubeTactileEnv(cfg, pollen=False)
    hud = TactileHUD(env)
    renderer = mujoco.Renderer(env.model, height=600, width=800)
    rows, frames, error = [], [], None
    start = time.monotonic()
    try:
        for i in range(args.samples):
            row = env.step(scripted_action(env))[-1]
            rows.append(row)
            if i % 25 == 0:
                progress = dict(samples=i+1, sim_time_s=row['time_s'], wall_s=time.monotonic()-start,
                                phase=row['phase'], left_n=row['normal_force_n'], right_n=row['passive_normal_force_n'],
                                lift_mm=row['lift_m']*1000, input_mm=row['reskin_query_mm'].tolist(),
                                channel_norm_ut=np.linalg.norm(row['magnetic_ut'].reshape(5,3), axis=1).tolist(),
                                status=row['reskin_status'])
                (args.output/'progress.json').write_text(json.dumps(progress, indent=2)+'\n')
                print(json.dumps(progress), flush=True)
            if i % 20 == 0:
                renderer.update_scene(env.data, camera=task_camera(env, True))
                frames.append(hud.composite(renderer.render(), row))
            if i in (0, 275, 350, 500, 575, 800):
                import imageio.v3 as iio
                renderer.update_scene(env.data, camera=task_camera(env, True))
                iio.imwrite(args.output/f'{i:03d}_{row["phase"]}.png', hud.composite(renderer.render(), row))
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        renderer.close()
        if rows:
            save_records(rows, args.output)
            contact = [r for r in rows if r['reskin_contact_count']]
            held = [r for r in rows if 4.5 < r['time_s'] < 5.5]
            report = dict(samples=len(rows), elapsed_wall_s=time.monotonic()-start, error=error,
                          physics_warnings=env.data.warning.number.tolist(), picked_up=bool(env.success),
                          max_lift_mm=max(r['lift_m'] for r in rows)*1000,
                          left_max_n=max(r['normal_force_n'] for r in rows), right_max_n=max(r['passive_normal_force_n'] for r in rows),
                          contact_frames=len(contact), input_in_training_domain_frames=sum(r['reskin_in_training_domain'] for r in contact),
                          depth_mm_range=[min(r['reskin_query_mm'][2] for r in contact),max(r['reskin_query_mm'][2] for r in contact)] if contact else None,
                          final_left_n=rows[-1]['normal_force_n'], final_right_n=rows[-1]['passive_normal_force_n'],
                          final_magnetic_ut=rows[-1]['magnetic_ut'].tolist(),
                          min_skin_volume_ratio=min(r['min_volume_ratio'] for r in rows),
                          min_passive_volume_ratio=min(r['passive_min_volume_ratio'] for r in rows),
                          all_field_values_finite=bool(np.isfinite([r['magnetic_ut'] for r in rows]).all()),
                          config=cfg, mapping=env.reskin_response.profile,
                          conclusion='End-to-end software/contact test only. Cube proxy, shallow-depth extrapolation and source XYZ are NOT validated WowSkin calibration.')
            if held:
                report.update(hold_force_mean_n=[float(np.mean([r[k] for r in held])) for k in ('normal_force_n','passive_normal_force_n')],
                              hold_channel_norm_mean_ut=np.linalg.norm(np.array([r['magnetic_ut'] for r in held]).reshape(-1,5,3),axis=2).mean(0).tolist(),
                              hold_depth_mm_range=[min(r['reskin_query_mm'][2] for r in held),max(r['reskin_query_mm'][2] for r in held)])
            (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            if frames:
                import imageio.v3 as iio
                iio.imwrite(args.output/'preview.gif', np.stack(frames), duration=200, loop=0)
            print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
