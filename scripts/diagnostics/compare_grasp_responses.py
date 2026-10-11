"""Same-state spoon/cable comparison: local trained MLP vs dipole response.

This records diagnostics, not demonstrations or a calibration. No force,
stiffness, friction or magnetic parameter is changed by the comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import imageio.v3 as iio
import mujoco
import numpy as np
import torch

from wowskin_lab.paths import ROOT
from wowskin_lab.envs.pick_cube import response_configuration, save_records
from wowskin_lab.envs.feeding import PandaFeedingEnv
from wowskin_lab.envs.cable import PandaCableEnv
from wowskin_lab.baselines.reskin_forward import MAPPED_OUTPUT
from wowskin_lab.ui.tactile import TactileHUD


def stats(v):
    v = np.asarray(v)
    return dict(min=float(v.min()), median=float(np.median(v)), max=float(v.max()))


def fields(v):
    v = np.asarray(v).reshape(-1, 5, 3)
    return dict(component_rms_ut=float(np.sqrt(np.mean(v*v))),
                chip_norm_mean_ut=np.linalg.norm(v, axis=2).mean(0).tolist(),
                mean_xyz_ut=v.mean(0).tolist())


def plot_comparison(output):
    """Plot saved states only; this never reruns or modifies the simulator."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tasks = [name for name in ('feeding', 'cable') if (output/name/'episode.npz').exists()]
    fig, axes = plt.subplots(3, len(tasks), squeeze=False, sharex='col', figsize=(6*len(tasks), 9), layout='constrained')
    for col, task in enumerate(tasks):
        with np.load(output/task/'episode.npz') as a:
            t = a['time_s']
            axes[0, col].plot(t, a['normal_force_n'], label='Active pad')
            axes[0, col].plot(t, a['passive_normal_force_n'], label='Passive pad')
            axes[0, col].set(title=task.capitalize(), ylabel='Normal load (N)')
            axes[0, col].legend()
            axes[1, col].axhspan(1.5, 2., color='gray', alpha=.2, label='Depth bounds of training data')
            axes[1, col].plot(t, a['reskin_query_mm'][:, 2], label='FEM compression')
            axes[1, col].set(ylabel='Depth (mm)', ylim=(0, 2.2))
            axes[1, col].legend(fontsize=8)
            for key, label in [('dipole_ideal_delta_ut', 'Dipole, unfiltered'), ('magnetic_ut', 'Local trained MLP')]:
                rms = np.sqrt(np.mean(a[key]**2, axis=1))
                axes[2, col].plot(t, np.maximum(rms, 1e-4), label=label)
            axes[2, col].set(yscale='log', ylabel='15-component RMS (uT)', xlabel='Simulation time (s)')
            axes[2, col].legend(fontsize=8)
    fig.suptitle('Same physical states; both magnetic models uncalibrated for WowSkin\nMLP contact position/shape also outside training data; larger output is not validation')
    fig.savefig(output/'comparison.png', dpi=150)
    plt.close(fig)


def action(env, task, frame):
    if task == 'feeding':
        env.phase = 'hold' if frame < 80 else 'lift' if frame < 140 else 'hold_after_lift'
        a = env.hold_action()
        if 80 <= frame < 140:
            a[2] = .0001
        return a
    t = env.data.time
    if t < 2.4:
        target, phase = [.45, -.12, .185], 'align'
    elif t < 3.3:
        target, phase = [.45, -.12, .1275], 'approach'
    elif t < 4.2:
        target, phase = [.45, -.12, .1275], 'close'
        env.command_gripper(True)
    elif t < 4.9:
        target, phase = [.45, -.12, .16], 'lift'
    elif t < 5.5:
        target, phase = [.45, -.12, .16], 'hold'
    else:
        target, phase = [.45, -.12, .16], 'release'
        env.command_gripper(False)
    env.target_xyz[:] = target
    env.phase = phase
    return env.hold_action()


def summarize(env, rows, elapsed, error):
    a = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    contact = a['reskin_contact_count'] > 0
    hold = contact & a['grasped'] & (a['phase'] == 'hold')
    model = env.reskin_response.model
    report = dict(task=env.task_name, samples=len(rows), wall_s=elapsed, error=error,
                  checkpoint=str(MAPPED_OUTPUT),
                  checkpoint_sha256=hashlib.sha256((MAPPED_OUTPUT/'model.safetensors').read_bytes()).hexdigest(),
                  physics='Same MuJoCo states; magnetic models cannot alter control or mechanics.',
                  mlp_input='Force-weighted contact XY and maximum top FEM compression; single-probe approximation.',
                  signal_comparison='Compare chip norms because cross-sensor XYZ axes are not identified. Dipole norms reordered by spatial chip position only.',
                  training_depths_mm=model.meta['measured_depths_mm'],
                  physics_warnings=env.data.warning.number.tolist(), config=env.cfg,
                  contact_frames=int(contact.sum()), grasped_frames=int(a['grasped'].sum()),
                  min_volume_ratio=float(min(a['min_volume_ratio'].min(), a['passive_min_volume_ratio'].min())),
                  groups={})
    for name, mask in [('contact', contact), ('hold', hold)]:
        if not mask.any():
            continue
        q = a['reskin_query_mm'][mask]
        # Zero-depth query is ONLY an extrapolation audit, not new training data
        # and not a newly invented reference subtraction for the MLP.
        zero = q.copy(); zero[:, 2] = 0.
        with torch.inference_mode():
            z = torch.tensor((zero-model.meta['x_mean'])/model.meta['x_scale'], dtype=torch.float32)
            zero_field = model.model(z).numpy()*model.meta['y_scale']+model.meta['y_mean']
        mlp = a['magnetic_ut'][mask].reshape(-1, 5, 3)
        dipole = a['dipole_ideal_delta_ut'][mask].reshape(-1, 5, 3)[:, [0, 4, 2, 1, 3]]
        d, y = np.linalg.norm(dipole, axis=2), np.linalg.norm(mlp, axis=2)
        # Diagnostic only: would one magnetization multiplier fit even norms?
        scale = float(np.sum(d*y)/max(np.sum(d*d), 1e-20))
        report['groups'][name] = dict(frames=int(mask.sum()),
            depth_mm=stats(q[:, 2]), query_xy_mm=[stats(q[:, j]) for j in range(2)],
            left_force_n=stats(a['normal_force_n'][mask]), right_force_n=stats(a['passive_normal_force_n'][mask]),
            input_in_training_domain_frames=int(a['reskin_in_training_domain'][mask].sum()),
            dipole=fields(dipole), mlp=fields(mlp), raw_mlp_at_zero_depth=fields(zero_field),
            mlp_change_from_zero_depth=fields(mlp-zero_field.reshape(-1, 5, 3)),
            hypothetical_norm_scale=scale,
            hypothetical_scale_residual_rms_ut=float(np.sqrt(np.mean((scale*d-y)**2))),
            parameter_update_applied=False)
    report['decision'] = ('No calibration applied. Out-of-domain depth/contact-shape predictions cannot identify '
                          'WowSkin magnetization, stiffness, force or friction. Preview checkpoint remains opt-in.')
    if 'head_peak_force_n' in a:
        report['head_peak_force_n'] = float(a['head_peak_force_n'].max())
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['feeding', 'cable'], required=True)
    p.add_argument('--output', type=Path, default=ROOT/'outputs/diagnostics/grasp_response_comparison')
    args = p.parse_args()
    out = args.output/args.task
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cfg = response_configuration(SimpleNamespace(task=args.task, ideal=True, reskin_model=MAPPED_OUTPUT, reskin_extrapolate=True))
    env = (PandaFeedingEnv if args.task == 'feeding' else PandaCableEnv)(cfg, pollen=False)
    renderer = mujoco.Renderer(env.model, width=1000, height=750)
    hud = TactileHUD(env)
    rows, error = [], None
    start = time.monotonic()
    try:
        for i in range(180 if args.task == 'feeding' else 610):
            row = env.step(action(env, args.task, i))[-1]
            rows.append(row)
            if i % 25 == 0:
                print(json.dumps(dict(task=args.task, frame=i, phase=row['phase'],
                    wall_s=time.monotonic()-start, left_n=row['normal_force_n'],
                    right_n=row['passive_normal_force_n'], query_mm=row['reskin_query_mm'].tolist())), flush=True)
            if (args.task == 'feeding' and i == 60) or (args.task == 'cable' and i in (510, 609)):
                renderer.update_scene(env.data, camera=env.viewer_camera(True))
                iio.imwrite(out/f'{i:03d}_{row["phase"]}.png', hud.composite(renderer.render(), row))
                if args.task == 'feeding':
                    renderer.update_scene(env.data, camera=env.viewer_camera(False))
                    iio.imwrite(out/'overview.png', renderer.render())
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        renderer.close()
        if rows:
            save_records(rows, out)
            report = summarize(env, rows, time.monotonic()-start, error)
            (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            plot_comparison(args.output)
            print(json.dumps({k: v for k, v in report.items() if k != 'config'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
