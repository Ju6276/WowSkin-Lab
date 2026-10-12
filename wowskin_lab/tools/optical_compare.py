"""Record native DIGIT algorithms on identical MuJoCo states, without scaling.

The optics have different projections and deformation models. Distribution
statistics compare responses, not pixel-aligned error against ground truth.
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wowskin_lab.core.optical_backends import DigitComparison
from wowskin_lab.envs.optical import OpticalBench
from wowskin_lab.paths import ROOT
from .optical_sensor import panel


def compare(destination, *, ideal=True, shadow=True, **tacto_options):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    report = dict(
        source_revisions=json.loads((ROOT/'configs/upstream.json').read_text()),
        physics_hz=2000, control_hz=100, observation_hz=10, ideal=ideal, fots_shadow=shadow,
        semantics='Same CAD, MuJoCo state, contact force and time. Native optics/projections differ; no pixel RMSE is an accuracy score.',
        depth='TACTO: source force-related optical shift plus geometry. FOTS: geometric overlap, no fitted amplitude.',
        units=dict(time_s='s', translation='m', rotation='rad', force_n='N', depth_m='m', marker_displacement_px='pixel'),
        cases={})
    for shape in ('sphere', 'box', 'cylinder'):
        case = destination/shape
        case.mkdir(exist_ok=True)
        env = OpticalBench(sensor='digit_compare', probe_shape=shape)
        optics = DigitComparison(env.model, env.sensors, ideal=ideal, shadow=shadow, **tacto_options)
        arrays, rows, stages = {}, [], []
        writer = None
        reference = {}
        def append(key, value):
            arrays.setdefault(key, []).append(np.asarray(value).copy())
        try:
            # Absolute actuator targets, all reached by mj_step; never set
            # qpos to fake deformation or attach the probe to the gel.
            trajectory = [
                ('unloaded', [0, 0, 0, 0, 0, 0]),
                ('press', [0, 0, -.012, 0, 0, 0]),
                ('slide', [.002, 0, -.012, 0, 0, 0]),
                ('twist', [.002, 0, -.012, 0, 0, .35]),
                ('tilt', [.002, 0, -.012, .15, 0, .35]),
                ('release', [.002, 0, 0, .15, 0, .35]),
                ('recontact', [0, 0, -.012, 0, 0, 0]),
                ('final_release', [0, 0, 0, 0, 0, 0]),
            ]
            for stage, target in trajectory:
                env.data.ctrl[:] = target
                for step in range(100):
                    env.step(np.zeros(6))
                    if step % 10 != 9:
                        continue
                    qpos = env.data.qpos.copy()
                    frames = optics.render(env.data)
                    np.testing.assert_array_equal(qpos, env.data.qpos)
                    stages.append(stage)
                    append('time_s', env.data.time)
                    append('qpos', env.data.qpos)
                    append('qvel', env.data.qvel)
                    append('ctrl', env.data.ctrl)
                    append('probe_position_m', env.data.geom_xpos[env.model.geom('probe').id])
                    append('probe_rotation', env.data.geom_xmat[env.model.geom('probe').id])
                    for name, frame in frames.items():
                        algorithm = name.rsplit(' / ', 1)[-1].lower()
                        reference.setdefault(algorithm, frame.rgb.copy())
                        valid = frame.valid_mask
                        for field in ('rgb', 'indentation_m', 'contact_mask', 'valid_mask',
                                      'camera_depth_m', 'baseline_depth_m', 'normal_force_n'):
                            append(algorithm+'/'+field, getattr(frame, field))
                        append(algorithm+'/contact_geom_mask',
                               np.isin(np.arange(env.model.ngeom), frame.contacting_geoms))
                        if frame.marker_displacement_px is not None:
                            append(algorithm+'/marker_displacement_px', frame.marker_displacement_px)
                            append(algorithm+'/marker_flow_rgb', frame.marker_flow_rgb)
                        append(algorithm+'/render_seconds', optics.render_seconds[algorithm])
                        rows.append(dict(stage=stage, time_s=float(env.data.time), algorithm=algorithm,
                            force_n=frame.normal_force_n,
                            max_depth_um=float(frame.indentation_m[valid].max()*1e6),
                            mean_positive_depth_um=float(np.maximum(frame.indentation_m[valid], 0).mean()*1e6),
                            contact_fraction=float(frame.contact_mask[valid].mean()),
                            rgb_change_mae=float(np.abs(frame.rgb.astype(float)-reference[algorithm]).mean()),
                            marker_max_px=(float(np.linalg.norm(frame.marker_displacement_px, axis=-1).max())
                                           if frame.marker_displacement_px is not None else None),
                            render_ms=optics.render_seconds[algorithm]*1000))
                    picture = panel(frames)
                    labelled = Image.new('RGB', (picture.shape[1], picture.shape[0]+30), (22, 28, 34))
                    labelled.paste(Image.fromarray(picture), (0, 30))
                    ImageDraw.Draw(labelled).text((5, 3), f'{shape} | {stage} | t = {env.data.time:.2f} s',
                                                 font=ImageFont.load_default(size=20), fill='white')
                    picture = np.asarray(labelled)
                    if writer is None:
                        writer = cv2.VideoWriter(str(case/'comparison.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                                 10, (picture.shape[1], picture.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError('Cannot create optical comparison video')
                    writer.write(picture[..., ::-1])
                Image.fromarray(picture).save(case/f'{stage}.png')
            np.savez_compressed(case/'frames.npz', stage=np.array(stages),
                                **{key: np.stack(value) for key, value in arrays.items()})
            with (case/'metrics.csv').open('w') as stream:
                writer_csv = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer_csv.writeheader()
                writer_csv.writerows(rows)
            by_stage = {}
            for stage, _ in trajectory:
                by_stage[stage] = {}
                for algo in ('tacto', 'fots'):
                    # Steady end of each stage; retain transients in NPZ/CSV.
                    selected = [r for r in rows if r['stage'] == stage and r['algorithm'] == algo][-3:]
                    by_stage[stage][algo] = {k: float(np.mean([r[k] for r in selected]))
                        for k in ('force_n', 'max_depth_um', 'contact_fraction', 'rgb_change_mae', 'render_ms')}
                    if algo == 'fots':
                        by_stage[stage][algo]['marker_max_px'] = float(np.mean([r['marker_max_px'] for r in selected]))
            report['cases'][shape] = dict(frames=len(stages), stages=by_stage,
                geom_names=[env.model.geom(i).name for i in range(env.model.ngeom)],
                joint_names=[env.model.joint(i).name for i in range(env.model.njnt)],
                physics_timestep_s=float(env.model.opt.timestep),
                force_difference_max_n=float(np.max(np.abs(np.array(arrays['tacto/normal_force_n'])-
                                                         np.array(arrays['fots/normal_force_n'])))))
        finally:
            if writer is not None:
                writer.release()
            optics.close()
    (destination/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(output=str(destination), cases={shape: v['frames'] for shape, v in report['cases'].items()},
                          force_difference_max_n={shape: v['force_difference_max_n'] for shape, v in report['cases'].items()}), indent=2))
    return report
