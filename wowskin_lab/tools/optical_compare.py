"""Matched commands on distinct source DIGIT models, not hardware accuracy."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from PIL import Image
from wowskin_lab.core.optical_backends import make_optics
from wowskin_lab.envs.optical import OpticalBench
from wowskin_lab.paths import ROOT
from .optical_sensor import panel


def compare(destination, *, ideal=True, shadow=True):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    report = dict(source_revisions=json.loads((ROOT/'configs/upstream.json').read_text()),
        semantics='Same probe commands, separate source mechanics and cameras; not a same-state or hardware-accuracy benchmark.',
        physics_hz=2000, observation_hz=10, ideal=ideal, shadow=shadow, cases={})
    trajectory = [('unloaded', [0,0,0,0,0,0]), ('press', [0,0,-.012,0,0,0]),
        ('slide', [.002,0,-.012,0,0,0]), ('twist', [.002,0,-.012,0,0,.35]),
        ('tilt', [.002,0,-.012,.15,0,.35]), ('release', [.002,0,0,.15,0,.35]),
        ('recontact', [0,0,-.012,0,0,0]), ('final_release', [0,0,0,0,0,0])]
    for shape in ('sphere', 'box', 'cylinder'):
        case = destination/shape
        case.mkdir(exist_ok=True)
        environments = {name: OpticalBench(sensor=backend, probe_shape=shape)
                        for name, backend in (('TACTO','tacto'), ('FOTS','digit_fots'))}
        optics, arrays, rows, stages = {}, {}, [], []
        writer = None
        def append(key, value):
            arrays.setdefault(key, []).append(np.asarray(value).copy())
        try:
            for name, backend in (('TACTO','tacto'), ('FOTS','digit_fots')):
                optics[name] = make_optics(environments[name], backend, ideal=ideal, shadow=shadow)
            for stage, target in trajectory:
                for env in environments.values():
                    env.data.ctrl[:] = target
                for step in range(100):
                    for env in environments.values():
                        env.step(np.zeros(6))
                    if step % 10 != 9:
                        continue
                    frames = {}
                    stages.append(stage)
                    for name, env in environments.items():
                        start = perf_counter()
                        frame = optics[name].render(env.data)['DIGIT']
                        elapsed = perf_counter()-start
                        frames['DIGIT / '+name] = frame
                        prefix = name.lower()+'/'
                        for field in ('rgb','indentation_m','camera_depth_m','baseline_depth_m','contact_mask',
                                      'valid_mask','normal_force_n','time_s','depth_model','marker_status'):
                            append(prefix+field, getattr(frame, field))
                        for field in ('render_input_depth','render_deformation','marker_flow_rgb','marker_displacement_px'):
                            value = getattr(frame, field)
                            if value is not None:
                                append(prefix+field, value)
                        for field in ('qpos','qvel','ctrl'):
                            append(prefix+field, getattr(env.data, field))
                        append(prefix+'render_seconds', elapsed)
                        append(prefix+'contact_geom_mask', np.isin(np.arange(env.model.ngeom), frame.contacting_geoms))
                        probe = env.model.geom('probe').id
                        append(prefix+'probe_position_m', env.data.geom_xpos[probe])
                        append(prefix+'probe_rotation', env.data.geom_xmat[probe])
                        rows.append(dict(algorithm=name, stage=stage, time_s=float(env.data.time),
                            normal_force_n=frame.normal_force_n, max_depth_um=float(frame.indentation_m.max()*1e6),
                            marker_max_px=(float(np.linalg.norm(frame.marker_displacement_px, axis=-1).max())
                                           if frame.marker_displacement_px is not None else None), render_ms=elapsed*1000))
                    picture = panel(frames)
                    if writer is None:
                        writer = cv2.VideoWriter(str(case/'comparison.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                                10, (picture.shape[1],picture.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError('Cannot create comparison video')
                    writer.write(picture[..., ::-1])
                Image.fromarray(picture).save(case/(stage+'.png'))
            np.savez_compressed(case/'frames.npz', stage=stages, **{k:np.stack(v) for k,v in arrays.items()})
            with (case/'metrics.csv').open('w') as stream:
                output = csv.DictWriter(stream, fieldnames=list(rows[0]))
                output.writeheader()
                output.writerows(rows)
            report['cases'][shape] = {stage: {name: next(r for r in reversed(rows)
                if r['stage']==stage and r['algorithm']==name) for name in environments} for stage,_ in trajectory}
        finally:
            if writer is not None:
                writer.release()
            for renderer in optics.values():
                renderer.close()
    (destination/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(output=str(destination), cases=list(report['cases'])), indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/diagnostics/digit_compare'))
    parser.add_argument('--ideal', action='store_true')
    parser.add_argument('--no-fots-shadow', action='store_true')
    args = parser.parse_args(argv)
    return compare(args.output, ideal=args.ideal, shadow=not args.no_fots_shadow)
