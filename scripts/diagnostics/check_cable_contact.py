"""Physical cable grasp/lift/tug diagnostic; no scripted routing-success claim."""
import json
import time
import mujoco
import imageio.v3 as iio
from wowskin_lab.paths import ROOT
from wowskin_lab.envs.cable import PandaCableEnv, camera, save_run
from wowskin_lab.core.physics import configuration
from wowskin_lab.baselines.reskin_forward import MAPPED_OUTPUT


def main():
    output=ROOT/'outputs/diagnostics/cable_contact'
    output.mkdir(parents=True,exist_ok=True)
    cfg=configuration()
    cfg.update(reskin_model=str(MAPPED_OUTPUT),reskin_extrapolate=True,noise_ut=0.,drift_ut_per_s=0.)
    env=PandaCableEnv(cfg,pollen=False)
    rows=[]; start=time.monotonic(); error=None
    renderer=mujoco.Renderer(env.model,width=1000,height=750)
    try:
        for i in range(600):
            t=env.data.time
            if t<2.4: target,phase=[.45,-.12,.185],'align'
            elif t<3.3: target,phase=[.45,-.12,.1275],'approach'
            elif t<4.2:
                target,phase=[.45,-.12,.1275],'close'
                env.command_gripper(True)
            elif t<4.9: target,phase=[.45,-.12,.16],'lift'
            elif t<5.4:
                target,phase=[.45,-.10,.16],'tangential_move'
            else:
                target,phase=[.45,-.10,.16],'release'
                env.command_gripper(False)
            env.target_xyz[:]=target;env.phase=phase
            row=env.step(env.hold_action())[-1];rows.append(row)
            if i%25==0:
                print(json.dumps(dict(frame=i,phase=phase,wall_s=time.monotonic()-start,
                    tcp=row['tcp_xyz_m'].tolist(),left_n=row['normal_force_n'],right_n=row['passive_normal_force_n'],
                    depth_mm=row['reskin_query_mm'][2],slip=int(row['slip']))),flush=True)
            if i in (0,329,419,489,539,599):
                renderer.update_scene(env.data,camera=camera(True))
                iio.imwrite(output/f'{i:03d}_{phase}.png',renderer.render())
    except Exception as exc:
        error=repr(exc)
        raise
    finally:
        renderer.close()
        save_run(env,rows,output,time.monotonic()-start)
        report=dict(error=error,contact_frames=sum(r['contact'] for r in rows),
                    both_pad_frames=sum(r['grasped'] for r in rows),slip_frames=sum(r['slip'] for r in rows),
                    max_normal_n=max((r['normal_force_n'] for r in rows),default=0.),
                    max_passive_n=max((r['passive_normal_force_n'] for r in rows),default=0.),
                    status='Contact diagnostic only; no routing success / real sensor validation.')
        (output/'contact_report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
