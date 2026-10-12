"""Prescribed 7-mm probe insertion/withdrawal through the passive clip.

No robot/tactile model in this material bench. Only the probe is kinematically
driven; all clip bends evolve under native spring/contact forces. This tests
deformation and recovery, not real force accuracy or cable retention success.
"""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wowskin_lab.envs.elastic_clip import add_elastic_clip, configuration, diagnostics
from wowskin_lab.envs.task_evaluation import task_config


def bench_xml(c, timestep=.0005):
    root = ET.fromstring('<mujoco model="Experimental passive C-clip"><compiler angle="radian"/>'
        '<option integrator="discrete" gravity="0 0 -9.81" solver="CG" iterations="30" tolerance="1e-8"/>'
        '<visual><global offwidth="960" offheight="640"/></visual><asset/><worldbody>'
        '<light pos=".2 .2 .3" dir="-.2 -.2 -.3"/></worldbody></mujoco>')
    root.find('option').set('timestep', str(timestep))
    w = root.find('worldbody')
    b = ET.SubElement(w, 'body', name='cclip_0')
    g = task_config('cable')['collision_proxy']['base']
    ET.SubElement(b, 'geom', name='cclip_0_base', type='box', pos=' '.join(map(str,g['pos'])),
        size=' '.join(map(str,g['size'])), contype='1', conaffinity='7', rgba='1 .85 .05 1')
    add_elastic_clip(root, b, dict(elastic_clip=c))
    p = ET.SubElement(w, 'body', name='probe', mocap='true', pos='.0065 .036 .010')
    ET.SubElement(p, 'geom', name='probe_geom', type='capsule', fromto='-.008 0 0 .008 0 0',
        size='.0035', contype='4', conaffinity='1', rgba='.9 .3 .1 1', friction='.25 .002 .0002')
    return ET.tostring(root, encoding='unicode')


def run(c, timestep=.0005, probe_height=.010):
    xml = bench_xml(c, timestep)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    probe_id = m.geom('probe_geom').id
    records, wrench = [], np.zeros(6)
    for i in range(round(6/timestep)):
        t = i*timestep
        y = np.interp(t, [0,.4,2.4,3.2,5.2,6], [.036,.036,.012,.012,.036,.036])
        d.mocap_pos[0] = [.0065,y,probe_height]
        mujoco.mj_step(m,d)
        if np.any(d.warning.number) or not np.isfinite(d.qpos).all():
            raise RuntimeError(f'Unstable clip at step {i}: {d.warning.number}')
        force = np.zeros(3)
        for j in range(d.ncon):
            contact = d.contact[j]
            if probe_id in contact.geom:
                mujoco.mj_contactForce(m,d,j,wrench)
                sign = 1 if contact.geom[1] == probe_id else -1
                force += sign*contact.frame.reshape(3,3).T@wrench[:3]
        r = diagnostics(m,d)
        records.append([d.time,y,r['clip_tip_distance_m'],r['clip_elastic_energy_j'],
                        r['clip_max_bend_rad'],*force, r['clip_lower_max_bend_rad'],
                        *r['clip_lower_tip_m'], *d.xpos[m.body('cclip_0').id]])
    a = np.array(records)
    report = dict(timestep_s=timestep, probe_height_m=probe_height, steps=len(a), warnings=d.warning.number.tolist(),
        initial_tip_distance_mm=a[0,2]*1000, maximum_tip_distance_mm=a[:,2].max()*1000,
        final_tip_distance_mm=a[-1,2]*1000,
        maximum_probe_force_n=np.linalg.norm(a[:,5:8],axis=1).max(),
        final_max_bend_rad=a[-1,4], final_elastic_energy_j=a[-1,3],
        lower_max_bend_rad=a[:,8].max(),
        lower_max_tip_displacement_mm=np.linalg.norm(a[:,9:12]-a[0,9:12],axis=1).max()*1000,
        lower_final_tip_displacement_mm=np.linalg.norm(a[-1,9:12]-a[0,9:12])*1000,
        mounting_base_displacement_m=np.linalg.norm(a[:,12:15]-a[0,12:15],axis=1).max(),
        test='Prescribed insertion and full withdrawal; no fitted material or real force validation')
    return xml, a, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--young-pa', type=float, default=configuration()['young_pa'])
    p.add_argument('--step-us', type=int, default=500)
    p.add_argument('--probe-height-mm', type=float, default=10.)
    args = p.parse_args()
    if args.step_us <= 0:
        p.error('--step-us must be positive')
    c = configuration(); c['young_pa'] = args.young_pa
    xml,a,report = run(c,args.step_us*1e-6,args.probe_height_mm*.001)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'scene.xml').write_text(xml)
    np.savez_compressed(args.output/'trace.npz', values=a,
        columns=['time_s','probe_y_m','tip_distance_m','elastic_energy_j','max_bend_rad','fx_n','fy_n','fz_n',
                 'lower_max_bend_rad','lower_tip_x_m','lower_tip_y_m','lower_tip_z_m',
                 'mount_x_m','mount_y_m','mount_z_m'])
    (args.output/'report.json').write_text(json.dumps(dict(parameters=c,**report),indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
