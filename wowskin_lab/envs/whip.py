"""Published 25-link whip dynamics, with a fixed laboratory anchor.

Source: Moses C. Nah / MIT Newman Lab, whip-project-targeting, pinned in
assets/whip_reference/manifest.json. Segment inertias, lengths, spring/damper
parameters and timestep are unchanged. This is a dynamics comparator, not a
Franka tactile grasp, and not the authors' targeting controller experiment.
"""
import argparse
import copy
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from wowskin_lab.paths import ROOT
from wowskin_lab.ui.keyboard import ViewerKeyboard

ASSETS = ROOT/'assets/whip_reference'


def whip_scene():
    source = ET.parse(ASSETS/'3D_model_w_whip.xml').getroot()
    root = ET.Element('mujoco', model='PublishedWhip: 25-link dynamics comparator')
    root.append(copy.deepcopy(source.find('compiler')))
    option = copy.deepcopy(source.find('option'))
    # MuJoCo 3 removed collision="predefined". Original XML has no explicit
    # pairs; disabling geom collision preserves its contact-free whip model.
    option.attrib.pop('collision', None)
    root.append(option)
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'geom', contype='0', conaffinity='0')
    root.append(copy.deepcopy(source.find('asset')))
    ET.SubElement(root, 'statistic', center='.5 0 1.1', extent='2.5')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='960', offheight='720')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -2 4', dir='0 .3 -1')
    # Reference grid is visual only, as there is no ground contact in source.
    ET.SubElement(world, 'geom', type='plane', size='3 3 .01', rgba='.3 .34 .38 1')
    anchor = ET.SubElement(world, 'body', name='laboratory_anchor', pos='0 0 2.1')
    ET.SubElement(anchor, 'geom', type='sphere', size='.025', rgba='.1 .65 .9 1')
    whip = copy.deepcopy(source.find(".//body[@name='body_whip_node1']"))
    whip.set('pos', '0 0 0')
    anchor.append(whip)
    return ET.tostring(root, encoding='unicode')


class WhipBench:
    def __init__(self):
        self.xml = whip_scene()
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.anchor = self.model.body('laboratory_anchor').id
        self.nodes = [self.model.geom(f'geom_whip_node{i}').id for i in range(1, 26)]
        self.reset()

    def reset(self, horizontal=True):
        mujoco.mj_resetData(self.model, self.data)
        self.model.body_pos[self.anchor] = [0, 0, 2.1]
        if horizontal:
            # Release experiment initial pose; original spring rest angles 0.
            j = self.model.joint('joint_whip_node1_Y').id
            self.data.qpos[self.model.jnt_qposadr[j]] = np.pi/2
        mujoco.mj_forward(self.model, self.data)

    def step(self, perturb=None):
        dt = self.model.opt.timestep
        for _ in range(round(.01/dt)):
            self.data.xfrc_applied[:] = 0.
            if perturb is not None:
                mujoco.mjv_applyPerturbForce(self.model, self.data, perturb)
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if np.any(self.data.warning.number) or not np.isfinite(self.data.qpos).all():
            raise RuntimeError('Whip physics warning; stop and inspect model')
        return dict(time_s=float(self.data.time), nodes_m=self.data.geom_xpos[self.nodes].copy(),
                    anchor_m=self.model.body_pos[self.anchor].copy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--seconds', type=float, default=0.)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/diagnostics/published_whip')
    args = parser.parse_args()
    bench = WhipBench()
    rows = [dict(time_s=0., nodes_m=bench.data.geom_xpos[bench.nodes].copy(),
                 anchor_m=bench.model.body_pos[bench.anchor].copy())]
    if args.headless:
        for _ in range(round((args.seconds or 3.)*100)):
            rows.append(bench.step())
    else:
        import mujoco.viewer
        from wowskin_lab.envs.franka import apply_free_camera
        keyboard = ViewerKeyboard({ord('R'), ord('H'), ord('P')}, 'PublishedWhip:')
        with mujoco.viewer.launch_passive(bench.model, bench.data, key_callback=keyboard.native_callback,
                                         show_left_ui=False, show_right_ui=False) as viewer, keyboard:
            camera = mujoco.MjvCamera();camera.lookat[:] = [.45, 0., 1.1]
            camera.distance, camera.azimuth, camera.elevation = 4.5, 100, -15
            apply_free_camera(viewer, camera)
            start = time.monotonic()
            while viewer.is_running() and (not args.seconds or time.monotonic()-start < args.seconds):
                tick = time.monotonic();keys = keyboard.sample()
                if ord('P') in keys.commands:
                    break
                if ord('R') in keys.commands or ord('H') in keys.commands:
                    bench.reset(horizontal=ord('R') in keys.commands)
                with viewer.lock():
                    perturb = copy.copy(viewer.perturb)
                rows.append(bench.step(perturb))
                viewer.set_texts([(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    'Published whip dynamics\nControls\nModel\nBoundary',
                    'MIT Newman Lab | original segment parameters\nR: horizontal release | H: vertical hang | P: exit\n25 x 72 mm | 0.30 kg | k=0.242 Nm/rad | c=0.092 Nm s/rad\nFixed anchor; no tactile grasp or ground contact')])
                viewer.sync();time.sleep(max(0., .01-(time.monotonic()-tick)))
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/'trajectory.npz', **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    report = dict(source=json.loads((ASSETS/'manifest.json').read_text()),
                  duration_sim_s=bench.data.time, segments=25, length_m=1.8, mass_kg=.3,
                  initial_tip_m=rows[0]['nodes_m'][-1].tolist(), final_tip_m=rows[-1]['nodes_m'][-1].tolist(),
                  warnings=bench.data.warning.number.tolist(),
                  local_changes=['fixed anchor replaces upstream forearm', 'horizontal release initial condition',
                                 'MuJoCo 3 contact-free compatibility', 'reference floor visual only'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('duration_sim_s', 'initial_tip_m', 'final_tip_m', 'warnings')}, indent=2))


if __name__ == '__main__':
    main()
