"""Single Panda cable workbench derived from pinned EBiM task-1 assets.

Upstream fixture meshes and cable constitutive parameters are retained. The
default clip's rigid collision proxy is aligned to the visible channel;
an explicit experimental planar elastic beam option replaces both lips.
Board layout, cable length/discretization and robot are local choices.
No grasp-assist, clip-assist, automatic contact-force closure or EBiM scoring.
"""
import argparse
from types import SimpleNamespace
import copy
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from wowskin_lab.core.arm_control import joint_pd_reset
from wowskin_lab.paths import ROOT
from wowskin_lab.envs.task_evaluation import CableClipEvaluator, task_config
from wowskin_lab.envs.pick_cube import (
    PandaPickCubeTactileEnv, pick_scene, TABLE_Z, MOTION_KEYS, ROTATE_AXES,
    manual_action, key_command, keyboard_translation, response_configuration,
)
from wowskin_lab.envs.franka import apply_free_camera, ViewerStyleGuard
from wowskin_lab.ui.keyboard import ViewerKeyboard, MOVE_AXES, ESC, BACKSPACE
from wowskin_lab.ui.tactile import TactileHUD

ASSETS = ROOT/'assets/ebim'
CABLE_LENGTH = .38
CABLE_COUNT = 25  # 24 capsules; original scene uses count=71 and length=3 m.
CABLE_LAYOUTS = ('routing', 'single_clip')
# No fixed TCP-height floor: the installed gripper must contact the actual table.
WORKSPACE = np.array([[.24, -.22, -np.inf], [.64, .22, .40]])


def cable_scene(cfg, layout):
    source = ET.parse(ASSETS/'source_scene.xml').getroot()
    root = ET.fromstring(pick_scene(cfg, layout))
    single_clip = cfg.get('cable_layout', 'routing') == 'single_clip'
    root.set('model', 'WowSkin-Lab | PandaCable: single clip (EBiM)' if single_clip
             else 'WowSkin-Lab | PandaCable: EBiM fixtures')
    world, asset = root.find('worldbody'), root.find('asset')
    world.remove(world.find("body[@name='cube']"))
    for geom in list(world.findall('geom')):
        if geom.get('name', '').startswith(('cell_pick_', 'cell_place_')):
            world.remove(geom)
    for name in ('round_peg', 'base_connector', 'wire_harness', 'cclip'):
        mesh = copy.deepcopy(source.find(f"asset/mesh[@name='{name}']"))
        mesh.set('file', str(ASSETS/mesh.get('file')))
        asset.append(mesh)
    texture = copy.deepcopy(source.find("asset/texture[@name='board_top_tex']"))
    texture.set('file', str(ASSETS/'board_top_texture.png'))
    asset.append(texture)
    asset.append(copy.deepcopy(source.find("asset/material[@name='board_fast_mat']")))
    table = world.find("geom[@name='table']")
    table.set('material', 'board_fast_mat')
    fixtures = source.find(".//body[@name='fixture_group']")
    poses = {'round_peg_0': (.40, -.025, TABLE_Z+.0008),
             'round_peg_1': (.49, .065, TABLE_Z+.0008),
             'cclip_0': (.53, .145, TABLE_Z+.0009),
             'adapter_0': (.2543, -.1179, TABLE_Z-.0066)}
    routing = cfg.get('cable_routing') if not single_clip else None
    if routing:
        # The same source sockets are placed at the two installed cable ends.
        for i, endpoint in enumerate(routing['endpoint_world_m']):
            poses[f'adapter_{i}'] = np.asarray(endpoint)-[.0057, -.0021, .0339]
        for i, position in enumerate(routing['posts_world_m']):
            poses[f'round_peg_{i}'] = position
    if single_clip:
        # Place the mouth about 50 mm on the +Y side of the initial cable
        # (y=-.12), retaining the user's reversed fixture orientation.
        poses['cclip_0'] = (.45, -.06, TABLE_Z+.0009)
    for name, pos in poses.items():
        if single_clip and name not in ('cclip_0', 'adapter_0'):
            continue
        body = copy.deepcopy(fixtures.find(f"body[@name='{name}']"))
        body.set('pos', ' '.join(map(str, pos)))
        if single_clip and name == 'cclip_0':
            # Reverse the original fixture about its vertical body axis.
            # The source mesh and its three collision boxes move together.
            body.set('quat', '0 0 0 1')
        if name == 'cclip_0':
            for suffix, geometry in cfg['clip_success'].get('collision_proxy', {}).items():
                geom = body.find(f"geom[@name='cclip_0_{suffix}']")
                for key in ('pos', 'size'):
                    geom.set(key, ' '.join(map(str, geometry[key])))
        for geom in body.iter('geom'):
            if geom.get('contype', '1') != '0':
                geom.set('contype', '1')
                geom.set('conaffinity', '6')
        if name == 'cclip_0' and cfg.get('clip_model') == 'elastic_beam':
            from .elastic_clip import add_elastic_clip
            add_elastic_clip(root, body, cfg)
        world.append(body)
    root.append(copy.deepcopy(source.find('extension')))
    cable = copy.deepcopy(fixtures.find('composite'))
    cable.set('prefix', 'wire_')
    cable.set('count', f'{CABLE_COUNT} 1 1')
    cable.set('curve', 's 0 0')
    cable.set('size', str(CABLE_LENGTH))
    cable.set('offset', f'.26 -.12 {TABLE_Z+.0273}')
    if routing:
        from .cable_routing import initial_vertices
        vertices = initial_vertices(routing)
        for name in ('count', 'curve', 'size', 'offset'):
            cable.attrib.pop(name, None)
        cable.set('vertex', ' '.join(f'{v:.12g}' for v in vertices.ravel()))
        # A bent starting pose must not turn this cable into a stress-free S spring.
        ET.SubElement(cable.find('plugin'), 'config', key='flat', value='true')
        ET.SubElement(world, 'site', name='routing_end_socket',
                      pos=' '.join(map(str, vertices[-1])), size='.001', rgba='0 0 0 0')
        equality = root.find('equality')
        if equality is None:
            equality = ET.SubElement(root, 'equality')
        ET.SubElement(equality, 'connect', name='installed_cable_end',
                      site1='wire_S_last', site2='routing_end_socket', solref='.004 1')
    cable.find('geom').set('contype', '4')
    cable.find('geom').set('conaffinity', '7')
    cable.find('geom').set('rgba', '.92 .65 .16 1')
    world.append(cable)
    return ET.tostring(root, encoding='unicode')


class PandaCableEnv(PandaPickCubeTactileEnv):
    task_name = 'cable'
    workspace = WORKSPACE
    contact_geometry = 'cable'

    def __init__(self, cfg=None, pollen=False):
        cfg = response_configuration(SimpleNamespace(task='cable')) if cfg is None else dict(cfg)
        cfg.setdefault('clip_success', task_config('cable'))
        if cfg.get('clip_model') == 'elastic_beam':
            from .elastic_clip import configuration as elastic_configuration
            cfg.setdefault('elastic_clip', elastic_configuration())
            if cfg.get('sensor'):
                cfg['optical_task_integrator'] = 'discrete'
                cfg['optical_task_integrator_reason'] = 'MuJoCo discrete stiffness integration for the clip hinge springs; source optical solver/contact/optics otherwise retained. Explicit spring integration diverges at 0.5 ms.'
        elif cfg.get('clip_model', 'rigid') != 'rigid':
            raise ValueError('Unknown clip model')
        self.cable_layout = cfg.setdefault('cable_layout', 'routing')
        if self.cable_layout not in CABLE_LAYOUTS:
            raise ValueError(f'Unknown cable layout: {self.cable_layout}')
        self.task_name = 'cable_insert' if self.cable_layout == 'single_clip' else 'cable'
        cfg.setdefault('gripper_control_mode', 'panda_robosuite_position')
        cfg.setdefault('scene_design', 'ebim_task1_single_arm_single_clip' if self.cable_layout == 'single_clip'
                       else 'ebim_two_connector_routing_v1')
        # Explicit old metadata reconstructs the earlier one-ended short layout.
        if self.cable_layout == 'routing' and cfg['scene_design'] != 'ebim_task1_single_arm':
            from .cable_routing import configuration as routing_configuration, initial_vertices
            cfg.setdefault('cable_routing', routing_configuration())
            vertices = initial_vertices(cfg['cable_routing'])
            self.cable_count = len(vertices)
            self.cable_length = float(np.linalg.norm(np.diff(vertices, axis=0), axis=1).sum())
        else:
            self.cable_count, self.cable_length = CABLE_COUNT, CABLE_LENGTH
        cfg.update(cable_length_m=self.cable_length, cable_capsules=self.cable_count-1)
        # Cable edges produce large local compression/shear. Improve top
        # tetrahedron aspect ratios. V/V0=0.15 is only a warning, not a stop.
        # Radius is a NUMERICAL
        # collision thickness (also affects the external contact surface), not
        # a measured silicone thickness or a change to EBiM cable parameters.
        if not cfg.get('sensor'):
            for key, value in dict(cap_top_inner_fraction=.85,
                                   fem_contact_radius_m=.0002, fem_contact_timeconst_s=.0002,
                                   fem_contact_solimp=[.98, .98, .001, .5, 2.]).items():
                cfg.setdefault(key, value)
        # The default 0.9/0.95 contact impedance lets thin rim tetrahedra
        # compress into the warning range during one-sided cable shear.
        # This is a numerical external contact constraint,
        # not measured silicone stiffness. Nonpositive volumes remain invalid.
        if not cfg.get('sensor'):
            cfg['silicone_numerics'] = 'Trilinear reduced skin, bottom nodes pinned; material uncalibrated'
        super().__init__(cfg, pollen=pollen, xml_factory=cable_scene,
                         contact_body='wire_B_last', contact_geom=f'wire_G{self.cable_count-2}', object_joint=None)
        self.contact_geoms = {self.model.geom(f'wire_G{i}').id for i in range(self.cable_count-1)}
        self.cable_bodies = np.array([self.model.geom_bodyid[self.model.geom(f'wire_G{i}').id]
                                     for i in range(self.cable_count-1)], dtype=int)
        self.cfg['task'] = 'single_arm_cable_insert_scene' if self.cable_layout == 'single_clip' else 'single_arm_cable_scene'
        self.cfg['cable_anchor_status'] = (cfg['cable_routing']['end_connection'] if cfg.get('cable_routing')
            else 'Source fixed first segment and blue adapter; other end free')
        self.cfg['ebim_revision'] = json.loads((ASSETS/'manifest.json').read_text())['revision']
        self.cfg['grasp_assist'] = False
        self.cfg['clip_assist'] = False

    @joint_pd_reset
    def reset(self, seed=None):
        result = super().reset(seed)
        m = self.model
        radius = float(m.geom_size[m.geom('wire_G0').id, 0])
        bounds = self.cfg['clip_success'].get('internal_bounds_clip_m')
        if bounds is None:  # Preserve explicitly recorded v1 scene semantics.
            base, back, roof = (m.geom(name).id for name in
                                ('cclip_0_base', 'cclip_0_backwall', 'cclip_0_hook_top'))
            p, z = m.geom_pos, m.geom_size
            bounds = [[p[roof,0]-z[roof,0], p[roof,0]+z[roof,0]],
                      [p[roof,1]-z[roof,1]+radius, p[back,1]-z[back,1]-radius],
                      [p[base,2]+z[base,2]+radius, p[roof,2]-z[roof,2]-radius]]
        self.clip_score = CableClipEvaluator(self.cfg['clip_success'],
            x_bounds=bounds[0], y_bounds=bounds[1], z_bounds=bounds[2])
        if self.cable_layout == 'routing' and self.cfg.get('cable_routing'):
            from .cable_routing import CableRoutingEvaluator
            self.routing_score = CableRoutingEvaluator(self.cfg['cable_routing'])
        return result

    def cable_segments(self):
        ids = [self.model.geom(f'wire_G{i}').id for i in range(self.cable_count-1)]
        half = self.model.geom_size[ids, 1]
        axis = self.data.geom_xmat[ids].reshape(-1,3,3)[:, :, 2]
        centres = self.data.geom_xpos[ids]
        return np.stack((centres-axis*half[:,None], centres+axis*half[:,None]), axis=1)

    def evaluate_clip(self):
        ends = self.cable_segments()
        clip = self.model.body('cclip_0').id
        local = (ends-self.data.xpos[clip])@self.data.xmat[clip].reshape(3,3)
        return self.clip_score.update(float(self.data.time), local)

    def evaluation_metrics(self, row=None):
        metrics = dict(self.clip_score.metrics)
        if hasattr(self, 'routing_score'):
            metrics.update(self.routing_score.metrics)
        if self.cfg.get('clip_model') == 'elastic_beam':
            from .elastic_clip import diagnostics
            metrics.update(diagnostics(self.model, self.data))
        return metrics

    def task_evaluation(self):
        return (self.routing_score if hasattr(self, 'routing_score') else self.clip_score).summary()

    def observation(self):
        obs = super().observation()
        obs['cable_last_segment_m'] = obs.pop('cube_position_m')
        return obs

    def viewer_camera(self, close=False, operator=False):
        return camera(close)

    def step(self, action):
        _, _, _, _, row = super().step(action)
        row['cable_last_segment_m'] = row.pop('cube_xyz_m')
        row['cable_centerline_m'] = self.data.xpos[self.cable_bodies].copy()
        row['cable_end_m'] = self.data.site_xpos[self.model.site('wire_S_last').id].copy()
        row.pop('lift_m')
        row.update(self.evaluate_clip())
        row['success'] = self.success = self.clip_score.success
        if hasattr(self, 'routing_score'):
            row.update(self.routing_score.update(float(self.data.time), self.cable_segments()))
            row['success'] = self.success = self.routing_score.success
            row['cable_end_anchor_error_m'] = float(np.linalg.norm(row['cable_end_m']-
                self.cfg['cable_routing']['endpoint_world_m'][1]))
        row['cable_layout'] = self.cable_layout
        if self.cfg.get('clip_model') == 'elastic_beam':
            from .elastic_clip import diagnostics
            row.update(diagnostics(self.model, self.data))
        row['task_status'] = 'clip inserted' if self.success else 'press cable under clip roof'
        if hasattr(self, 'routing_score'):
            row['task_status'] = 'two-post route complete' if self.success else 'route around both post guides'
        self.last_row = row
        return self.observation(), float(self.success), False, False, row


def camera(close=False):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [.43, .025, .14] if close else [.28, .0, .18]
    cam.distance, cam.azimuth, cam.elevation = (.70, 135, -55) if close else (1.5, 135, -30)
    return cam


def save_run(env, rows, output, elapsed):
    output.mkdir(parents=True, exist_ok=True)
    (output/'scene.xml').write_text(env.xml)
    if rows:
        np.savez_compressed(output/'episode.npz', **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    report = dict(task='PandaCable scene preview, not an EBiM scored rollout', samples=len(rows),
                  sim_time_s=float(env.data.time), wall_s=elapsed,
                  physics_warnings=env.data.warning.number.tolist(), config=env.cfg,
                  cable_length_m=env.cable_length, cable_capsules=env.cable_count-1,
                  grasp_assist=False, clip_assist=False, routing_evaluator=hasattr(env, 'routing_score'), task_evaluation=env.task_evaluation(),
                  caveat='ReSkin MLP line-contact transfer unvalidated; no cable depth labels in Pollen.')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'config'}, indent=2), flush=True)


def viewer(env, args):
    import mujoco.viewer
    # The standalone diagnostic viewer uses the same fixed operator convention
    # as `teleop --task cable`, including the non-macOS key callback.
    env.cfg.update(teleop_frame='base', teleop_base_lateral_sign=1., disable_workspace_limits=True)
    keyboard = ViewerKeyboard(MOTION_KEYS | set(map(ord, 'XZCRVTP-=,.')) | {ESC, BACKSPACE},
                              title_contains='PandaCable:')
    hud = TactileHUD(env)
    if hud.reskin_mode:
        hud.subtitle = 'Experimental cable proxy [uT]'
        hud.footer = 'NOT calibrated | cable/probe mismatch'
    rows, moving, close, fault = [], False, False, ''
    start = time.monotonic()
    print('PandaCable: BASE W:+X S:-X A:-Y D:+Y E:+Z Q:-Z | local workspace OFF | I/K L/J U/O rotate | X/Z grip | R reset | V view | T tug | P exit', flush=True)
    try:
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=keyboard.native_callback,
                                          show_left_ui=False, show_right_ui=False) as handle, keyboard:
            apply_free_camera(handle, camera())
            guard = None if keyboard.continuous else ViewerStyleGuard(handle, 'WASDEQXZCRVTIJKLUO,.')
            next_hud = 0.
            while handle.is_running() and (not args.seconds or time.monotonic()-start < args.seconds):
                tick = time.monotonic()
                snapshot = keyboard.sample()
                if ord('P') in snapshot.commands:
                    break
                fallback = env.hold_action()
                for key in snapshot.commands:
                    if key in (ord('V'), ESC):
                        close = not close if key == ord('V') else False
                        apply_free_camera(handle, camera(close))
                    else:
                        key_command(env, key)
                        hud.key(key)
                        if key in (ord('R'), BACKSPACE):
                            fault, moving = '', False
                            keyboard.clear()
                        if not keyboard.continuous:
                            if key in MOVE_AXES: fallback[:3] += np.array(MOVE_AXES[key])*.005
                            if key in ROTATE_AXES: fallback[3:6] += np.array(ROTATE_AXES[key])*.03
                if not fault:
                    try:
                        with handle.lock():
                            azimuth = handle.cam.azimuth
                            perturb = copy.copy(handle.perturb)
                        env.update_mouse_perturbation(perturb)
                        if keyboard.continuous:
                            action, moving = manual_action(env, snapshot.held, moving, azimuth)
                        else:
                            fallback[:3] = keyboard_translation(env, fallback[:3], azimuth)
                            fallback[6] = env.gripper_fraction(target=True)
                            action = fallback
                        rows.append(env.step(action)[-1])
                    except RuntimeError as exc:
                        fault = str(exc)
                        print(f'Paused: {fault}; R resets', flush=True)
                if env.last_row and time.monotonic() >= next_hud:
                    next_hud = time.monotonic()+.1
                    r = env.last_row
                    handle.set_texts([(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        'PandaCable\nMove / rotate\nGripper / view\nContact N\nTask status',
                        'EBiM assets | one Franka\nWASD/EQ | IK LJ UO\nX/Z | V close view | P exit\n'
                        f"{r['normal_force_n']:.3f} / {r['passive_normal_force_n']:.3f}\n"+
                        (fault or ('SUCCESS: cable in clip' if env.success else 'Press cable into clip')))])
                    hud.show(handle, r)
                before = env.data.time
                guard.sync(handle) if guard else handle.sync()
                if env.data.time < before-1e-9:
                    env.reset()
                    fault, moving = '', False
                time.sleep(max(0., .01-(time.monotonic()-tick)))
    finally:
        save_run(env, rows, args.output/time.strftime('viewer_%Y%m%d_%H%M%S'), time.monotonic()-start)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.set_defaults(task='cable')
    p.add_argument('mode', choices=['viewer', 'check'], nargs='?', default='viewer')
    p.add_argument('--reskin-model', type=Path)
    p.add_argument('--reskin-extrapolate', action='store_true')
    p.add_argument('--magnetic-fit', type=Path, help='Default: the teleoperation fitted-dipole configuration')
    p.add_argument('--ideal', action='store_true', help='Disable synthetic tactile noise, filtering and delay')
    p.add_argument('--pollen', action='store_true')
    p.add_argument('--cable-layout', choices=CABLE_LAYOUTS, default='routing')
    p.add_argument('--seconds', type=float, default=0.)
    p.add_argument('--samples', type=int, default=50)
    p.add_argument('--output', type=Path, default=ROOT/'outputs/diagnostics/cable_scene')
    args = p.parse_args()
    if args.reskin_extrapolate and not args.reskin_model:
        p.error('--reskin-extrapolate requires --reskin-model')
    if args.samples < 1 or args.seconds < 0:
        p.error('samples must be positive; seconds must be nonnegative')
    if args.magnetic_fit and args.reskin_model:
        p.error('Choose --magnetic-fit or --reskin-model, not both')
    if not args.reskin_model and not args.magnetic_fit:
        args.magnetic_fit = ROOT/'configs/magnetic_fit_pollen_nonslip.json'
    cfg = response_configuration(args)
    if args.cable_layout == 'single_clip' and args.output == ROOT/'outputs/diagnostics/cable_scene':
        args.output = ROOT/'outputs/diagnostics/cable_insert_scene'
    env = PandaCableEnv(cfg, pollen=args.pollen)
    if args.mode == 'viewer':
        return viewer(env, args)
    start, rows = time.monotonic(), []
    for _ in range(args.samples):
        rows.append(env.step(env.hold_action())[-1])
    save_run(env, rows, args.output, time.monotonic()-start)
    import imageio.v3 as iio
    with mujoco.Renderer(env.model, width=1000, height=750) as renderer:
        for close in (False, True):
            renderer.update_scene(env.data, camera=camera(close))
            iio.imwrite(args.output/('workbench.png' if close else 'overview.png'), renderer.render())


if __name__ == '__main__':
    main()
