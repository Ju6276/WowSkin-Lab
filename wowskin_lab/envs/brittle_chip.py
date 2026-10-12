"""Experimental potato chip: prefragmented bodies joined by breakable welds.

Method: https://github.com/google-deepmind/mujoco/issues/577
This is a local bond-failure model, not continuum fracture or calibrated food.
Each seam has its own force/moment criterion. No fragment teleportation, release
impulse or timed break. Internal faces of a connected rigid cluster are not
collision surfaces; detached clusters collide with one another.
"""
import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from wowskin_lab.paths import ROOT

CONFIG = ROOT / 'configs/brittle_chip.json'


def configuration(path=CONFIG):
    return validate_configuration(json.loads(Path(path).read_text()))


def validate_configuration(c):
    if c['schema'] != 'experimental_brittle_chip_v1':
        raise ValueError('Unsupported brittle chip configuration')
    if c['failure_rule'] != 'max(norm(force)/force_limit, norm(moment_at_shared_edge)/moment_limit) >= 1':
        raise ValueError('Unsupported bond failure rule')
    for k in ('thickness_m', 'density_kg_m3', 'bond_force_limit_n',
              'bond_moment_limit_nm', 'timestep_s', 'probe_speed_m_s',
              'constraint_time_constant_s', 'weld_torquescale_m', 'support_radius_m',
              'probe_radius_m', 'probe_mass_kg', 'probe_servo_kp_n_m', 'demo_travel_m'):
        if not np.isfinite(c[k]) or c[k] <= 0:
            raise ValueError(f'{k} must be positive and finite')
    if int(c['sectors']) != c['sectors'] or not 4 <= c['sectors'] <= 10:
        raise ValueError('sectors must be an integer 4..10 (at most 30 collision-mask bits)')
    if not 0 < c['inner_radius_fraction'] < 1:
        raise ValueError('inner_radius_fraction must lie between 0 and 1')
    diameters = np.asarray(c['diameters_m'], dtype=float)
    if diameters.shape != (2,) or not np.isfinite(diameters).all() or np.any(diameters <= 0):
        raise ValueError('diameters_m must contain two positive finite lengths')
    if c['internal_collision_rule'] != 'No internal-face collision within a connected weld cluster; all separated clusters and all external objects collide':
        raise ValueError('Unsupported internal collision rule')
    return c


def _numbers(values):
    return ' '.join(f'{x:.12g}' for x in np.asarray(values).ravel())


def chip_mesh(c):
    """Two oval rings triangulated into closed, non-overlapping prisms."""
    n = int(c['sectors'])
    a, b = np.asarray(c['diameters_m']) / 2
    xy = [[0., 0.]]
    for radius in (c['inner_radius_fraction'], 1.):
        for angle in np.arange(n) * (2*np.pi/n):
            xy.append([radius*a*np.cos(angle), radius*b*np.sin(angle)])
    xy = np.asarray(xy)
    z = c['saddle_height_m']*((xy[:, 0]/a)**2 - (xy[:, 1]/b)**2)
    points = np.column_stack((xy, z))
    triangles = []
    for i in range(n):
        j = (i+1) % n
        triangles.extend(((0, 1+i, 1+j), (1+i, 1+n+i, 1+n+j),
                          (1+i, 1+n+j, 1+j)))
    # Outward winding: bottom -Z, top +Z; six rectangular-side triangles.
    faces = [[0, 2, 1], [3, 4, 5], [0, 1, 4], [0, 4, 3],
             [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5]]
    pieces, shared = [], {}
    for index, tri in enumerate(triangles):
        surface = points[list(tri)]
        center = surface.mean(axis=0)
        offset = np.array([0., 0., c['thickness_m']/2])
        vertices = np.concatenate((surface-offset, surface+offset)) - center
        pieces.append((center, vertices, faces))
        for edge in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            shared.setdefault(tuple(sorted(edge)), []).append(index)
    bonds = [(owners[0], owners[1], points[list(edge)].mean(axis=0))
             for edge, owners in shared.items() if len(owners) == 2]
    return pieces, bonds


def chip_scene(c):
    root = ET.Element('mujoco', model='BrittleChip: prefragmented weld experiment')
    ET.SubElement(root, 'compiler', angle='radian')
    ET.SubElement(root, 'option', timestep=str(c['timestep_s']), integrator='implicitfast',
                  solver='Newton', jacobian='dense', iterations='100', tolerance='1e-10')
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'geom', friction=_numbers(c['friction']), condim='3',
                  solref=f"{c['constraint_time_constant_s']} 1", solimp='.99 .999 .0001')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='960', offheight='720')
    ET.SubElement(visual, 'map', znear='.005')
    ET.SubElement(visual, 'quality', shadowsize='2048')
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'texture', name='floor', type='2d', builtin='checker',
                  rgb1='.28 .32 .36', rgb2='.32 .36 .40', width='256', height='256')
    ET.SubElement(asset, 'material', name='floor', texture='floor', texrepeat='10 10',
                  texuniform='true', reflectance='.05')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='.1 -.1 .25', dir='-.2 .2 -1')
    ET.SubElement(world, 'light', pos='-.15 .05 .18', diffuse='.4 .4 .4', castshadow='false')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='.15 .15 .001', material='floor')
    for sign in (-1, 1):
        x, z = sign*c['support_half_span_m'], c['support_axis_height_m']
        ET.SubElement(world, 'geom', name=f'support_{sign}', type='capsule',
                      fromto=_numbers([x, -.018, z, x, .018, z]),
                      size=str(c['support_radius_m']), rgba='.24 .28 .31 1')
        ET.SubElement(world, 'geom', type='box', pos=_numbers([x, 0, z/2]),
                      size=_numbers([.003, .020, z/2-.001]), rgba='.20 .23 .26 1')
    pieces, bonds = chip_mesh(c)
    for i, (center, vertices, faces) in enumerate(pieces):
        ET.SubElement(asset, 'mesh', name=f'piece_{i}', vertex=_numbers(vertices), face=_numbers(faces))
        pos = center + [0, 0, c['chip_center_height_m']]
        body = ET.SubElement(world, 'body', name=f'piece_{i}', pos=_numbers(pos))
        ET.SubElement(body, 'freejoint', name=f'piece_{i}')
        # Subtle baked colour variation; the visible and collision meshes coincide.
        shade = .03*np.sin(i*2.4)
        ET.SubElement(body, 'geom', name=f'piece_{i}', type='mesh', mesh=f'piece_{i}',
                      density=str(c['density_kg_m3']), rgba=_numbers([.86+shade, .59+shade, .19, 1]))
    equality = ET.SubElement(root, 'equality')
    for i, (a, b, seam) in enumerate(bonds):
        ET.SubElement(equality, 'weld', name=f'bond_{i}', body1=f'piece_{a}', body2=f'piece_{b}',
                      anchor=_numbers(seam-pieces[b][0]),
                      torquescale=str(c['weld_torquescale_m']),
                      solref=f"{c['constraint_time_constant_s']} 1", solimp=_numbers(c['weld_solimp']))
    probe = ET.SubElement(world, 'body', name='probe', pos=f"0 0 {c['probe_home_height_m']}", gravcomp='1')
    ET.SubElement(probe, 'joint', name='probe_z', type='slide', axis='0 0 1')
    ET.SubElement(probe, 'geom', name='probe', type='sphere', size=str(c['probe_radius_m']),
                  mass=str(c['probe_mass_kg']), rgba='.12 .4 .6 1')
    actuator = ET.SubElement(root, 'actuator')
    ET.SubElement(actuator, 'position', joint='probe_z', kp=str(c['probe_servo_kp_n_m']),
                  kv=str(c['probe_servo_kv_ns_m']))
    return ET.tostring(root, encoding='unicode'), pieces, bonds


def weld_wrench(model, data, eq_id, body_id, point_world):
    """World force and moment at a specified seam point, on body_id.

    efc_force's rotational rows are *not* directly Nm (weld torquescale and
    quaternion Jacobians matter). First project this equality's rows by J.T,
    then invert the body's spatial point Jacobian, using virtual work.
    Only independent free bodies are supported here.
    """
    joint = model.body_jntadr[body_id]
    if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError('Bond body must have its own free joint')
    rows = np.flatnonzero((data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                          & (data.efc_id == eq_id))
    if len(rows) != 6:
        raise ValueError(f'Expected 6 active weld rows, got {len(rows)}')
    if mujoco.mj_isSparse(model):
        raise ValueError('Brittle chip requires a dense constraint Jacobian')
    start = model.jnt_dofadr[joint]
    dofs = np.arange(start, start+6)
    jac = data.efc_J.reshape(data.nefc, model.nv)
    generalized = jac[np.ix_(rows, dofs)].T @ data.efc_force[rows]
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jp, jr, np.asarray(point_world), body_id)
    return np.linalg.solve(np.vstack((jp[:, dofs], jr[:, dofs])).T, generalized)


class BrittleChipBench:
    def __init__(self, config=None):
        self.config = configuration() if config is None else validate_configuration(dict(config))
        self.xml, self.pieces, self.bonds = chip_scene(self.config)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.body_ids = [self.model.body(f'piece_{i}').id for i in range(len(self.pieces))]
        self.eq_ids = [self.model.equality(f'bond_{i}').id for i in range(len(self.bonds))]
        self.probe_geom = self.model.geom('probe').id
        self.piece_geoms = {self.model.geom(f'piece_{i}').id for i in range(len(self.pieces))}
        self._seam_bodies = np.array([self.body_ids[a] for a, _, _ in self.bonds])
        self._seam_offsets = np.array([p-self.pieces[a][0] for a, _, p in self.bonds])
        self._seam_dofs = np.array([self.model.jnt_dofadr[self.model.body_jntadr[b]]+np.arange(6)
                                   for b in self._seam_bodies])
        self._update_fragment_collision_masks()
        self._place_on_supports()
        self.reset()

    def _place_on_supports(self):
        """Initial condition only: lower the intact chip to first geometric touch.

        Uses actual convex collision shapes, not an invisible supporting weld.
        No settling with fracture disabled; all loads count from the first step.
        """
        z_ids = [self.model.jnt_qposadr[self.model.body_jntadr[b]]+2 for b in self.body_ids]
        supports = [self.model.geom(f'support_{s}').id for s in (-1, 1)]
        initial_z = self.model.qpos0[z_ids].copy()
        for _ in range(100):
            mujoco.mj_forward(self.model, self.data)
            gap = min(mujoco.mj_geomDistance(self.model, self.data, a, b, .1, None)
                      for a in self.piece_geoms for b in supports)
            if gap < -1e-7:
                raise ValueError('Chip intersects supports in the initial configuration')
            if gap < 1e-8:
                self.model.qpos0[z_ids] = self.data.qpos[z_ids]
                self.initial_support_shift_m = float(self.data.qpos[z_ids[0]]-initial_z[0])
                return
            self.data.qpos[z_ids] -= .9*gap
        raise RuntimeError('Could not place chip on supports without initial overlap')

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.events = []
        self.peak_probe_force_n = 0.
        self.max_load_ratio = 0.
        self.target_m = 0.
        self._bond_cache = None
        self._update_fragment_collision_masks()
        mujoco.mj_forward(self.model, self.data)

    def break_overloaded_bonds(self):
        """Evaluate all seams in one solved state; disable failed bonds together.

        Call immediately after mj_step, whose derived positions/Jacobians still
        describe its pre-integration force solve. Do not update kinematics first.
        """
        ids, wrenches = self.bond_wrenches()
        if not len(ids):
            return []
        forces = np.linalg.norm(wrenches[:, :3], axis=1)
        moments = np.linalg.norm(wrenches[:, 3:], axis=1)
        ratios = np.maximum(forces/self.config['bond_force_limit_n'],
                            moments/self.config['bond_moment_limit_nm'])
        self.max_load_ratio = max(self.max_load_ratio, float(ratios.max()))
        failed = np.flatnonzero(ratios >= 1)
        for k in failed:
            self.events.append(dict(time_s=float(self.data.time), bond=int(ids[k]),
                                    force_n=float(forces[k]), moment_nm=float(moments[k]),
                                    load_ratio=float(ratios[k])))
        eqs = [self.eq_ids[i] for i in ids[failed]]
        self.data.eq_active[eqs] = False
        if eqs:
            self._bond_cache = None
            self._update_fragment_collision_masks()
        return eqs

    def bond_wrenches(self):
        """Batch equivalent of weld_wrench for independent MuJoCo free bodies.

        Their translation velocities use world axes, angular velocities local
        axes. Convert the generalized torque to world axes and shift it from
        the body origin to the seam midpoint. No assumptions about COM location.
        """
        if self._bond_cache is None:
            ids = np.flatnonzero(self.data.eq_active[self.eq_ids])
            types, eq_id = self.data.efc_type, self.data.efc_id
            rows = [np.flatnonzero((types == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                                   & (eq_id == self.eq_ids[i])) for i in ids]
            if any(len(r) != 6 for r in rows):
                raise RuntimeError('Active seam does not have six weld constraint rows')
            self._bond_cache = ids, np.asarray(rows, dtype=int).reshape(-1, 6)
        ids, rows = self._bond_cache
        if not len(ids):
            return ids, np.empty((0, 6))
        jac = self.data.efc_J.reshape(self.data.nefc, self.model.nv)
        blocks = jac[rows[:, :, None], self._seam_dofs[ids, None, :]]
        qforce = np.einsum('bij,bi->bj', blocks, self.data.efc_force[rows])
        rotation = self.data.xmat[self._seam_bodies[ids]].reshape(-1, 3, 3)
        lever = np.einsum('bij,bj->bi', rotation, self._seam_offsets[ids])
        moment = np.einsum('bij,bj->bi', rotation, qforce[:, 3:])-np.cross(lever, qforce[:, :3])
        return ids, np.column_stack((qforce[:, :3], moment))

    def _component_roots(self):
        parents = list(range(len(self.pieces)))
        def find(i):
            while parents[i] != i:
                i = parents[i]
            return i
        for eq, (a, b, _) in zip(self.eq_ids, self.bonds):
            if self.data.eq_active[eq]:
                parents[find(a)] = find(b)
        return [find(i) for i in range(len(self.pieces))]

    def _update_fragment_collision_masks(self):
        """Suppress only internal surfaces of each connected rigid cluster.

        One bit per prism; bit 0 is the environment. After a split, enable
        collisions across all separated clusters, including the new seam.
        Model body masks mirror the geom masks for MuJoCo's broad phase.
        External probe/support/floor contacts are always enabled.
        """
        roots = self._component_roots()
        bits = [1 << (i+1) for i in range(len(roots))]
        all_bits = (1 << (len(roots)+1))-1
        for i, body in enumerate(self.body_ids):
            inside = sum(bits[j] for j, r in enumerate(roots) if r == roots[i])
            geom = self.model.geom(f'piece_{i}').id
            self.model.geom_contype[geom] = self.model.body_contype[body] = bits[i]
            self.model.geom_conaffinity[geom] = self.model.body_conaffinity[body] = all_bits ^ inside

    def components(self):
        """Number of separate connected fragment groups (not merely broken seams)."""
        return len(set(self._component_roots()))

    def step(self, target_m=None, duration=.01):
        if target_m is not None:
            if not np.isfinite(target_m):
                raise ValueError('Probe target must be finite')
            self.target_m = float(target_m)
        self.data.ctrl[0] = self.target_m
        contact_wrench = np.zeros(6)
        for _ in range(max(1, round(duration/self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)
            if np.any(self.data.warning.number) or not np.isfinite(self.data.qpos).all():
                raise RuntimeError('Brittle chip solver warning; inspect the physical model')
            probe_force = 0.
            for j, contact in enumerate(self.data.contact):
                if self.probe_geom in contact.geom and any(g in self.piece_geoms for g in contact.geom):
                    mujoco.mj_contactForce(self.model, self.data, j, contact_wrench)
                    probe_force += max(0., contact_wrench[0])
            self.peak_probe_force_n = max(self.peak_probe_force_n, probe_force)
            self.break_overloaded_bonds()
        mujoco.mj_forward(self.model, self.data)
        return self.report()

    def report(self):
        return dict(time_s=float(self.data.time), pieces=len(self.pieces), bonds=len(self.bonds),
                    broken_bonds=int(np.count_nonzero(~self.data.eq_active[self.eq_ids].astype(bool))),
                    fragment_groups=self.components(), peak_probe_force_n=self.peak_probe_force_n,
                    peak_bond_load_ratio=self.max_load_ratio,
                    mass_kg=float(sum(self.model.body_mass[self.body_ids])),
                    initial_support_shift_m=self.initial_support_shift_m,
                    warnings=self.data.warning.number.tolist())


def camera():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0., 0., .022]
    cam.distance, cam.azimuth, cam.elevation = .18, 110., -30.
    return cam


def demo_target(t, config):
    travel, speed = config['demo_travel_m'], config['probe_speed_m_s']
    elapsed = max(0., t-.5)
    down_time = travel/speed
    if elapsed <= down_time:
        return -speed*elapsed
    return -max(0., travel-speed*(elapsed-down_time))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--seconds', type=float, default=9., help='Headless demo duration in simulated seconds')
    parser.add_argument('--output', type=Path, help='Optional local diagnostics; no recording by default')
    parser.add_argument('--video', action='store_true', help='Headless demo MP4 in --output')
    args = parser.parse_args()
    if args.video and (not args.headless or args.output is None):
        parser.error('--video requires --headless and --output')
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be positive')
    bench = BrittleChipBench(configuration(args.config))
    rows = []
    print('Experimental brittle chip; strengths NOT calibrated. No robot or tactile recorder attached.', flush=True)
    if args.headless:
        from contextlib import ExitStack
        with ExitStack() as stack:
            renderer, video = None, None
            if args.video:
                import cv2
                args.output.mkdir(parents=True, exist_ok=True)
                renderer = stack.enter_context(mujoco.Renderer(bench.model, height=720, width=960))
                video = cv2.VideoWriter(str(args.output/'demo.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (960, 720))
                stack.callback(video.release)
                if not video.isOpened():
                    raise RuntimeError('Could not create chip demo video')
            for i in range(round(args.seconds*100)):
                rows.append(bench.step(demo_target(bench.data.time, bench.config)))
                if renderer is not None and i % 4 == 0:
                    renderer.update_scene(bench.data, camera=camera())
                    rgb = renderer.render()
                    video.write(rgb[..., ::-1])
                    if i == 0:
                        cv2.imwrite(str(args.output/'intact.png'), rgb[..., ::-1])
            if renderer is not None:
                renderer.update_scene(bench.data, camera=camera())
                cv2.imwrite(str(args.output/'after.png'), renderer.render()[..., ::-1])
    else:
        from mujoco import viewer as mj_viewer
        from wowskin_lab.ui.keyboard import ViewerKeyboard
        from wowskin_lab.envs.franka import apply_free_camera
        keyboard = ViewerKeyboard({ord('D'), ord('U'), ord('R'), ord('P'), 32}, 'BrittleChip:')
        demo_start = None
        with mj_viewer.launch_passive(bench.model, bench.data, key_callback=keyboard.native_callback,
                                         show_left_ui=False, show_right_ui=False) as viewer, keyboard:
            apply_free_camera(viewer, camera())
            while viewer.is_running():
                tick = time.monotonic()
                keys = keyboard.sample()
                if ord('P') in keys.commands:
                    break
                if ord('R') in keys.commands:
                    bench.reset(); demo_start = None
                if 32 in keys.commands:
                    demo_start = bench.data.time if demo_start is None else None
                manual = keys.held if keyboard.continuous else keys.commands
                movement = int(ord('U') in manual)-int(ord('D') in manual)
                if movement:
                    demo_start = None
                target = bench.target_m + movement*bench.config['probe_speed_m_s']*.01
                if demo_start is not None:
                    target = demo_target(bench.data.time-demo_start, bench.config)
                    if bench.data.time-demo_start > .5+2*bench.config['demo_travel_m']/bench.config['probe_speed_m_s']:
                        demo_start = None
                r = bench.step(target)
                if args.output is not None:
                    rows.append(r)
                viewer.set_texts([(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    'Brittle potato chip\nBond status\nProbe force (peak)\nControls\nModel',
                    f"{'BROKEN' if r['broken_bonds'] else 'INTACT'} | {r['fragment_groups']} fragment groups\n"
                    f"{r['broken_bonds']} / {r['bonds']} bonds broken\n{r['peak_probe_force_n']:.3f} N\n"
                    'D: down | U: up | Space: press cycle/stop | R: reset | P: exit\n'
                    'Prefragmented welds | experimental, NOT calibrated')])
                viewer.sync()
                time.sleep(max(0., .01-(time.monotonic()-tick)))
    report = dict(**bench.report(), config=bench.config, events=bench.events,
                  limitations='Predefined seams and uncalibrated force/moment thresholds; no crack propagation, fracture energy or sensor model.')
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        (args.output/'trajectory.json').write_text(json.dumps(rows)+'\n')
    print(json.dumps(bench.report(), indent=2))


if __name__ == '__main__':
    main()
