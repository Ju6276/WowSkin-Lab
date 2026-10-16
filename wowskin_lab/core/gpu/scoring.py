"""CUDA task geometry and completion state, with the existing score definitions.

Scores are privileged diagnostics; they never modify physics or policy inputs.
Only the resulting scalar metrics are mirrored to the native summary objects.
"""
import weakref

import numpy as np
import warp as wp

from .observations import point_velocity

Checks = wp.types.vector(length=7, dtype=wp.int32)

@wp.struct
class Scene:
    xpos: wp.array2d[wp.vec3]
    xmat: wp.array2d[wp.mat33]
    gpos: wp.array2d[wp.vec3]
    gmat: wp.array2d[wp.mat33]
    com: wp.array2d[wp.vec3]
    cvel: wp.array2d[wp.spatial_vector]
    root: wp.array[int]
    ncon: wp.array[int]
    geom: wp.array[wp.vec2i]
    flex: wp.array[wp.vec2i]
    dist: wp.array[float]
    kind: wp.array[int]
    forces: wp.array2d[wp.float64]
    force_column: int


@wp.func
def grasped(s: Scene):
    return s.forces[0, s.force_column] > wp.float64(.01) and s.forces[1, s.force_column] > wp.float64(.01)


@wp.func
def dwell_state(time: wp.float64, eligible: bool, required: wp.float64, state: wp.array[wp.float64]):
    if eligible:
        if state[1] < wp.float64(0.0):
            state[1] = time
    else:
        state[1] = wp.float64(-1.0)
    dwell = wp.float64(0.0)
    if state[1] >= wp.float64(0.0):
        dwell = wp.max(wp.float64(0.0), time-state[1])
    if eligible and dwell+wp.float64(1.e-9) >= required:
        if state[0] == wp.float64(0.0):
            state[2] = time
        state[0] = wp.float64(1.0)
    return dwell


@wp.kernel
def _cube(s: Scene, body: int, geom: int, hand_geoms: wp.array[int],
          half: wp.vec3d, c: wp.array[wp.float64], time: wp.array[wp.float64],
          state: wp.array[wp.float64], out: wp.array[wp.float64]):
    center = wp.vec3d(s.gpos[0, geom])
    rotation = wp.mat33d(s.gmat[0, geom])
    inside = bool(True)
    zmin = wp.float64(1.e30)
    for i in range(8):
        local = wp.vec3d(half[0]*wp.float64(2*(i//4)-1), half[1]*wp.float64(2*((i//2)%2)-1), half[2]*wp.float64(2*(i%2)-1))
        point = center+rotation*local
        inside = inside and wp.abs(point[0]-c[0]) <= c[2] and wp.abs(point[1]-c[1]) <= c[2]
        zmin = wp.min(zmin, point[2])
    if state[3] != wp.float64(0.0) and grasped(s) and center[2]-c[3] >= c[4]:
        state[4] = wp.float64(1.0)
    contacts = int(0)
    for i in range(wp.min(s.ncon[0], s.geom.shape[0])):
        if s.dist[i] > 0.0 or (s.kind[i] & 5) == 0:
            continue
        for side in range(2):
            if s.geom[i][side] != geom:
                continue
            other = s.geom[i][1-side]
            if (other < 0 and s.flex[i][1-side] >= 0) or (other >= 0 and hand_geoms[other] != 0):
                contacts += 1
                break
    velocity = point_velocity(body, wp.vec3d(s.xpos[0, body]), s.root, s.com, s.cvel)
    stable = wp.length(velocity) <= c[7] and wp.length(wp.vec3d(wp.spatial_top(s.cvel[0, body]))) <= c[8]
    on_table = wp.abs(zmin-c[5]) <= c[6]
    released = contacts == 0
    dwell = dwell_state(time[0], state[4] != wp.float64(0.0) and inside and on_table and released and stable, c[9], state)
    out[0] = state[0]
    out[1] = state[3]
    out[2] = state[4]
    out[3] = wp.float64(inside)
    out[4] = wp.float64(on_table)
    out[5] = wp.float64(released)
    out[6] = wp.float64(stable)
    out[7] = dwell
    out[8] = wp.length(wp.vec2d(center[0]-c[0], center[1]-c[1]))


@wp.kernel
def _peg(s: Scene, peg: int, socket: int, c: wp.array[wp.float64], time: wp.array[wp.float64],
         state: wp.array[wp.float64], out: wp.array[wp.float64]):
    pr = wp.mat33d(s.xmat[0, peg])
    hr = wp.transpose(wp.mat33d(s.xmat[0, socket]))
    local = wp.vec3d(c[0], c[1], c[2])
    tip = hr*(wp.vec3d(s.xpos[0, peg])+pr*local-wp.vec3d(s.xpos[0, socket]))
    rotation = hr*pr
    depth = tip[2]-c[3]
    lateral = wp.max(wp.abs(tip[0]-c[0]), wp.abs(tip[1]-c[1]))
    angle = wp.acos(wp.clamp((wp.trace(rotation)-wp.float64(1.0))/wp.float64(2.0), wp.float64(-1.0), wp.float64(1.0)))*wp.float64(180.0/wp.pi)
    aligned = lateral <= c[4] and angle <= c[5]
    inserted = aligned and depth >= c[6] and depth <= c[7]
    dwell = dwell_state(time[0], inserted, c[8], state)
    out[0] = state[0]
    out[1] = wp.float64(inserted)
    out[2] = wp.float64(aligned)
    out[3] = depth
    out[4] = lateral
    out[5] = angle
    out[6] = dwell


@wp.func
def segment(s: Scene, geom: int, half: wp.float64):
    center = wp.vec3d(s.gpos[0, geom])
    axis = wp.mat33d(s.gmat[0, geom])*wp.vec3d(0.0, 0.0, half)
    return center-axis, center+axis


@wp.kernel
def _cable(s: Scene, clip: int, geoms: wp.array[int], half: wp.array[wp.float64],
           c: wp.array[wp.float64], gates: wp.array2d[wp.float64], time: wp.array[wp.float64],
           state: wp.array[wp.float64], route_state: wp.array[wp.float64], out: wp.array[wp.float64]):
    rotation = wp.transpose(wp.mat33d(s.xmat[0, clip]))
    origin = wp.vec3d(s.xpos[0, clip])
    count = int(0)
    for k in range(3):
        x = c[0]+wp.float64(k)*wp.float64(.5)*(c[1]-c[0])
        found = bool(False)
        for i in range(geoms.shape[0]):
            a, b = segment(s, geoms[i], half[i])
            a = rotation*(a-origin)
            b = rotation*(b-origin)
            dx = b[0]-a[0]
            if wp.abs(dx) < wp.float64(1.e-10):
                continue
            fraction = (x-a[0])/dx
            point = a+fraction*(b-a)
            if fraction >= wp.float64(0.0) and fraction <= wp.float64(1.0) and point[1] >= c[2]-c[6] and point[1] <= c[3]+c[6] and point[2] >= c[4]-c[6] and point[2] <= c[5]+c[6]:
                found = bool(True)
                break
        count += int(found)
    inside = count == 3
    dwell = dwell_state(time[0], inside, c[7], state)
    out[0] = state[0]
    out[1] = wp.float64(inside)
    out[2] = dwell
    out[3] = wp.float64(count)
    previous = wp.float64(-1.0)
    count = int(0)
    for gate in range(gates.shape[0]):
        first = wp.float64(1.e30)
        for i in range(geoms.shape[0]):
            a, b = segment(s, geoms[i], half[i])
            dy = b[1]-a[1]
            if wp.abs(dy) < wp.float64(1.e-12):
                continue
            t = (gates[gate, 2]-a[1])/dy
            p = a+t*(b-a)
            order = wp.float64(i)+t
            if t >= wp.float64(0.0) and t <= wp.float64(1.0) and order > previous and p[0] >= gates[gate, 0] and p[0] <= gates[gate, 1] and p[2] >= gates[gate, 3] and p[2] <= gates[gate, 4]:
                first = wp.min(first, order)
        if first == wp.float64(1.e30):
            break
        previous = first
        count += 1
    inside = gates.shape[0] > 0 and count == gates.shape[0]
    dwell = dwell_state(time[0], inside, c[8], route_state)
    out[4] = route_state[0]
    out[5] = wp.float64(inside)
    out[6] = wp.float64(count)
    out[7] = wp.float64(gates.shape[0])
    out[8] = dwell


@wp.kernel
def _feeding(s: Scene, spoon: int, head: int, beans: wp.array[int], head_geoms: wp.array[int],
             c: wp.array[wp.float64], required: wp.array[int], time: wp.array[wp.float64],
             state: wp.array[wp.float64], out: wp.array[wp.float64]):
    sp = wp.vec3d(s.xpos[0, spoon])
    sr = wp.mat33d(s.xmat[0, spoon])
    hp = wp.vec3d(s.xpos[0, head])
    hr = wp.mat33d(s.xmat[0, head])
    target = wp.vec3d(c[0], c[1], c[2])
    radii = wp.vec3d(c[3], c[4], c[5])
    origin = wp.transpose(hr)*(sp-hp)-target
    axes = wp.transpose(hr)*sr
    local = wp.vec3d(c[6], c[7], c[8])
    if c[9] != wp.float64(0.0):
        # Exact convex box/ellipsoid distance: enumerate the 27 active sets
        # (interior, faces, edges, corners) of the same bounded least squares.
        a = axes
        for i in range(3):
            for j in range(3):
                a[i, j] = axes[i, j]/radii[i]
        b = wp.cw_div(origin, radii)
        hessian = wp.transpose(a)*a
        rhs = -(wp.transpose(a)*b)
        best = wp.float64(1.e30)
        for active in range(27):
            mode = wp.vec3i(active%3, (active//3)%3, active//9)
            fixed = wp.vec3d(0.0)
            for i in range(3):
                if mode[i] > 0:
                    fixed[i] = c[10+i] if mode[i] == 1 else c[13+i]
            mat = hessian
            right = rhs-hessian*fixed
            for i in range(3):
                if mode[i] > 0:
                    right[i] = wp.float64(0.0)
                    for j in range(3):
                        mat[i, j] = wp.float64(0.0)
                        mat[j, i] = wp.float64(0.0)
                    mat[i, i] = wp.float64(1.0)
            point = fixed+wp.inverse(mat)*right
            valid = bool(True)
            for i in range(3):
                valid = valid and point[i] >= c[10+i]-wp.float64(1.e-12) and point[i] <= c[13+i]+wp.float64(1.e-12)
            value = wp.length_sq(a*point+b)
            if valid and value < best:
                best = value
                local = point
    error = origin+axes*local
    inside = wp.length(wp.cw_div(error, radii)) <= wp.float64(1.0)
    count = int(0)
    for i in range(beans.shape[0]):
        bean = wp.transpose(sr)*(wp.vec3d(s.xpos[0, beans[i]])-sp)
        radial = wp.pow((bean[0]-c[16])/c[18], wp.float64(2.0))+wp.pow((bean[1]-c[17])/c[19], wp.float64(2.0))
        count += int(radial <= wp.float64(1.0) and bean[2] >= c[20] and bean[2] <= c[21])
    tilt = wp.acos(wp.clamp(sr[2, 2], wp.float64(-1.0), wp.float64(1.0)))*wp.float64(180.0/wp.pi)
    approach = wp.acos(wp.clamp(wp.dot(sr*wp.vec3d(1.0, 0.0, 0.0), hr*wp.vec3d(0.0, 1.0, 0.0)), wp.float64(-1.0), wp.float64(1.0)))*wp.float64(180.0/wp.pi)
    bowl = sp+sr*wp.vec3d(c[22], c[23], c[24])
    speed = wp.length(point_velocity(spoon, bowl, s.root, s.com, s.cvel))
    contacts = int(0)
    for i in range(wp.min(s.ncon[0], s.geom.shape[0])):
        if s.dist[i] > 0.0 or (s.kind[i] & 5) == 0:
            continue
        hit = bool(False)
        for j in range(2):
            g = s.geom[i][j]
            if g >= 0 and head_geoms[g] != 0:
                hit = bool(True)
        contacts += int(hit)
    checks = Checks(int(inside), int(count >= int(c[25])), int(grasped(s)),
        int(tilt <= c[26]), int(approach <= c[27]), int(speed <= c[28]), int(c[29] == wp.float64(0.0) or contacts == 0))
    eligible = bool(True)
    for i in range(7):
        eligible = eligible and (required[i] == 0 or checks[i] != 0)
    dwell = dwell_state(time[0], eligible, c[30], state)
    out[0] = wp.length(error)
    out[1] = wp.float64(inside)
    out[2] = wp.float64(count)
    out[3] = wp.float64(grasped(s))
    out[4] = tilt
    out[5] = approach
    out[6] = speed
    out[7] = wp.float64(contacts)
    out[8] = state[0]
    out[9] = wp.float64(eligible)
    out[10] = dwell


class GpuScoring:
    def __init__(self, env, sim, observations):
        self.env = weakref.proxy(env)
        self.scene = Scene()
        for name, value in dict(xpos=sim.data.xpos, xmat=sim.data.xmat, gpos=sim.data.geom_xpos,
            gmat=sim.data.geom_xmat, com=sim.data.subtree_com, cvel=sim.data.cvel,
            root=sim.model.body_rootid, ncon=sim.data.nacon, geom=sim.data.contact.geom,
            flex=sim.data.contact.flex, dist=sim.data.contact.dist, kind=sim.data.contact.type,
            forces=observations.result if env.is_optical else observations.contacts,
            force_column=3 if env.is_optical else 0).items():
            setattr(self.scene, name, value)
        self.time = wp.zeros(1, dtype=wp.float64)
        self.state = wp.array([0., -1., -1., 0., 0.], dtype=wp.float64)
        self.route_state = wp.array([0., -1., -1.], dtype=wp.float64)
        self.out = wp.zeros(11, dtype=wp.float64)
        self.last = None
        self.ready_time = None
        self.metrics = None
        m = env.model
        self.task = 'cable' if env.task_name == 'cable_insert' else env.task_name
        if self.task == 'pick_cube':
            e = env.pick_place
            c = e.config
            config = [*c['target_center_xy_m'], c['target_half_width_m']-c['marking_half_width_m'],
                e.initial_center_z, c['required_lift_m'], e.table_z, c['table_height_tolerance_m'],
                c['max_speed_m_s'], c['max_angular_speed_rad_s'], c['continuous_dwell_s']]
            mask = np.zeros(m.ngeom, dtype=np.int32)
            for g in range(m.ngeom):
                b = int(m.geom_bodyid[g])
                while b and b != env.hand:
                    b = int(m.body_parentid[b])
                mask[g] = b == env.hand
            self.mask = wp.array(mask)
            self.state.assign([float(e.success), -1. if e.since is None else e.since, -1., float(e.started_in_source), float(e.lifted)])
            self.kernel = _cube
            args = [env.probe_body, env.probe_geom, self.mask, wp.vec3d(*m.geom_size[env.probe_geom])]
        elif self.task == 'peg_insert':
            c = env.peg_score.config
            config = [*c['tip_local_m'], c['socket_entry_local_z_m'], c['max_lateral_error_m'],
                c['max_orientation_error_deg'], c['required_insertion_m'], c['maximum_insertion_m'], c['continuous_dwell_s']]
            self.kernel, args = _peg, [env.probe_body, m.body('hole').id]
        elif self.task == 'cable':
            e = env.clip_score
            config = [*e.x_bounds, *e.y_bounds, *e.z_bounds, e.config['collision_tolerance_m'], e.config['continuous_dwell_s'], 0.]
            ids = [m.geom(f'wire_G{i}').id for i in range(env.cable_count-1)]
            self.geoms, self.half = wp.array(ids, dtype=int), wp.array(m.geom_size[ids, 1], dtype=wp.float64)
            gates = []
            if hasattr(env, 'routing_score'):
                config[-1] = env.routing_score.config['scoring']['continuous_dwell_s']
                gates = [[*x, y, *z] for x, y, z in env.routing_score.gates()]
            self.gates = wp.array(np.asarray(gates).reshape(-1, 5), dtype=wp.float64)
            self.kernel, args = _cable, [m.body('cclip_0').id, self.geoms, self.half]
        elif self.task == 'feeding':
            c = env.delivery.config
            bounds = c.get('delivery_part_bounds_spoon_m', [[0.]*3]*2)
            target = np.asarray(c['mouth_center_head_m'])+[0., -c['target_offset_outside_lips_m'], 0.]
            v = c['bean_volume_spoon']
            config = [*target, *c['target_ellipsoid_radii_m'], *c.get('delivery_point_spoon_m', c['bowl_center_spoon_m']),
                float('delivery_part_bounds_spoon_m' in c), *bounds[0], *bounds[1],
                *v['center_xy_m'], *v['radii_xy_m'], *v['z_range_m'], *c['bowl_center_spoon_m'],
                c['required_beans'], c['max_tilt_deg'], c['max_approach_angle_deg'], c['max_bowl_speed_m_s'],
                float(c['require_no_head_contact']), c['continuous_dwell_s']]
            names = ('region', 'beans', 'grasp', 'tilt', 'approach', 'speed', 'head_contact')
            required = c.get('completion_checks', list(names))
            if not required or any(k not in names for k in required):
                raise ValueError('Invalid feeding completion_checks')
            self.required = wp.array([int(k in required) for k in names], dtype=int)
            self.beans = wp.array(env.bean_bodies, dtype=int)
            mask = np.zeros(m.ngeom, dtype=np.int32)
            mask[list(env.head_geoms)] = 1
            self.mask = wp.array(mask)
            self.kernel, args = _feeding, [env.probe_body, m.body('configurable_visual_head').id, self.beans, self.mask]
        else:
            raise ValueError(f'GPU scoring unsupported for {self.task}')
        self.config = wp.array(config, dtype=wp.float64)
        self.args = [self.scene, *args, self.config]
        if self.task == 'cable':
            self.args += [self.gates]
        elif self.task == 'feeding':
            self.args += [self.required]
        self.args += [self.time, self.state]
        if self.task == 'cable':
            self.args += [self.route_state]
        self.args += [self.out]
        initial, route_initial = self.state.numpy(), self.route_state.numpy()
        self.compute()
        wp.synchronize()
        self.state.assign(initial)
        self.route_state.assign(route_initial)

    def compute(self):
        wp.launch(self.kernel, 1, self.args)

    def update(self):
        time = float(self.env.data.time)
        if time == self.last:
            return dict(self.metrics)
        if self.last is not None and time < self.last:
            raise ValueError('Reset the GPU evaluator when simulation time resets')
        if time != self.ready_time:
            self.time.assign([time])
            self.compute()
        values, state = self.out.numpy(), self.state.numpy()
        self.last = time
        if self.task == 'pick_cube':
            keys = ('cube_success', 'cube_started_in_source', 'cube_lifted_from_source', 'cube_inside_target',
                'cube_on_table', 'cube_released', 'cube_stable', 'cube_dwell_s', 'cube_target_distance_m')
            values = [*map(bool, values[:7]), *map(float, values[7:9])]
            evaluator = self.env.pick_place
            evaluator.lifted = bool(state[4])
        elif self.task == 'peg_insert':
            keys = ('peg_success', 'peg_inserted', 'peg_aligned', 'peg_depth_m', 'peg_lateral_error_m', 'peg_orientation_error_deg', 'peg_dwell_s')
            values = [*map(bool, values[:3]), *map(float, values[3:7])]
            evaluator = self.env.peg_score
        elif self.task == 'cable':
            keys = ('clip_success', 'clip_inside', 'clip_dwell_s', 'clip_span_count')
            evaluator = self.env.clip_score
            if hasattr(self.env, 'routing_score'):
                route = self.env.routing_score
                route.metrics = dict(routing_success=bool(values[4]), routing_inside=bool(values[5]),
                    routing_gate_count=int(values[6]), routing_gate_total=int(values[7]), routing_dwell_s=float(values[8]))
                rs = self.route_state.numpy()
                route.success, route.since = bool(rs[0]), None if rs[1] < 0 else float(rs[1])
            values = [bool(values[0]), bool(values[1]), float(values[2]), int(values[3])]
        else:
            keys = ('delivery_distance_m', 'delivery_inside_target', 'beans_in_spoon', 'spoon_grasped',
                'spoon_tilt_deg', 'spoon_approach_angle_deg', 'spoon_bowl_speed_m_s', 'head_contact_count',
                'delivery_success', 'delivery_eligible', 'delivery_dwell_s')
            values = [float(values[0]), bool(values[1]), int(values[2]), bool(values[3]), *map(float, values[4:7]),
                int(values[7]), bool(values[8]), bool(values[9]), float(values[10])]
            evaluator = self.env.delivery
            evaluator.first_success_time = None if state[2] < 0 else float(state[2])
            evaluator.last_time = time
        evaluator.success, evaluator.since = bool(state[0]), None if state[1] < 0 else float(state[1])
        self.metrics = evaluator.metrics = dict(zip(keys, values))
        return dict(self.metrics)
