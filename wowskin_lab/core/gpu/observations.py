"""GPU tactile geometry, magnetic forward model and optical contact readout.

These kernels preserve the source observation definitions. No force clipping,
made-up contact depths, material changes, or substituted optical images.
"""
import weakref

import numpy as np
import mujoco_warp as mjw
import warp as wp
from mujoco_warp._src import types


@wp.func
def point_velocity(body: int, point: wp.vec3d, root: wp.array[int],
                   com: wp.array2d[wp.vec3], cvel: wp.array2d[wp.spatial_vector]):
    velocity = cvel[0, body]
    return wp.vec3d(wp.spatial_bottom(velocity)) + wp.cross(
        wp.vec3d(wp.spatial_top(velocity)), point-wp.vec3d(com[0, root[body]]))


@wp.kernel
def _optical_contacts(ncon: wp.array[int], world: wp.array[int], kind: wp.array[int],
                      geom: wp.array[wp.vec2i], pos: wp.array[wp.vec3], frame: wp.array[wp.mat33],
                      force: wp.array[wp.spatial_vector], body: wp.array[int], root: wp.array[int],
                      com: wp.array2d[wp.vec3], cvel: wp.array2d[wp.spatial_vector],
                      gel: wp.array2d[int], objects: wp.array[int], result: wp.array2d[wp.float64]):
    side = wp.tid()
    total = wp.float64(0.0)
    weighted = wp.float64(0.0)
    count = wp.float64(0.0)
    target = wp.float64(0.0)
    target_weighted = wp.float64(0.0)
    target_count = wp.float64(0.0)
    for i in range(wp.min(ncon[0], geom.shape[0])):
        if world[i] != 0 or (kind[i] & int(types.ContactType.CONSTRAINT | types.ContactType.PASSIVE)) == 0:
            continue
        a = geom[i][0]
        b = geom[i][1]
        if a < 0 or b < 0:
            continue
        other = int(-1)
        if gel[side, a] != 0:
            other = b
        elif gel[side, b] != 0:
            other = a
        fn = wp.max(wp.float64(force[i][0]), wp.float64(0.0))
        if other < 0 or fn <= wp.float64(0.0):
            continue
        p = wp.vec3d(pos[i])
        relative = point_velocity(body[b], p, root, com, cvel)-point_velocity(body[a], p, root, com, cvel)
        normal = wp.vec3d(frame[i][0])
        speed = wp.length(relative-normal*wp.dot(relative, normal))
        total += fn
        weighted += fn*speed
        count += wp.float64(1.0)
        if objects[other] != 0:
            target += fn
            target_weighted += fn*speed
            target_count += wp.float64(1.0)
    result[side, 0] = total
    result[side, 1] = weighted/wp.max(total, wp.float64(1.e-30))
    result[side, 2] = count
    result[side, 3] = target
    result[side, 4] = target_weighted/wp.max(target, wp.float64(1.e-30))
    result[side, 5] = target_count


@wp.kernel
def _deformation(vertices: wp.array2d[wp.vec3], xpos: wp.array2d[wp.vec3],
                 xmat: wp.array2d[wp.mat33], mount: int, tets: wp.array[wp.vec4i],
                 inverse: wp.array[wp.mat33d], reference: wp.array[wp.vec3d],
                 centres: wp.array[wp.vec3d], moments: wp.array[wp.vec3d], ratios: wp.array[wp.float64]):
    i = wp.tid()
    ids = tets[i]
    r = wp.transpose(wp.mat33d(xmat[0, mount]))
    p = wp.vec3d(xpos[0, mount])
    a = r*(wp.vec3d(vertices[0, ids[0]])-p)
    b = r*(wp.vec3d(vertices[0, ids[1]])-p)
    c = r*(wp.vec3d(vertices[0, ids[2]])-p)
    d = r*(wp.vec3d(vertices[0, ids[3]])-p)
    ds = wp.matrix_from_cols(b-a, c-a, d-a)
    deformation = ds*inverse[i]
    ratios[i] = wp.determinant(deformation)
    # For positive-volume states the polar rotation is U V^T, as in NumPy.
    u, sigma, v = wp.svd3(deformation)
    moments[i] = u*wp.transpose(v)*reference[i]
    centres[i] = (a+b+c+d)*wp.float64(.25)


@wp.kernel
def _field(sensors: wp.array[wp.vec3d], rotations: wp.array[wp.mat33d],
           gains: wp.array[wp.float64], centres: wp.array[wp.vec3d],
           moments: wp.array[wp.vec3d], result: wp.array[wp.vec3d]):
    i = wp.tid()
    value = wp.vec3d(0.0)
    for j in range(centres.shape[0]):
        r = sensors[i]-centres[j]
        distance = wp.length(r)
        if distance < wp.float64(1.e-5):
            value = wp.vec3d(wp.float64(wp.nan))
            break
        unit = r/distance
        value += (wp.float64(3.0)*unit*wp.dot(unit, moments[j])-moments[j])/(distance*distance*distance)
    result[i] = wp.transpose(rotations[i])*value*(wp.float64(.1)*gains[i])


@wp.kernel
def _quality(ratios: wp.array[wp.float64], output: wp.array[wp.float64], index: int):
    minimum = wp.float64(1.e30)
    for i in range(ratios.shape[0]):
        if not wp.isfinite(ratios[i]):
            minimum = wp.float64(-1.0)
            break
        minimum = wp.min(minimum, ratios[i])
    output[index] = minimum


@wp.kernel
def _vertex_velocities(weights: wp.array2d[wp.float64], sides: wp.array[int],
                       nodes: wp.array2d[int], local: wp.array2d[wp.vec3d],
                       xpos: wp.array2d[wp.vec3], xmat: wp.array2d[wp.mat33],
                       root: wp.array[int], com: wp.array2d[wp.vec3],
                       cvel: wp.array2d[wp.spatial_vector], velocities: wp.array[wp.vec3d]):
    i = wp.tid()
    side = sides[i]
    if side < 0:
        return
    value = wp.vec3d(0.0)
    for j in range(8):
        body = nodes[side, j]
        point = wp.vec3d(xpos[0, body])+wp.mat33d(xmat[0, body])*local[side, j]
        value += weights[i, j]*point_velocity(body, point, root, com, cvel)
    velocities[i] = value


@wp.kernel
def _skin_contacts(ncon: wp.array[int], world: wp.array[int], kind: wp.array[int],
                   geom: wp.array[wp.vec2i], flex: wp.array[wp.vec2i],
                   vert: wp.array[wp.vec2i], elem: wp.array[wp.vec2i],
                   pos: wp.array[wp.vec3], frame: wp.array[wp.mat33],
                   force: wp.array[wp.spatial_vector], body: wp.array[int], root: wp.array[int],
                   com: wp.array2d[wp.vec3], cvel: wp.array2d[wp.spatial_vector],
                   vertices: wp.array2d[wp.vec3], velocities: wp.array[wp.vec3d],
                   tets: wp.array[wp.vec4i], pads: wp.array[int], objects: wp.array[int],
                   result: wp.array2d[wp.float64]):
    pad = wp.tid()
    total = wp.float64(0.0)
    weighted = wp.float64(0.0)
    count = wp.float64(0.0)
    all_force = wp.float64(0.0)
    net = wp.vec3d(0.0)
    for i in range(wp.min(ncon[0], geom.shape[0])):
        if world[i] != 0 or (kind[i] & int(types.ContactType.CONSTRAINT | types.ContactType.PASSIVE)) == 0:
            continue
        side = int(-1)
        if geom[i][0] < 0 and flex[i][0] == pads[pad]:
            side = 0
        elif geom[i][1] < 0 and flex[i][1] == pads[pad]:
            side = 1
        if side < 0:
            continue
        fn = wp.max(wp.float64(force[i][0]), wp.float64(0.0))
        all_force += fn
        other = int(-1)
        for j in range(2):
            g = geom[i][j]
            if g >= 0 and objects[g] != 0:
                other = g
                break
        if other < 0 or fn < wp.float64(1.e-8):
            continue
        skin_v = wp.vec3d(0.0)
        v = vert[i][side]
        if v >= 0:
            skin_v = velocities[v]
        else:
            ids = tets[elem[i][side]]
            a = wp.vec3d(vertices[0, ids[0]])
            ds = wp.matrix_from_cols(wp.vec3d(vertices[0, ids[1]])-a,
                wp.vec3d(vertices[0, ids[2]])-a, wp.vec3d(vertices[0, ids[3]])-a)
            # Inverse uses an absolute determinant cutoff. Millimetre-sized
            # tetrahedra must be scaled before inversion, not treated as a
            # singular matrix merely because their SI volume is small.
            scale = wp.sqrt(wp.ddot(ds, ds))
            bary = wp.inverse(ds/scale)*((wp.vec3d(pos[i])-a)/scale)
            weights = wp.vec4d(wp.float64(1.0)-bary[0]-bary[1]-bary[2], bary[0], bary[1], bary[2])
            norm = wp.float64(0.0)
            for j in range(4):
                weights[j] = wp.clamp(weights[j], wp.float64(0.0), wp.float64(1.0))
                norm += weights[j]
            for j in range(4):
                skin_v += weights[j]/norm*velocities[ids[j]]
        relative = point_velocity(body[other], wp.vec3d(pos[i]), root, com, cvel)-skin_v
        normal = wp.vec3d(frame[i][0])
        weighted += fn*wp.length(relative-normal*wp.dot(relative, normal))
        total += fn
        sign = wp.float64(-1.0)
        if geom[i][1] == other:
            sign = wp.float64(1.0)
        net += sign*(wp.transpose(wp.mat33d(frame[i]))*wp.vec3d(wp.spatial_top(force[i])))
        count += wp.float64(1.0)
    result[pad, 0] = total
    result[pad, 1] = weighted/wp.max(total, wp.float64(1.e-30))
    result[pad, 2] = count
    for j in range(3):
        result[pad, 3+j] = net[j]
    result[pad, 6] = all_force


@wp.kernel
def _robot_velocity(site: int, hand: int, base: int,
                    sites: wp.array2d[wp.vec3], xpos: wp.array2d[wp.vec3],
                    xmat: wp.array2d[wp.mat33], root: wp.array[int],
                    com: wp.array2d[wp.vec3], cvel: wp.array2d[wp.spatial_vector],
                    output: wp.array2d[wp.float64]):
    point = wp.vec3d(sites[0, site])
    origin = wp.vec3d(xpos[0, base])
    linear = point_velocity(hand, point, root, com, cvel)
    angular = wp.vec3d(wp.spatial_top(cvel[0, hand]))
    base_linear = point_velocity(base, origin, root, com, cvel)
    base_angular = wp.vec3d(wp.spatial_top(cvel[0, base]))
    rotation = wp.transpose(wp.mat33d(xmat[0, base]))
    relative = rotation*(linear-base_linear-wp.cross(base_angular, point-origin))
    spin = rotation*(angular-base_angular)
    for i in range(3):
        output[0, i] = relative[i]
        output[0, i+3] = spin[i]
        output[1, i] = linear[i]
        output[1, i+3] = angular[i]


class GpuObservations:
    def __init__(self, env, sim):
        self.env, self.sim = weakref.proxy(env), sim
        self.contact_ids = wp.array(np.arange(sim.data.naconmax), dtype=int)
        self.forces = wp.zeros(sim.data.naconmax, dtype=wp.spatial_vector)
        self.optical = env.is_optical
        self.values = None
        from wowskin_lab.data.robot_state import BASE_BODY
        self.base = env.model.body(BASE_BODY).id
        self.tcp_velocity = wp.empty((2, 6), dtype=wp.float64)
        objects = np.zeros(env.model.ngeom, dtype=np.int32)
        objects[list(env.contact_geoms)] = 1
        self.objects = wp.array(objects)
        if self.optical:
            gels = np.zeros((2, env.model.ngeom), dtype=np.int32)
            for i, side in enumerate(('left', 'right')):
                gels[i, list(env.optical_gel_geoms[side])] = 1
            self.gels = wp.array(gels)
            self.result = wp.zeros((2, 6), dtype=wp.float64)
        else:
            self.pads = []
            for flex, mount, tets, inverse in (
                (env.skin_flex, env.mount_body, env.tets, env.dm_inverse),
                (env.passive_flex, env.passive_mount, env.passive_tets, env.passive_dm_inverse)):
                n = len(tets)
                reference = env.reference_moments if flex == env.skin_flex else np.zeros((n, 3))
                self.pads.append(dict(mount=mount, tets=wp.array(tets+env.model.flex_vertadr[flex], dtype=wp.vec4i),
                    inverse=wp.array(inverse, dtype=wp.mat33d), reference=wp.array(reference, dtype=wp.vec3d),
                    centres=wp.empty(n, dtype=wp.vec3d), moments=wp.empty(n, dtype=wp.vec3d),
                    ratios=wp.empty(n, dtype=wp.float64)))
            self.sensors = wp.array(env.sensor_xyz, dtype=wp.vec3d)
            self.rotations = wp.array(env.rotations, dtype=wp.mat33d)
            self.gains = wp.array(env.magnetic_gains, dtype=wp.float64)
            self.magnetic = wp.empty(5, dtype=wp.vec3d)
            self.quality = wp.empty(2, dtype=wp.float64)
            m = env.model
            self.flex_ids = (env.skin_flex, env.passive_flex)
            if any(m.flex_interp[f] != 1 or m.flex_nodenum[f] != 8 for f in self.flex_ids):
                raise ValueError('GPU tactile slip readout requires the existing trilinear skin')
            nodes, local = [], []
            sides = np.full(m.nflexvert, -1, dtype=np.int32)
            weights = np.zeros((m.nflexvert, 8))
            tets = np.zeros((m.nflexelem, 4), dtype=np.int32)
            corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
            for side, f in enumerate(self.flex_ids):
                ns = int(m.flex_nodeadr[f])
                nodes.append(m.flex_nodebodyid[ns:ns+8])
                local.append(m.flex_node[ns:ns+8])
                start, size = int(m.flex_vertadr[f]), int(m.flex_vertnum[f])
                sides[start:start+size] = side
                xyz = m.flex_vert0[start:start+size]
                weights[start:start+size] = np.prod(np.where(corners[None], xyz[:, None], 1-xyz[:, None]), axis=2)
                element_start = int(m.flex_elemadr[f])
                native_tets = env.tets if side == 0 else env.passive_tets
                tets[element_start:element_start+len(native_tets)] = native_tets+start
            self.nodes = wp.array(np.asarray(nodes), dtype=int)
            self.local_nodes = wp.array(np.asarray(local), dtype=wp.vec3d)
            self.weights, self.sides = wp.array(weights, dtype=wp.float64), wp.array(sides)
            self.flexes = wp.array(self.flex_ids, dtype=int)
            self.tets = wp.array(tets, dtype=wp.vec4i)
            self.velocities = wp.zeros(m.nflexvert, dtype=wp.vec3d)
            self.contacts = wp.empty((2, 7), dtype=wp.float64)
        self.compute()
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            self.compute()
        self.graph = capture.graph

    def compute(self):
        m, d = self.sim.model, self.sim.data
        wp.launch(_robot_velocity, 1, [self.env.tcp_site, self.env.hand, self.base,
            d.site_xpos, d.xpos, d.xmat, m.body_rootid, d.subtree_com, d.cvel, self.tcp_velocity])
        mjw.contact_force(m, d, self.contact_ids, False, self.forces)
        if self.optical:
            wp.launch(_optical_contacts, 2, [d.nacon, d.contact.worldid, d.contact.type,
                d.contact.geom, d.contact.pos, d.contact.frame, self.forces,
                m.geom_bodyid, m.body_rootid, d.subtree_com, d.cvel, self.gels, self.objects, self.result])
        else:
            for i, pad in enumerate(self.pads):
                wp.launch(_deformation, len(pad['tets']), [d.flexvert_xpos, d.xpos, d.xmat, pad['mount'],
                    pad['tets'], pad['inverse'], pad['reference'], pad['centres'], pad['moments'], pad['ratios']])
                wp.launch(_quality, 1, [pad['ratios'], self.quality, i])
            pad = self.pads[0]
            wp.launch(_field, 5, [self.sensors, self.rotations, self.gains, pad['centres'], pad['moments'], self.magnetic])
            wp.launch(_vertex_velocities, self.env.model.nflexvert, [self.weights, self.sides,
                self.nodes, self.local_nodes, d.xpos, d.xmat, m.body_rootid, d.subtree_com, d.cvel, self.velocities])
            wp.launch(_skin_contacts, 2, [d.nacon, d.contact.worldid, d.contact.type, d.contact.geom,
                d.contact.flex, d.contact.vert, d.contact.elem, d.contact.pos, d.contact.frame,
                self.forces, m.geom_bodyid, m.body_rootid, d.subtree_com, d.cvel, d.flexvert_xpos,
                self.velocities, self.tets, self.flexes, self.objects, self.contacts])

    def read(self, *, compute=True):
        if compute:
            wp.capture_launch(self.graph)
        self.velocity_values = self.tcp_velocity.numpy()
        if self.optical:
            self.values = self.result.numpy()
        else:
            self.values = self.magnetic.numpy()
            self.quality_values = self.quality.numpy()
            self.contact_values = self.contacts.numpy()

    def skin_contact_state(self, flex):
        values = self.contact_values[self.flex_ids.index(flex)]
        force, speed, count = values[:3]
        contact = force >= self.env.cfg['contact_threshold_n']
        return dict(contact=int(contact), slip=int(contact and speed >= self.env.cfg['slip_speed_m_per_s']),
                    normal_force_n=float(force), slip_speed_m_per_s=float(speed),
                    contacts=int(count), force_on_probe_n=values[3:6].copy())

    def contact_state(self, side, objects):
        if objects is not None and set(objects) != set(self.env.contact_geoms):
            return None  # An explicit diagnostic filter uses the native mirror.
        offset = 0 if objects is None else 3
        force, speed, count = self.values[0 if side == 'left' else 1, offset:offset+3]
        return dict(normal_force_n=float(force), slip_speed_m_per_s=float(speed),
                    contacts=int(count), contact=bool(count))
