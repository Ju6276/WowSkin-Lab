"""Observation snapshots of a single Warp world, without exporting solver workspaces.

The native MjData is a read-only observation/rendering mirror, not a CPU dynamics
state. Forces are the solved Warp constraints in native contact order. Use the
upstream get_data_into explicitly when inspecting mass/constraint matrices.
"""
import mujoco
import numpy as np
import warp as wp
from mujoco_warp._src import types


# Kinematics, measured state, forces, and render inputs. In particular there is
# no mass matrix, LDL factor, constraint Jacobian, or flex Hessian here.
FIELDS = (
    'qpos', 'qvel', 'act', 'qacc', 'qacc_warmstart', 'ctrl',
    'qfrc_applied', 'xfrc_applied', 'mocap_pos', 'mocap_quat',
    'xpos', 'xquat', 'xmat', 'xipos', 'ximat', 'xanchor', 'xaxis',
    'geom_xpos', 'geom_xmat', 'site_xpos', 'site_xmat',
    'cam_xpos', 'cam_xmat', 'light_xpos', 'light_xdir',
    'subtree_com', 'cdof', 'cvel', 'flexvert_xpos',
    'flexedge_length', 'flexedge_velocity', 'qfrc_bias', 'qfrc_actuator',
    'actuator_force', 'sensordata', 'ten_length', 'wrap_xpos',
)
CONTACT_FIELDS = (
    ('dist', 1), ('pos', 3), ('frame', 9), ('includemargin', 1),
    ('friction', 5), ('solref', 2), ('solreffriction', 2), ('solimp', 5),
    ('adhesion', 1), ('dim', 1), ('geom', 2), ('flex', 2),
    ('elem', 2), ('vert', 2), ('exclude', 1), ('efc_address', 1),
)
WIDTH = sum(size for _, size in CONTACT_FIELDS)


@wp.struct
class ContactArrays:
    # Preserve the pinned upstream array types without duplicating vec5 aliases.
    __annotations__ = dict(types.Contact.__annotations__)


@wp.struct
class ExportArrays:
    contact: ContactArrays
    ne: wp.array[int]
    nf: wp.array[int]
    nl: wp.array[int]
    nacon: wp.array[int]
    solver_niter: wp.array[int]
    force: wp.array2d[float]
    naconmax: int
    njmax: int


@wp.kernel
def _contacts(d: ExportArrays, cone: int, has_flex: bool,
              elemadr: wp.array[int], vertadr: wp.array[int],
              rows: wp.array2d[float], forces: wp.array[float], header: wp.array[int]):
    count = int(0)
    offset = d.ne[0] + d.nf[0] + d.nl[0]
    for i in range(offset):
        forces[i] = d.force[0, i]
    for i in range(wp.min(d.nacon[0], d.naconmax)):
        if d.contact.worldid[i] != 0 or (d.contact.type[i] & int(types.ContactType.CONSTRAINT | types.ContactType.PASSIVE)) == 0:
            continue
        rows[count, 0] = d.contact.dist[i]
        for j in range(3):
            rows[count, 1+j] = d.contact.pos[i][j]
            for k in range(3):
                rows[count, 4+3*j+k] = d.contact.frame[i][j, k]
        rows[count, 13] = d.contact.includemargin[i]
        for j in range(5):
            rows[count, 14+j] = d.contact.friction[i][j]
            rows[count, 23+j] = d.contact.solimp[i][j]
        for j in range(2):
            rows[count, 19+j] = d.contact.solref[i][j]
            rows[count, 21+j] = d.contact.solreffriction[i][j]
            rows[count, 30+j] = float(d.contact.geom[i][j])
            f = int(-1)
            e = int(-1)
            v = int(-1)
            if has_flex and d.contact.geom[i][j] < 0:
                f = d.contact.flex[i][j]
                e = d.contact.elem[i][j]
                v = d.contact.vert[i][j]
                if f >= 0:
                    if e >= 0:
                        e -= elemadr[f]
                    if v >= 0:
                        v -= vertadr[f]
            rows[count, 32+j] = float(f)
            rows[count, 34+j] = float(e)
            rows[count, 36+j] = float(v)
        rows[count, 28] = d.contact.adhesion[i]
        dim = d.contact.dim[i]
        rows[count, 29] = float(dim)
        rows[count, 38] = float(0)
        if (d.contact.type[i] & int(types.ContactType.PASSIVE)) != 0:
            rows[count, 38] = float(4)
        rows[count, 39] = float(-1)
        if d.contact.efc_address[i][0] >= 0:
            rows[count, 39] = float(offset)
            size = dim
            if cone == int(types.ConeType.PYRAMIDAL):
                size = wp.max(1, 2*(dim-1))
            for j in range(size):
                if offset+j < forces.shape[0]:
                    forces[offset+j] = d.force[0, d.contact.efc_address[i, j]]
            offset += size
        count += 1
    header[0] = count
    header[1] = wp.min(offset, d.njmax)
    header[2] = d.ne[0]
    header[3] = d.nf[0]
    header[4] = d.nl[0]
    header[5] = d.solver_niter[0]


class ObservationSnapshot:
    def __init__(self, sim):
        self.sim = sim
        self.arrays = ExportArrays()
        self.arrays.contact = ContactArrays()
        for name in types.Contact.__annotations__:
            setattr(self.arrays.contact, name, getattr(sim.data.contact, name))
        for name in ('ne', 'nf', 'nl', 'nacon', 'solver_niter', 'naconmax', 'njmax'):
            setattr(self.arrays, name, getattr(sim.data, name))
        self.arrays.force = sim.data.efc.force
        self.fields = []
        offset = 0
        for name in FIELDS:
            source = getattr(sim.data, name)
            native = getattr(sim.initial_data, name)
            if not native.size:
                continue
            flat = source.flatten().view(wp.float32).flatten()
            if flat.size != native.size:
                raise ValueError(f'Observation shape mismatch for {name}: {flat.size} != {native.size}')
            self.fields.append((name, offset, flat))
            offset += flat.size
        self.device = wp.empty(offset, dtype=float)
        self.host = wp.empty(offset, dtype=float, device='cpu', pinned=True)
        self.header = wp.empty(6, dtype=int)
        self.host_header = wp.empty_like(self.header, device='cpu', pinned=True)
        self.contacts = wp.empty((sim.data.naconmax, WIDTH), dtype=float)
        self.host_contacts = wp.empty_like(self.contacts, device='cpu', pinned=True)
        self.force = wp.empty(sim.data.njmax, dtype=float)
        self.host_force = wp.empty_like(self.force, device='cpu', pinned=True)
        self.elemadr = wp.array(sim.native_model.flex_elemadr, dtype=int)
        self.vertadr = wp.array(sim.native_model.flex_vertadr, dtype=int)
        self.pack()
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            self.pack()
        self.graph = capture.graph

    def pack(self):
        for _, offset, source in self.fields:
            wp.copy(self.device, source, dest_offset=offset)
        wp.launch(_contacts, 1, [self.arrays, int(self.sim.native_model.opt.cone),
                  bool(self.sim.native_model.nflex), self.elemadr, self.vertadr,
                  self.contacts, self.force, self.header])

    def download(self, result):
        wp.capture_launch(self.graph)
        wp.copy(self.host, self.device)
        wp.copy(self.host_header, self.header)
        wp.synchronize_stream()
        ncon, nefc, ne, nf, nl, iterations = map(int, self.host_header.numpy())
        wp.copy(self.host_contacts.flatten(), self.contacts.flatten(), count=ncon*WIDTH)
        wp.copy(self.host_force, self.force, count=nefc)
        wp.synchronize_stream()
        values = self.host.numpy()
        for name, offset, source in self.fields:
            target = getattr(result, name)
            target[:] = values[offset:offset+source.size].reshape(target.shape)
        # Allocate native contact storage, but no dense/sparse Jacobian export.
        # The untouched solver matrices are deliberately not CPU solve inputs.
        if ncon != result.ncon or nefc != result.nefc:
            mujoco._functions._realloc_con_efc(result, ncon=ncon, nefc=nefc, nJ=0)
        result.ncon, result.ne, result.nf, result.nl = ncon, ne, nf, nl
        result.solver_niter[0] = iterations
        rows = self.host_contacts.numpy()[:ncon]
        offset = 0
        for name, width in CONTACT_FIELDS:
            target = getattr(result.contact, name)
            target[:] = rows[:, offset:offset+width].reshape(target.shape)
            offset += width
        result.contact.geom1[:] = result.contact.geom[:, 0]
        result.contact.geom2[:] = result.contact.geom[:, 1]
        result.efc_force[:] = self.host_force.numpy()[:nefc]
