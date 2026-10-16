# Copyright 2022 DeepMind Technologies Limited
# Copyright 2026 WowSkin-Lab contributors
# SPDX-License-Identifier: Apache-2.0
"""Warp translation of MuJoCo 3.15.0's elasticity/cable.cc passive forces.

Retains the native chain, geometry, inertia, joints and constitutive parameters.
Only the CPU plugin callback is replaced, explicitly, with GPU kernels. This is
a local adapter, not upstream body-plugin support. See NOTICE and LICENSE.
"""
import copy
from functools import lru_cache
import importlib.metadata
from unittest.mock import patch

import mujoco
import numpy as np
import warp as wp
from mujoco_warp._src import math, support


@lru_cache(maxsize=1)
def _cable_slot():
    # Slot numbers vary with loaded plugins. Resolve the name via the compiler.
    probe = mujoco.MjModel.from_xml_string('''<mujoco>
      <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
      <worldbody><composite type="cable" count="4 1 1" size=".1" curve="s 0 0" initial="none">
        <plugin plugin="mujoco.elasticity.cable">
          <config key="twist" value="1"/><config key="bend" value="1"/>
        </plugin><geom type="capsule" size=".003"/>
      </composite></worldbody></mujoco>''')
    return int(probe.plugin[0])


def parameters(model):
    """Extract exactly the reference configuration used by the CPU plugin."""
    if mujoco.__version__ != '3.15.0' or importlib.metadata.version('mujoco-warp') != '3.15.0':
        raise ValueError('Cable adapter requires MuJoCo and MuJoCo Warp 3.15.0')
    if not model.nplugin or np.any(model.plugin != _cable_slot()):
        raise ValueError('Only mujoco.elasticity.cable plugins can be translated')
    for field in ('geom_plugin', 'sensor_plugin', 'actuator_plugin'):
        if np.any(getattr(model, field) >= 0):
            raise ValueError(f'Cable adapter does not support {field}')
    if model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP):
        raise ValueError('Cable adapter has not been validated with sleeping')
    reference = mujoco.MjData(model)
    reference.qpos[:] = model.qpos0
    reference.mocap_quat[:] = 0
    mujoco.mj_kinematics(model, reference)
    bodies, previous, following, qadr, stiffness, omega0 = [], [], [], [], [], []
    configs = []
    for instance in range(model.nplugin):
        ids = np.flatnonzero(model.body_plugin == instance)
        if len(ids) < 2 or not np.all(np.diff(ids) == 1):
            raise ValueError('Cable bodies must form a contiguous chain of at least two segments')
        if not np.array_equal(model.body_parentid[ids[1:]], ids[:-1]):
            raise ValueError('Cable body topology is not a chain')
        start = int(model.plugin_attradr[instance])
        end = int(model.plugin_attradr[instance+1]) if instance+1 < model.nplugin else model.npluginattr
        attrs = bytes(model.plugin_attr[start:end]).decode().split('\0')
        if len(attrs) != 5 or attrs[-1] != '':
            raise ValueError('Unexpected cable plugin attribute layout')
        twist, bend = float(attrs[0] or 0), float(attrs[1] or 0)
        if not np.isfinite([twist, bend]).all():
            raise ValueError('Non-finite cable stiffness')
        flat = attrs[2] == 'true'
        configs.append(dict(instance=instance, segments=len(ids), twist=twist, bend=bend, flat=flat))
        base = len(bodies)
        for b, body in enumerate(ids):
            first, last = b == 0, b == len(ids)-1
            joint = int(model.body_jntadr[body])
            if not first and (model.body_jntnum[body] != 1 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_BALL):
                raise ValueError('Cable interior segments require one ball joint')
            adr = -1 if first else int(model.jnt_qposadr[joint] + model.body_dofnum[body] - 3)
            rest = np.zeros(3)
            if not first and not flat:
                mujoco.mju_subQuat(rest, model.body_quat[body], reference.qpos[adr:adr+4])
            geom = int(model.body_geomadr[body])
            if geom < 0:
                raise ValueError('Cable segment has no cross-section geom')
            kind, size = int(model.geom_type[geom]), model.geom_size[geom]
            if kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
                inertia_y = inertia_z = np.pi*size[0]**4/4
                torsion = 2*inertia_y
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                h, w = size[1:3]
                a, c = max(h, w), min(h, w)
                torsion = a*c**3*(16/3 - 3.36*c/a*(1-c**4/a**4/12))
                inertia_y, inertia_z = (2*w)**3*2*h/12, (2*h)**3*2*w/12
            else:
                raise ValueError('Unsupported cable cross-section')
            length = 1.0 if first else np.linalg.norm(reference.xpos[body]-reference.xpos[body-1])
            if length <= 0:
                raise ValueError('Zero cable reference segment length')
            bodies.append(body);previous.append(-1 if first else base+b-1)
            following.append(-1 if last else base+b+1);qadr.append(adr)
            stiffness.append([torsion*twist, inertia_y*bend, inertia_z*bend, length])
            omega0.append(rest)
    return dict(bodies=bodies, previous=previous, following=following, qadr=qadr,
                stiffness=stiffness, omega0=omega0, configs=configs)


@wp.func
def _stress(world: int, index: int, pullback: bool, body: wp.array[int], adr: wp.array[int],
            stiffness: wp.array[wp.vec4], rest: wp.array[wp.vec3],
            body_quat: wp.array2d[wp.quat], qpos: wp.array2d[float]) -> wp.vec3:
    a = adr[index]
    q = math.mul_quat(body_quat[world % body_quat.shape[0],body[index]], wp.quat(qpos[world,a], qpos[world,a+1], qpos[world,a+2], qpos[world,a+3]))
    omega = math.quat_to_vel(q) - rest[index]
    k = stiffness[index]
    torque = wp.vec3(-k[0]*omega[0]/k[3], -k[1]*omega[1]/k[3], -k[2]*omega[2]/k[3])
    if pullback:
        torque = math.rot_vec_quat(torque, math.quat_inv(q))
    return torque


@wp.kernel
def _torques(body: wp.array[int], previous: wp.array[int], following: wp.array[int],
             adr: wp.array[int], stiffness: wp.array[wp.vec4], rest: wp.array[wp.vec3],
             body_quat: wp.array2d[wp.quat], qpos: wp.array2d[float],
             xquat: wp.array2d[wp.quat], forces: wp.array2d[wp.spatial_vector]):
    world, i = wp.tid()
    local = wp.vec3(0.0)
    k = stiffness[i]
    if k[0] != 0.0 or k[1] != 0.0 or k[2] != 0.0:
        if previous[i] >= 0:
            local += _stress(world, i, True, body, adr, stiffness, rest, body_quat, qpos)
        if following[i] >= 0:
            local -= _stress(world, following[i], False, body, adr, stiffness, rest, body_quat, qpos)
    torque = math.rot_vec_quat(local, xquat[world, body[i]])
    forces[world, body[i]] = wp.spatial_vector(0.0, 0.0, 0.0, torque[0], torque[1], torque[2])


class CableForces:
    def __init__(self, model):
        p = parameters(model)
        self.configs = p.pop('configs')
        self.count = len(p['bodies'])
        self.arrays = {key: wp.array(value, dtype=wp.vec4 if key == 'stiffness' else wp.vec3 if key == 'omega0' else int)
                       for key, value in p.items()}
        self.forces = None

    def import_model(self, native_model):
        import mujoco_warp as mjw
        from mujoco_warp._src import io
        # Detach only CPU dispatch on a private import copy. The actual force
        # law is reattached below, before any GPU step. Never mutate the source.
        imported = copy.copy(native_model)
        imported.body_plugin[:] = -1
        # io filters its plugin arrays to SDF/geom plugins, but 3.15.0 retains
        # the unfiltered count. All plugins here are validated cable plugins;
        # there are zero GPU SDF plugins. Scope this metadata fix to import.
        create_array = io._create_array
        def arrays_without_cpu_plugins(value, spec, sizes, batch_size=1):
            return create_array(value, spec, dict(sizes, nplugin=0), batch_size)
        with patch.object(io, '_create_array', arrays_without_cpu_plugins):
            gpu = mjw.put_model(imported)
        gpu.nplugin = 0
        if gpu.callback.passive is not None:
            raise ValueError('Refusing to overwrite an existing passive callback')
        gpu.callback.passive = self.apply
        return gpu

    def allocate(self, data, nbody):
        self.forces = wp.zeros((data.nworld, nbody), dtype=wp.spatial_vector)

    def apply(self, model, data):
        a = self.arrays
        wp.launch(_torques, (data.nworld, self.count),
                  [a['bodies'], a['previous'], a['following'], a['qadr'], a['stiffness'],
                   a['omega0'], model.body_quat, data.qpos, data.xquat, self.forces])
        support.apply_ft(model, data, self.forces, data.qfrc_passive, True)
