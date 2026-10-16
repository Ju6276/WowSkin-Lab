"""CUDA physics behind the existing environment and recorder interfaces.

Initialization/reset stays on CPU. The runtime controller and observations use
CUDA; native rendering/recording receives an observation mirror without solver
workspaces. No CPU physics fallback on a GPU error.
"""
import importlib.metadata

import weakref

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from mujoco_warp._src import types

from .runtime import GpuSimulation, capture_step
from ..stepping import robot_control


@wp.kernel
def _audit(q: wp.array2d[float], v: wp.array2d[float], flags: wp.array[int],
           iterations: wp.array[int], health: wp.array[int]):
    i = wp.tid()
    if i == 0:
        wp.atomic_or(health, 0, flags[0])
        wp.atomic_max(health, 1, iterations[0])
    if i < q.shape[1] and not wp.isfinite(q[0, i]):
        wp.atomic_or(health, 0, 1073741824)
    if i < v.shape[1] and not wp.isfinite(v[0, i]):
        wp.atomic_or(health, 0, 1073741824)


@wp.kernel
def _head_interval_start(metrics: wp.array[wp.float64]):
    metrics[4] = wp.float64(0.0)


@wp.kernel
def _set_time(endpoint: wp.array[wp.float64], time: wp.array[float]):
    time[0] = float(endpoint[0])


@wp.kernel
def _external_force(body: int, wrench: wp.array[wp.spatial_vector], until: wp.array[wp.float64],
                    tick: wp.array[wp.int64], start: wp.float64, dt: wp.float64,
                    applied: wp.array2d[wp.spatial_vector]):
    value = wrench[0]
    if start+wp.float64(tick[0])*dt < until[0]:
        value[2] -= .5
    applied[0, body] = value
    tick[0] += wp.int64(1)


@wp.kernel
def _head_contact_metrics(ncon: wp.array[int], world: wp.array[int],
                          kind: wp.array[int], geom: wp.array[wp.vec2i],
                          frame: wp.array[wp.mat33], force: wp.array[wp.spatial_vector],
                          head: wp.array[int], timestep: wp.float64, metrics: wp.array[wp.float64]):
    # Match Feeding.after_physics_step: sum magnitudes, signed world force,
    # every-step peaks and impulse. Keep the original contact ordering.
    total = wp.float64(0.0)
    net = wp.vec3d(0.0)
    for i in range(wp.min(ncon[0], geom.shape[0])):
        if world[i] != 0 or (kind[i] & int(types.ContactType.CONSTRAINT | types.ContactType.PASSIVE)) == 0:
            continue
        left = int(0)
        right = int(0)
        if geom[i][0] >= 0:
            left = head[geom[i][0]]
        if geom[i][1] >= 0:
            right = head[geom[i][1]]
        if left != 0 or right != 0:
            local = wp.vec3d(wp.spatial_top(force[i]))
            total += wp.length(local)
            sign = wp.float64(-1.0)
            if right != 0:
                sign = wp.float64(1.0)
            net += sign * (wp.transpose(wp.mat33d(frame[i])) * local)
    metrics[0] = total
    metrics[1] = net[0]
    metrics[2] = net[1]
    metrics[3] = net[2]
    metrics[4] = wp.max(metrics[4], total)
    metrics[5] = wp.max(metrics[5], total)
    metrics[6] += total * timestep


class WarpBackend:
    def __init__(self, env):
        from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv
        if not isinstance(env, PandaPickCubeTactileEnv):
            raise ValueError('Warp collection currently requires a robot task environment')
        if hasattr(env, 'fracture'):
            raise ValueError('Warp chip fracture is not supported: dynamic fragment collision masks and sparse weld readout require a separate adapter. Use CPU for chip_tray.')
        if mujoco.__version__ != '3.15.0' or importlib.metadata.version('mujoco-warp') != '3.15.0':
            raise ValueError('GPU collection requires MuJoCo and MuJoCo Warp 3.15.0')
        self.env = weakref.proxy(env)
        if env.predictor is not None:
            env.predictor.model.to('cuda')
        if env.reskin_response is not None:
            env.reskin_response.model.model.to('cuda')
        self.faulted = False
        self.steps = 0
        self.start_time = float(env.data.time)
        m, d = env.model, env.data
        if not hasattr(env, '_gpu_source_options'):
            env._gpu_source_options = {name: getattr(m.opt, name) for name in
                ('disableflags', 'jacobian', 'noslip_iterations', 'iterations', 'tolerance')}
        changes = []
        if env.cfg.get('gpu_disable_multiccd'):
            m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
            changes.append('MULTICCD explicitly disabled; original margins retained')
        if env.cfg.get('gpu_disable_nativeccd'):
            m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
            changes.append('NATIVECCD explicitly disabled; original mesh contact margins retained')
        if env.cfg.get('gpu_disable_noslip'):
            m.opt.noslip_iterations = 0
            changes.append('Extra no-slip solver explicitly disabled; friction coefficients retained')
        elif m.opt.noslip_iterations:
            raise ValueError('MuJoCo Warp has no no-slip solver. Use CPU, or explicitly pass --gpu-disable-noslip and validate grasp/slip behavior.')
        if m.nv > 60:
            m.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
            changes.append('Sparse Jacobian required by Warp above 60 DoFs')
        if env.cfg.get('gpu_solver_iterations') is not None:
            m.opt.iterations = env.cfg['gpu_solver_iterations']
            changes.append(f'Explicit constraint iteration limit {m.opt.iterations}')
        if env.cfg.get('gpu_solver_tolerance') is not None:
            m.opt.tolerance = env.cfg['gpu_solver_tolerance']
            changes.append(f'Explicit constraint tolerance {m.opt.tolerance}')
        # Rebuild CPU constraint storage once before upload; no CPU dynamics
        # solve is performed on the normal acquisition path after this point.
        mujoco.mj_forward(m, d)
        self.gpu_pd = (env.cfg.get('arm_controller', 'ik_pd') == 'ik_pd'
                       and env.control.__func__ is PandaPickCubeTactileEnv.control
                       and not env.cfg.get('finger_contact_stop')
                       and env.cfg.get('gripper_control_mode') in
                       ('panda_menagerie_position', 'panda_robosuite_position', 'ebim_robotiq_position'))
        arm = dict(qids=env.arm_q, vids=env.arm_v, aids=env.arm_controls,
                   target=env.joint_target) if self.gpu_pd else None
        self.sim = GpuSimulation(m, d, arm=arm, cable_plugin_warp=bool(m.nplugin))
        if self.sim.cable:
            changes.append('Native cable elasticity translated to Warp passive-force kernels')
        self.health = wp.zeros(2, dtype=int)
        self.ticks = wp.zeros(1, dtype=wp.int64)
        self.mouse = wp.zeros(1, dtype=wp.spatial_vector)
        self.tug_until = wp.zeros(1, dtype=wp.float64)
        self.sim.compile()
        self.last_diagnostics = {}
        self.max_solver_iterations = 0
        # Reject an invalid initial solve before observation kernels or graph
        # construction can consume its contacts and constraint addresses.
        mjw.forward(self.sim.model, self.sim.data)
        self._check()
        self.validate()
        from .snapshot import ObservationSnapshot
        self.snapshot = ObservationSnapshot(self.sim)
        self.ik = None
        if self.gpu_pd:
            from .ik import GpuIK
            self.ik = GpuIK(env, self.sim.arm['target'])
        from .observations import GpuObservations
        self.observations = GpuObservations(env, self.sim)
        from .scoring import GpuScoring
        self.scoring = GpuScoring(env, self.sim, self.observations)
        self.head_metrics = None
        from wowskin_lab.envs.feeding import PandaFeedingEnv
        callback = getattr(env, 'after_physics_step', None)
        if getattr(callback, '__func__', None) is PandaFeedingEnv.after_physics_step:
            mask = np.zeros(m.ngeom, dtype=np.int32)
            mask[list(env.head_geoms)] = 1
            self.head_geoms = wp.array(mask)
            self.head_contacts = wp.array(np.arange(self.sim.data.naconmax), dtype=int)
            self.head_forces = wp.zeros(self.sim.data.naconmax, dtype=wp.spatial_vector)
            self.head_metrics = wp.array([env.head_force_n, *env.head_net_force,
                env.head_interval_peak_n, env.head_peak_n, env.head_contact_impulse_ns], dtype=wp.float64)
            # Compile observation kernels before Graph capture, without taking
            # another physics step or counting this warm-up as an observation.
            saved = self.head_metrics.numpy()
            self._head_metrics()
            wp.synchronize()
            self.head_metrics.assign(saved)
        # A full physics step and its validity check form one serial graph.
        self.step_graph, self.step_scratch = capture_step(self.sim.model, self.sim.data, self._step)
        self.forward_graph, self.forward_scratch = capture_step(self.sim.model, self.sim.data, self._forward)
        self.endpoint_time = wp.array([self.start_time], dtype=wp.float64)
        wp.launch(_set_time, 1, [self.endpoint_time, self.sim.data.time])
        wp.synchronize()
        # One graph for the same sequential upstream steps. Conditional
        # solver graphs cannot be nested via cudaGraphLaunch during capture.
        self.interval_graph, self.interval_scratch = capture_step(
            self.sim.model, self.sim.data, self._interval)
        self.health.zero_()
        env.cfg['physics_execution'] = dict(
            backend='mujoco_warp', device=str(wp.get_device()),
            gpu_name=wp.get_device().name, mujoco=mujoco.__version__,
            contact_capacity=self.sim.data.naconmax, constraint_capacity=self.sim.data.njmax,
            mujoco_warp=importlib.metadata.version('mujoco-warp'), warp=wp.__version__,
            precision='float32', cuda_graph=True, cable_elasticity='warp_native_cable' if self.sim.cable else None,
            graph_scope='one complete control interval with sequential source physics steps',
            head_contact_statistics='GPU every physics step' if self.head_metrics is not None else None,
            compatibility_changes=changes, solver=mujoco.mjtSolver(m.opt.solver).name,
            iterations=int(m.opt.iterations), tolerance=float(m.opt.tolerance), timestep_s=float(m.opt.timestep),
            reset='native CPU initialization/settling', ik='CUDA Graph, source DLS controller' if self.ik else 'CPU',
            observation='Batched observation/render snapshot; solver matrices remain on GPU',
            tactile_readout='CUDA magnetic forward model, volume checks, contact forces and slip',
            task_scoring='CUDA geometry and completion state',
            scene_renderer=env.cfg.get('camera_backend', 'opengl'),
            clock='integer executed physics ticks; float64 acquisition timestamps',
            fault_policy='stop on any GPU overflow, iteration limit, or nonfinite state; no CPU fallback')
        self._check()
        self.download()

    def _check(self):
        d = self.sim.data
        wp.launch(_audit, max(self.env.model.nq, self.env.model.nv),
                  [d.qpos, d.qvel, d.overflow, d.solver_niter, self.health])

    def _step(self):
        if self.gpu_pd:
            wp.launch(_external_force, 1, [self.env.probe_body, self.mouse, self.tug_until,
                self.ticks, self.start_time, float(self.env.model.opt.timestep), self.sim.data.xfrc_applied])
        self.sim.raw_step()
        self._check()
        if self.head_metrics is not None:
            self._head_metrics()

    def _head_metrics(self):
        m, d = self.sim.model, self.sim.data
        mjw.contact_force(m, d, self.head_contacts, False, self.head_forces)
        wp.launch(_head_contact_metrics, 1, [d.nacon, d.contact.worldid, d.contact.type,
                  d.contact.geom, d.contact.frame, self.head_forces, self.head_geoms,
                  float(self.env.model.opt.timestep), self.head_metrics])

    def _forward(self):
        mjw.forward(self.sim.model, self.sim.data)
        self._check()

    def _interval(self):
        for _ in range(self.env.steps_per_sample):
            self._step()
        wp.launch(_set_time, 1, [self.endpoint_time, self.sim.data.time])
        self._forward()
        self.observations.compute()
        wp.copy(self.scoring.time, self.endpoint_time)
        self.scoring.compute()

    def validate(self):
        flags, iterations = map(int, self.health.numpy())
        self.max_solver_iterations = max(self.max_solver_iterations, iterations)
        self.last_diagnostics = dict(overflow_flags=flags, physics_steps=self.steps,
                                     max_solver_iterations=self.max_solver_iterations)
        if flags:
            self.faulted = True
            names = [name for name, value in mjw.OverflowType.__members__.items()
                     if name not in ('ALL', 'NONE') and flags & int(value)]
            if flags & 1073741824:
                names.append('NONFINITE')
            raise RuntimeError(f'MuJoCo Warp physics fault ({", ".join(names)}); reset required; refusing to export invalid data')

    def upload_controls(self):
        d, gpu = self.env.data, self.sim.data
        for name in ('ctrl', 'xfrc_applied', 'qfrc_applied', 'mocap_pos', 'mocap_quat', 'eq_active'):
            target = getattr(gpu, name)
            value = np.asarray(getattr(d, name))
            if value.size:
                target.assign(value[None])
        if self.gpu_pd and self.ik is None:
            self.sim.arm['target'].assign(self.env.joint_target)
        if self.gpu_pd:
            self.mouse.assign(np.asarray(self.env.mouse_wrench)[None])
            self.tug_until.assign([float(self.env.tug_until)])

    def download(self, *, full=False, computed=False):
        self.validate()
        self.observations.read(compute=not computed)
        self.scoring.ready_time = self.start_time+self.steps*float(self.env.model.opt.timestep) if computed else None
        if not full:
            self.snapshot.download(self.env.data)
            self.env.data.time = self.start_time+self.steps*float(self.env.model.opt.timestep)
            return
        mjw.get_data_into(self.env.data, self.env.model, self.sim.data)
        # Warp 3.15 exports global flex element/vertex IDs; native MjContact
        # uses indices local to each flex. Correct the readout convention only.
        # No CPU collision/force solve is used to rebuild the contacts.
        m, d = self.env.model, self.env.data
        d.contact.geom1[:] = d.contact.geom[:, 0]
        d.contact.geom2[:] = d.contact.geom[:, 1]
        # Rigid narrowphase does not overwrite flex metadata, even in a mixed
        # model. Reused contact slots can otherwise label spoon/bean contacts
        # as silicone. A geom side is always rigid, regardless of stale tags.
        rigid = d.contact.geom >= 0
        for name in ('flex', 'elem', 'vert'):
            getattr(d.contact, name)[rigid] = -1
        if m.nflex:
            for field, offsets in (('elem', m.flex_elemadr), ('vert', m.flex_vertadr)):
                indices = getattr(d.contact, field)
                valid = (d.contact.flex >= 0) & (indices >= 0)
                indices[valid] -= offsets[d.contact.flex[valid]]
        else:
            # Warp skips these fields for rigid-only models. Reallocated
            # native contacts otherwise retain zeros/stale flex IDs, which
            # falsely mark table contacts as skin contacts in task scoring.
            for field in ('flex', 'elem', 'vert'):
                getattr(d.contact, field)[:] = -1
        # The clock counts actual executed steps. It is not an interpolation
        # or an extra integration, and avoids float32 timestamp accumulation.
        self.env.data.time = self.start_time+self.steps*float(self.env.model.opt.timestep)

    def advance(self, controller):
        if self.faulted:
            raise RuntimeError('MuJoCo Warp physics fault: reset required before continuing')
        if controller is not robot_control:
            raise ValueError('Warp task sampling requires the standard robot controller')
        env = self.env
        self.health.zero_()
        callback = getattr(env, 'after_physics_step', None)
        if self.head_metrics is not None:
            wp.launch(_head_interval_start, 1, [self.head_metrics])
            callback = None
        fast = self.gpu_pd and callback is None
        try:
            if fast:
                controller(env)
                self.upload_controls()
                self.endpoint_time.assign([self.start_time+(self.steps+env.steps_per_sample)*env.model.opt.timestep])
                wp.capture_launch(self.interval_graph)
                self.steps += env.steps_per_sample
            else:
                # Preserve each-step force feedback and task peak metrics.
                for _ in range(env.steps_per_sample):
                    controller(env)
                    self.upload_controls()
                    wp.capture_launch(self.step_graph)
                    self.steps += 1
                    self.download(full=not self.gpu_pd or callback is not None)
                    if callback is not None:
                        callback()
            # Contact forces and poses for the recorder refer to the endpoint.
            if not fast:
                self.sim.data.time.assign(np.array([self.start_time+self.steps*env.model.opt.timestep], dtype=np.float32))
                wp.capture_launch(self.forward_graph)
            self.download(full=not self.gpu_pd or callback is not None, computed=fast)
            if self.ik is not None:
                self.ik.read_target()
            if self.head_metrics is not None:
                stats = self.head_metrics.numpy()
                env.head_force_n = float(stats[0])
                env.head_net_force = stats[1:4].copy()
                env.head_interval_peak_n, env.head_peak_n, env.head_contact_impulse_ns = map(float, stats[4:])
        except Exception:
            self.faulted = True
            raise
