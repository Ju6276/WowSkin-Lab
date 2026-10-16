"""GPU execution, reset, and fault handling through the public task interface."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from wowskin_lab.core.stepping import prepare
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, response_configuration


def setUpModule():
    global wp, GpuSimulation
    try:
        import warp as wp
        import mujoco_warp
    except ImportError:
        raise unittest.SkipTest('Optional NVIDIA dependencies: requirements/gpu.txt')
    if not wp.is_cuda_available():
        raise unittest.SkipTest('NVIDIA CUDA device required')
    from wowskin_lab.core.gpu.runtime import GpuSimulation


class GpuEnvironmentChecks(unittest.TestCase):
    def environment(self):
        cfg = response_configuration(SimpleNamespace(task='pick_cube', ideal=True,
            physics_backend='warp', gpu_disable_multiccd=True))
        return PandaPickCubeTactileEnv(cfg, pollen=False)

    def test_motion_clock_and_reset_without_cpu_physics(self):
        env = self.environment()
        source_flags = env.model.opt.disableflags
        backend = prepare(env)
        initial = env.data.qpos.copy()
        # IK may compute kinematics on its own data; the live dynamics must
        # never call native step/forward after preparation.
        native_forward = mujoco.mj_forward
        def forbid_live_forward(model, data):
            if data is env.data:
                raise AssertionError('CPU forward on live GPU state')
            return native_forward(model, data)
        with patch.object(mujoco, 'mj_step', side_effect=AssertionError('CPU step')), \
             patch.object(mujoco, 'mj_forward', side_effect=forbid_live_forward):
            for i in range(10):
                action = env.hold_action()
                action[0] = .0001
                env.step(action)
                self.assertAlmostEqual(env.data.time, (i+1)*.01, places=10)
                np.testing.assert_array_equal(env.data.qpos,
                    backend.sim.data.qpos.numpy()[0])
        self.assertGreater(np.linalg.norm(env.data.qpos-initial), 1e-5)
        self.assertEqual(backend.last_diagnostics['overflow_flags'], 0)
        env.reset(seed=3)
        self.assertIsNone(env._gpu_physics)
        self.assertEqual(env.model.opt.disableflags, source_flags)
        second = prepare(env)
        self.assertIsNot(second, backend)
        env.step(env.hold_action())
        self.assertAlmostEqual(env.data.time, .01, places=10)

    def test_native_step_and_graph_match(self):
        env = self.environment()
        env.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        env.control()
        arm = dict(qids=env.arm_q, vids=env.arm_v, aids=env.arm_controls, target=env.joint_target)
        compare(env.model, env.data, arm)

    def test_viewer_worker_can_step_and_recapture_after_reset(self):
        from concurrent.futures import ThreadPoolExecutor
        env = self.environment()
        initial = prepare(env)  # Recorder prepares on the UI thread.
        with ThreadPoolExecutor(max_workers=1) as worker:
            worker.submit(env.step, env.hold_action()).result()
            self.assertAlmostEqual(env.data.time, .01, places=10)
            worker.submit(env.reset).result()
            worker.submit(env.step, env.hold_action()).result()
            self.assertIsNot(env._gpu_physics, initial)
            self.assertAlmostEqual(env.data.time, .01, places=10)

    def test_fault_requires_reset_and_does_not_fall_back(self):
        env = self.environment()
        backend = prepare(env)
        backend.health.assign(np.array([1073741824, 0], dtype=np.int32))
        with self.assertRaisesRegex(RuntimeError, 'NONFINITE'):
            backend.validate()
        before = env.data.time
        with self.assertRaisesRegex(RuntimeError, 'reset required'):
            env.step(env.hold_action())
        self.assertEqual(env.data.time, before)
        env.reset()
        env.step(env.hold_action())
        self.assertFalse(env._gpu_physics.faulted)


def compare(m,d,arm=None):
    sim=GpuSimulation(m,d,arm=arm)
    # Exercise motion, rather than checking equality of stationary arrays.
    d.qvel[:]=np.random.default_rng(31).normal(0,.001,m.nv);mujoco.mj_forward(m,d);sim.reset()
    sim.raw_step();wp.synchronize();gpu_q=sim.data.qpos.numpy()[0];gpu_v=sim.data.qvel.numpy()[0]
    reference=mujoco.MjData(m);mujoco.mj_copyData(reference,m,d)
    if arm:
        torque=600*(arm['target']-reference.qpos[arm['qids']])-45*reference.qvel[arm['vids']]+reference.qfrc_bias[arm['vids']]
        limits=m.actuator_ctrlrange[arm['aids']];reference.ctrl[arm['aids']]=np.clip(torque,limits[:,0],limits[:,1])
    mujoco.mj_step(m,reference)
    qerr=float(abs(gpu_q-reference.qpos).max());verr=float(abs(gpu_v-reference.qvel).max())
    assert qerr<5e-6 and verr<5e-3,(qerr,verr)
    sim.reset()
    for _ in range(5):sim.raw_step()
    wp.synchronize();rq=sim.data.qpos.numpy();rv=sim.data.qvel.numpy()
    sim.reset();sim.compile()
    for _ in range(5):sim.step()
    wp.synchronize();cq=float(abs(rq-sim.data.qpos.numpy()).max());cv=float(abs(rv-sim.data.qvel.numpy()).max())
    assert cq<5e-6 and cv<5e-3,(cq,cv)
    diag=sim.diagnostics();assert diag['finite'] and not diag['overflow_flags'],diag
    return dict(native_gpu_qpos_error=qerr,native_gpu_qvel_error=verr,raw_graph_qpos_error=cq,raw_graph_qvel_error=cv,diagnostics=diag)


def check_contact_force_readback(env):
    """Compare native readout directly with Warp's force kernel, not another CPU reader."""
    import warp as wp
    from mujoco_warp._src import support, types
    sim = env._gpu_physics.sim
    d = sim.data
    count = int(d.nacon.numpy()[0])
    exported = int(types.ContactType.CONSTRAINT | types.ContactType.PASSIVE)
    ids = np.flatnonzero((d.contact.worldid.numpy()[:count] == 0) &
                        ((d.contact.type.numpy()[:count] & exported) != 0))
    assert len(ids) == env.data.ncon and len(ids) > 0
    forces = wp.zeros(len(ids), dtype=wp.spatial_vector)
    support.contact_force(sim.model, d, wp.array(ids, dtype=int), False, forces)
    expected = forces.numpy()
    actual = np.zeros_like(expected, dtype=np.float64)
    for i in range(len(ids)):
        mujoco.mj_contactForce(env.model, env.data, i, actual[i])
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


class ContactReadbackChecks(unittest.TestCase):
    def test_rigid_table_contacts_are_not_mislabelled_as_flex_grasps(self):
        from wowskin_lab.envs.task_evaluation import gripper_contacts
        for sensor in ('tacto', 'digit_fots', 'gelsight'):
            with self.subTest(sensor=sensor):
                cfg = response_configuration(SimpleNamespace(task='pick_cube', ideal=True,
                    sensor_backend=sensor, physics_backend='warp', gpu_disable_multiccd=True,
                    gpu_disable_nativeccd=True, gpu_disable_noslip=True))
                env = PandaPickCubeTactileEnv(cfg, pollen=False)
                self.assertEqual(env.model.nflex, 0)
                prepare(env)
                for _ in range(5):
                    env.step(env.hold_action())
                    self.assertGreater(env.data.ncon, 0)
                    for field in ('flex', 'elem', 'vert'):
                        np.testing.assert_array_equal(getattr(env.data.contact, field), -1)
                    np.testing.assert_array_equal(env.data.contact.geom1, env.data.contact.geom[:, 0])
                    np.testing.assert_array_equal(env.data.contact.geom2, env.data.contact.geom[:, 1])
                    self.assertEqual(gripper_contacts(env, {env.probe_geom}), 0)

    def test_loaded_pads_use_local_flex_contact_indices(self):
        from wowskin_lab.envs.feeding import PandaFeedingEnv
        cfg = response_configuration(SimpleNamespace(task='feeding', ideal=True,
            physics_backend='warp', gpu_disable_multiccd=True,
            gpu_disable_nativeccd=True))
        env = PandaFeedingEnv(cfg, pollen=False)
        prepare(env)
        env.step(env.hold_action())
        check_contact_force_readback(env)
        for contact in env.data.contact:
            for side in (0, 1):
                flex = contact.flex[side]
                if flex >= 0:
                    self.assertLess(contact.elem[side], env.model.flex_elemnum[flex])
                    self.assertLess(contact.vert[side], env.model.flex_vertnum[flex])
        self.assertTrue(np.isfinite(env.pad_forces()).all())

    def test_chip_fails_before_unported_fracture_can_be_recorded(self):
        from wowskin_lab.envs.chip_tray import PandaChipEnv
        cfg = response_configuration(SimpleNamespace(task='chip_tray', ideal=True,
            physics_backend='warp', gpu_disable_multiccd=True))
        env = PandaChipEnv(cfg, pollen=False)
        with self.assertRaisesRegex(ValueError, 'dynamic fragment collision masks'):
            prepare(env)
