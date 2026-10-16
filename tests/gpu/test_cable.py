"""Independent parity against the native MuJoCo cable plugin, including CUDA Graph."""
import unittest
import mujoco
import numpy as np


def setUpModule():
    global mjw, wp, GpuSimulation
    try:
        import mujoco_warp as mjw
        import warp as wp
    except ImportError:
        raise unittest.SkipTest('Optional NVIDIA dependencies: requirements/gpu.txt')
    if not wp.is_cuda_available():
        raise unittest.SkipTest('NVIDIA CUDA device required')
    from wowskin_lab.core.gpu.runtime import GpuSimulation


def cable_model(*, flat=False, box=False):
    # Curved reference and anisotropic section exercise frame/sign conventions.
    geom = 'box' if box else 'capsule'
    size = '.004 .006 .003' if box else '.0035'
    return mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep=".0005" gravity="0 0 0" solver="CG" integrator="discrete"/>
      <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
      <worldbody><composite type="cable" count="8 1 1" size=".16 .025 .012"
          curve="s sin(s) cos(s)" initial="free">
        <plugin plugin="mujoco.elasticity.cable">
          <config key="twist" value="1e7"/><config key="bend" value="2e6"/>
          <config key="flat" value="{str(flat).lower()}"/>
        </plugin><joint kind="main" damping=".0001"/>
        <geom type="{geom}" size="{size}" contype="0" conaffinity="0"/>
      </composite></worldbody></mujoco>''')


def force_check(model, *, seed=17):
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    # Rotate the whole cable, bend and twist joints; exercise large rotation too.
    delta = np.zeros(model.nv)
    for body in np.flatnonzero(model.body_plugin >= 0):
        start, count = model.body_dofadr[body], model.body_dofnum[body]
        if count:
            delta[start:start+count] = rng.normal(0, .15, count)
    mujoco.mj_integratePos(model, data.qpos, delta, 1.)
    data.qvel[:] = rng.normal(0, .002, model.nv)
    mujoco.mj_forward(model, data)
    original_slots = model.body_plugin.copy()
    sim = GpuSimulation(model, data, cable_plugin_warp=True)
    mjw.forward(sim.model, sim.data)
    wp.synchronize()
    forward_diagnostics = sim.diagnostics()
    full_passive_error = float(np.max(abs(sim.data.qfrc_passive.numpy()[0]-data.qfrc_passive)))
    # Isolate the plugin contribution from native MuJoCo's independently
    # exposed components. Pad f32/f64 differences are not cable-port errors.
    assert not model.jnt_actgravcomp.any()
    expected = data.qfrc_passive.copy()
    for field in ('qfrc_spring', 'qfrc_damper', 'qfrc_gravcomp', 'qfrc_fluid', 'qfrc_adhesion'):
        expected -= getattr(data, field)
    sim.data.qfrc_passive.zero_()
    sim.cable.apply(sim.model, sim.data)
    actual = sim.data.qfrc_passive.numpy()[0]
    absolute = float(np.max(abs(actual-expected)))
    relative = float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected), 1e-12))
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-6)
    np.testing.assert_array_equal(model.body_plugin, original_slots)
    return sim, dict(max_force_error=absolute, relative_force_error=relative,
                     reference_force_norm=float(np.linalg.norm(expected)), source_plugin_unchanged=True,
                     full_passive_error_including_other_models=full_passive_error,
                     forward_diagnostics=forward_diagnostics)


def rollout_check(model, steps=200):
    data = mujoco.MjData(model)
    rng = np.random.default_rng(23)
    # Smaller perturbations for time integration; no contact makes differences
    # attributable to elastic dynamics rather than collision manifold choices.
    mujoco.mj_integratePos(model, data.qpos, rng.normal(0, .01, model.nv), 1.)
    mujoco.mj_forward(model, data)
    sim = GpuSimulation(model, data, cable_plugin_warp=True)
    sim.compile()
    reference = mujoco.MjData(model)
    mujoco.mj_copyData(reference, model, data)
    for _ in range(steps):
        sim.step()
        mujoco.mj_step(model, reference)
    wp.synchronize()
    q = sim.data.qpos.numpy()[0];v = sim.data.qvel.numpy()[0]
    qerr = float(np.max(abs(q-reference.qpos)));verr = float(np.max(abs(v-reference.qvel)))
    np.testing.assert_allclose(q, reference.qpos, rtol=1e-3, atol=2e-5)
    np.testing.assert_allclose(v, reference.qvel, rtol=1e-2, atol=2e-3)
    assert sim.diagnostics()['finite']
    # Independently verify graph capture retains the cable callback.
    raw = GpuSimulation(model, data, cable_plugin_warp=True)
    for _ in range(steps):
        raw.raw_step()
    wp.synchronize()
    np.testing.assert_allclose(raw.data.qpos.numpy()[0], q, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(raw.data.qvel.numpy()[0], v, rtol=1e-5, atol=1e-6)
    return dict(steps=steps, qpos_error=qerr, qvel_error=verr, raw_graph_match=True)

class CableGpuChecks(unittest.TestCase):
    def test_force_and_trajectory_match_native_plugin(self):
        for flat in (False, True):
            for box in (False, True):
                with self.subTest(flat=flat, box=box):
                    model = cable_model(flat=flat, box=box)
                    force_check(model)
                    rollout_check(model)

    def test_project_cable_layouts_preserve_elastic_forces(self):
        from types import SimpleNamespace
        from wowskin_lab.envs.pick_cube import response_configuration
        from wowskin_lab.envs.cable import PandaCableEnv
        for layout in ('single_clip', 'routing'):
            with self.subTest(layout=layout):
                cfg = response_configuration(SimpleNamespace(task='cable',
                    gripper='robotiq_2f85', cable_layout=layout, ideal=True))
                model = PandaCableEnv(cfg, pollen=False).model
                model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
                model.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
                force_check(model, seed=37)
