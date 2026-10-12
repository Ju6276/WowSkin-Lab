"""Physics and data-contract checks, not a real-world accuracy validation."""
import unittest
import mujoco

import numpy as np

from wowskin_lab.core.physics import TactileSim, configuration, dipole_field
from wowskin_lab.tools.sensor_diagnostics import run_episode


@unittest.skipUnless(mujoco.__version__.startswith('3.5.'), 'Legacy full-node sensor bench for MuJoCo 3.5')
class PhysicsChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim, cls.rows = run_episode(configuration())

    def test_contact_labels_and_force_direction(self):
        r = self.rows
        # Holding still should transmit load without gross slip.
        hold = (r["time_s"] > .9) & (r["time_s"] < 1.15)
        self.assertTrue(np.all(r["contact"][hold] == 1))
        self.assertTrue(np.all(r["slip"][hold] == 0))
        self.assertAlmostEqual(float(r["force_on_probe_n"][hold, 2].mean()), .05, delta=.005)
        slide = (r["time_s"] > 2.7) & (r["time_s"] < 3.1)
        self.assertGreater(float(r["slip"][slide].mean()), .9)
        # The lifted probe moves in air: motion alone must not imply slip.
        air = r["time_s"] > 4.1
        self.assertTrue(np.all(r["contact"][air] == 0))
        self.assertTrue(np.all(r["slip"][air] == 0))

    def test_mesh_and_signal_are_valid(self):
        r = self.rows
        self.assertEqual(r["magnetic_ut"].shape, (450, 15))
        self.assertTrue(np.isfinite(r["magnetic_ut"]).all())
        self.assertGreater(float(r["min_volume_ratio"].min()), .8)
        np.testing.assert_allclose(np.diff(r["time_s"]), .01, atol=1e-9)
        pinned = self.sim.model.flex_vertbodyid == 0
        np.testing.assert_allclose(self.sim.data.flexvert_xpos[pinned], self.sim.rest[pinned], atol=1e-12)
        self.assertFalse(np.any(self.sim.data.warning.number))

    def test_magnetism_has_no_label_input(self):
        sim = TactileSim(configuration())
        base = sim.magnetic_field().copy()
        sim.cfg["slip_speed_m_per_s"] = 100
        sim.cfg["contact_threshold_n"] = 100
        np.testing.assert_array_equal(sim.magnetic_field(), base)
        sim.data.qpos[:3] += np.array([.0001, 0, 0])
        import mujoco
        mujoco.mj_forward(sim.model, sim.data)
        self.assertGreater(np.linalg.norm(sim.magnetic_field()-base), 1e-6)


class DipoleMathChecks(unittest.TestCase):
    def test_dipole_analytical_axis_and_translation(self):
        source = np.zeros((1, 3))
        moment = np.array([[0., 0., 1e-5]])
        sensors = np.array([[0, 0, .01], [.01, 0, 0]])
        field = dipole_field(sensors, source, moment)
        np.testing.assert_allclose(field, [[0, 0, 2], [0, 0, -1]], atol=1e-10)
        offset = np.array([.02, -.03, .01])
        np.testing.assert_allclose(dipole_field(sensors+offset, source+offset, moment), field)


if __name__ == "__main__":
    unittest.main()
