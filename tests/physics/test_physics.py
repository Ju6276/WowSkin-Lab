"""Physics and data-contract checks, not a real-world accuracy validation."""
import unittest
import mujoco

import numpy as np

from wowskin_lab.core.physics import TactileSim, configuration, dipole_field
from wowskin_lab.tools.sensor_diagnostics import run_episode


def reduced_configuration():
    cfg = configuration()
    cfg.update(skin_dof='trilinear', integrator='discrete')
    return cfg


@unittest.skipUnless(mujoco.__version__.startswith('3.15.'), 'Reduced sensor bench for MuJoCo 3.15')
class PhysicsChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim, cls.rows = run_episode(reduced_configuration())

    def test_contact_labels_and_force_direction(self):
        r = self.rows
        t = r["time_s"]
        # Holding still should transmit load without gross slip.
        hold = (t > .9) & (t < 1.15)
        self.assertTrue(np.all(r["contact"][hold] == 1))
        self.assertTrue(np.all(r["slip"][hold] == 0))
        self.assertAlmostEqual(float(r["force_on_probe_n"][hold, 2].mean()), .05, delta=.005)
        # Slip is a contact-gated label. The reduced skin chatters on the return
        # stroke, so check the label contract instead of a contact duty cycle.
        self.assertTrue(np.all(r["contact"][r["slip"] == 1] == 1))
        for lo, hi in ((1.4, 1.8), (2.7, 3.1)):
            touching = (t > lo) & (t < hi) & (r["contact"] == 1)
            self.assertGreater(int(touching.sum()), 20)
            self.assertGreater(float(r["slip"][touching].mean()), .9)
        # The lifted probe moves in air: motion alone must not imply slip.
        air = t > 4.1
        self.assertTrue(np.all(r["contact"][air] == 0))
        self.assertTrue(np.all(r["slip"][air] == 0))

    def test_mesh_and_signal_are_valid(self):
        r = self.rows
        self.assertEqual(r["magnetic_ut"].shape, (450, 15))
        self.assertTrue(np.isfinite(r["magnetic_ut"]).all())
        self.assertGreater(float(r["min_volume_ratio"].min()), .8)
        np.testing.assert_allclose(np.diff(r["time_s"]), .01, atol=1e-9)
        self.assertFalse(np.any(self.sim.data.warning.number))

    def test_magnetism_has_no_label_input(self):
        sim = TactileSim(reduced_configuration())
        base = sim.magnetic_field().copy()
        sim.cfg["slip_speed_m_per_s"] = 100
        sim.cfg["contact_threshold_n"] = 100
        np.testing.assert_array_equal(sim.magnetic_field(), base)
        self.assertLess(3, sim.probe_dof)  # qpos[:3] is a skin node, not the probe.
        sim.data.qpos[:3] += np.array([.0001, 0, 0])
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
