"""Sensor bench checks: no classifier is involved."""
import unittest
from argparse import Namespace
from collections import Counter

import numpy as np

from wowskin_lab.core.physics import TactileSim, configuration
from wowskin_lab.envs.bench import normal_press, sensor_reading, bench_configuration


class SensorChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = bench_configuration(Namespace(seed=0, x_mm=0., y_mm=0., load_n=.05, ideal=True))
        cls.sim = TactileSim(cfg)
        cls.rows = [sensor_reading(cls.sim, normal_press) for _ in range(350)]

    def test_load_deformation_and_recovery(self):
        hold = [r for r in self.rows if 1.4 < r["time_s"] < 1.9]
        self.assertAlmostEqual(np.mean([r["normal_force_n"] for r in hold]), .05, delta=.005)
        self.assertGreater(np.mean([r["max_displacement_m"] for r in hold]), 1e-6)
        self.assertGreater(np.linalg.norm(hold[-1]["ideal_delta_ut"]), .01)
        tail = self.rows[-1]
        self.assertLess(tail["normal_force_n"], 1e-6)
        self.assertLess(tail["max_displacement_m"], 2e-6)
        self.assertLess(np.linalg.norm(tail["ideal_delta_ut"]), .01)
        self.assertNotIn("slip", tail)
        self.assertEqual(tail["magnetic_ut"].shape, (15,))

    def test_measured_base_and_fitted_dimensions(self):
        rest = self.sim.rest
        np.testing.assert_allclose(np.ptp(rest[:, :2], axis=0), [.0432, .0446], atol=1e-9)
        self.assertAlmostEqual(rest[:, 2].max(), .0135)
        self.assertAlmostEqual(rest[:, 2].min(), .0095)
        model, data = self.sim.model, self.sim.data
        gid = model.geom('standard_shell').id
        mid = model.geom_dataid[gid]
        verts = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
        world = verts @ data.geom_xmat[gid].reshape(3,3).T + data.geom_xpos[gid]
        self.assertAlmostEqual(world[:,2].min(), 0., places=8)
        self.assertAlmostEqual(world[:,2].max(), .0115, places=8)
        self.assertEqual(configuration()['total_height_m'], .0135)  # Shared mounted/bench profile.

    def test_sleeve_is_conforming_and_positive_volume(self):
        sim = self.sim
        faces = Counter()
        for tet in sim.tets:
            for skip in range(4):
                faces[tuple(sorted(np.delete(tet, skip)))] += 1
        self.assertTrue(all(count in (1, 2) for count in faces.values()))
        # The skirt/top interface must be internal, not overlapping free surfaces.
        for face, count in faces.items():
            if np.allclose(sim.rest[list(face),2], .0115, atol=1e-12):
                xy = sim.rest[list(face),:2]
                if np.any(np.abs(xy[:,0]) > .0422/2+1e-8) or np.any(np.abs(xy[:,1]) > .0436/2+1e-8):
                    self.assertEqual(count, 2)
        self.assertGreater(sim.volume.min(), 0.)
        np.testing.assert_array_equal(sim.data.warning.number, 0)

    def test_default_noise_does_not_grow_with_elapsed_time(self):
        cfg = bench_configuration(Namespace(seed=0, x_mm=0., y_mm=0., load_n=.05, ideal=False))
        self.assertEqual(cfg['drift_ut_per_s'], 0.)
        readings = []
        for start_time in (0., 10000.):
            sim = TactileSim(cfg)
            sim.data.time = start_time
            sim.rng = np.random.default_rng(91)
            for _ in range(cfg['delay_samples']+1):
                row = sim.sample(lambda s: s.apply_probe_load([0.,0.], None))
            readings.append(row['magnetic_ut'])
        np.testing.assert_allclose(*readings, atol=1e-9)
        self.assertGreater(np.linalg.norm(readings[0]), .1)

    def test_zero_rejects_contact_and_clears_unloaded_history(self):
        sim = TactileSim(configuration())
        for _ in range(150):
            sim.sample(normal_press)
        with self.assertRaisesRegex(ValueError, "Release"):
            sim.zero()
        for _ in range(200):
            sim.sample(normal_press)
        sim.zero()
        np.testing.assert_allclose(sim.magnetic_field()-sim.baseline, 0, atol=1e-12)
        self.assertEqual(sim.zero_time, sim.data.time)
        self.assertEqual(len(sim.delay), sim.cfg["delay_samples"])
        np.testing.assert_array_equal(sim.filtered, np.zeros((5, 3)))


if __name__ == "__main__":
    unittest.main()
