"""Published checkpoint fidelity and physically mounted sensor invariants."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
import torch

from wowskin_lab.envs.franka import FrankaTactileSim
from wowskin_lab.baselines.pollen import CHANNELS, RESOURCE_DIR, PollenPredictor, read_csv


class MountedSensorChecks(unittest.TestCase):
    def test_magnetic_field_invariant_under_rigid_arm_motion(self):
        sim = FrankaTactileSim()
        before = sim.magnetic_field().copy()
        position = sim.data.xpos[sim.mount_body].copy()
        sim.data.qpos[sim.arm_q[0]] += .3
        sim.data.qpos[sim.arm_q[1]] += .1
        mujoco.mj_forward(sim.model, sim.data)
        self.assertGreater(np.linalg.norm(sim.data.xpos[sim.mount_body]-position), .01)
        np.testing.assert_allclose(sim.magnetic_field(), before, atol=1e-7)
        pinned = sim.model.flex_vertbodyid[sim.skin_vertices] == sim.mount_body
        self.assertEqual(int(pinned.sum()), 97)
        np.testing.assert_allclose(sim.skin_vertices_local()[pinned], sim.rest[pinned], atol=1e-12)
        pinned_right = sim.model.flex_vertbodyid[sim.passive_vertices] == sim.passive_mount
        self.assertEqual(int(pinned_right.sum()), 97)
        np.testing.assert_allclose(sim.passive_vertices_local()[pinned_right], sim.passive_rest[pinned_right], atol=1e-12)

    def test_matching_pad_geometry_and_passive_magnetic_isolation(self):
        sim = FrankaTactileSim()
        m, d = sim.model, sim.data
        self.assertEqual(m.nflex, 2)
        np.testing.assert_allclose(sim.rest, sim.passive_rest, atol=1e-12)
        np.testing.assert_allclose(np.ptp(sim.passive_rest, axis=0),
                                   [sim.cfg["width_m"], sim.cfg["length_m"], sim.cfg["skin_thickness_m"]], atol=1e-10)
        left, right = (d.xmat[bid].reshape(3, 3) for bid in (sim.mount_body, sim.passive_mount))
        self.assertAlmostEqual(float(left[:, 2] @ right[:, 2]), -1.)
        # The mount displacement is along the pad normal, with matching height.
        offset = (d.xpos[sim.passive_mount]-d.xpos[sim.mount_body]) @ left
        np.testing.assert_allclose(offset[:2], 0., atol=1e-8)
        self.assertGreater(offset[2], 2*sim.cfg["total_height_m"])
        self.assertEqual(int((m.site_bodyid == sim.mount_body).sum()), 5)
        self.assertEqual(int((m.site_bodyid == sim.passive_mount).sum()), 0)
        before = sim.magnetic_field().copy()
        # Deform only the passive pad. It must not add magnetic sources or chips.
        bid = m.flex_vertbodyid[sim.passive_vertices][-1]
        q = m.jnt_qposadr[m.body_jntadr[bid]]
        d.qpos[q:q+3] += [.00001, -.00002, -.00003]
        mujoco.mj_forward(m, d)
        np.testing.assert_allclose(sim.magnetic_field(), before, atol=1e-12)

    def test_passive_pad_contact_deformation_and_release(self):
        sim = FrankaTactileSim()
        sim.set_probe_side("right")
        unloaded = sim.passive_vertices_local().copy()
        sim.mode = "press"
        pressed = [sim.reading() for _ in range(80)]
        self.assertAlmostEqual(np.mean([r["passive_normal_force_n"] for r in pressed[-20:]]), .05, delta=.006)
        self.assertGreater(np.linalg.norm(sim.passive_vertices_local()-unloaded, axis=1).max(), 5e-6)
        self.assertLess(max(r["normal_force_n"] for r in pressed), 1e-6)
        sim.mode = "slide"
        sliding = [sim.reading() for _ in range(60)]
        self.assertTrue(any(r["passive_slip"] for r in sliding))
        sim.mode = "off"
        released = [sim.reading() for _ in range(50)]
        self.assertLess(released[-1]["passive_normal_force_n"], 1e-6)
        self.assertLess(np.linalg.norm(sim.passive_vertices_local()-unloaded, axis=1).max(), 2e-6)
        rows = pressed+sliding+released
        self.assertGreater(min(r["passive_min_volume_ratio"] for r in rows), .9)
        self.assertTrue(np.isfinite([r["magnetic_ut"] for r in rows]).all())
        self.assertFalse(np.any(sim.data.warning.number))

    def test_contact_sliding_release_and_solver_stability(self):
        sim = FrankaTactileSim()
        rows = [sim.reading() for _ in range(350)]
        def phase(lo, hi, key):
            return np.array([r[key] for r in rows if lo < r["time_s"] < hi])
        self.assertAlmostEqual(float(phase(.9, 1.2, "normal_force_n").mean()), .05, delta=.006)
        self.assertEqual(float(phase(.9, 1.2, "slip").mean()), 0.)
        self.assertGreater(float(phase(1.8, 2.3, "slip").mean()), .9)
        self.assertEqual(float(phase(3.3, 3.5, "contact").mean()), 0.)
        self.assertGreater(min(r["min_volume_ratio"] for r in rows), .9)
        self.assertGreater(min(r["passive_min_volume_ratio"] for r in rows), .9)
        self.assertLess(max(r["passive_normal_force_n"] for r in rows), 1e-6)
        self.assertTrue(np.isfinite([r["magnetic_ut"] for r in rows]).all())
        self.assertFalse(np.any(sim.data.warning.number))


if __name__ == "__main__":
    unittest.main()
