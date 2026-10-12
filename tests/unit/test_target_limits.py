import unittest
from types import SimpleNamespace
import numpy as np
from wowskin_lab.teleop.target_limits import straight_step, workspace_bounds


class TargetLimits(unittest.TestCase):
    def test_boundary_stops_diagonal_without_sideways_slide(self):
        bounds = np.array([[0., -1., -1.], [1., 1., 1.]])
        p = np.array([.0005, 0., 0.])
        delta = np.array([-.001, .001, 0.])
        actual = p.copy()
        np.testing.assert_allclose(straight_step(p, delta, actual, bounds), delta*.5)
        p += delta*.5
        np.testing.assert_allclose(straight_step(p, delta, actual, bounds), 0.)
        # Reversing at the boundary must work immediately.
        np.testing.assert_allclose(straight_step(p, -delta, actual, bounds), -delta)

    def test_lead_does_not_feed_tracking_error_into_target(self):
        p = np.array([0., 0., .2])
        actual = np.array([-.0055, .003, .2])
        delta = np.array([.001, .001, 0.])
        result = straight_step(p, delta, actual, [[-np.inf]*3, [np.inf]*3])
        np.testing.assert_allclose(result, delta*.5)
        np.testing.assert_array_equal(p, [0., 0., .2])
        self.assertLessEqual(np.max(np.abs(p+result-actual)), .006+1e-12)

    def test_workspace_off_keeps_lead(self):
        e = SimpleNamespace(cfg={'disable_workspace_limits': True})
        bounds = workspace_bounds(e, [[0]*3, [1]*3])
        self.assertTrue(np.isinf(bounds).all())
        p = np.array([2., 2., 2.])
        np.testing.assert_allclose(straight_step(p, [.01, 0, 0], p, bounds), [.006, 0, 0])


if __name__ == '__main__':
    unittest.main()
