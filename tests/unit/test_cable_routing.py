import unittest
import numpy as np
from wowskin_lab.envs.cable_routing import configuration, initial_vertices, CableRoutingEvaluator


class RoutingScoreTests(unittest.TestCase):
    def test_ordered_passages_dwell_wrong_side_and_above_cap(self):
        c = configuration()
        score = CableRoutingEvaluator(c)
        points = np.array([[xb.mean(), y, zb.mean()] for xb, y, zb in score.gates()])
        segments = np.stack((points[:-1], points[1:]), axis=1)
        self.assertTrue(score.update(0., segments)['routing_inside'])
        self.assertFalse(score.success)
        self.assertTrue(score.update(.31, segments)['routing_success'])
        score.reset()
        self.assertFalse(score.update(1., segments[::-1, ::-1])['routing_inside'])
        high = segments.copy(); high[:, :, 2] += .03
        self.assertFalse(score.update(2., high)['routing_inside'])
        initial = initial_vertices(c)
        self.assertFalse(score.update(3., np.stack((initial[:-1], initial[1:]), axis=1))['routing_inside'])


if __name__ == '__main__':
    unittest.main()
