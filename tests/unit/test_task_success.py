"""Task outcomes reject shortcuts and require continuous geometric completion."""
import itertools
import json
import unittest
import numpy as np
from wowskin_lab.envs.task_evaluation import CubePickPlaceEvaluator, CableClipEvaluator, task_config


class TaskSuccessTests(unittest.TestCase):
    def corners(self, xy, height=.12):
        return np.array(list(itertools.product([-.02, .02], repeat=3))) + [*xy, height]

    def cube(self):
        c = task_config('pick_cube')
        e = CubePickPlaceEvaluator(c)
        e.reset(self.corners(c['source_center_xy_m']), .1)
        return e

    def update(self, e, time, corners, **kw):
        return e.update(time, corners=corners, **dict(
            dict(grasped=False, gripper_contact_count=0, speed=0., angular_speed=0.), **kw))

    def test_cube_requires_pick_transport_release_and_settle(self):
        e = self.cube(); c = e.config
        target = self.corners(c['target_center_xy_m'])
        # Merely pushing a cube into the destination is not pick-and-place.
        self.update(e, 1., target); self.assertFalse(self.update(e, 2., target)['cube_success'])
        self.update(e, 3., self.corners(c['source_center_xy_m'], .22), grasped=True, gripper_contact_count=2)
        self.update(e, 4., target, grasped=True, gripper_contact_count=2)
        self.assertFalse(e.success)
        self.update(e, 5., target)
        self.assertFalse(self.update(e, 5.29, target)['cube_success'])
        self.assertTrue(self.update(e, 5.30, target)['cube_success'])
        json.dumps(e.summary())
        e.reset(self.corners(c['source_center_xy_m']), .1)
        self.assertFalse(e.success)

    def test_cube_rejects_edge_overhang_motion_and_wrong_start(self):
        for changes in (dict(speed=.1), dict(angular_speed=.5), dict(gripper_contact_count=1)):
            e = self.cube(); c = e.config
            self.update(e, 0., self.corners(c['source_center_xy_m'], .22), grasped=True)
            for t in (1., 2.): self.update(e, t, self.corners(c['target_center_xy_m']), **changes)
            self.assertFalse(e.success)
        e = self.cube(); c = e.config
        self.update(e, 0., self.corners(c['source_center_xy_m'], .22), grasped=True)
        overhang = self.corners(np.asarray(c['target_center_xy_m'])+[.03, 0])
        for t in (1., 2.): self.update(e, t, overhang)
        self.assertFalse(e.success)
        # Airborne at the destination and starting outside source also fail.
        for t in (3., 4.): self.update(e, t, self.corners(c['target_center_xy_m'], .22))
        self.assertFalse(e.success)
        e.reset(self.corners([.65, .2]), .1)
        self.update(e, 0., self.corners([.65, .2], .22), grasped=True)
        for t in (1., 2.): self.update(e, t, self.corners(c['target_center_xy_m']))
        self.assertFalse(e.success)

    def test_clip_above_roof_front_side_and_incomplete_span_fail(self):
        e = CableClipEvaluator(task_config('cable'), x_bounds=[-.005, .005],
                               y_bounds=[-.0015, .0015], z_bounds=[.0035, .0055])
        for a, b in (([-.02, 0, .015], [.02, 0, .015]),
                     ([-.02, -.01, .0045], [.02, -.01, .0045]),
                     ([-.02, .01, .0045], [.02, .01, .0045]),
                     ([-.02, 0, .0045], [0, 0, .0045])):
            e.reset()
            for t in (0., 1.): e.update(t, [[a, b]])
            self.assertFalse(e.success)
        correct = [[[-.02, 0, .0045], [.02, 0, .0045]]]
        e.reset(); e.update(0., correct)
        self.assertFalse(e.update(.29, correct)['clip_success'])
        self.assertTrue(e.update(.30, correct)['clip_success'])
        json.dumps(e.summary())
        e.reset(); e.update(0., correct); e.update(.29, [])
        self.assertFalse(e.update(.5, correct)['clip_success'])


if __name__ == '__main__':
    unittest.main()
