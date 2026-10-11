import unittest
import numpy as np
from wowskin_lab.envs.feeding_evaluation import delivery_config, DeliveryEvaluator, delivery_checks, completion_checks
from wowskin_lab.ui.teleop_hud import feeding_feedback


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = delivery_config()
        self.e = DeliveryEvaluator(self.cfg)
        self.rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        target = np.asarray(self.cfg['mouth_center_head_m'])+[0, -self.cfg['target_offset_outside_lips_m'], 0]
        spoon = target-self.rotation@self.cfg['delivery_point_spoon_m']
        self.args = dict(head_pos=np.zeros(3), head_rotation=np.eye(3), spoon_pos=spoon,
                         spoon_rotation=self.rotation, bean_positions=np.array([
                             spoon+self.rotation@[.031+.009*(i//2), -.0035+.007*(i%2), 0]
                             for i in range(6)]), bowl_speed=0., grasped=True, head_contacts=0)

    def test_delivery_requires_continuous_hold_and_reset(self):
        self.assertFalse(self.e.update(0., **self.args)['delivery_success'])
        self.assertFalse(self.e.update(.49, **self.args)['delivery_success'])
        self.assertTrue(self.e.update(.5, **self.args)['delivery_success'])
        self.e.reset()
        self.assertFalse(self.e.success)
        self.e.update(0., **self.args)
        self.e.update(.4, **dict(self.args, bean_positions=np.zeros((6, 3))))
        self.assertFalse(self.e.update(.6, **self.args)['delivery_success'])

    def test_empty_fast_tilted_wrong_side_contact_not_success(self):
        # Historical data keep their original strict scoring, even after the
        # current default has been simplified by the user.
        legacy = {k:v for k,v in self.cfg.items() if k != 'completion_checks'}
        self.e = DeliveryEvaluator(legacy)
        scenarios = [dict(bean_positions=np.zeros((6, 3))), dict(bowl_speed=.1),
                     dict(spoon_rotation=np.eye(3)), dict(head_contacts=1), dict(grasped=False),
                     dict(spoon_pos=np.array([0., .15, 0.]))]
        for changes in scenarios:
            with self.subTest(changes=changes):
                self.e.reset()
                self.e.update(0., **dict(self.args, **changes))
                result = self.e.update(1., **dict(self.args, **changes))
                self.assertFalse(result['delivery_success'])

    def test_inside_region_reports_tilt_instead_of_silent_failure(self):
        # Recorded failure: all conditions except the local 15-degree tilt
        # limit passed. Feedback must report the exact failing predicate.
        metrics = dict(delivery_inside_target=True, beans_in_spoon=6, spoon_grasped=True,
                       spoon_tilt_deg=16.982481, spoon_approach_angle_deg=24.40664,
                       spoon_bowl_speed_m_s=.000182, head_contact_count=0)
        historical = {k:v for k,v in dict(self.cfg, max_tilt_deg=15.).items() if k != 'completion_checks'}
        checks = delivery_checks(metrics, historical)
        self.assertEqual([k for k,v in checks.items() if not v], ['tilt'])
        self.assertEqual(feeding_feedback(metrics, historical), 'tilt 17.0 > 15 deg')
        self.assertTrue(all(delivery_checks(metrics, self.cfg).values()))
        metrics['spoon_tilt_deg'] = 20.1
        self.assertFalse(delivery_checks(metrics, self.cfg)['tilt'])

    def test_position_food_completion_retains_contact_and_motion_metrics(self):
        # Valid geometric food/tip positions with an intentionally bad
        # orientation, high speed, no grasp, and a head contact. These must
        # remain visible in metrics without vetoing the new completion rule.
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_euler('xyz', [35., 20., 45.], degrees=True).as_matrix()
        target = np.asarray(self.cfg['mouth_center_head_m'])+[0, -self.cfg['target_offset_outside_lips_m'], 0]
        spoon = target-rotation@self.cfg['delivery_point_spoon_m']
        beans = np.array([spoon+rotation@[.031+.009*(i//2), -.0035+.007*(i%2), 0.] for i in range(6)])
        args = dict(self.args, spoon_pos=spoon, spoon_rotation=rotation, bean_positions=beans,
                    bowl_speed=.1, grasped=False, head_contacts=2)
        self.e.update(0., **args)
        row = self.e.update(.5, **args)
        self.assertTrue(row['delivery_success'])
        self.assertEqual(row['head_contact_count'], 2)
        self.assertFalse(row['spoon_grasped'])
        self.assertEqual(row['spoon_bowl_speed_m_s'], .1)
        self.assertGreater(row['spoon_tilt_deg'], self.cfg['max_tilt_deg'])
        self.assertEqual(list(completion_checks(row, self.cfg)), ['region', 'beans'])
        self.assertEqual(feeding_feedback(row, self.cfg), 'All conditions met; hold still')

    def test_goal_edge_touches_lips_and_uses_tip_not_bowl(self):
        self.e = DeliveryEvaluator({k:v for k,v in self.cfg.items() if k != 'delivery_part_bounds_spoon_m'})
        self.assertAlmostEqual(self.cfg['target_offset_outside_lips_m'],
                               self.cfg['target_ellipsoid_radii_m'][1])
        good = self.e.update(0., **self.args)
        self.assertTrue(good['delivery_inside_target'])
        self.e.reset()
        tip_to_bowl = self.rotation@(np.asarray(self.cfg['delivery_point_spoon_m']) -
                                     self.cfg['bowl_center_spoon_m'])
        # Bowl at target centre pushes the tip beyond the head-side boundary.
        bad = self.e.update(0., **dict(self.args, spoon_pos=self.args['spoon_pos']+tip_to_bowl))
        self.assertFalse(bad['delivery_inside_target'])

    def test_front_part_can_overlap_with_tip_outside_but_handle_cannot(self):
        shift = self.rotation@np.array([.025, 0., 0.])
        args = dict(self.args, spoon_pos=self.args['spoon_pos']+shift,
                    bean_positions=self.args['bean_positions']+shift)
        legacy = DeliveryEvaluator({k:v for k,v in self.cfg.items() if k != 'delivery_part_bounds_spoon_m'})
        self.assertFalse(legacy.update(0., **args)['delivery_inside_target'])
        self.assertTrue(self.e.update(0., **args)['delivery_inside_target'])
        self.assertTrue(self.e.update(.5, **args)['delivery_success'])
        self.e.reset()
        shift = self.rotation@np.array([.090, 0., 0.])
        args = dict(self.args, spoon_pos=self.args['spoon_pos']+shift,
                    bean_positions=self.args['bean_positions']+shift)
        self.assertFalse(self.e.update(0., **args)['delivery_inside_target'])


if __name__ == '__main__':
    unittest.main()
