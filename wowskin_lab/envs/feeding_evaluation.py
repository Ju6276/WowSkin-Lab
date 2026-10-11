"""Local delivery criterion shared by teleoperation and policy rollouts.

Uses privileged geometry for scoring only, never as a policy observation.
"""
import json
import numpy as np
from scipy.optimize import lsq_linear
from wowskin_lab.paths import ROOT


def delivery_config():
    return json.loads((ROOT/'configs/feeding.json').read_text())


def delivery_checks(metrics, config):
    """Evaluate geometry and quality checks independently of completion policy."""
    c, m = config, metrics
    return dict(region=bool(m['delivery_inside_target']),
                beans=m['beans_in_spoon'] >= c['required_beans'],
                grasp=bool(m['spoon_grasped']),
                tilt=m['spoon_tilt_deg'] <= c['max_tilt_deg'],
                approach=m['spoon_approach_angle_deg'] <= c['max_approach_angle_deg'],
                speed=m['spoon_bowl_speed_m_s'] <= c['max_bowl_speed_m_s'],
                head_contact=not c['require_no_head_contact'] or m['head_contact_count'] == 0)


def completion_checks(metrics, config):
    """Select the explicitly configured requirements; legacy configs stay strict."""
    checks = delivery_checks(metrics, config)
    required = config.get('completion_checks', list(checks))
    if not required or any(key not in checks for key in required):
        raise ValueError('Invalid feeding completion_checks')
    return {key: checks[key] for key in required}


class DeliveryEvaluator:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.last_time = None
        self.since = None
        self.success = False
        self.first_success_time = None
        self.metrics = {}

    def update(self, time, *, head_pos, head_rotation, spoon_pos, spoon_rotation,
               bean_positions, bowl_speed, grasped, head_contacts):
        c = self.config
        # Older datasets retain their original bowl-centre scoring config.
        point_local = c.get('delivery_point_spoon_m', c['bowl_center_spoon_m'])
        target = np.asarray(c['mouth_center_head_m'])+[0, -c['target_offset_outside_lips_m'], 0]
        if 'delivery_part_bounds_spoon_m' in c:
            # Exact overlap between the configured oriented front-part box
            # and the mouth ellipsoid, including edges/faces (not just corners).
            bounds = np.asarray(c['delivery_part_bounds_spoon_m'])
            radii = np.asarray(c['target_ellipsoid_radii_m'])
            origin = head_rotation.T@(spoon_pos-head_pos)-target
            axes = head_rotation.T@spoon_rotation
            result = lsq_linear(axes/radii[:, None], -origin/radii,
                                bounds=(bounds[0], bounds[1]), tol=1e-10)
            if not result.success:
                raise RuntimeError('Feeding front-part overlap calculation failed')
            point_local = result.x
        point = spoon_pos+spoon_rotation@np.asarray(point_local)
        error = head_rotation.T@(point-head_pos)-target
        inside = float(np.linalg.norm(error/np.asarray(c['target_ellipsoid_radii_m']))) <= 1
        beans = (bean_positions-spoon_pos)@spoon_rotation
        volume = c['bean_volume_spoon']
        radial = np.sum(((beans[:, :2]-volume['center_xy_m'])/volume['radii_xy_m'])**2, axis=1)
        count = int(np.sum((radial <= 1) & (beans[:, 2] >= volume['z_range_m'][0]) &
                           (beans[:, 2] <= volume['z_range_m'][1])))
        tilt = float(np.rad2deg(np.arccos(np.clip(spoon_rotation[2, 2], -1, 1))))
        inward = head_rotation@np.array([0., 1., 0.])
        approach = float(np.rad2deg(np.arccos(np.clip(spoon_rotation[:, 0]@inward, -1, 1))))
        metrics = dict(delivery_distance_m=float(np.linalg.norm(error)),
                       delivery_inside_target=bool(inside), beans_in_spoon=count,
                       spoon_grasped=bool(grasped), spoon_tilt_deg=tilt,
                       spoon_approach_angle_deg=approach, spoon_bowl_speed_m_s=float(bowl_speed),
                       head_contact_count=int(head_contacts))
        eligible = all(completion_checks(metrics, c).values())
        if self.last_time is not None and time < self.last_time:
            raise ValueError('Reset the delivery evaluator when simulation time resets')
        if eligible:
            if self.since is None:
                self.since = time
        else:
            self.since = None
        dwell = 0. if self.since is None else float(time-self.since)
        if eligible and dwell+1e-9 >= c['continuous_dwell_s']:
            if not self.success:
                self.first_success_time = float(time)
            self.success = True
        self.last_time = float(time)
        self.metrics = dict(metrics, delivery_success=self.success, delivery_eligible=eligible,
                            delivery_dwell_s=dwell)
        return self.metrics.copy()

    def summary(self):
        return dict(criterion=self.config['schema'], success=self.success,
                    first_success_sim_time_s=self.first_success_time, final_metrics=self.metrics,
                    scope=self.config['source'])
