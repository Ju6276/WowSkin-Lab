"""Geometry-based task scores. Never alter simulation state or policy input."""
import itertools
import json
import numpy as np
from wowskin_lab.paths import ROOT


def task_config(task):
    return json.loads((ROOT/'configs/task_success.json').read_text())[task]


def cube_corners(env):
    half = env.model.geom_size[env.probe_geom, :3]
    local = np.array(list(itertools.product([-1., 1.], repeat=3)))*half
    return local@env.data.geom_xmat[env.probe_geom].reshape(3, 3).T+env.data.geom_xpos[env.probe_geom]


def gripper_contacts(env, object_geoms):
    count = 0
    for contact in env.data.contact:
        if contact.dist > 0:
            continue
        for side in (0, 1):
            if contact.geom[side] not in object_geoms:
                continue
            other = 1-side
            if contact.flex[other] >= 0:
                count += 1  # The two sensor skins are this environment's flexes.
                break
            geom = int(contact.geom[other])
            body = int(env.model.geom_bodyid[geom]) if geom >= 0 else 0
            while body and body != env.hand:
                body = int(env.model.body_parentid[body])
            if body == env.hand:
                count += 1
                break
    return count


class CubePickPlaceEvaluator:
    def __init__(self, config):
        self.config = config

    def _inside(self, corners, name):
        c = self.config
        half = c[f'{name}_half_width_m']-c['marking_half_width_m']
        return bool(np.all(np.abs(corners[:, :2]-c[f'{name}_center_xy_m']) <= half))

    def reset(self, corners, table_z):
        self.table_z = table_z
        self.initial_center_z = float(corners[:, 2].mean())
        self.started_in_source = self._inside(corners, 'source')
        self.lifted = self.success = False
        self.since = None
        self.metrics = {}
        self.update(0., corners=corners, grasped=False, gripper_contact_count=0,
                    speed=0., angular_speed=0.)

    def update(self, time, *, corners, grasped, gripper_contact_count, speed, angular_speed):
        c = self.config
        self.lifted |= bool(self.started_in_source and grasped and
            corners[:, 2].mean()-self.initial_center_z >= c['required_lift_m'])
        inside = self._inside(corners, 'target')
        on_table = abs(float(corners[:, 2].min())-self.table_z) <= c['table_height_tolerance_m']
        released = gripper_contact_count == 0
        stable = bool(speed <= c['max_speed_m_s'] and angular_speed <= c['max_angular_speed_rad_s'])
        eligible = bool(self.lifted and inside and on_table and released and stable)
        self.since = (time if self.since is None else self.since) if eligible else None
        dwell = 0. if self.since is None else max(0., time-self.since)
        self.success |= bool(eligible and dwell+1e-9 >= c['continuous_dwell_s'])
        self.metrics = dict(cube_success=self.success, cube_started_in_source=self.started_in_source,
            cube_lifted_from_source=self.lifted, cube_inside_target=inside, cube_on_table=on_table,
            cube_released=released, cube_stable=stable, cube_dwell_s=float(dwell),
            cube_target_distance_m=float(np.linalg.norm(corners[:, :2].mean(0)-c['target_center_xy_m'])))
        return dict(self.metrics)

    def summary(self):
        return dict(criterion=self.config['schema'], config=self.config, success=self.success,
                    metrics=self.metrics, full_pick_and_place_scored=True)


class CableClipEvaluator:
    def __init__(self, config, *, x_bounds, y_bounds, z_bounds):
        self.config = config
        self.x_bounds, self.y_bounds, self.z_bounds = map(np.asarray, (x_bounds, y_bounds, z_bounds))
        self.reset()

    def reset(self):
        self.success = False
        self.since = None
        self.metrics = dict(clip_success=False, clip_inside=False, clip_dwell_s=0., clip_span_count=0)

    def update(self, time, segments):
        """Intersect actual capsule centre lines with three tunnel cross-sections."""
        tol = self.config['collision_tolerance_m']
        spans = 0
        for x in (self.x_bounds[0], self.x_bounds.mean(), self.x_bounds[1]):
            found = False
            for a, b in np.asarray(segments):
                dx = b[0]-a[0]
                if abs(dx) < 1e-10:
                    continue
                fraction = (x-a[0])/dx
                if not 0 <= fraction <= 1:
                    continue
                p = a+fraction*(b-a)
                if (self.y_bounds[0]-tol <= p[1] <= self.y_bounds[1]+tol and
                        self.z_bounds[0]-tol <= p[2] <= self.z_bounds[1]+tol):
                    found = True
                    break
            spans += found
        inside = spans == 3
        self.since = (time if self.since is None else self.since) if inside else None
        dwell = 0. if self.since is None else max(0., time-self.since)
        self.success |= bool(inside and dwell+1e-9 >= self.config['continuous_dwell_s'])
        self.metrics = dict(clip_success=self.success, clip_inside=inside,
                            clip_dwell_s=float(dwell), clip_span_count=int(spans))
        return dict(self.metrics)

    def summary(self):
        return dict(criterion=self.config['schema'], config=self.config, success=self.success,
                    metrics=self.metrics, routing_scored=False, release_retention_scored=False,
                    centerline_bounds_clip_m=dict(x=self.x_bounds.tolist(), y=self.y_bounds.tolist(),
                                                  z=self.z_bounds.tolist()))
