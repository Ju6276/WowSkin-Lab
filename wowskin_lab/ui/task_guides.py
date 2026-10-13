"""Operator-only goal geometry. Never inserted into MjModel or camera renderers."""
from dataclasses import dataclass
import itertools

import mujoco
import numpy as np


@dataclass
class Region:
    name: str
    center: np.ndarray
    half: np.ndarray
    rotation: np.ndarray
    color: tuple
    ellipsoid: bool = False


def goal_regions(env, model, data, row):
    """Read the evaluators' actual configuration, in the display snapshot frame."""
    cyan, amber, green = (.12, .83, 1., .13), (1., .70, .18, .09), (.15, 1., .35, .15)
    color = green if row.get('success') else cyan
    if hasattr(env, 'delivery'):
        c = env.delivery.config
        head = model.body('configurable_visual_head').id
        rotation = data.xmat[head].reshape(3, 3)
        local = np.asarray(c['mouth_center_head_m'])+[0, -c['target_offset_outside_lips_m'], 0]
        label = 'SPOON TIP HERE' if 'delivery_point_spoon_m' in c else 'BOWL HERE'
        if 'delivery_part_bounds_spoon_m' in c:
            label = 'SPOON FRONT HERE'
        regions = [Region(label, data.xpos[head]+rotation@local,
                          np.asarray(c['target_ellipsoid_radii_m']), rotation, color, True)]
        if 'delivery_part_bounds_spoon_m' in c:
            bounds = np.asarray(c['delivery_part_bounds_spoon_m'])
            body = env.probe_body
            rot = data.xmat[body].reshape(3, 3)
            regions.append(Region('', data.xpos[body]+rot@bounds.mean(0),
                                  (bounds[1]-bounds[0])/2, rot, amber))
        return regions
    if hasattr(env, 'peg_score'):
        c = env.peg_score.config
        hole = model.body('hole').id
        rotation = data.xmat[hole].reshape(3, 3)
        lower = c['socket_entry_local_z_m']+c['required_insertion_m']
        upper = c['socket_entry_local_z_m']+c['maximum_insertion_m']
        local = np.r_[c['tip_local_m'][:2], (lower+upper)/2]
        half = np.array([c['max_lateral_error_m']]*2+[(upper-lower)/2])
        return [Region('PEG TIP TARGET', data.xpos[hole]+rotation@local, half, rotation, color)]
    if hasattr(env, 'routing_score'):
        return [Region(f'ROUTE {i//3+1}' if i % 3 == 1 else '',
                       np.array([xb.mean(), y, zb.mean()]),
                       np.array([np.ptp(xb)/2, .0005, np.ptp(zb)/2]), np.eye(3), color)
                for i, (xb, y, zb) in enumerate(env.routing_score.gates())]
    if hasattr(env, 'clip_score'):
        c = env.clip_score
        bounds = np.array([c.x_bounds, c.y_bounds, c.z_bounds])
        # X spans are exact cross-sections; only Y/Z accept contact tolerance.
        half = np.diff(bounds, axis=1).ravel()/2
        half[1:] += c.config['collision_tolerance_m']
        clip = model.body('cclip_0').id
        rotation = data.xmat[clip].reshape(3, 3)
        return [Region('CABLE CENTRE HERE', data.xpos[clip]+rotation@bounds.mean(1),
                       half, rotation, color)]
    if env.pick_place is None:
        return []
    c, table = env.pick_place.config, env.pick_place.table_z
    h = float(model.geom_size[env.probe_geom, 2])
    # The target prism shows XY containment at the resting cube height.
    # Bottom-height tolerance is a separate thin slab, not a new condition.
    target_half = c['target_half_width_m']-c['marking_half_width_m']
    source_half = c['source_half_width_m']-c['marking_half_width_m']
    return [
        Region('PICK', np.r_[c['source_center_xy_m'], table+h],
               np.array([source_half, source_half, h]), np.eye(3), amber),
        Region('PLACE + RELEASE', np.r_[c['target_center_xy_m'], table+h],
               np.array([target_half, target_half, h]), np.eye(3), color),
        Region('BOTTOM ON TABLE', np.r_[c['target_center_xy_m'], table],
               np.array([target_half, target_half, c['table_height_tolerance_m']]), np.eye(3), color),
        Region(f"LIFT {c['required_lift_m']*1000:g} mm", np.r_[c['source_center_xy_m'],
               env.pick_place.initial_center_z+c['required_lift_m']],
               np.array([.022, .022, .0002]), np.eye(3), amber),
    ]


class TaskGuides:
    def __init__(self, env):
        self.env = env

    @staticmethod
    def _geom(scene, kind, size, pos, rotation, rgba, label=''):
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError('Operator guide scene capacity exceeded')
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, kind, np.asarray(size, float), np.asarray(pos, float),
                           np.asarray(rotation, float).ravel(), np.asarray(rgba, np.float32))
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.objtype, geom.objid = mujoco.mjtObj.mjOBJ_UNKNOWN, -1
        geom.label = label
        scene.ngeom += 1
        return geom

    def append(self, scene, model, data, row):
        for region in goal_regions(self.env, model, data, row):
            r = region
            kind = mujoco.mjtGeom.mjGEOM_ELLIPSOID if r.ellipsoid else mujoco.mjtGeom.mjGEOM_BOX
            self._geom(scene, kind, r.half, r.center, r.rotation, r.color)
            corners = np.array(list(itertools.product([-1., 1.], repeat=3)))
            world = (corners*r.half)@r.rotation.T+r.center
            for a in range(8):
                for b in range(a+1, 8):
                    if np.count_nonzero(corners[a] != corners[b]) != 1:
                        continue
                    geom = self._geom(scene, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3),
                                      r.center, np.eye(3), (*r.color[:3], .75))
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2., world[a], world[b])
            # Labels sit outside the target, so small cable slots remain findable.
            label_pos = world.mean(0)+[0, 0, np.max(world[:,2])-r.center[2]+.025]
            self._geom(scene, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3), label_pos,
                       np.eye(3), (*r.color[:3], 1.), r.name)
        if hasattr(self.env, 'delivery') and 'delivery_part_bounds_spoon_m' not in self.env.delivery.config:
            c = self.env.delivery.config
            body = self.env.probe_body
            local = c.get('delivery_point_spoon_m', c['bowl_center_spoon_m'])
            point = data.xpos[body]+data.xmat[body].reshape(3, 3)@np.asarray(local)
            self._geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [.002]*3, point,
                       np.eye(3), (.95, .82, .12, 1.))

    def show(self, handle, model, data, row):
        if not handle.is_running():
            return
        # This scene is exclusively ours. Preview/recording renderers do not
        # receive it, even when their camera is moved to the same viewpoint.
        with handle.lock():
            handle.user_scn.ngeom = 0
            if getattr(self.env, 'is_optical', False):
                from wowskin_lab.envs.optical import sensor_geometry, sensor_prefix
                geometry = sensor_geometry(self.env.cfg['sensor'])
                prefix = sensor_prefix(self.env.cfg['sensor'])
                for side in ('left', 'right'):
                    b = model.body(f'{prefix}_{side}_module').id
                    local = geometry['source_surface']
                    point = data.xpos[b]+data.xmat[b].reshape(3,3)@local
                    color = (.39,.86,1.,1.) if side == 'left' else (1.,.74,.39,1.)
                    self._geom(handle.user_scn, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3), point, np.eye(3), color, 'L' if side=='left' else 'R')
            self.append(handle.user_scn, model, data, row)
