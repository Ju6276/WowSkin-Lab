"""Explicit local two-socket layout and geometry-only route scoring.

Scoring reads the centreline. It never applies force, changes a pose, or attaches
the cable to the guides. The cable plugin uses a straight stress-free shape.
"""
import json
import numpy as np
from wowskin_lab.paths import ROOT


def configuration():
    return json.loads((ROOT/'configs/cable_routing.json').read_text())


def initial_vertices(c):
    start, end = np.asarray(c['endpoint_world_m'], float)
    t = np.linspace(0., 1., 4097)
    direction = end-start
    normal = np.array([-direction[1], direction[0], 0.])
    normal /= np.linalg.norm(normal)
    points = (start+t[:, None]*direction+
              c['initial_lateral_sine_amplitude_m']*np.sin(2*np.pi*t)[:, None]*normal)
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    samples = np.linspace(0., distance[-1], c['capsules']+1)
    return np.stack([np.interp(samples, distance, points[:, j]) for j in range(3)], axis=1)


class CableRoutingEvaluator:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.success = False
        self.since = None
        self.metrics = dict(routing_success=False, routing_inside=False,
                            routing_gate_count=0, routing_gate_total=6, routing_dwell_s=0.)

    def gates(self):
        c = self.config['scoring']
        for post, sign in zip(self.config['posts_world_m'], self.config['post_sides_x']):
            xb = np.sort(post[0]+sign*np.asarray(c['side_offset_x_m']))
            for dy in c['cross_section_offsets_y_m']:
                yield xb, post[1]+dy, np.asarray(c['center_height_world_m'])

    def update(self, time, segments):
        # Passage positions must occur in order along the connected cable.
        segments = np.asarray(segments)
        previous, count = -1., 0
        gates = list(self.gates())
        for xb, y, zb in gates:
            hits = []
            for i, (a, b) in enumerate(segments):
                dy = b[1]-a[1]
                if abs(dy) < 1e-12:
                    continue
                t = (y-a[1])/dy
                if 0. <= t <= 1. and i+t > previous:
                    p = a+t*(b-a)
                    if xb[0] <= p[0] <= xb[1] and zb[0] <= p[2] <= zb[1]:
                        hits.append(i+t)
            if not hits:
                break
            previous = min(hits)
            count += 1
        inside = count == len(gates)
        self.since = (time if self.since is None else self.since) if inside else None
        dwell = 0. if self.since is None else max(0., time-self.since)
        self.success |= bool(inside and dwell+1e-9 >= self.config['scoring']['continuous_dwell_s'])
        self.metrics = dict(routing_success=self.success, routing_inside=inside,
                            routing_gate_count=count, routing_gate_total=len(gates), routing_dwell_s=dwell)
        return dict(self.metrics)

    def summary(self):
        return dict(criterion=self.config['scoring']['schema'], config=self.config,
                    success=self.success, metrics=self.metrics, routing_scored=True,
                    clip_required=False, release_retention_scored=False)
