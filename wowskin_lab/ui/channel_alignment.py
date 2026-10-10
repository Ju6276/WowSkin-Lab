"""Explicit display adapter; does not rewrite simulation/training raw channels.

Positions follow the manufacturer's display. Physical correspondence and the
reference PCB's magnetic axes still require an experiment on the user's device.
"""
import numpy as np
from .manufacturer import upstream

OFFICIAL_TO_REFERENCE = np.array([0, 4, 2, 1, 3])
POSITIONS = ('center', 'left', 'right', 'up', 'down')


def official_values(sim, values):
    local = np.asarray(values, dtype=float).reshape(5, 3)
    board = np.einsum('nij,nj->ni', sim.rotations, local)[OFFICIAL_TO_REFERENCE]
    screen = board[:, :2] * [1., -1.]
    theta = upstream()['chip_xy_rotations']
    c, s = np.cos(theta), np.sin(theta)
    result = board.copy()
    result[:, 0] = c*screen[:, 0] + s*screen[:, 1]
    result[:, 1] = -s*screen[:, 0] + c*screen[:, 1]
    return result.reshape(15)
