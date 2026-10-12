"""Limit motion distance without rotating a requested Cartesian direction."""
import numpy as np


def workspace_bounds(env, default):
    if env.cfg.get('disable_cartesian_limits') or env.cfg.get('disable_workspace_limits'):
        return np.array([[-np.inf]*3, [np.inf]*3])
    return np.asarray(getattr(env, 'workspace', default), float)


def straight_step(target, step, actual, bounds, lead_m=.006):
    """Scale the entire increment at a boundary, never clip individual axes.

    The target is not recentered on tracking error. When external contact moves
    the actual pose outside the lead box, steps toward it remain possible;
    steps farther away stop. Physical tracking still has finite error.
    """
    target, step, actual = (np.asarray(x, float) for x in (target, step, actual))
    lower = np.maximum(np.asarray(bounds[0]), actual-lead_m)
    upper = np.minimum(np.asarray(bounds[1]), actual+lead_m)
    scale = 1.
    for i, delta in enumerate(step):
        if delta > 0:
            scale = min(scale, max(0., (upper[i]-target[i])/delta))
        elif delta < 0:
            scale = min(scale, max(0., (lower[i]-target[i])/delta))
    return step*scale
