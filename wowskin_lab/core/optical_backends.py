"""Dispatch hardware-specific source optical pipelines."""

from .optical_tactile import OpticalTactile


def make_optics(env, sensor, *, ideal=False, shadow=True, **tacto_options):
    if sensor == 'digit_fots':
        from .fots_mujoco import NativeFOTSOptical
        return NativeFOTSOptical(env.model, env.sensors, ideal=ideal, shadow=shadow,
            device='cpu')
    if sensor == 'gelsight':
        from .gelsight_mujoco import GelSightMujocoOptical
        return GelSightMujocoOptical(env.model, env.sensors, ideal=ideal)
    if sensor == 'tacto':
        return OpticalTactile(env.model, env.sensors, ideal=ideal, **tacto_options)
    raise ValueError(f'Unknown optical sensor combination: {sensor}')
