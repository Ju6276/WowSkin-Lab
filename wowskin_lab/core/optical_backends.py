"""Explicit optical combinations and simultaneous DIGIT comparison."""
from time import perf_counter

from .optical_tactile import OpticalTactile


class DigitComparison:
    label = 'Same DIGIT / TACTO + FOTS (same physics state)'

    def __init__(self, model, sensors, *, ideal=False, shadow=True, **tacto_options):
        from .fots import FOTSOptical
        self.tacto = OpticalTactile(model, sensors, ideal=ideal, **tacto_options)
        try:
            fots_sites = {name: (site.removesuffix('_camera')+'_surface', gel)
                          for name, (site, gel) in sensors.items()}
            self.fots = FOTSOptical(model, fots_sites, hardware='digit', shadow=shadow, ideal=ideal)
        except Exception:
            self.tacto.close()
            raise
        self.render_seconds = {}

    def render(self, data):
        # No second environment, stepping, force fitting or depth rescaling.
        start = perf_counter()
        tacto = self.tacto.render(data)
        self.render_seconds['tacto'] = perf_counter()-start
        start = perf_counter()
        fots = self.fots.render(data)
        self.render_seconds['fots'] = perf_counter()-start
        output = {}
        for name in tacto:
            assert tacto[name].time_s == fots[name].time_s
            assert tacto[name].contacting_geoms == fots[name].contacting_geoms
            assert tacto[name].normal_force_n == fots[name].normal_force_n
            output[name+' / TACTO'] = tacto[name]
            output[name+' / FOTS'] = fots[name]
        return output

    def reset(self):
        self.fots.reset()

    def close(self):
        self.tacto.close()
        self.fots.close()


def make_optics(env, sensor, *, ideal=False, shadow=True, **tacto_options):
    if sensor in ('fots', 'digit_fots'):
        from .fots import FOTSOptical
        return FOTSOptical(env.model, env.sensors, hardware='digit' if sensor == 'digit_fots' else 'gelsight',
                           ideal=ideal, shadow=shadow)
    if sensor == 'gelsight':
        from .gelsight_mujoco import GelSightMujocoOptical
        return GelSightMujocoOptical(env.model, env.sensors, reference_data=env.data, ideal=ideal)
    if sensor == 'digit_compare':
        return DigitComparison(env.model, env.sensors, ideal=ideal, shadow=shadow, **tacto_options)
    if sensor == 'tacto':
        return OpticalTactile(env.model, env.sensors, ideal=ideal, **tacto_options)
    raise ValueError(f'Unknown optical sensor combination: {sensor}')
