"""Optical observations in the same operator/RGB-camera layout as AnySkin."""
from types import SimpleNamespace
from .tactile import TactileHUD


class OpticalHUD(TactileHUD):
    def __init__(self, env):
        super().__init__(env)
        self.optics = None
        self.data = None
        self.show_mask = False
        self.last_episode = None

    def bind(self, model, data):
        from wowskin_lab.core.optical_backends import make_optics
        snapshot = SimpleNamespace(model=model, data=data, sensors=self.sim.sensors)
        self.optics = make_optics(snapshot, self.sim.cfg['sensor'], ideal=self.sim.cfg.get('optical_ideal', False),
                                 shadow=self.sim.cfg.get('optical_shadow', True))
        self.data = data

    def render(self, row, probability=None):
        from wowskin_lab.tools.optical_sensor import panel
        if self.last_episode != row.get('episode_id') and hasattr(self.optics, 'reset'):
            self.optics.reset()
        self.last_episode = row.get('episode_id')
        frames = self.optics.render(self.data)
        return panel({f'ROBOT {side.upper()}': frame
                      for side, frame in frames.items()}, show_mask=self.show_mask, mask_key=',',
                     depth_max_mm=self.sim.cfg.get('optical_depth_display_max_mm'))

    def key(self, code):
        if code == ord(','):
            self.show_mask = not self.show_mask

    def close(self):
        if self.optics:
            self.optics.close()
