"""Shared command contract for keyboard, Pico input and future robot backends."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CartesianCommand:
    world_translation_m: tuple
    local_rotation_vector_rad: tuple
    closure: float

    @classmethod
    def from_array(cls, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError('Expected finite [dx_m, dy_m, dz_m, drx_rad, dry_rad, drz_rad, closure]')
        return cls(tuple(values[:3]), tuple(values[3:6]), float(values[6]))

    def as_array(self):
        return np.array([*self.world_translation_m, *self.local_rotation_vector_rad, self.closure])
