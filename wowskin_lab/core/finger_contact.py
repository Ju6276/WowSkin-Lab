"""Explicit contact-triggered closure hold, independent of tactile readout.

This is a local controller option, not a material or force calibration. It
changes the actuator target only; contacts, forces, and deformation stay live.
"""
import numpy as np


class FingerContactStop:
    def __init__(self):
        self.reset()

    def reset(self):
        self.limit = None
        self.contact = False

    @property
    def active(self):
        return self.limit is not None

    def apply(self, requested, measured, mutual_contact):
        self.contact = bool(mutual_contact)
        # Opening explicitly releases the hold. A continuing CLOSE command
        # cannot ratchet the target deeper as the pads settle or separate.
        opening = self.limit is not None and requested < self.limit - 1e-12
        if opening:
            self.limit = None
        elif self.limit is None and self.contact and requested > measured:
            self.limit = float(np.clip(measured, 0., 1.))
        return min(requested, self.limit) if self.active else requested


def pads_touch(env):
    """Only actual pad-to-pad contacts, never pad-to-object/table contacts."""
    flex = env.data.contact.flex
    if not len(flex):
        return False
    a, b = env.skin_flex, env.passive_flex
    mutual = (((flex[:, 0] == a) & (flex[:, 1] == b)) |
              ((flex[:, 0] == b) & (flex[:, 1] == a)))
    return bool(np.any(mutual & (env.data.contact.dist <= 0) &
                       (env.data.contact.efc_address >= 0)))
