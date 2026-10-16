"""Optical contacts use the current MuJoCo contact IDs after GPU readback."""
import unittest
from unittest.mock import patch

import mujoco

from wowskin_lab.core.optical_tactile import contact_forces


class OpticalContactChecks(unittest.TestCase):
    def test_current_geom_pair_overrides_stale_legacy_fields(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom type="plane" size="1 1 .1"/>
          <body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1"/></body>
        </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertEqual(data.ncon, 1)
        data.contact[0].geom[:] = [23, 41]
        self.assertEqual(data.contact[0].geom1, 0)
        self.assertEqual(data.contact[0].geom2, 1)
        def measured_force(model, data, index, force):
            force[:] = [2.5, 0, 0, 0, 0, 0]
        with patch.object(mujoco, 'mj_contactForce', side_effect=measured_force):
            self.assertEqual(contact_forces(model, data, 23), {41: 2.5})
            self.assertEqual(contact_forces(model, data, 41), {23: 2.5})
            self.assertEqual(contact_forces(model, data, [23, 41]), {})
            self.assertEqual(contact_forces(model, data, 0), {})
