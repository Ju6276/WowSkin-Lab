"""Force-loop regression against contact dynamics, not an object-width table."""
import unittest
import json

import mujoco
import numpy as np

from wowskin_lab.core.grip_force import GripForceFeedback
from wowskin_lab.paths import ROOT


class GripForceTests(unittest.TestCase):
    def fixture(self, obstacle_top):
        # One driven finger approaching an immobile object. Changing the
        # object thickness is unknown to the controller; only force is read.
        model = mujoco.MjModel.from_xml_string(f'''<mujoco>
          <option timestep=".001" gravity="0 0 0" integrator="implicitfast"/>
          <worldbody>
            <geom type="box" pos="0 0 {obstacle_top-.01}" size=".03 .03 .01"/>
            <body pos="0 0 .06"><joint name="finger" type="slide" axis="0 0 1" range="-.04 0"/>
              <geom type="box" size=".01 .01 .005" mass=".05"/>
            </body>
          </worldbody>
          <actuator><position joint="finger" kp="50" kv="5"/></actuator>
        </mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        config = json.loads((ROOT/'configs/chip_tray.json').read_text())['grip_force_feedback']
        ctrl = GripForceFeedback(dict(config, target_n=.1))
        ctrl.enabled = True
        return model, data, ctrl

    def settle(self, model, data, ctrl, seconds):
        wrench = np.zeros(6)
        last_forces = []
        for _ in range(round(seconds/model.opt.timestep)):
            force = 0.
            for i in range(data.ncon):
                mujoco.mj_contactForce(model, data, i, wrench)
                force += max(0., wrench[0])
            data.ctrl[0] = ctrl.update(data.ctrl[0], [force, 0.], model.opt.timestep,
                                       model.jnt_range[0])
            mujoco.mj_step(model, data)
            last_forces.append(force)
        self.assertFalse(np.any(data.warning.number))
        self.assertAlmostEqual(np.mean(last_forces[-1000:]), ctrl.target_n, delta=.003)
        return float(data.ctrl[0])

    def test_unknown_thickness_and_changing_force_target(self):
        positions = []
        for height in (.03, .04):
            m, d, c = self.fixture(height)
            positions.append(self.settle(m, d, c, 25.))
            c.selected_target_n = .16
            tighter = self.settle(m, d, c, 8.)
            self.assertLess(tighter, positions[-1])
            c.selected_target_n = .03
            looser = self.settle(m, d, c, 10.)
            self.assertGreater(looser, tighter)
        self.assertAlmostEqual(positions[1]-positions[0], .01, delta=.0002)

    def test_invalid_feedback_is_not_zero_filled(self):
        _, _, ctrl = self.fixture(.03)
        with self.assertRaises(RuntimeError):
            ctrl.update(-.01, [float('nan'), .1], .001, [-.04, 0.])
        ctrl.selected_target_n = float('nan')
        with self.assertRaises(ValueError):
            ctrl.update(-.01, [.1, .1], .001, [-.04, 0.])


if __name__ == '__main__':
    unittest.main()
