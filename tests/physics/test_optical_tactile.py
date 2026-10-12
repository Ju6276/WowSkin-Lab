"""Optional native OpenGL checks: requires requirements/tacto.txt and pinned assets."""
import tempfile
from pathlib import Path
import unittest

import mujoco
import numpy as np

from wowskin_lab.core.optical_tactile import OpticalTactile
from wowskin_lab.envs.optical import OpticalBench, OpticalPandaEnv
from wowskin_lab.tools.optical_sensor import demo


def contact_scene(euler='0 0 0'):
    # Two contacts on left gel; right gel unloaded. Additional sphere/sphere
    # contact elsewhere must not count as a sensor contact. Gravity disabled
    # so a global rigid rotation must preserve mechanics and optical images.
    xml = f'''<mujoco><compiler angle="radian"/><option gravity="0 0 0"/>
      <worldbody><body euler="{euler}">
        <geom name="left" type="box" pos="-.03 0 -.002" size=".01 .015 .002"/>
        <site name="left_cam" pos="-.03 0 .022"/>
        <geom name="right" type="box" pos=".03 0 -.002" size=".01 .015 .002"/>
        <site name="right_cam" pos=".03 0 .022"/>
        <body pos="-.035 -.005 .0035"><joint type="slide" axis="0 0 1"/>
          <geom name="ball1" type="sphere" size=".004" mass=".02"/></body>
        <body pos="-.025 .005 .0035"><joint type="slide" axis="0 0 1"/>
          <geom name="ball2" type="sphere" size=".004" mass=".02"/></body>
        <body pos=".2 0 0"><joint type="slide" axis="0 0 1"/>
          <geom type="sphere" size=".005" mass=".02"/></body>
        <body pos=".2 0 .008"><joint type="slide" axis="0 0 1"/>
          <geom type="sphere" size=".005" mass=".02"/></body>
      </body></worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class OpticalChecks(unittest.TestCase):
    def test_multi_contact_and_global_rotation(self):
        frames = []
        for euler in ('0 0 0', '.4 -.7 1.1'):
            model, data = contact_scene(euler)
            optics = OpticalTactile(model, {'left': ('left_cam', 'left'), 'right': ('right_cam', 'right')}, ideal=True)
            try:
                frame = optics.render(data)
                self.assertEqual(set(frame['left'].contacting_geoms), {model.geom('ball1').id, model.geom('ball2').id})
                self.assertEqual(frame['right'].normal_force_n, 0.)
                self.assertGreater(frame['left'].normal_force_n, 0.)
                self.assertGreater(frame['left'].indentation_m.max(), 0.)
                self.assertEqual(float(frame['right'].indentation_m.max()), 0.)
                frames.append(frame['left'])
            finally:
                optics.close()
        self.assertAlmostEqual(frames[0].normal_force_n, frames[1].normal_force_n, places=7)
        np.testing.assert_allclose(frames[0].indentation_m, frames[1].indentation_m, atol=1e-7)
        np.testing.assert_allclose(frames[0].rgb, frames[1].rgb, atol=1)

    def test_press_and_unload(self):
        env = OpticalBench()
        optics = OpticalTactile(env.model, env.sensors, ideal=True)
        try:
            baseline = optics.render(env.data)['DIGIT']
            env.data.ctrl[2] = -.012
            for _ in range(150):
                env.step(np.zeros(3))
            pressed = optics.render(env.data)['DIGIT']
            self.assertGreater(pressed.normal_force_n, 1.)
            self.assertGreater(pressed.indentation_m.max(), .0001)
            self.assertGreater(np.abs(pressed.rgb.astype(float)-baseline.rgb).mean(), 1.)
            env.data.ctrl[2] = 0.
            for _ in range(150):
                env.step(np.zeros(3))
            released = optics.render(env.data)['DIGIT']
            self.assertEqual(released.normal_force_n, 0.)
            np.testing.assert_array_equal(released.rgb, baseline.rgb)
            np.testing.assert_array_equal(released.indentation_m, baseline.indentation_m)
        finally:
            optics.close()

    def test_panda_physical_grasp_lift_release(self):
        env = OpticalPandaEnv()
        self.assertEqual(env.model.nflex, 0)
        optics = OpticalTactile(env.model, env.sensors, ideal=True)
        try:
            with tempfile.TemporaryDirectory() as temp:
                report = demo(env, optics, Path(temp))
            self.assertGreater(report['lift_m'], .09)
            self.assertLess(report['released_cube_z_m'], .121)
            for side in ('left', 'right'):
                self.assertGreater(report['lifted'][side]['normal_force_n'], .1)
                self.assertGreater(report['lifted'][side]['max_indentation_mm'], .01)
                self.assertEqual(report['released'][side]['normal_force_n'], 0.)
                self.assertEqual(report['released'][side]['max_indentation_mm'], 0.)
        finally:
            optics.close()


if __name__ == '__main__':
    unittest.main()
