"""Optional FOTS checks: source parity, metric depth and physical grasp/release."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import mujoco
import numpy as np
import torch

from wowskin_lab.core.fots import FOTS_ROOT, FOTSOptical, FOTSRenderer
from wowskin_lab.envs.optical import OpticalBench, OpticalPandaEnv
from wowskin_lab.tools.optical_sensor import demo
from tests.physics.test_optical_tactile import contact_scene


class FOTSChecks(unittest.TestCase):
    def test_matches_original_optical_pipeline(self):
        seed_function = torch.seed
        renderer = FOTSRenderer()
        self.assertIs(torch.seed, seed_function)
        rows, cols = np.mgrid[:320, :240]
        depth = np.maximum(0., .0006-((rows-160)**2+(cols-120)**2)*1e-7)
        actual = renderer.render(depth)
        shadow_actual = renderer.render(depth, shadow=True)
        trajectory = [[0., 0., 0.], [.8, -.6, .2]]
        marker_actual, displacement = renderer.markers(actual, [(depth, trajectory)])
        self.assertGreater(np.abs(displacement).max(), 1.)
        self.assertFalse(np.array_equal(shadow_actual, actual))
        # Independently execute the original entry point in its own process;
        # its cwd, bare imports and torch.seed mutation never affect this app.
        with tempfile.TemporaryDirectory() as directory:
            np.save(Path(directory)/'depth.npy', depth)
            source = '''import os, sys, importlib.util, functools, numpy as np, torch
# Import the real MLP file directly: its parent package eagerly imports the
# unrelated physical DIGIT USB driver. No optical code is replaced.
spec = importlib.util.spec_from_file_location('mlp_calib.src.train.mlp_model', 'mlp_calib/src/train/mlp_model.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
# Checkpoint was saved on CUDA; device relocation leaves weights unchanged.
torch.load = functools.partial(torch.load, map_location='cpu', weights_only=True)
from tactile_render import get_simapproach
from utils.marker_motion import MarkerMotion
import params as pr
np.mat = np.asmatrix  # NumPy 2 compatibility for the untouched shadow module.
p = os.environ['FOTS_TEST_DIR']
d = np.load(p+'/depth.npy')
with torch.inference_mode():
    renderer = get_simapproach()
    image = renderer.generate(d*1000*pr.mm_to_pixel, d>0, shadow=False)
    shadow = renderer.generate(d*1000*pr.mm_to_pixel, d>0, shadow=True)
marker = MarkerMotion(frame0_blur=image, depth=d*1000, mask=d>0,
                      traj=[[0.,0.,0.],[.8,-.6,.2]], lamb=[.00125,.00021,.00038])
np.save(p+'/rgb.npy', image[..., ::-1])
np.save(p+'/shadow.npy', shadow[..., ::-1])
np.save(p+'/marker.npy', marker._marker_motion()[..., ::-1])
'''
            subprocess.run([sys.executable, '-c', source], cwd=FOTS_ROOT,
                           env=dict(os.environ, FOTS_TEST_DIR=directory, PYTHONDONTWRITEBYTECODE='1'), check=True)
            np.testing.assert_array_equal(actual, np.load(Path(directory)/'rgb.npy'))
            np.testing.assert_array_equal(shadow_actual, np.load(Path(directory)/'shadow.npy'))
            np.testing.assert_array_equal(marker_actual, np.load(Path(directory)/'marker.npy'))
        np.testing.assert_array_equal(actual, renderer.render(depth))

    def test_metric_depth_multi_contact_and_rotated_sensor(self):
        frames = []
        for rotation in ('0 0 0', '.4 -.7 1.1'):
            model, data = contact_scene(rotation)
            # FOTS sites are at the gel surface, unlike TACTO's camera sites.
            model.site_pos[:, 2] = 0
            mujoco.mj_forward(model, data)
            optics = FOTSOptical(model, {'left': ('left_cam', 'left'), 'right': ('right_cam', 'right')})
            try:
                frame = optics.render(data)
                self.assertEqual(set(frame['left'].contacting_geoms), {model.geom('ball1').id, model.geom('ball2').id})
                self.assertEqual(frame['right'].normal_force_n, 0.)
                self.assertEqual(float(frame['right'].indentation_m.max()), 0.)
                # Sphere radius 4 mm; centre 3.5 mm above plane: overlap .5 mm.
                self.assertAlmostEqual(float(frame['left'].indentation_m.max()), .0005, delta=.00002)
                self.assertEqual(frame['left'].rgb.shape, (320, 240, 3))
                frames.append(frame['left'])
            finally:
                optics.close()
        np.testing.assert_allclose(frames[0].indentation_m, frames[1].indentation_m, atol=2e-8)
        np.testing.assert_allclose(frames[0].rgb, frames[1].rgb, atol=1)
        np.testing.assert_allclose(frames[0].marker_displacement_px, frames[1].marker_displacement_px, atol=1e-3)

    def test_shear_twist_history_release_and_reset(self):
        # A broad box ensures several markers contact it. Move/rotate the
        # object tangentially, then unload: histories cannot leak to next touch.
        xml = '''<mujoco><option gravity="0 0 0"/>
          <worldbody><geom name="gel" type="box" pos="0 0 -.002" size=".01 .015 .002"/>
            <site name="cam"/>
            <body pos="0 0 .0045"><freejoint/>
              <geom name="probe" type="box" size=".004 .004 .005" mass=".02"/>
            </body></worldbody></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        optics = FOTSOptical(model, {'sensor': ('cam', 'gel')})
        def frame():
            mujoco.mj_forward(model, data)
            return optics.render(data)['sensor']
        try:
            first = frame()
            data.qpos[0] += .001
            data.qpos[3:7] = [np.cos(.15), 0., 0., np.sin(.15)]
            data.time = .1
            moved = frame()
            self.assertGreater(np.abs(moved.marker_displacement_px-first.marker_displacement_px).max(), 1.)
            repeat = frame()
            np.testing.assert_array_equal(moved.marker_displacement_px, repeat.marker_displacement_px)
            # No visible movement on the same simulation frame may accumulate.
            data.qpos[2] = .02
            data.time = .2
            released = frame()
            np.testing.assert_array_equal(released.marker_displacement_px, 0.)
            self.assertEqual(optics.history['sensor'], {})
            data.qpos[2] = .0045
            data.time = .3
            recontact = frame()
            optics.reset()
            fresh = frame()
            np.testing.assert_array_equal(recontact.marker_displacement_px, fresh.marker_displacement_px)
        finally:
            optics.close()

    def test_press_and_unload(self):
        env = OpticalBench(sensor='fots')
        optics = FOTSOptical(env.model, env.sensors)
        try:
            baseline = optics.render(env.data)['GELSIGHT']
            env.data.ctrl[2] = -.012
            for _ in range(150):
                env.step(np.zeros(3))
            pressed = optics.render(env.data)['GELSIGHT']
            centre_z = env.data.geom_xpos[env.model.geom('probe').id, 2]
            expected_depth = .05-(centre_z-.005)
            self.assertGreater(pressed.normal_force_n, 1.)
            self.assertGreater(expected_depth, .0001)
            self.assertAlmostEqual(float(pressed.indentation_m.max()), expected_depth, delta=.00002)
            self.assertGreater(np.abs(pressed.rgb.astype(float)-baseline.rgb).mean(), .1)
            env.data.ctrl[2] = 0.
            for _ in range(150):
                env.step(np.zeros(3))
            released = optics.render(env.data)['GELSIGHT']
            self.assertEqual(released.normal_force_n, 0.)
            np.testing.assert_array_equal(released.rgb, baseline.rgb)
            np.testing.assert_array_equal(released.indentation_m, baseline.indentation_m)
        finally:
            optics.close()

    def test_panda_grasp_lift_release(self):
        env = OpticalPandaEnv(sensor='fots')
        optics = FOTSOptical(env.model, env.sensors)
        try:
            with tempfile.TemporaryDirectory() as directory:
                report = demo(env, optics, Path(directory))
            self.assertGreater(report['lift_m'], .09)
            self.assertLess(report['released_cube_z_m'], .121)
            for side in ('left', 'right'):
                self.assertGreater(report['lifted'][side]['normal_force_n'], .1)
                self.assertGreater(report['lifted'][side]['max_indentation_mm'], 0.)
                self.assertLess(report['lifted'][side]['max_indentation_mm'], 1.)
                self.assertEqual(report['released'][side]['max_indentation_mm'], 0.)
        finally:
            optics.close()


if __name__ == '__main__':
    unittest.main()
