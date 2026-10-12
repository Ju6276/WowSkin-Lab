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
import params as pr
p = os.environ['FOTS_TEST_DIR']
d = np.load(p+'/depth.npy')
with torch.inference_mode():
    image = get_simapproach().generate(d*1000*pr.mm_to_pixel, d>0, shadow=False)
np.save(p+'/rgb.npy', image[..., ::-1])
'''
            subprocess.run([sys.executable, '-c', source], cwd=FOTS_ROOT,
                           env=dict(os.environ, FOTS_TEST_DIR=directory, PYTHONDONTWRITEBYTECODE='1'), check=True)
            np.testing.assert_array_equal(actual, np.load(Path(directory)/'rgb.npy'))
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
