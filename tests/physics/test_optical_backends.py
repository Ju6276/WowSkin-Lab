"""Cross-algorithm identity and native GelSight migration regression checks."""
from types import ModuleType
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import mujoco
import numpy as np

from wowskin_lab.core.gelsight_mujoco import SOURCE, GelSightRenderer, source_config
from wowskin_lab.core.optical_backends import make_optics
from wowskin_lab.envs.optical import OpticalBench, OpticalPandaEnv
from wowskin_lab.tools.optical_sensor import demo


class BackendChecks(unittest.TestCase):
    def test_optical_physics_does_not_load_anyskin_profiles(self):
        # Exercise actual model construction; a sensor-profile change must not
        # affect the independent optical hardware through the shared arm code.
        with patch('wowskin_lab.envs.pick_cube.configuration',
                   side_effect=AssertionError('AnySkin profile loaded')):
            for sensor in ('tacto', 'digit_fots', 'gelsight'):
                with self.subTest(sensor=sensor):
                    env = OpticalPandaEnv(sensor=sensor)
                    self.assertEqual(env.model.nflex, 0)
                    self.assertNotIn('skin_model_source', env.cfg)
                    self.assertTrue(env.cfg['disable_workspace_limits'])
    def test_gelsight_contact_and_solver_match_source(self):
        source = mujoco.MjModel.from_xml_path(str(SOURCE/'models/gripper_testbed_PickandPlace.xml'))
        env = OpticalPandaEnv(sensor='gelsight')
        for attr in ('solver', 'integrator', 'iterations', 'tolerance', 'cone',
                     'jacobian', 'impratio', 'noslip_iterations', 'enableflags', 'disableflags'):
            self.assertEqual(getattr(env.model.opt, attr), getattr(source.opt, attr), attr)
        for i in range(source.ngeom):
            name = source.geom(i).name
            if not name.startswith('gelsight0:'):
                continue
            for side in ('left', 'right'):
                target = env.model.geom(f'source_gelsight_{side}_{name.split(":")[1]}').id
                for attr in ('friction', 'solref', 'solimp', 'condim', 'margin', 'gap', 'priority', 'solmix'):
                    values = getattr(env.model, 'geom_'+attr)[target]
                    np.testing.assert_array_equal(values, getattr(source, 'geom_'+attr)[i], err_msg=name+' '+attr)
                for attr in ('pos', 'quat', 'size', 'type'):
                    np.testing.assert_array_equal(getattr(env.model,'geom_'+attr)[target],getattr(source,'geom_'+attr)[i])
        for side in ('left','right'):
            old, new = source.body('gelsight_sensor_body0').id, env.model.body(f'source_gelsight_{side}_module').id
            for attr in ('mass', 'inertia', 'ipos', 'iquat'):
                np.testing.assert_allclose(getattr(env.model,'body_'+attr)[new],getattr(source,'body_'+attr)[old],atol=1e-10)
            oldcam, newcam = source.camera('gelsight0:camera').id, env.model.camera(f'optical_source_gelsight_{side}_camera').id
            for attr in ('pos', 'quat', 'fovy'):
                np.testing.assert_array_equal(getattr(env.model,'cam_'+attr)[newcam],getattr(source,'cam_'+attr)[oldcam])
            self.assertEqual(env.model.body(f'{side}_finger').geomnum,0)
        np.testing.assert_allclose(np.linalg.norm(np.diff(env.sensor_surfaces(),axis=0)),.08,atol=1e-6)

    def test_gelsight_original_image_function_and_noise(self):
        renderer = GelSightRenderer(ideal=True)
        module = ModuleType('reference_gelsight')
        path = SOURCE/'utils/gelsight_driver.py'
        exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
        params, lights = source_config()
        background = cv2.resize(cv2.imread(str(SOURCE/'models/assets/background.png')),
                                (160, 120), interpolation=cv2.INTER_AREA)
        original = module.SimulationApproach(background_img=background, light_sources=lights, **params)
        original.texture_sigma = 0
        rows, cols = np.mgrid[:120, :160]
        depth = .9-np.maximum(0., .02-((rows-60)**2+(cols-80)**2)*.0001)
        rgb, deformed = renderer.render(depth)
        expected, expected_depth = original.generate(depth.copy(), return_depth=True)
        np.testing.assert_array_equal(rgb, expected[..., ::-1])
        np.testing.assert_array_equal(deformed, expected_depth)
        noisy = GelSightRenderer()
        self.assertGreater(noisy.original.texture_sigma, 0.)
        first, second = noisy.render(depth), noisy.render(depth)
        self.assertFalse(np.array_equal(first[0], second[0]))
        # Upstream returns the noiseless smoothed depth, while texture noise
        # only enters normals/lighting. Do not invent noise in saved geometry.
        np.testing.assert_array_equal(first[1], second[1])

    def test_gelsight_transparency_metric_depth_and_unload(self):
        env = OpticalBench(sensor='gelsight')
        optics = make_optics(env, 'gelsight', ideal=True)
        try:
            key = next(iter(env.sensors))
            original = optics.render(env.data)[key]
            env.data.ctrl[2] = -.012
            for _ in range(150):
                env.step(np.zeros(3))
            pressed = optics.render(env.data)[key]
            self.assertEqual(pressed.rgb.shape, (120, 160, 3))
            self.assertGreater(pressed.normal_force_n, .1)
            self.assertTrue(.001 < pressed.indentation_m.max() < .003)
            # A new recorder created under load must retain the same unloaded
            # reference, rather than subtracting away the existing contact.
            started_under_load = make_optics(env, 'gelsight', ideal=True)
            try:
                fresh = started_under_load.render(env.data)[key]
                np.testing.assert_array_equal(fresh.indentation_m,pressed.indentation_m)
                np.testing.assert_array_equal(fresh.rgb,pressed.rgb)
            finally:
                started_under_load.close()
            self.assertGreater(np.abs(pressed.rgb.astype(float)-original.rgb).mean(), .1)
            self.assertEqual(pressed.render_depth_unit, 'legacy_opengl_buffer')
            self.assertTrue(((pressed.render_input_depth >= 0)&(pressed.render_input_depth <= 1)).all())
            # This known 3 cm optical range must not be labelled as ~0.98 m.
            self.assertTrue(.02 < np.median(pressed.camera_depth_m) < .04)
            env.data.ctrl[2] = 0
            for _ in range(150):
                env.step(np.zeros(3))
            released = optics.render(env.data)[key]
            self.assertEqual(released.normal_force_n, 0)
            np.testing.assert_allclose(released.indentation_m, original.indentation_m, atol=2e-7)
            np.testing.assert_array_equal(released.rgb, original.rgb)
        finally:
            optics.close()

    def test_gelsight_panda_source_solver_lift_release(self):
        env = OpticalPandaEnv(sensor='gelsight')
        self.assertEqual(env.model.opt.solver, mujoco.mjtSolver.mjSOL_NEWTON)
        self.assertEqual(env.model.opt.iterations, 100)
        self.assertEqual(env.model.opt.noslip_iterations, 0)
        optics = make_optics(env, 'gelsight', ideal=True)
        try:
            with tempfile.TemporaryDirectory() as directory:
                report = demo(env, optics, Path(directory))
            # Reject the previous unstable catapult, which a lower-only
            # lift test would have incorrectly counted as a successful grip.
            self.assertTrue(.09 < report['lift_m'] < .12)
            self.assertTrue(.119 < report['released_cube_z_m'] < .121)
            for side in ('left', 'right'):
                self.assertGreater(report['lifted'][side]['normal_force_n'], .1)
                self.assertEqual(report['released'][side]['normal_force_n'], 0)
                self.assertLess(report['released'][side]['max_indentation_mm'], .001)
        finally:
            optics.close()

    def test_gelsight_depth_and_rgb_independent_of_host_scene_clipping(self):
        env = OpticalBench(sensor='gelsight')
        optics = make_optics(env, 'gelsight', ideal=True)
        try:
            env.data.ctrl[2] = -.012
            for _ in range(150):
                env.step(np.zeros(3))
            key = next(iter(env.sensors))
            before = optics.render(env.data)[key]
            self.assertGreater(before.normal_force_n, .1)
            # A different room size / task camera setup must not alter a
            # stationary sensor's optical calibration or physical contacts.
            env.model.stat.extent *= 10
            env.model.vis.map.znear = .002
            env.model.vis.map.zfar = 800
            after = optics.render(env.data)[key]
            self.assertEqual(before.normal_force_n, after.normal_force_n)
            np.testing.assert_allclose(before.render_input_depth, after.render_input_depth, atol=1e-7, rtol=0)
            np.testing.assert_allclose(before.camera_depth_m, after.camera_depth_m, atol=2e-7, rtol=0)
            np.testing.assert_allclose(before.rgb, after.rgb, atol=1, rtol=0)
            # Internal-camera overrides must not change task-camera settings.
            self.assertAlmostEqual(env.model.vis.map.znear, .002)
            self.assertEqual(env.model.vis.map.zfar, 800)
        finally:
            optics.close()


if __name__ == '__main__':
    unittest.main()
