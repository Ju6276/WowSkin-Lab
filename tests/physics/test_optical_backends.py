"""Cross-algorithm identity and native GelSight migration regression checks."""
from types import ModuleType
from pathlib import Path
import tempfile
import unittest

import cv2
import mujoco
import numpy as np

from wowskin_lab.core.gelsight_mujoco import SOURCE, GelSightRenderer, source_config
from wowskin_lab.core.optical_backends import make_optics
from wowskin_lab.envs.optical import OpticalBench, OpticalPandaEnv
from wowskin_lab.tools.optical_sensor import demo


class BackendChecks(unittest.TestCase):
    def test_digit_suspended_hold_does_not_creep_or_accumulate_flow(self):
        env = OpticalPandaEnv(sensor='digit_fots')
        optics = make_optics(env, 'digit_fots', ideal=True)
        self.assertEqual(env.model.opt.impratio, 20)
        self.assertEqual(env.model.opt.noslip_iterations, 5)
        self.assertEqual(env.model.opt.solver, mujoco.mjtSolver.mjSOL_NEWTON)
        def move(xyz, seconds):
            env.target_xyz = np.asarray(xyz, float)
            for i in range(round(seconds*env.cfg['sample_hz'])):
                env.step(env.hold_action())
                if i % 10 == 0:
                    optics.render(env.data)
        try:
            move([.4, 0, .17], 1.5)
            move([.4, 0, .125], 1.5)
            env.command_gripper(True)
            move([.4, 0, .125], 1)
            move([.4, 0, .23], 2)
            initial = env.data.xpos[env.probe_body].copy()
            self.assertGreater(initial[2], .22)
            max_flow = 0.
            for _ in range(60):
                for _ in range(100):
                    env.step(env.hold_action())
                frames = optics.render(env.data)
                for frame in frames.values():
                    self.assertGreater(frame.normal_force_n, .1)
                    max_flow = max(max_flow, float(np.linalg.norm(frame.marker_displacement_px, axis=-1).max()))
            self.assertLess(np.linalg.norm(env.data.xpos[env.probe_body]-initial), .0001)
            self.assertLess(max_flow, .2)
        finally:
            optics.close()

    def test_digit_same_physics_and_both_algorithms_release(self):
        a = OpticalPandaEnv(sensor='tacto')
        for sensor in ('digit_fots', 'digit_compare'):
            b = OpticalPandaEnv(sensor=sensor)
            for name in ('body_pos', 'body_quat', 'geom_pos', 'geom_quat', 'geom_friction',
                         'geom_contype', 'geom_conaffinity', 'body_mass', 'body_inertia', 'jnt_range'):
                np.testing.assert_array_equal(getattr(a.model, name), getattr(b.model, name))
            np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
        optics = make_optics(b, 'digit_compare', ideal=True)
        try:
            with tempfile.TemporaryDirectory() as directory:
                report = demo(b, optics, Path(directory))
            self.assertTrue(.09 < report['lift_m'] < .12)
            self.assertTrue(.119 < report['released_cube_z_m'] < .121)
            for side in ('left', 'right'):
                t = report['lifted'][side+' / TACTO']
                f = report['lifted'][side+' / FOTS']
                self.assertEqual(t['normal_force_n'], f['normal_force_n'])
                self.assertGreater(t['normal_force_n'], .1)
                self.assertGreater(t['max_indentation_mm'], f['max_indentation_mm'])
                self.assertGreater(f['max_indentation_mm'], 0)
                for algorithm in ('TACTO', 'FOTS'):
                    self.assertEqual(report['released'][side+' / '+algorithm]['max_indentation_mm'], 0)
        finally:
            optics.close()

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

    def test_digit_marker_shear_twist_and_recontact(self):
        env = OpticalBench(sensor='digit_compare', probe_shape='box')
        optics = make_optics(env, 'digit_compare', ideal=True)
        def capture(target):
            env.data.ctrl[:] = target
            frames = None
            for i in range(100):
                env.step(np.zeros(6))
                if i % 10 == 9:
                    frames = optics.render(env.data)
            return frames['DIGIT / FOTS']
        try:
            pressed = capture([0, 0, -.012, 0, 0, 0])
            moved = capture([.002, 0, -.012, 0, 0, .35])
            self.assertEqual(pressed.marker_displacement_px.shape[-1], 2)
            self.assertEqual(pressed.marker_displacement_px.size, 9*8*2)
            self.assertGreater(np.abs(pressed.marker_displacement_px-moved.marker_displacement_px).max(), 1)
            released = capture([0, 0, 0, 0, 0, 0])
            np.testing.assert_array_equal(released.marker_displacement_px, 0)
            self.assertEqual(optics.fots.history['DIGIT'], {})
            capture([0, 0, -.012, 0, 0, 0])
            optics.reset()
            self.assertEqual(optics.fots.history, {})
        finally:
            optics.close()

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
