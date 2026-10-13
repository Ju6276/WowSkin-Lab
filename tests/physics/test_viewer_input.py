"""Held-key motion does not depend on OS repeat or accumulate after release."""
import unittest

import numpy as np

from wowskin_lab.envs.pick_cube import MOTION_KEYS, ROTATE_AXES, PandaPickCubeTactileEnv, manual_action, rotation_vector, view_translation, task_camera
from wowskin_lab.ui.keyboard import KeyState, MOVE_AXES, UP, direction


class InputChecks(unittest.TestCase):
    def test_cable_interactive_step_and_faster_keyboard_release(self):
        from types import SimpleNamespace
        from wowskin_lab.envs.pick_cube import response_configuration
        from wowskin_lab.envs.cable import PandaCableEnv
        cfg = response_configuration(SimpleNamespace(task='cable',ideal=True))
        self.assertAlmostEqual(cfg['timestep_s'],.0005)
        reference = response_configuration(SimpleNamespace(task='cable',physics_step_us=20))
        cube = response_configuration(SimpleNamespace(task='pick_cube'))
        self.assertAlmostEqual(reference['timestep_s'],.00002)
        self.assertAlmostEqual(cube['timestep_s'],.0005)
        cfg.update(keyboard_speed_m_s=.12,tcp_speed_m_s=.12,
                   teleop_frame='base',teleop_base_lateral_sign=1.,disable_workspace_limits=True)
        env = PandaCableEnv(cfg,pollen=False)
        self.assertEqual(env.steps_per_sample,20)
        keys = KeyState(MOVE_AXES)
        keys.event(ord('D'),True)
        start=env.data.site_xpos[env.tcp_site].copy()
        moving=False
        targets=[]
        for _ in range(100):
            a,moving=manual_action(env,keys.sample().held,moving,camera_azimuth=135)
            env.step(a)
            targets.append(env.target_xyz.copy())
        end=env.data.site_xpos[env.tcp_site].copy()
        distance=float((end-start)[1])
        # 120 mm/s is an input ceiling. The retained 6 mm target lead can
        # reduce travel when joint PD tracking lags; it must not turn the path.
        self.assertGreater(distance,.03)
        self.assertLess(distance,.125)
        targets=np.asarray(targets)
        np.testing.assert_allclose(targets[:,[0,2]],np.broadcast_to(start[[0,2]],(100,2)),atol=1e-9)
        self.assertTrue(np.all(np.diff(targets[:,1])>=-1e-10))
        self.assertLess(np.max(np.diff(targets[:,1])),.001201)
        self.assertLess(np.linalg.norm(end-env.target_xyz),.008)
        keys.event(ord('D'),False)
        for _ in range(50):
            a,moving=manual_action(env,keys.sample().held,moving,camera_azimuth=135)
            env.step(a)
        residual=float(np.linalg.norm(env.data.site_xpos[env.tcp_site]-end))
        self.assertFalse(moving)
        self.assertLess(residual,.002)
        self.assertAlmostEqual(env.data.time,1.5,places=7)
        self.assertGreater(env.min_volume_ratio,.95)
        np.testing.assert_array_equal(env.data.warning.number,0)
        print(f'Cable keyboard: {distance*1000:.2f} mm / sim-second; release residual {residual*1000:.3f} mm')

    def test_hold_repeat_release_and_focus_loss(self):
        keys = KeyState(set(MOVE_AXES) | {ord('F')})
        self.assertFalse(keys.event(ord('H'), True))
        keys.event(ord('W'), True)
        self.assertEqual(keys.sample().commands, (ord('W'),))
        for _ in range(100):
            np.testing.assert_array_equal(direction(keys.sample().held), [0, 1, 0])
        keys.event(ord('W'), True, repeat=True)
        self.assertEqual(keys.sample().commands, ())
        keys.event(ord('W'), False)
        np.testing.assert_array_equal(direction(keys.sample().held), [0, 0, 0])
        keys.event(ord('F'), True)
        keys.event(ord('F'), True, repeat=True)
        self.assertEqual(keys.sample().commands, (ord('F'),))
        keys.event(ord('W'), True)
        keys.clear()  # Window/application focus loss discards queued input too.
        keys.event(ord('W'), True, repeat=True)
        self.assertEqual(keys.sample().held, frozenset())

    def test_combined_keys_cancel_and_normalize(self):
        np.testing.assert_array_equal(direction({ord('W'), UP}), [0, 1, 0])
        np.testing.assert_array_equal(direction({ord('W'), ord('S')}), [0, 0, 0])
        self.assertAlmostEqual(np.linalg.norm(direction({ord('W'), ord('D')})), 1.)
        np.testing.assert_array_equal(direction({ord('I'), ord('K')}, ROTATE_AXES), [0, 0, 0])
        self.assertAlmostEqual(np.linalg.norm(direction({ord('I'), ord('L')}, ROTATE_AXES)), 1.)

    def test_three_local_rotation_axes_hold_then_release(self):
        env = PandaPickCubeTactileEnv(pollen=False)
        for axis, code in enumerate(map(ord, 'ILU')):
            with self.subTest(axis=axis):
                env.reset()
                start = env.data.site_xmat[env.tcp_site].reshape(3, 3).copy()
                position = env.data.site_xpos[env.tcp_site].copy()
                keys = KeyState(MOTION_KEYS)
                keys.event(code, True)
                moving = False
                for _ in range(50):
                    action, moving = manual_action(env, keys.sample().held, moving)
                    env.step(action)
                end = env.data.site_xmat[env.tcp_site].reshape(3, 3).copy()
                rotated = rotation_vector(start.T @ end)
                self.assertGreater(rotated[axis], .15)
                self.assertLess(rotated[axis], .23)
                self.assertLess(np.linalg.norm(np.delete(rotated, axis)), .01)
                self.assertLess(np.linalg.norm(env.data.site_xpos[env.tcp_site]-position), .002)
                keys.event(code, False)
                for _ in range(30):
                    action, moving = manual_action(env, keys.sample().held, moving)
                    env.step(action)
                stopped = rotation_vector(end.T @ env.data.site_xmat[env.tcp_site].reshape(3, 3))
                self.assertFalse(moving)
                self.assertLess(np.linalg.norm(stopped), .01)
                np.testing.assert_array_equal(env.data.warning.number, 0)
                print(f'Local axis {axis}: {np.rad2deg(rotated[axis]):.2f} deg; '
                      f'after release: {np.rad2deg(np.linalg.norm(stopped)):.3f} deg')

    def test_view_direction_agrees_with_render_camera(self):
        import mujoco
        env = PandaPickCubeTactileEnv(pollen=False)
        scene = mujoco.MjvScene(env.model, maxgeom=4000)
        for azimuth in (0, 90, 135, 225):
            camera = task_camera(env)
            camera.azimuth = azimuth
            mujoco.mjv_updateScene(env.model, env.data, mujoco.MjvOption(), None,
                                  camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
            forward = np.array(scene.camera[0].forward)
            right = np.cross(forward, scene.camera[0].up)
            forward[2] = right[2] = 0
            forward /= np.linalg.norm(forward)
            right /= np.linalg.norm(right)
            np.testing.assert_allclose(view_translation([0, 1, 0], azimuth), forward, atol=1e-6)
            np.testing.assert_allclose(view_translation([1, 0, 0], azimuth), right, atol=1e-6)

    def test_physics_moves_while_held_then_stops(self):
        env = PandaPickCubeTactileEnv(pollen=False)
        keys = KeyState(MOVE_AXES)
        start = env.data.site_xpos[env.tcp_site].copy()
        keys.event(ord('W'), True)  # Exactly one down event, no repeat events.
        moving = False
        for _ in range(100):
            action, moving = manual_action(env, keys.sample().held, moving, camera_azimuth=135)
            env.step(action)
        end = env.data.site_xpos[env.tcp_site].copy()
        # Default teleoperation is robot-base aligned, independent of the
        # camera. The old assertion still assumed the removed view default.
        forward = np.array([1., 0., 0.])
        distance = np.dot(end-start, forward)
        self.assertGreater(distance, .03)
        self.assertLess(distance, .045)
        self.assertLess(abs((end-start)[1]), .001)
        keys.event(ord('W'), False)
        release = end.copy()
        for _ in range(50):
            action, moving = manual_action(env, keys.sample().held, moving, camera_azimuth=135)
            env.step(action)
        self.assertFalse(moving)
        self.assertLess(np.linalg.norm(env.data.site_xpos[env.tcp_site]-release), .001)
        print(f'Held 1 sim-second: {distance*1000:.2f} mm; '
              f'after release: {np.linalg.norm(env.data.site_xpos[env.tcp_site]-release)*1000:.3f} mm')


if __name__ == '__main__':
    unittest.main()
