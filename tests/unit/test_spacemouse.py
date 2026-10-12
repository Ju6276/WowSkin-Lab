"""HID input, stop behavior, and canonical 7D simulation command contract."""
import unittest
from types import SimpleNamespace

import numpy as np

from wowskin_lab.teleop.spacemouse import MouseState, SpaceMouseController


def report(kind, values):
    return [kind, *np.asarray(values, dtype='<i2').tobytes()]


def fake_env():
    env = SimpleNamespace(cfg={'sample_hz': 100}, tcp_site=0,
                          data=SimpleNamespace(site_xpos=np.array([[.4,0,.2]]),
                                               site_xmat=np.eye(3).reshape(1,9), qpos=np.zeros(9)),
                          arm_q=np.arange(7), finger_q=np.array([7,8]),
                          target_xyz=np.array([.4,0,.2]), target_rotation=np.eye(3), closure=0.)
    env.gripper_fraction = lambda target=False: env.closure
    env.command_gripper = lambda close: setattr(env, 'closure', float(close))
    env.hold_action = lambda: np.r_[np.zeros(6), env.closure]
    return env


class SpaceMouseChecks(unittest.TestCase):
    def test_front_facing_base_lateral_input_and_pitch_are_separate(self):
        from wowskin_lab.envs.pick_cube import keyboard_translation
        from scipy.spatial.transform import Rotation
        for yaw in (0.,135.,270.):
            env, state, controller = fake_env(), MouseState(), SpaceMouseController()
            env.cfg.update(teleop_frame='base',teleop_base_lateral_sign=1.,disable_workspace_limits=True)
            env.model = SimpleNamespace(body=lambda name: SimpleNamespace(id=0))
            env.data.xmat = np.eye(3).reshape(1,9)
            env.data.xpos = np.zeros((1,3))
            rotation = np.diag([1.,-1.,-1.])
            env.data.site_xmat = rotation.reshape(1,9)
            controller.action(env,state.sample(),yaw,now=1.)
            for sign in (-1.,1.):
                state.feed(report(1,[350*sign,0,0,350,0,0]),2.)
                action = controller.action(env,state.sample(),yaw,now=2.)
                np.testing.assert_allclose(action[:3],[0.,sign*.0012,0.],atol=1e-12)
                np.testing.assert_allclose(action[:3],.0012*keyboard_translation(env,[sign,0,0],yaw),atol=1e-12)
                next_rotation = rotation @ Rotation.from_rotvec(action[3:6]).as_matrix()
                self.assertGreater((next_rotation[:,2]-rotation[:,2])[0],0.)

    def test_base_frame_translation_and_rotation_ignore_view(self):
        from wowskin_lab.envs.pick_cube import keyboard_translation
        base = np.array([[0.,-1.,0.], [1.,0.,0.], [0.,0.,1.]])
        results = []
        for azimuth in (0., 135., 270.):
            env, state, controller = fake_env(), MouseState(), SpaceMouseController()
            env.cfg.update(teleop_frame='base', disable_workspace_limits=True)
            env.model = SimpleNamespace(body=lambda name: SimpleNamespace(id=0))
            env.data.xmat = base.reshape(1,9)
            env.data.xpos = np.zeros((1,3))
            np.testing.assert_allclose(keyboard_translation(env, [0,1,0], azimuth), base[:,0])
            controller.action(env, state.sample(), azimuth, now=1.)
            state.feed(report(1, [0,-350,0,350,0,0]), 2.)
            results.append(controller.action(env, state.sample(), azimuth, now=2.))
        for action in results:
            np.testing.assert_allclose(action[:3], .0012*base[:,0], atol=1e-12)
            np.testing.assert_allclose(action, results[0], atol=1e-12)

    def test_unlimited_mode_keeps_target_lead_and_centered_target_but_focus_loss_stops(self):
        env, state, controller = fake_env(), MouseState(), SpaceMouseController()
        env.cfg['disable_cartesian_limits'] = True
        controller.action(env, state.sample(), 90, now=1.)
        # The arm is lagging well behind its target.
        env.target_xyz[0] = .48
        state.feed(report(1, [350, 0, 0, 0, 0, 0]), 2.)
        command = controller.action(env, state.sample(), 90, now=2.)
        self.assertGreater(command[0], 0)
        self.assertEqual(env.target_xyz[0], .48)
        state.feed(report(1, [0]*6), 2.1)
        controller.action(env, state.sample(), 90, now=2.1)
        self.assertEqual(env.target_xyz[0], .48)
        state.feed(report(1, [350, 0, 0, 0, 0, 0]), 2.2)
        controller.action(env, state.sample(), 90, now=2.2)
        controller.action(env, state.sample(), 90, focused=False, now=2.2)
        np.testing.assert_allclose(env.target_xyz, env.data.site_xpos[0])

    def test_combined_split_signed_axes_and_status_does_not_refresh_motion(self):
        s = MouseState()
        s.feed(report(1, [350,-350,-175,70,-35,0]), 10.)
        np.testing.assert_allclose(s.sample().axes, [1,-1,-.5,.2,-.1,0])
        s.feed([23,9,1], 11.)
        np.testing.assert_array_equal(s.sample().received, [10,10])
        s.feed(report(1, [0,35,0]), 12.)
        s.feed(report(2, [-350,0,350]), 12.1)
        snap = s.sample()
        np.testing.assert_allclose(snap.axes, [0,.1,0,-1,0,1])
        np.testing.assert_allclose(snap.received, [12,12.1])
        s.feed([1,0,1], 13.)  # malformed does not refresh or zero old axes
        np.testing.assert_array_equal(s.sample().received, snap.received)

    def test_buttons_only_on_press_and_both_hold(self):
        s = MouseState()
        s.feed([3,1,0]);s.feed([3,1,0]);s.feed([3,0,0]);s.feed([3,2,0])
        self.assertEqual(s.sample().presses, ('close','open'))
        self.assertEqual(s.sample().presses, ())
        s.feed([3,3,0])
        self.assertEqual(s.sample().presses, ('hold',))

    def test_seven_d_camera_frame_release_timeout_and_focus_rearm(self):
        env, s, controller = fake_env(), MouseState(), SpaceMouseController()
        controller.action(env, s.sample(), 90, now=1.)
        self.assertTrue(controller.armed)
        s.feed(report(1, [350,0,0,0,0,0]), 2.)
        command = controller.action(env, s.sample(), 90, now=2.)
        np.testing.assert_allclose(command, [.0012,0,0,0,0,0,0], atol=1e-10)
        self.assertTrue(controller.moving)
        s.feed(report(1,[0]*6), 2.1)
        np.testing.assert_array_equal(controller.action(env,s.sample(),90,now=2.1),np.zeros(7))
        self.assertFalse(controller.moving)
        s.feed(report(1,[0,0,-350,0,0,0]),3.)
        self.assertGreater(controller.action(env,s.sample(),90,now=3.)[2],0)
        self.assertFalse(np.any(controller.action(env,s.sample(),90,now=3.3)))
        self.assertFalse(controller.armed)
        s.feed(report(1,[350,0,0,0,0,0]),3.4)
        self.assertFalse(np.any(controller.action(env,s.sample(),90,now=3.4)))
        s.feed(report(1,[0]*6),3.5)
        controller.action(env,s.sample(),90,now=3.5)
        s.feed(report(1,[0]*6),4.)
        s.feed([3,1,0],4.)
        self.assertEqual(controller.action(env,s.sample(),90,now=4.)[6],1.)
        controller.action(env,s.sample(),90,focused=False,now=4.)
        self.assertFalse(controller.armed)
        s.feed([3,2,0],4.1)
        controller.action(env,s.sample(),90,focused=False,now=4.1)
        self.assertEqual(env.closure,1.)  # Background presses discarded.

    def test_rotation_is_converted_to_local_frame(self):
        env, s, controller = fake_env(), MouseState(), SpaceMouseController()
        env.data.site_xmat[:] = np.diag([1,-1,-1]).reshape(1,9)
        controller.action(env,s.sample(),90,now=1.)
        s.feed(report(1,[0,0,0,0,0,350]),2.)
        a=controller.action(env,s.sample(),90,now=2.)
        np.testing.assert_allclose(a[3:6],[0,0,-.005],atol=1e-10)

    def test_forward_cap_tilt_moves_downward_tool_away_in_each_view(self):
        from scipy.spatial.transform import Rotation
        # Check the physical motion of a tool tip, rather than just checking
        # a sign array. Front/back and oblique views must agree on away.
        for azimuth in (0, 90, 135, 180, 270):
            for input_sign in (-1, 1):
                env, s, controller = fake_env(), MouseState(), SpaceMouseController()
                rotation = np.diag([1., -1., -1.])
                env.data.site_xmat[:] = rotation.reshape(1, 9)
                controller.action(env, s.sample(), azimuth, now=1.)
                s.feed(report(1, [0, 0, 0, input_sign*350, 0, 0]), 2.)
                command = controller.action(env, s.sample(), azimuth, now=2.)
                next_rotation = rotation @ Rotation.from_rotvec(command[3:6]).as_matrix()
                tip_motion = next_rotation[:, 2] - rotation[:, 2]
                angle = np.deg2rad(azimuth)
                away = np.array([np.cos(angle), np.sin(angle), 0.])
                right = np.array([np.sin(angle), -np.cos(angle), 0.])
                self.assertGreater(input_sign * np.dot(tip_motion, away), 0)
                self.assertAlmostEqual(np.dot(tip_motion, right), 0)
                np.testing.assert_array_equal(command[:3], np.zeros(3))

    def test_both_buttons_hold_uses_source_servo_coordinate_mapping(self):
        env, s, controller = fake_env(), MouseState(), SpaceMouseController()
        env.cfg['gripper_control_mode'] = 'panda_menagerie_position'
        env.model = SimpleNamespace(jnt_range=np.array([[-.0349, 0.]]))
        env.finger_joints = [0, 0]
        env.data.qpos[env.finger_q] = -.025
        controller.action(env, s.sample(), 90, now=1.)
        s.feed([3,3,0], 2.)
        controller.action(env, s.sample(), 90, now=2.)
        self.assertAlmostEqual(env.grip_target, -.025/-.04*-.0349)
        env.cfg['gripper_control_mode'] = 'panda_robosuite_position'
        s.feed([3,0,0], 2.1); s.feed([3,3,0], 2.2)
        controller.action(env, s.sample(), 90, now=2.2)
        self.assertAlmostEqual(env.grip_target, -.025)


if __name__ == '__main__':
    unittest.main()
