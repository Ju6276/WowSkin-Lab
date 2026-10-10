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
