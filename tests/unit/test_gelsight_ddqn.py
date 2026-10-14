"""The discrete RL adapter must keep real control time and terminal semantics."""
from types import SimpleNamespace
import unittest

import numpy as np

from wowskin_lab.learning.gelsight_ddqn_env import PandaGelSightDDQN, discrete_command, source_reward


class GelSightDDQNAdapterChecks(unittest.TestCase):
    def environment(self, *, success_at=None):
        bridge = PandaGelSightDDQN.__new__(PandaGelSightDDQN)
        actions = []
        data = SimpleNamespace(time=0., qpos=np.zeros(1), warning=SimpleNamespace(number=np.zeros(1)),
                               xpos=np.array([[.4, 0., .12]]), site_xpos=np.array([[.4, 0., .2]]))
        env = SimpleNamespace(data=data, success=False, cfg={'sample_hz': 100}, probe_body=0, tcp_site=0)
        def step(command):
            actions.append(command.copy())
            data.time += .01
            data.site_xpos[0] += command[:3]
            env.success = len(actions) == success_at
            return None, 0., False, False, dict(cube_xyz_m=data.xpos[0], lift_m=0., grasped=False)
        env.step = step
        env.gripper_fraction = lambda target: .2
        env.task_evaluation = lambda: dict(success=env.success)
        bridge.env, bridge.goal = env, np.array([.49, .135, .12])
        bridge.translation, bridge.grip_step = .005, .1
        bridge.action_ticks, bridge.max_steps = 10, 2
        bridge.steps, bridge.ended = 0, False
        bridge.observation = lambda: data.site_xpos.copy()
        return bridge, actions

    def test_target_increment_is_not_repeated_and_timeout_bootstraps(self):
        env, actions = self.environment()
        _, _, terminated, truncated, info = env.step(0)
        self.assertFalse(terminated or truncated)
        self.assertEqual(len(actions), 10)
        np.testing.assert_allclose(np.sum(actions, axis=0)[:3], [.005, 0., 0.])
        np.testing.assert_allclose(np.asarray(actions)[:, 6], .2)
        self.assertAlmostEqual(info['time_s'], .1)
        _, _, terminated, truncated, _ = env.step(1)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_real_success_ends_without_waiting_for_time_limit(self):
        env, actions = self.environment(success_at=3)
        _, _, terminated, truncated, info = env.step(6)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info['success'])
        self.assertEqual(len(actions), 3)
        self.assertAlmostEqual(info['time_s'], .03)
        np.testing.assert_allclose(np.asarray(actions)[:, 6], .3)

    def test_reward_and_gripper_bounds(self):
        self.assertEqual(source_reward([0., 0., 0.], [3., 0., 0.], [3., 4., 0.]), -7.)
        self.assertEqual(discrete_command(6, .98)[6], 1.)
        self.assertEqual(discrete_command(7, .02)[6], 0.)
        for action in (-1, 8, .5, True):
            with self.assertRaises(ValueError):
                discrete_command(action, 0.)
