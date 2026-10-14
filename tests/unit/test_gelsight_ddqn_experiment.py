"""Resuming must retain the learning state, not merely a policy's weights."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from wowskin_lab.learning.gelsight_ddqn_experiment import load_resume, save_resume, evaluate_policy


class Network:
    def __init__(self, value):
        self.weights = [np.array([value], dtype=np.float32)]
        self.optimizer = self
    def _make_train_function(self):
        pass
    def get_weights(self):
        return self.weights
    def set_weights(self, values):
        self.weights = values


class ExperimentChecks(unittest.TestCase):
    def test_resume_restores_both_networks_replay_epsilon_and_random_stream(self):
        with TemporaryDirectory() as directory:
            memory = SimpleNamespace(memory_count=5, memory_size=3,
                state_memory=[np.arange(6).reshape(3, 2).astype(float)],
                next_state_memory=[np.arange(6).reshape(3, 2).astype(float)+10],
                action_memory=np.eye(3, dtype=np.int8), reward_memory=np.arange(3.), terminal_memory=np.ones(3))
            learner = SimpleNamespace(q_evaluation=Network(2.), q_target=Network(7.), memory=memory, epsilon=.42)
            learner.get_epsilon = lambda: learner.epsilon
            learner.set_epsilon = lambda value: setattr(learner, 'epsilon', value)
            tf = SimpleNamespace(keras=SimpleNamespace(backend=SimpleNamespace(batch_get_value=lambda v:v)))
            keys = ('source', 'environment_fingerprint', 'environment', 'learning', 'seed', 'runtime_versions',
                    'validation_seed', 'test_seed', 'eval_episodes', 'test_episodes', 'checkpoint_every', 'eval_every')
            report = dict.fromkeys(keys, 1)
            np.random.seed(21)
            path = Path(directory)/'resume.h5'
            save_resume(path, learner, report, tf)
            expected = np.random.rand(5)
            memory.state_memory[0][:] = -1
            memory.next_state_memory[0][:] = -2
            memory.memory_count = 0
            learner.q_target.set_weights([np.zeros(1)])
            learner.epsilon = 1.
            load_resume(path, learner, report, tf)
            np.testing.assert_array_equal(np.random.rand(5), expected)
            np.testing.assert_array_equal(memory.state_memory[0], np.arange(6).reshape(3, 2))
            np.testing.assert_array_equal(memory.next_state_memory[0], np.arange(6).reshape(3, 2)+10)
            np.testing.assert_array_equal(learner.q_target.get_weights()[0], [7.])
            self.assertEqual(memory.memory_count, 5)
            self.assertEqual(learner.epsilon, .42)
            with self.assertRaisesRegex(ValueError, 'environment_fingerprint'):
                load_resume(path, learner, {**report, 'environment_fingerprint': 2}, tf)

    def test_evaluation_restores_rng_and_counts_real_outcomes(self):
        class Environment:
            def reset(self, seed):
                self.steps = 0
                return [np.zeros(1)]
            def step(self, action):
                self.steps += 1
                np.random.rand()  # Simulate an observation implementation using the global RNG.
                return [np.zeros(1)], -1., False, self.steps == 2, dict(
                    steps=self.steps, lift_m=0., grasped=False, success=False)
        with TemporaryDirectory() as directory:
            np.random.seed(33)
            expected = np.random.rand(4)
            np.random.seed(33)
            result = evaluate_policy(Environment(), None, Path(directory), seeds=[1000, 1001], random=True, videos=0)
            np.testing.assert_array_equal(np.random.rand(4), expected)
            self.assertEqual(result['successes'], 0)
            self.assertEqual(result['mean_reward'], -2.)
            self.assertEqual([e['steps'] for e in result['episodes']], [2, 2])
