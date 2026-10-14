"""Exercise the actual pinned replay/DDQN methods without starting TensorFlow."""
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from wowskin_lab.learning.gelsight_ddqn import SOURCE, prepare, verify_workspace


def source_class(path, name):
    # Execute the real class, omitting only the module's optional TensorFlow
    # import. Network API calls below use controlled numerical test doubles.
    module = ast.parse(path.read_text())
    node = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == name)
    namespace = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


@unittest.skipUnless((SOURCE/'Test_bed/rl_algos/ddqn_model.py').is_file(), 'Optional pinned GelSightMujoco source not installed')
class GelSightDDQNSourceChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)/'workspace'
        prepare(self.workspace)

    def test_multimodal_replay_preserves_distinct_current_and_next_states(self):
        cls = source_class(self.workspace/'source/rl_algos/replaybuffer.py', 'ReplayBuffer')
        buffer = cls(n_actions=8, tact_dims=(2, 2, 3), num_inp=3,
                     nn_type=[False, True, False, True], max_size=4)
        state = [np.full((2, 2, 3), 1.), np.full((2, 2, 3), 2.), np.full(3, 3.)]
        following = [v+10 for v in state]
        buffer.store_transition(state, 2, -.4, following, False)
        for index in range(3):
            np.testing.assert_array_equal(buffer.state_memory[index][0], state[index])
            np.testing.assert_array_equal(buffer.next_state_memory[index][0], following[index])
            self.assertFalse(np.shares_memory(buffer.state_memory[index], buffer.next_state_memory[index]))
        self.assertEqual(buffer.terminal_memory[0], 1.)
        buffer.store_transition(state, 3, -.5, following, True)
        self.assertEqual(buffer.terminal_memory[1], 0.)

    def test_source_ddqn_selects_with_online_network_and_values_with_target(self):
        cls = source_class(self.workspace/'source/rl_algos/ddqn_model.py', 'ddqn')
        class Network:
            def __init__(self, predictions):
                self.predictions = iter(predictions)
            def predict(self, states):
                return next(self.predictions).copy()
            def fit(self, states, targets, **kwargs):
                self.fitted = targets
                return 'fit'
        class Buffer:
            memory_count = 3
            def sample_buffer(self, size):
                return (np.zeros((2, 1)), np.array([[1, 0], [0, 1]]),
                        np.array([-1., -2.]), np.ones((2, 1)), np.array([1., 0.]))
        online = Network([np.zeros((2, 2)), np.array([[1., 2.], [3., 0.]])])
        target = Network([np.array([[100., 5.], [9., 100.]])])
        agent = cls(.9, 2, 1., 2, .99, .1, online, target, Buffer())
        agent.learn()
        np.testing.assert_allclose(online.fitted, [[3.5, 0.], [0., -2.]])

    def test_prepared_source_tampering_is_rejected(self):
        path = self.workspace/'source/rl_algos/ddqn_model.py'
        path.write_text(path.read_text()+'\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'modified'):
            verify_workspace(self.workspace)
