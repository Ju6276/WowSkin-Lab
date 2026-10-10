"""Time-index sentinels verify historical patched loaders against known labels."""
import subprocess
import sys
import unittest


SCRIPT = r'''
import sys, tempfile, pickle
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tests.legacy.test_recording import fixture
from wowskin_lab.legacy.policy_export import export_dataset
from wowskin_lab.legacy.policy_smoke import build

method = sys.argv[1]
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    fixture(root, count=35)
    dst = root/'data.pkl'
    export_dataset(root, dst, method)
    dataset, _, _ = build(method, dst, agent_needed=False)
    with patch('numpy.random.randint', return_value=12):
        sample = dataset._sample()
    if method == 'ftf':
        # The official loader trims the initial stationary frame. History
        # therefore covers original t=4..13, with next-time targets 5..14.
        stats = dataset.stats
        force = sample['future_force_states'] * (stats['force_states']['max']-stats['force_states']['min']+1e-5) + stats['force_states']['min']
        expected_time = np.arange(5,15)[:,None]+np.arange(10)
        np.testing.assert_allclose(force, expected_time, atol=1e-5)
        actions = sample['actions'] * (stats['actions']['max']-stats['actions']['min']+1e-5) + stats['actions']['min']
        np.testing.assert_allclose(actions[:,:,0], expected_time**2*.001, atol=1e-6)
        past = sample['past_force_states']*(stats['force_states']['max']-stats['force_states']['min']+1e-5)+stats['force_states']['min']
        np.testing.assert_allclose(past, np.arange(4,14), atol=1e-5)
    else:
        # Compare chunk and non-chunk samplers at identical observation time.
        dataset._temporal_agg = False
        with patch('numpy.random.randint', return_value=12):
            single = dataset._sample()
        np.testing.assert_allclose(sample['actions'][:,0], single['actions'])
        # No second baseline subtraction of this already-baselined signal.
        episode = dataset._episodes[0][0]
        np.testing.assert_allclose(episode['observation']['sensor0_states'][1,0], 0.)
        np.testing.assert_allclose(episode['observation']['sensor0_states'][4,0], 3.)
print(method, 'time alignment passed')
'''


class UpstreamAlignmentChecks(unittest.TestCase):
    def test_legacy_visk_label_alignment(self):
        self.check_method('visk')

    def test_legacy_ftf_history_and_contact_alignment(self):
        self.check_method('ftf')

    def check_method(self, method):
        result = subprocess.run([sys.executable, '-c', SCRIPT, method],
                                text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)


if __name__ == '__main__':
    unittest.main()
