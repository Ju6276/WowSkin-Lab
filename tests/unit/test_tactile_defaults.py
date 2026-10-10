import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from wowskin_lab.__main__ import main


class TactileDefaultsTests(unittest.TestCase):
    def command(self, args):
        captured = []
        module = SimpleNamespace(main=lambda: captured.extend(sys.argv[1:]))
        with patch('wowskin_lab.__main__.importlib.import_module', return_value=module):
            main(args)
        return captured

    def test_all_three_scenes_use_same_fitted_response(self):
        for task in ('pick_cube', 'cable', 'feeding'):
            args = self.command(['teleop', '--task', task])
            self.assertIn('--magnetic-fit', args)
            self.assertTrue(args[args.index('--magnetic-fit')+1].endswith('magnetic_fit_pollen_nonslip.json'))
            self.assertIn('fitted_dipole_preview', args[args.index('--record-dir')+1])

    def test_explicit_network_does_not_get_gain_profile(self):
        args = self.command(['teleop', '--reskin-model=outputs/reskin_forward_mapped'])
        self.assertNotIn('--magnetic-fit', args)
