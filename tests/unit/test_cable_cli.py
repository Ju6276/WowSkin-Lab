"""The standalone cable entry point must construct a real sensor config."""
from pathlib import Path
import unittest
from unittest.mock import patch

from wowskin_lab.envs import cable
from wowskin_lab.paths import ROOT


class CableCLITests(unittest.TestCase):
    def run_main(self, *args):
        with patch('sys.argv', ['cable', 'viewer', *args]), \
                patch.object(cable, 'PandaCableEnv') as environment, \
                patch.object(cable, 'viewer') as viewer:
            cable.main()
        viewer.assert_called_once()
        self.assertIs(viewer.call_args.args[0], environment.return_value)
        return environment.call_args.args[0]

    def test_viewer_entry_point_configures_selected_layout(self):
        config = self.run_main('--cable-layout', 'single_clip')
        self.assertEqual(config['cable_layout'], 'single_clip')
        self.assertEqual(config['sample_hz'], 100)
        self.assertEqual(config['skin_dof'], 'trilinear')
        self.assertEqual(config['integrator'], 'discrete')
        self.assertGreater(config['young_pa'], 0)

    def test_magnetic_model_and_ideal_match_teleoperation_defaults(self):
        config = self.run_main()
        self.assertEqual(Path(config['magnetic_fit']), (ROOT/'configs/magnetic_fit_pollen_nonslip.json').resolve())
        self.assertGreater(config['noise_ut'], 0)
        config = self.run_main('--ideal')
        self.assertEqual((config['noise_ut'], config['filter_tau_s'], config['delay_samples']), (0., 0., 0))

    def test_network_model_replaces_default_fit(self):
        config = self.run_main('--reskin-model', 'outputs/reskin_forward_mapped')
        self.assertNotIn('magnetic_fit', config)
        self.assertEqual(Path(config['reskin_model']), Path('outputs/reskin_forward_mapped').resolve())
        with patch('sys.stderr'), self.assertRaises(SystemExit):
            self.run_main('--reskin-model', 'outputs/reskin_forward_mapped',
                          '--magnetic-fit', 'configs/magnetic_fit_pollen_nonslip.json')


if __name__ == '__main__':
    unittest.main()
