"""The standalone cable entry point must construct a real sensor config."""
import unittest
from unittest.mock import patch

from wowskin_lab.envs import cable


class CableCLITests(unittest.TestCase):
    def test_viewer_entry_point_configures_selected_layout(self):
        with patch('sys.argv', ['cable', 'viewer', '--cable-layout', 'single_clip']), \
                patch.object(cable, 'PandaCableEnv') as environment, \
                patch.object(cable, 'viewer') as viewer:
            cable.main()
        config = environment.call_args.args[0]
        self.assertEqual(config['cable_layout'], 'single_clip')
        self.assertEqual(config['sample_hz'], 100)
        self.assertEqual(config['skin_dof'], 'trilinear')
        self.assertEqual(config['integrator'], 'discrete')
        self.assertGreater(config['young_pa'], 0)
        viewer.assert_called_once()
        self.assertIs(viewer.call_args.args[0], environment.return_value)


if __name__ == '__main__':
    unittest.main()
