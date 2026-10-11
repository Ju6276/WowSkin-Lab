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

    def test_all_three_scenes_use_fitted_dipole(self):
        for task in ('pick_cube', 'cable', 'feeding'):
            args = self.command(['teleop', '--task', task])
            self.assertIn('--magnetic-fit', args)
            self.assertNotIn('--reskin-extrapolate', args)
            self.assertNotIn('--reskin-model', args)
            self.assertTrue(args[args.index('--magnetic-fit')+1].endswith('magnetic_fit_pollen_nonslip.json'))
            self.assertTrue(args[args.index('--record-dir')+1].endswith('data/episodes'))
            self.assertEqual(args[args.index('--skin-dof')+1], 'trilinear')
            self.assertEqual(args[args.index('--record-format')+1], 'lerobot')

    def test_explicit_fitted_dipole_is_preserved(self):
        args = self.command(['teleop', '--magnetic-fit', 'configs/magnetic_fit_pollen_nonslip.json'])
        self.assertNotIn('--reskin-model', args)

    def test_explicit_network_does_not_get_gain_profile(self):
        args = self.command(['teleop', '--reskin-model=outputs/reskin_forward_mapped'])
        self.assertNotIn('--magnetic-fit', args)

    def test_task_integration_and_explicit_step_override(self):
        from wowskin_lab.envs.pick_cube import response_configuration
        for task, step in (('pick_cube', .0001), ('cable', .0001), ('feeding', .00005)):
            cfg = response_configuration(SimpleNamespace(task=task))
            self.assertAlmostEqual(cfg['timestep_s'], step)
        cfg = response_configuration(SimpleNamespace(task='pick_cube', physics_step_us=20))
        self.assertAlmostEqual(cfg['timestep_s'], .00002)
        self.assertEqual(cfg['constraint_jacobian'], 'sparse')

    def test_shared_dataset_records_model_choice_in_episode_metadata(self):
        args = self.command(['teleop', '--skin-dof', 'trilinear'])
        self.assertTrue(args[args.index('--record-dir')+1].endswith('data/episodes'))
