import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from wowskin_lab.__main__ import main


class TactileDefaultsTests(unittest.TestCase):
    def test_old_full_config_cannot_be_reinterpreted_as_reduced(self):
        from wowskin_lab.core.geometry import standard_xml
        from wowskin_lab.core.physics import configuration
        with self.assertRaisesRegex(ValueError, 'Only trilinear'):
            standard_xml(dict(configuration(), skin_dof='full'))

    def command(self, args):
        captured = []
        module = SimpleNamespace(main=lambda: captured.extend(sys.argv[1:]))
        with patch('wowskin_lab.__main__.importlib.import_module', return_value=module):
            main(args)
        return captured

    def test_all_scenes_use_fitted_dipole(self):
        for task in ('pick_cube', 'cable', 'feeding', 'peg_insert'):
            args = self.command(['teleop', '--task', task])
            self.assertIn('--magnetic-fit', args)
            self.assertNotIn('--reskin-extrapolate', args)
            self.assertNotIn('--reskin-model', args)
            self.assertTrue(args[args.index('--magnetic-fit')+1].endswith('magnetic_fit_pollen_nonslip.json'))
            self.assertTrue(args[args.index('--record-dir')+1].endswith('data/episodes'))
            self.assertNotIn('--skin-dof', args)
            self.assertEqual(args[args.index('--record-format')+1], 'lerobot')

    def test_explicit_fitted_dipole_is_preserved(self):
        args = self.command(['teleop', '--magnetic-fit', 'configs/magnetic_fit_pollen_nonslip.json'])
        self.assertNotIn('--reskin-model', args)

    def test_each_optical_selection_reaches_all_tasks_without_magnetic_options(self):
        for sensor, optics, backend in (('digit','tacto','tacto'), ('digit','fots','digit_fots'), ('gelsight',None,'gelsight')):
            for task in ('pick_cube','cable','feeding','peg_insert','chip_tray'):
                selection = ['--sensor', sensor]+(['--optics',optics] if optics else [])
                args = self.command(['teleop','--task',task,*selection])
                self.assertEqual(args[args.index('--sensor-backend')+1],backend)
                self.assertEqual(args[args.index('--task')+1],task)
                self.assertNotIn('--skin-dof',args)
                self.assertNotIn('--magnetic-fit',args)
                self.assertEqual(args[args.index('--record-format')+1],'lerobot')
                self.assertTrue(args[args.index('--record-dir')+1].endswith('data/episodes_'+backend))

    def test_explicit_network_does_not_get_gain_profile(self):
        args = self.command(['teleop', '--reskin-model=outputs/reskin_forward_mapped'])
        self.assertNotIn('--magnetic-fit', args)

    def test_task_integration_and_explicit_step_override(self):
        from wowskin_lab.envs.pick_cube import response_configuration
        for task, step in (('pick_cube', .0005), ('cable', .0005), ('feeding', .0005), ('peg_insert', .0005)):
            cfg = response_configuration(SimpleNamespace(task=task))
            self.assertAlmostEqual(cfg['timestep_s'], step)
            self.assertEqual(cfg['skin_dof'], 'trilinear')
            self.assertTrue(cfg['disable_workspace_limits'])
            self.assertEqual(cfg['teleop_frame'], 'base')
            self.assertEqual(cfg['teleop_base_lateral_sign'], 1.)
        with self.assertRaisesRegex(ValueError, 'Only trilinear'):
            response_configuration(SimpleNamespace(task='pick_cube', skin_dof='full'))
        cfg = response_configuration(SimpleNamespace(task='pick_cube', physics_step_us=20))
        self.assertAlmostEqual(cfg['timestep_s'], .00002)
        self.assertEqual(cfg['constraint_jacobian'], 'sparse')
        cfg = response_configuration(SimpleNamespace(task='pick_cube', workspace_limits=True))
        self.assertFalse(cfg['disable_workspace_limits'])

    def test_shared_dataset_records_model_choice_in_episode_metadata(self):
        args = self.command(['teleop'])
        self.assertTrue(args[args.index('--record-dir')+1].endswith('data/episodes'))
