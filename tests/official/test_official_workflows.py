"""Check provenance and the actual composed upstream configs; no substitute training."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hydra import compose, initialize_config_dir

from wowskin_lab.learning.official import prepare, verify_workspace


class OfficialWorkflowChecks(unittest.TestCase):
    def test_preserve_all_three_upstream_training_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for route in ('visk', 'ftf-human', 'ftf-robot'):
                with self.subTest(route=route):
                    directory = root/route
                    prepare(route, directory, root/'raw_data', 'pick_bread')
                    manifest = json.loads((directory/'manifest.json').read_text())
                    train = next(c for c in manifest['commands'] if c['stage'] == 'train')
                    cfg_dir = directory/'source'/Path(manifest['training_config']).parent
                    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
                        cfg = compose(config_name='config', overrides=train['argv'][2:])
                    if route == 'visk':
                        self.assertEqual(cfg.batch_size, 256)
                        self.assertEqual(cfg.num_queries, 10)
                        self.assertEqual(cfg.dataloader.bc_dataset.subsample, 5)
                        self.assertEqual(list(cfg.suite.aux_keys), ['sensor0'])
                        self.assertEqual(manifest['visk_profile'], 'paper-single')
                        self.assertEqual(list(cfg.suite.pixel_keys), ['pixels1', 'pixels2', 'pixels52'])
                        self.assertEqual(manifest['camera_ids'], [1, 2, 52])
                        self.assertTrue((directory/'THREE_VIEW_CAMERAS.patch').is_file())
                        self.assertTrue((directory/'PAPER_SINGLE_SENSOR.patch').is_file())
                        self.assertEqual(cfg.suite.num_train_steps, 10000)
                    else:
                        self.assertEqual(cfg.batch_size, 64)
                        self.assertEqual(cfg.num_queries, 20)
                        self.assertEqual(cfg.suite.history_len, 10)
                        self.assertEqual(cfg.suite.num_train_steps, 100100)
                        self.assertEqual(cfg.suite.task.task_name, '0309_pick_bread')
                        self.assertEqual(cfg.suite.num_object_points, 3)
                        self.assertEqual(cfg.suite.subsample, 3 if route.endswith('human') else 15)
                        if route.endswith('robot'):
                            self.assertTrue(cfg.suite.binarize_gripper_data)
                        stages = {c['stage'] for c in manifest['commands']}
                        self.assertIn('track', stages)
                        self.assertEqual('retarget' in stages, route.endswith('human'))
                    self.assertTrue(verify_workspace(directory)['source_matches_prepared_manifest'])
                    # Detect any unrecorded algorithm edit; path setup must not
                    # make policy changes pass as untouched official source.
                    file = directory/'source'/('agent/bc.py' if route == 'visk' else 'point_policy/agent/p3po.py')
                    file.write_text(file.read_text()+'\n# accidental algorithm edit\n')
                    self.assertFalse(verify_workspace(directory)['source_matches_prepared_manifest'])

    def test_published_default_remains_available_as_a_separate_provenance_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/'published'
            prepare('visk', directory, Path(tmp)/'data', 'pick_cube', visk_profile='published-default')
            with initialize_config_dir(config_dir=str(directory/'source/cfgs'), version_base=None):
                cfg = compose(config_name='config')
            self.assertEqual(list(cfg.suite.aux_keys), ['sensor0', 'sensor1'])
            self.assertFalse((directory/'PAPER_SINGLE_SENSOR.patch').exists())

    def test_old_train_command_cannot_silently_run_simplified_model(self):
        result = subprocess.run([sys.executable, '-m', 'wowskin_lab', 'train-ftf', 'old.pkl'],
                                text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('simplified pipeline has been removed', result.stderr)

    def test_training_rejects_wrong_task_and_unrecorded_source_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace, output = root/'workspace', root/'training'
            prepare('visk', workspace, root/'data', 'feeding')
            command = [sys.executable, 'scripts/train_visk_mac.py', '--workspace', str(workspace),
                       '--output', str(output), '--device', 'cpu']
            result = subprocess.run(command+['--task', 'pick_cube'], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Task differs from the workspace', result.stderr)
            self.assertFalse(output.exists())
            # The inferred feeding task should reach the missing-dataset check,
            # rather than defaulting to cube or starting an optimization run.
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertIn('Run official process-visk-sim', result.stderr)
            source = workspace/'source/agent/bc.py'
            source.write_text(source.read_text()+'\n# unexpected algorithm edit\n')
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('unrecorded source edits', result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
