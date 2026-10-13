"""The recording CLI must preserve the selected physical clip model."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from wowskin_lab.data.auto_collect import main


class ScriptedCollectionTests(unittest.TestCase):
    def test_clip_selection_reaches_scene_configuration(self):
        for model in ('rigid', 'elastic_beam'):
            with self.subTest(model=model), TemporaryDirectory() as directory:
                output = Path(directory)/'recording'
                # Stop at the expensive scene boundary, after real argument
                # parsing, dispatch and response-configuration construction.
                with patch('wowskin_lab.envs.cable.PandaCableEnv',
                           side_effect=RuntimeError('scene boundary')) as scene:
                    with self.assertRaisesRegex(RuntimeError, 'scene boundary'):
                        main(['--output', str(output), '--task', 'cable',
                              '--cable-layout', 'single_clip', '--sensor-backend', 'tacto',
                              '--clip-model', model, '--episodes', '1'])
                config = scene.call_args.args[0]
                self.assertEqual(config['clip_model'], model)
                self.assertEqual(config['cable_layout'], 'single_clip')
                self.assertEqual(config['sensor'], 'tacto')
                self.assertFalse(output.exists())

    def test_elastic_clip_is_not_silently_ignored_for_other_tasks(self):
        for task in ('feeding', 'pick_cube'):
            with self.subTest(task=task), TemporaryDirectory() as directory:
                output = Path(directory)/'recording'
                with self.assertRaisesRegex(ValueError, '--clip-model requires --task cable'):
                    main(['--output', str(output), '--task', task, '--sensor-backend', 'tacto',
                          '--clip-model', 'elastic_beam'])
                self.assertFalse(output.exists())
