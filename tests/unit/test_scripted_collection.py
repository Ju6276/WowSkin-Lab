"""The recording CLI must preserve the selected physical clip model."""
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

from wowskin_lab.data.auto_collect import collect, main


class ScriptedCollectionTests(unittest.TestCase):
    def test_cube_records_each_episode_reset_seed(self):
        recorded, reset = [], []

        class Environment:
            def __init__(self, cfg, **kwargs):
                self.cfg, self.is_optical = cfg, True
                self.data = SimpleNamespace(xpos=np.array([[.4, 0., .12]]), time=0., warning=SimpleNamespace(number=np.zeros(1)))
                self.probe_body, self.target_xyz, self.success = 0, np.zeros(3), True
                self.pick_place = SimpleNamespace(config={'target_center_xy_m': [.4, 0.]})
            def reset(self, seed):
                reset.append(seed)
            def step(self, action):
                self.data.time += .01
                return [dict(time_s=self.data.time, lift_m=0., grasped=False)]

        class Recorder:
            def __init__(self, env, directory, **kwargs):
                self.env, self.directory = env, directory
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def start(self):
                recorded.append(self.env.cfg['seed'])
            def step(self, row):
                pass
            def finish(self, status):
                return self.directory/f'capture_{len(recorded):06d}'

        with TemporaryDirectory() as directory, \
                patch('wowskin_lab.data.auto_collect.PandaPickCubeTactileEnv', Environment), \
                patch('wowskin_lab.data.auto_collect.LearningSimRecorder', Recorder), \
                patch('wowskin_lab.data.auto_collect.require_training_tactile'), patch('builtins.print'):
            report = collect(Path(directory)/'data', episodes=3, seed=11,
                             sensor_backend='tacto', record_format='dataset')
            self.assertEqual(reset, [11, 12, 13])
            self.assertEqual(recorded, reset)
            self.assertEqual([e['seed'] for e in report['episodes']], reset)
            self.assertTrue((Path(directory)/'data/controller_source.py').is_file())

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
