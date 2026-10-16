"""Backend options cannot silently alter CPU runs or bypass CLI dispatch."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from wowskin_lab.core.stepping import configuration, prepare
from wowskin_lab.data.auto_collect import main


class PhysicsBackendChecks(unittest.TestCase):
    def test_cpu_default_does_not_load_gpu_package(self):
        import subprocess
        import sys
        subprocess.run([sys.executable, '-c',
            'import sys; from types import SimpleNamespace; '
            'from wowskin_lab.core.stepping import prepare; '
            'assert prepare(SimpleNamespace(cfg={})) is None; '
            'assert "warp" not in sys.modules; assert "mujoco_warp" not in sys.modules'], check=True)

    def test_gpu_options_are_explicit_and_validated(self):
        for option in ({'gpu_disable_multiccd': True}, {'gpu_disable_noslip': True},
                       {'gpu_solver_iterations': 100}, {'gpu_disable_nativeccd': True},
                       {'gpu_solver_tolerance': 1e-6}, {'camera_backend': 'warp'}):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, 'require'):
                configuration(SimpleNamespace(**option))
        for invalid in (0., -1., float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'finite and positive'):
                configuration(SimpleNamespace(physics_backend='warp', gpu_solver_tolerance=invalid))
        with self.assertRaisesRegex(ValueError, 'positive'):
            configuration(SimpleNamespace(physics_backend='warp', gpu_solver_iterations=0))

    def test_gpu_cli_reaches_cube_and_cable_configuration(self):
        for task, module in (('pick_cube', 'wowskin_lab.data.auto_collect.PandaPickCubeTactileEnv'),
                             ('cable', 'wowskin_lab.envs.cable.PandaCableEnv')):
            with self.subTest(task=task), TemporaryDirectory() as tmp, \
                    patch(module, side_effect=RuntimeError('scene boundary')) as scene:
                args = ['--output', str(Path(tmp)/'data'), '--task', task,
                        '--sensor-backend', 'tacto', '--physics-backend', 'warp',
                        '--camera-backend', 'warp',
                        '--gpu-disable-multiccd', '--gpu-disable-noslip', '--gpu-solver-iterations', '100']
                if task == 'cable':
                    args += ['--cable-layout', 'single_clip']
                with self.assertRaisesRegex(RuntimeError, 'scene boundary'):
                    main(args)
                cfg = scene.call_args.args[0]
                self.assertEqual(cfg['physics_backend'], 'warp')
                self.assertEqual(cfg['camera_backend'], 'warp')
                self.assertTrue(cfg['gpu_disable_multiccd'])
                self.assertTrue(cfg['gpu_disable_noslip'])
                self.assertEqual(cfg['gpu_solver_iterations'], 100)
