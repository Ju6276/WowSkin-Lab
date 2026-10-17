"""Collection stays on CPU MuJoCo and does not load a GPU package."""
import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from wowskin_lab.core.stepping import configuration
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

    def test_configuration_is_cpu_and_rejects_gpu_options(self):
        cfg = configuration(SimpleNamespace())
        self.assertEqual(cfg, dict(physics_backend='cpu', camera_backend='opengl'))
        for option in ({'physics_backend': 'warp'}, {'camera_backend': 'warp'},
                       {'gpu_disable_multiccd': True}, {'gpu_disable_noslip': True},
                       {'gpu_disable_nativeccd': True}, {'gpu_solver_iterations': 100},
                       {'gpu_solver_tolerance': 1e-6}):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, 'removed'):
                configuration(SimpleNamespace(**option))

    def test_cli_rejects_gpu_backend(self):
        with TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['--output', str(Path(tmp)/'data'), '--task', 'pick_cube',
                  '--sensor-backend', 'tacto', '--physics-backend', 'warp'])
