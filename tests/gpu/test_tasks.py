"""Short moving task trajectories compared with native CPU physics.

Both sides explicitly use the documented GPU-compatible collision options;
this isolates execution differences, not equivalence to the source defaults.
Full scripted completion is assessed separately by record-auto.
"""
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
from wowskin_lab.core.stepping import prepare
from wowskin_lab.envs.pick_cube import response_configuration, PandaPickCubeTactileEnv
from wowskin_lab.envs.cable import PandaCableEnv
from wowskin_lab.envs.feeding import PandaFeedingEnv
from wowskin_lab.envs.peg_insert import PandaPegInsertEnv
from wowskin_lab.envs.chip_tray import PandaChipEnv


def setUpModule():
    try:
        import warp as wp
        import mujoco_warp
    except ImportError:
        raise unittest.SkipTest('Optional NVIDIA dependencies: requirements/gpu.txt')
    if not wp.is_cuda_available():
        raise unittest.SkipTest('NVIDIA CUDA device required')


class GpuTaskChecks(unittest.TestCase):
    def compare(self, cls, task, sensor, gripper='panda', layout=None):
        args = dict(task=task, ideal=True, gripper=gripper, cable_layout=layout,
                    gpu_disable_multiccd=True, gpu_disable_nativeccd=True,
                    gpu_disable_noslip=True, physics_backend='warp')
        if sensor != 'anyskin':
            args['sensor_backend'] = sensor
        gpu = cls(response_configuration(SimpleNamespace(**args)), pollen=False)
        cpu = cls(dict(gpu.cfg, physics_backend='cpu'), pollen=False)
        if task == 'chip_tray':
            with self.assertRaisesRegex(ValueError, 'dynamic fragment collision masks'):
                prepare(gpu)
            return
        backend = prepare(gpu)
        # Matched contact algorithms, same source timestep/material/controller.
        cpu.model.opt.disableflags = gpu.model.opt.disableflags
        cpu.model.opt.noslip_iterations = gpu.model.opt.noslip_iterations
        cpu.model.opt.jacobian = gpu.model.opt.jacobian
        mujoco.mj_copyData(cpu.data, cpu.model, gpu.data)
        mujoco.mj_forward(cpu.model, cpu.data)
        for _ in range(10):
            action = gpu.hold_action()
            action[0] = .00002
            gpu.step(action)
            cpu.step(action)
        print(f'GPU parity {task}/{layout}/{sensor}/{gripper}: '
              f'qpos_max={np.max(abs(gpu.data.qpos-cpu.data.qpos)):.6g}, '
              f'tcp_m={np.linalg.norm(gpu.data.site_xpos[gpu.tcp_site]-cpu.data.site_xpos[cpu.tcp_site]):.6g}, '
              f'object_m={np.linalg.norm(gpu.data.xpos[gpu.probe_body]-cpu.data.xpos[cpu.probe_body]):.6g}, '
              f'pad_N_gpu={gpu.pad_forces()}, pad_N_cpu={cpu.pad_forces()}', flush=True)
        np.testing.assert_allclose(gpu.data.qpos, cpu.data.qpos, rtol=0, atol=2e-3)
        np.testing.assert_allclose(gpu.data.site_xpos[gpu.tcp_site], cpu.data.site_xpos[cpu.tcp_site], rtol=0, atol=2e-4)
        self.assertAlmostEqual(cpu.data.time, gpu.data.time, places=9)
        self.assertEqual(backend.last_diagnostics['overflow_flags'], 0)
        self.assertTrue(np.isfinite(gpu.pad_forces()).all())
        # Tactile collection also requires comparable forces, not just a
        # nearby robot pose. A force mismatch remains a visible test failure.
        np.testing.assert_allclose(gpu.pad_forces(), cpu.pad_forces(), rtol=.1, atol=.05)
        if task == 'feeding':
            self.assertTrue(np.isfinite(gpu.head_interval_peak_n))

    def test_all_panda_tasks_and_sensors(self):
        for sensor in ('anyskin', 'tacto', 'digit_fots', 'gelsight'):
            for task, cls, layout in (
                ('pick_cube', PandaPickCubeTactileEnv, None),
                ('cable', PandaCableEnv, 'single_clip'),
                ('cable', PandaCableEnv, 'routing'),
                ('feeding', PandaFeedingEnv, None),
                ('peg_insert', PandaPegInsertEnv, None),
                ('chip_tray', PandaChipEnv, None)):
                with self.subTest(task=task, sensor=sensor, layout=layout):
                    self.compare(cls, task, sensor, layout=layout)

    def test_robotiq_original_pads_and_cable(self):
        for task, cls, layout in (
            ('pick_cube', PandaPickCubeTactileEnv, None),
            ('cable', PandaCableEnv, 'single_clip'),
            ('cable', PandaCableEnv, 'routing'),
            ('feeding', PandaFeedingEnv, None),
            ('peg_insert', PandaPegInsertEnv, None)):
            with self.subTest(task=task, layout=layout):
                self.compare(cls, task, 'anyskin', gripper='robotiq_2f85', layout=layout)
