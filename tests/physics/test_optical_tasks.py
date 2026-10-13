"""Shared tasks use selectable native DIGIT sensors, never magnetic/FEM stand-ins."""
import json
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import h5py
import numpy as np

from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, response_configuration
from wowskin_lab.envs.cable import PandaCableEnv
from wowskin_lab.envs.feeding import PandaFeedingEnv
from wowskin_lab.envs.peg_insert import PandaPegInsertEnv
from wowskin_lab.envs.chip_tray import PandaChipEnv
from wowskin_lab.envs.digit_sources import source, CONTACT_FIELDS
from wowskin_lab.data.raw_recording import RawRobotRecorder
from wowskin_lab.data.optical_recording import optical_path, validate_optical
from wowskin_lab.ui.teleop_hud import operator_overlay


class OpticalTaskChecks(unittest.TestCase):
    def test_all_tasks_keep_source_sensor_and_record_real_optical_frames(self):
        for backend in ('tacto', 'digit_fots', 'gelsight'):
            if backend != 'gelsight':
                _, original, _ = source(backend)
            for cls in (PandaPickCubeTactileEnv, PandaCableEnv, PandaFeedingEnv, PandaPegInsertEnv, PandaChipEnv):
                with self.subTest(backend=backend, task=cls.task_name):
                    cfg = response_configuration(SimpleNamespace(task=cls.task_name, sensor_backend=backend,
                        cable_layout='single_clip', clip_model='elastic_beam', ideal=True))
                    env = cls(cfg, pollen=False)
                    if backend != 'gelsight':
                        for side in ('left','right'):
                            for g in range(original.ngeom):
                                target = env.model.geom(f'digit_{side}_g{g}').id
                                for attr in (*CONTACT_FIELDS, 'type', 'size'):
                                    np.testing.assert_allclose(getattr(env.model, 'geom_'+attr)[target], getattr(original,'geom_'+attr)[g], atol=1e-8)
                    self.assertFalse(any(k.startswith(('fem_', 'magnetization', 'noise_ut')) for k in env.cfg))
                    self.assertNotIn('magnetic_ut', env.observation())
                    solver = env.model.opt.solver
                    for _ in range(20):
                        row = env.step(env.hold_action())[-1]
                    self.assertEqual(env.model.opt.solver, solver)
                    self.assertTrue(np.isfinite(env.data.qpos).all())
                    self.assertFalse(np.any(env.data.warning.number))
                    labels, values = operator_overlay(env,row,title=cls.task_name,input_kind='Keyboard',input_status='Ready',
                        recorder=None,recording_notice='',diagnostics=True,realtime_factor=1,loop_hz=100)
                    self.assertEqual(len(labels),len(values))
                    with TemporaryDirectory(prefix='wowskin_optical_task_') as directory:
                        with RawRobotRecorder(env,directory,source='contract_test_not_evaluation') as recorder:
                            recorder.start()
                            for _ in range(4): recorder.step(env.step(env.hold_action())[-1])
                            path = recorder.finish('interrupted')
                        meta = json.loads((path/'metadata.json').read_text())
                        self.assertEqual(meta['tactile_sensors'],2)
                        self.assertNotIn('magnetic_baseline_ut',meta)
                        with h5py.File(path/'measurements.h5') as f:
                            self.assertNotIn('sensor_states',f)
                            self.assertEqual(f['optical_contact_force_n'].shape, (5,2))
                            times=f['camera_timestamp'][:]
                        validate_optical(optical_path(path,meta),times)
                        with h5py.File(optical_path(path,meta)) as f:
                            for side in ('left','right'):
                                g=f['frames/000000/'+side]
                                self.assertEqual(g['rgb'].shape,(120,160,3) if backend=='gelsight' else (320,240,3))
                                self.assertEqual('marker_displacement_px' in g,backend=='digit_fots' and g.attrs['marker_status']!='multiple_bodies_unsupported')
                    env.reset()
                    self.assertEqual(env.model.opt.solver,solver)
