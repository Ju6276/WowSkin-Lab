"""Runtime GPU observations and controls against independent native readers."""
from types import SimpleNamespace
import copy
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from tests.gpu.test_environment import setUpModule
from wowskin_lab.core.stepping import prepare
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, response_configuration


def environment(task='pick_cube', sensor='tacto'):
    from wowskin_lab.envs.feeding import PandaFeedingEnv
    cls = PandaFeedingEnv if task == 'feeding' else PandaPickCubeTactileEnv
    return cls(response_configuration(SimpleNamespace(task=task, sensor_backend=sensor,
        ideal=True, physics_backend='warp', gpu_disable_multiccd=True,
        gpu_disable_nativeccd=True, gpu_disable_noslip=True)), pollen=False)


class PipelineChecks(unittest.TestCase):
    def test_backend_does_not_keep_discarded_environment_alive(self):
        import weakref
        env = environment()
        backend = prepare(env)
        reference = weakref.ref(env)
        del env
        self.assertIsNone(reference())
        # Keep this reference alive to exercise the backend's ownership rule.
        self.assertIsNotNone(backend.sim)

    def test_cuda_fots_keeps_source_image_values(self):
        from wowskin_lab.core.fots_mujoco import NativeFOTSRenderer
        cpu, gpu = NativeFOTSRenderer(), NativeFOTSRenderer('cuda')
        depth = cpu.baseline.copy()
        depth[130:190, 90:150] -= .0001
        for shadow in (False, True):
            expected = cpu.render(depth, shadow=shadow).astype(float)
            actual = gpu.render(depth, shadow=shadow).astype(float)
            # Same author algorithm/weights, different floating point device.
            self.assertLessEqual(float(np.abs(actual-expected).max()), 1.)
            # Source bg_render cancels near-integer pixels; sub-ULP model
            # differences can move those pixels across uint8 truncation.
            # Compare unquantized source MLP output as well as the 1-code RGB bound.
        import torch
        source = torch.linspace(-1., 1., 320*240*5).reshape(-1, 5)
        with torch.inference_mode():
            expected = cpu.model(source).numpy()
            actual = gpu.model(source.cuda()).cpu().numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)

    def test_gpu_scores_match_native_on_same_moving_state(self):
        from wowskin_lab.envs.cable import PandaCableEnv
        from wowskin_lab.envs.peg_insert import PandaPegInsertEnv
        from wowskin_lab.envs.feeding import PandaFeedingEnv
        for task, cls, layout in (('pick_cube', PandaPickCubeTactileEnv, None),
                ('feeding', PandaFeedingEnv, None), ('cable', PandaCableEnv, 'single_clip'),
                ('cable', PandaCableEnv, 'routing'), ('peg_insert', PandaPegInsertEnv, None)):
            with self.subTest(task=task, layout=layout):
                # Source optical geometry; the test compares readers on ONE
                # solved state, not separate contact solvers or task success.
                env = cls(response_configuration(SimpleNamespace(task=task, cable_layout=layout,
                    sensor_backend='gelsight', ideal=True, physics_backend='warp', gpu_disable_multiccd=True,
                    gpu_disable_nativeccd=True, gpu_disable_noslip=True)), pollen=False)
                backend = prepare(env)
                attr, method = {'pick_cube': ('pick_place', 'evaluate_pick_place'),
                    'feeding': ('delivery', 'evaluate_delivery'), 'cable': ('clip_score', 'evaluate_clip'),
                    'peg_insert': ('peg_score', 'evaluate_peg')}[task]
                reference = copy.deepcopy(getattr(env, attr))
                route = copy.deepcopy(getattr(env, 'routing_score', None))
                for _ in range(3):
                    action = env.hold_action()
                    action[0] = .00003
                    row = env.step(action)[-1]
                    actual = backend.scoring.update()
                    with patch.object(env, '_gpu_physics', None), patch.object(env, attr, reference):
                        expected = getattr(env, method)(row) if task in ('pick_cube', 'feeding') else getattr(env, method)()
                        if route is not None:
                            route.update(env.data.time, env.cable_segments())
                    for key in actual:
                        np.testing.assert_allclose(actual[key], expected[key], atol=3e-6, rtol=1e-5, err_msg=key)
                    if route is not None:
                        for key in route.metrics:
                            np.testing.assert_allclose(env.routing_score.metrics[key], route.metrics[key], atol=1e-9)

    def test_warp_scene_cameras_read_gpu_without_native_scene_update(self):
        from wowskin_lab.core.gpu.cameras import WarpCameras
        env = environment()
        prepare(env)
        cameras = WarpCameras(env, width=160, height=120)
        with patch.object(mujoco, 'mjv_updateScene', side_effect=AssertionError('CPU scene update')):
            for name in ('lab_front', 'lab_side', 'lab_wrist'):
                cameras.update_scene(env.data, camera=name)
                rgb = cameras.render()
                self.assertEqual(rgb.shape, (120, 160, 3))
                self.assertEqual(rgb.dtype, np.uint8)
                self.assertGreater(float(rgb.std()), 1.)
                cameras.enable_depth_rendering()
                depth = cameras.render()
                self.assertTrue(np.isfinite(depth).all())
                self.assertGreater(float(depth.max()), .01)
                cameras.disable_depth_rendering()
        cameras.close()

    def test_compact_snapshot_matches_upstream_export_on_loaded_contacts(self):
        import mujoco_warp as mjw
        from wowskin_lab.core.gpu.snapshot import FIELDS, CONTACT_FIELDS
        for sensor in (None, 'tacto'):
            with self.subTest(sensor=sensor):
                env = environment('feeding', sensor)
                backend = prepare(env)
                env.step(env.hold_action())
                reference = mujoco.MjData(env.model)
                mjw.get_data_into(reference, env.model, backend.sim.data)
                for name in FIELDS:
                    np.testing.assert_array_equal(getattr(env.data, name), getattr(reference, name), err_msg=name)
                for name, _ in CONTACT_FIELDS:
                    expected = getattr(reference.contact, name).copy()
                    if name in ('flex', 'elem', 'vert') and not env.model.nflex:
                        expected[:] = -1
                    elif name in ('elem', 'vert'):
                        valid = (reference.contact.flex >= 0) & (reference.contact.geom < 0) & (expected >= 0)
                        offsets = env.model.flex_elemadr if name == 'elem' else env.model.flex_vertadr
                        expected[valid] -= offsets[reference.contact.flex[valid]]
                    if name in ('flex', 'elem', 'vert'):
                        expected[reference.contact.geom >= 0] = -1
                    np.testing.assert_array_equal(getattr(env.data.contact, name), expected, err_msg=name)
                self.assertGreater(env.data.ncon, 0)
                for name in ('flex', 'elem', 'vert'):
                    np.testing.assert_array_equal(getattr(env.data.contact, name)[env.data.contact.geom >= 0], -1)
                for i in range(env.data.ncon):
                    actual, expected = np.zeros(6), np.zeros(6)
                    mujoco.mj_contactForce(env.model, env.data, i, actual)
                    mujoco.mj_contactForce(env.model, reference, i, expected)
                    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)

    def test_gpu_ik_tracks_native_controller_with_translation_and_rotation(self):
        from wowskin_lab.core.gpu.ik import GpuIK
        from wowskin_lab.envs.pick_cube import rotation_matrix
        import warp as wp
        wp.init()
        wp.set_device('cuda')
        for task in ('pick_cube', 'feeding'):
            env = environment(task)
            target = wp.array(env.joint_target, dtype=float)
            ik = GpuIK(env, target)
            for i in range(60):
                env.target_xyz += [.0001, .00005, -.00003]
                env.target_rotation = env.target_rotation @ rotation_matrix([0., .0001, -.0001])
                ik.solve()
                env.solve_ik()
                np.testing.assert_allclose(target.numpy(), env.joint_target, atol=3e-5, rtol=0)
                np.testing.assert_allclose(ik.servo.numpy(), np.r_[env.servo_xyz, env.servo_rotation.ravel()], atol=1e-10)

    def test_runtime_does_not_use_native_ik_or_full_export(self):
        import mujoco_warp as mjw
        env = environment()
        prepare(env)
        with patch.object(mujoco, 'mj_kinematics', side_effect=AssertionError('CPU IK')), \
             patch.object(mjw, 'get_data_into', side_effect=AssertionError('Full state export')):
            for _ in range(10):
                action = env.hold_action()
                action[0] = .0001
                env.step(action)

    def test_optical_force_and_slip_match_native_jacobians(self):
        from wowskin_lab.core.optical_physics import contact_state
        from wowskin_lab.data.robot_state import robot_sample
        env = environment('feeding')
        backend = prepare(env)
        for i in range(5):
            action = env.hold_action()
            action[:3] = [.0001, 0, .0001]
            env.step(action)
            actual = robot_sample(env)
            with patch.object(env, '_gpu_physics', None):
                expected = robot_sample(env)
            for key in ('tcp_velocity', 'tcp_velocity_world'):
                np.testing.assert_allclose(actual[key], expected[key], atol=2e-7, rtol=1e-5)
            for side in ('left', 'right'):
                for objects in (None, env.contact_geoms):
                    actual = contact_state(env, side, objects)
                    with patch.object(env, '_gpu_physics', None):
                        expected = contact_state(env, side, objects)
                    for key in actual:
                        np.testing.assert_allclose(actual[key], expected[key], atol=2e-7, rtol=1e-5, err_msg=key)
        self.assertGreater(backend.observations.values[:, 0].sum(), .01)

    def test_magnetic_polar_rotation_and_quality_match_numpy(self):
        env = environment('feeding', None)
        prepare(env)
        for _ in range(3):
            env.step(env.hold_action())
            actual, quality = env.magnetic_field(), env.tactile_quality()
            with patch.object(env, '_gpu_physics', None):
                expected, reference = env.magnetic_field(), env.tactile_quality()
            np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
            for key in quality:
                self.assertAlmostEqual(quality[key], reference[key], places=10)
            for flex in (env.skin_flex, env.passive_flex):
                actual_contact = env.contact_state(flex)
                with patch.object(env, '_gpu_physics', None):
                    reference_contact = env.contact_state(flex)
                for key in actual_contact:
                    np.testing.assert_allclose(actual_contact[key], reference_contact[key],
                        atol=2e-7, rtol=1e-5, err_msg=key)

    def test_display_uses_detached_gpu_kinematics_without_cpu_forward(self):
        from wowskin_lab.ui.physics_worker import PhysicsWorker
        env = environment('feeding')
        prepare(env)
        env.step(env.hold_action())
        env.cfg['camera_backend'] = 'warp'
        model, data = PhysicsWorker.display(env)
        handle = SimpleNamespace(is_running=lambda: False)
        with PhysicsWorker(env, handle, None, model, data) as worker:
            worker.checkpoint(force=True)
            saved = env.data.geom_xpos.copy()
            frame = worker.camera_frame()
            self.assertEqual(len(frame[1]), 3)
            self.assertEqual(len(frame[2]), 3)
            np.testing.assert_array_equal(frame[0].cam_xpos, env.data.cam_xpos)
            saved_cameras = frame[0].cam_xpos.copy()
            env.data.cam_xpos[:] += 1
            env.data.geom_xpos[:] += 1  # A later live update cannot change the frame.
            with patch.object(mujoco, 'mj_forward', side_effect=AssertionError('CPU display solve')):
                worker.present()
            np.testing.assert_array_equal(data.geom_xpos, saved)
            np.testing.assert_array_equal(frame[0].cam_xpos, saved_cameras)


if __name__ == '__main__':
    unittest.main()
