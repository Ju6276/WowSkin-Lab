"""Check contact instead of a hidden height/empty-gap stop (MuJoCo 3.15)."""
from types import SimpleNamespace
from pathlib import Path
import time
import unittest
import mujoco
import numpy as np
from wowskin_lab.envs.cable import PandaCableEnv
from wowskin_lab.envs.pick_cube import response_configuration, hold_current_pose, limit_vector, rotation_matrix, rotation_vector


@unittest.skipUnless(mujoco.__version__ == '3.15.0', 'Requires reduced environment')
class ContactIntegrityChecks(unittest.TestCase):
    fixture = Path(__file__).resolve().parents[1]/'fixtures/cable_reduced_clip_commands.npz'

    def make_env(self, gripper='panda'):
        cfg = response_configuration(SimpleNamespace(task='cable', skin_dof='trilinear',
            cable_layout='single_clip', ideal=True, gripper=gripper))
        return PandaCableEnv(cfg, pollen=False)

    def test_reduced_skirt_mesh_resolves_recorded_deformation(self):
        from wowskin_lab.core.geometry import cap_mesh
        env = self.make_env()
        self.assertEqual(env.cfg['mesh_outline_samples'], 48)
        ratios = []
        with np.load(self.fixture) as source:
            for samples in (12, 48, 96):
                cfg = dict(env.cfg, mesh_outline_samples=samples)
                _, rest, tets, _, _ = cap_mesh(cfg)
                u = (rest-source['deformation_origin'])/source['deformation_scale']
                a, b, c = u.T
                basis = np.array([np.ones(len(u)), a, b, c, a*b, a*c, b*c, a*b*c]).T
                deformed = np.einsum('ij,jk->ik', basis, source['deformation_coefficients'])
                self.assertTrue(np.isfinite(deformed).all())
                r, d = rest[tets], deformed[tets]
                dm = np.stack([r[:, k]-r[:, 0] for k in (1, 2, 3)], axis=-1)
                ds = np.stack([d[:, k]-d[:, 0] for k in (1, 2, 3)], axis=-1)
                ratios.append(float(np.linalg.det(ds @ np.linalg.inv(dm)).min()))
        self.assertLess(ratios[0], .15)  # Reproduce the coarse-mesh failure.
        self.assertGreater(min(ratios[1:]), .7)
        self.assertLess(abs(ratios[1]-ratios[2]), .01)

    def test_recorded_cable_clip_motion_and_release(self):
        self.replay_clip_motion()

    def test_second_recorded_clip_motion_and_release(self):
        self.replay_clip_motion('second_')

    def test_degraded_cable_skin_keeps_recording_and_recovers_on_release(self):
        # Deliberately replay the old coarse mesh that triggered the user's
        # pause. Production stays at 48; no artificial deformation injection.
        from wowskin_lab.data.recording import require_training_tactile
        cfg = response_configuration(SimpleNamespace(task='cable', skin_dof='trilinear',
            cable_layout='single_clip', ideal=True, mesh_outline_samples=12))
        cfg['mesh_outline_samples'] = 12
        env = PandaCableEnv(cfg, pollen=False)
        with np.load(self.fixture) as source:
            env.data.qpos[env.arm_q] = source['initial_arm_q']
            env.joint_target = source['initial_arm_q'].copy()
            actions = source['actions'].copy()
        mujoco.mj_forward(env.model, env.data)
        hold_current_pose(env)
        for _ in range(100):
            env.step(env.hold_action())
        moving_before = False
        last_valid = None
        for action in np.vstack([actions, np.repeat(actions[-1:], 50, axis=0)]):
            moving = bool(np.any(action[:6]))
            if moving != moving_before:
                hold_current_pose(env)
            if moving:
                actual = env.data.site_xpos[env.tcp_site]
                env.target_xyz = np.clip(env.target_xyz, actual-.006, actual+.006)
                rotation = env.data.site_xmat[env.tcp_site].reshape(3, 3)
                lead = rotation_vector(rotation.T @ env.target_rotation)
                env.target_rotation = rotation @ rotation_matrix(limit_vector(lead, .03))
            moving_before = moving
            row = env.step(action)[-1]
            if row['mesh_quality_warning']:
                break
            last_valid = row
        self.assertTrue(row['mesh_quality_warning'], 'Must reproduce the real positive-volume degradation')
        self.assertTrue(row['tactile_valid'])
        self.assertGreater(row['min_volume_ratio'], 0)
        self.assertTrue(np.isfinite(row['magnetic_ut']).all())
        self.assertFalse(np.array_equal(row['magnetic_ut'], last_valid['magnetic_ut']))
        np.testing.assert_allclose(row['ideal_delta_ut'], (env.magnetic_field()-env.baseline).reshape(15))
        require_training_tactile(env)
        degraded_time = row['time_s']
        hold_current_pose(env)
        env.command_gripper(False)
        recovered = []
        import tempfile
        import h5py
        import json
        from wowskin_lab.data.raw_recording import RawRobotRecorder
        with tempfile.TemporaryDirectory() as tmp:
            with RawRobotRecorder(env, tmp, source='mesh_warning_regression', record_depth=False) as recorder:
                recorder.start()  # Recording is allowed even below the warning threshold.
                saved = recorder.writer.path
                for i in range(200):
                    action = env.hold_action()
                    if i < 100:
                        action[2] = .0005
                    current = env.step(action)[-1]
                    recovered.append(current)
                    recorder.step(current)
                    self.assertTrue(recorder.active)
                recorder.finish('success')
            self.assertEqual(json.loads((saved/'metadata.json').read_text())['status'], 'success')
            with h5py.File(saved/'measurements.h5') as data:
                self.assertEqual(len(data['timestamp']), 201)
                self.assertTrue(data['mesh_quality_warning'][0])
                self.assertFalse(data['mesh_quality_warning'][-1])
                np.testing.assert_allclose(np.diff(data['timestamp']), .01, atol=1e-8)
                self.assertTrue(np.isfinite(data['sensor_states']).all())
        self.assertAlmostEqual(recovered[-1]['time_s']-degraded_time, 2., places=5)
        self.assertTrue(all(r['tactile_valid'] for r in recovered))
        self.assertFalse(any(r['mesh_quality_warning'] for r in recovered[-30:]))
        require_training_tactile(env)
        np.testing.assert_array_equal(env.data.warning.number, 0)
        print(f"Warning-only recording: t={degraded_time:.2f}s, min V/V0={row['min_volume_ratio']:.3f}, "
              f"clear after release={next(r['time_s'] for r in recovered if not r['mesh_quality_warning'])-degraded_time:.2f}s", flush=True)

    def replay_clip_motion(self, prefix=''):
        env = self.make_env()
        env.cfg.update(tcp_speed_m_s=.12, tcp_angular_speed_rad_s=.5)
        with np.load(self.fixture) as source:
            env.data.qpos[env.arm_q] = source[prefix+'initial_arm_q']
            env.joint_target = source[prefix+'initial_arm_q'].copy()
            actions = source[prefix+'actions'].copy()
        mujoco.mj_forward(env.model, env.data)
        hold_current_pose(env)
        for _ in range(100):
            env.step(env.hold_action())
        was_moving, clip_contact, grasped = False, False, False
        rows, costs = [], []
        # Include 0.5 seconds beyond the original pause command.
        for action in np.vstack([actions, np.repeat(actions[-1:], 50, axis=0)]):
            moving = bool(np.any(action[:6]))
            if moving != was_moving:
                hold_current_pose(env)
            if moving:
                actual = env.data.site_xpos[env.tcp_site]
                env.target_xyz = np.clip(env.target_xyz, actual-.006, actual+.006)
                rot = env.data.site_xmat[env.tcp_site].reshape(3, 3)
                lead = rotation_vector(rot.T @ env.target_rotation)
                env.target_rotation = rot @ rotation_matrix(limit_vector(lead, .03))
            was_moving = moving
            start = time.perf_counter()
            rows.append(env.step(action)[-1])
            costs.append(time.perf_counter()-start)
            grasped |= rows[-1]['normal_force_n'] > 1. and rows[-1]['passive_normal_force_n'] > 1.
            clip_contact |= any(any(g >= 0 and env.model.geom(g).name.startswith('cclip_')
                                    for g in c.geom) for c in env.data.contact)
        self.assertTrue(grasped)
        self.assertTrue(clip_contact)  # Must not pass by avoiding the fixture.
        hold_current_pose(env)
        env.command_gripper(False)
        for i in range(200):
            action = env.hold_action()
            if i < 100:
                action[2] = .0005
            rows.append(env.step(action)[-1])
        self.assertGreater(min(min(r['min_volume_ratio'], r['passive_min_volume_ratio']) for r in rows), .15)
        self.assertGreater(rows[-1]['min_volume_ratio'], .95)
        self.assertGreater(rows[-1]['passive_min_volume_ratio'], .95)
        self.assertLess(rows[-1]['normal_force_n'], .001)
        self.assertLess(rows[-1]['passive_normal_force_n'], .001)
        self.assertTrue(np.isfinite([r['magnetic_ut'] for r in rows]).all())
        self.assertTrue(np.isfinite([r['slip_speed_m_per_s'] for r in rows]).all())
        np.testing.assert_array_equal(env.data.warning.number, 0)
        print(f"Reduced cable replay {prefix or 'first'}: min V/V0={min(r['min_volume_ratio'] for r in rows):.3f}; "
              f"released={rows[-1]['min_volume_ratio']:.3f}; "
              f"step median/p95={np.quantile(costs, [.5, .95])*1000} ms", flush=True)

    def test_panda_native_closure_has_pad_contact_instead_of_empty_gap(self):
        e = self.make_env()
        np.testing.assert_allclose(e.model.jnt_range[e.finger_joints], [[-.04, 0.], [-.04, 0.]])
        e.command_gripper(True)
        rows = [e.step(e.hold_action())[-1] for _ in range(200)]
        pairs = [set(map(int, c.flex)) for c in e.data.contact]
        self.assertIn({e.skin_flex, e.passive_flex}, pairs)
        self.assertLess(rows[-1]['gripper_gap_m'], .001)
        self.assertGreater(min(r['min_volume_ratio'] for r in rows), .9)
        np.testing.assert_array_equal(e.data.warning.number, 0)

    def test_table_contact_and_retraction_without_tcp_height_floor(self):
        for gripper in ('panda', 'robotiq_2f85'):
            with self.subTest(gripper=gripper):
                e = self.make_env(gripper)
                self.assertEqual(e.workspace[0, 2], -np.inf)
                rows, touched = [], False
                table = e.model.geom('table').id
                for _ in range(450):
                    t = e.data.time
                    z = .185 if t < 2. else max(.100, .185-.06*(t-2)) if t < 3.8 else .185
                    e.target_xyz[:] = [.45, .15, z]
                    rows.append(e.step(e.hold_action())[-1])
                    touched |= any(table in c.geom and
                        (e.skin_flex in c.flex or e.passive_flex in c.flex) for c in e.data.contact)
                self.assertTrue(touched)
                self.assertLess(min(r['tcp_xyz_m'][2] for r in rows), .127)
                self.assertGreater(rows[-1]['tcp_xyz_m'][2], .14)
                self.assertGreater(min(min(r['min_volume_ratio'], r['passive_min_volume_ratio']) for r in rows), .15)
                np.testing.assert_array_equal(e.data.warning.number, 0)


if __name__ == '__main__':
    unittest.main()
