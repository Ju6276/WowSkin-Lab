"""Run using .venv/bin/python -m unittest tests.physics.test_reduced_skin."""
from types import SimpleNamespace
import unittest
import mujoco
import numpy as np
from wowskin_lab.paths import ROOT
from wowskin_lab.core.geometry import cap_mesh
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, response_configuration, scripted_action

@unittest.skipUnless(mujoco.__version__ == '3.15.0', 'Isolated reduced-skin environment required')
class ReducedSkinChecks(unittest.TestCase):
    def make_env(self):
        cfg = response_configuration(SimpleNamespace(task='pick_cube', skin_dof='trilinear', ideal=True))
        return PandaPickCubeTactileEnv(cfg, pollen=False)

    def test_rest_geometry_attachment_and_velocity(self):
        e = self.make_env()
        m, d = e.model, e.data
        self.assertEqual(m.nv, 39)
        np.testing.assert_allclose(e.rest, cap_mesh(e.cfg)[1], atol=1e-10)
        for flex, parent in ((e.skin_flex, e.mount_body), (e.passive_flex, e.passive_mount)):
            adr = m.flex_nodeadr[flex]
            np.testing.assert_array_equal(m.flex_nodebodyid[adr:adr+8:2], parent)
        d.qvel[:] = np.random.default_rng(7).normal(0, .02, m.nv)
        mujoco.mj_forward(m, d)
        with np.errstate(divide='raise', over='raise', invalid='raise'):
            velocities = e.flex_vertex_velocities(e.skin_flex)
        self.assertTrue(np.isfinite(velocities).all())
        pos = d.flexvert_xpos[e.skin_vertices].copy()
        mujoco.mj_integratePos(m, d.qpos, d.qvel, 1e-7)
        mujoco.mj_forward(m, d)
        np.testing.assert_allclose((d.flexvert_xpos[e.skin_vertices]-pos)/1e-7, velocities, atol=1e-7)
        field = e.magnetic_field().copy()
        d.qpos[e.arm_q[0]] += .4
        mujoco.mj_forward(m, d)
        np.testing.assert_allclose(e.magnetic_field(), field, atol=1e-7)

    def test_matching_pad_geometry_and_passive_magnetic_isolation(self):
        e = self.make_env()
        m, d = e.model, e.data
        self.assertEqual(m.nflex, 2)
        np.testing.assert_allclose(e.rest, e.passive_rest, atol=1e-12)
        left, right = (d.xmat[bid].reshape(3, 3) for bid in (e.mount_body, e.passive_mount))
        self.assertAlmostEqual(float(left[:, 2] @ right[:, 2]), -1.)
        # The mount displacement is along the pad normal, with matching height.
        offset = (d.xpos[e.passive_mount]-d.xpos[e.mount_body]) @ left
        np.testing.assert_allclose(offset[:2], 0., atol=1e-8)
        self.assertGreater(offset[2], 2*e.cfg['total_height_m'])
        self.assertEqual(int((m.site_bodyid == e.mount_body).sum()), 5)
        self.assertEqual(int((m.site_bodyid == e.passive_mount).sum()), 0)

        def moved(flex):
            adr = m.flex_nodeadr[flex]
            node = m.flex_nodebodyid[adr+1]  # Top node; bottom nodes are welded to the mount.
            q = m.jnt_qposadr[m.body_jntadr[node]]
            before = e.magnetic_field().copy()
            saved = d.qpos.copy()
            d.qpos[q:q+3] += [.00001, -.00002, -.00003]
            mujoco.mj_forward(m, d)
            change = np.linalg.norm(e.magnetic_field()-before)
            d.qpos[:] = saved
            mujoco.mj_forward(m, d)
            return change
        # Deforming only the passive pad adds no magnetic sources or chips.
        self.assertLess(moved(e.passive_flex), 1e-12)
        self.assertGreater(moved(e.skin_flex), 1e-6)

    def test_pick_hold_tug_release(self):
        e = self.make_env()
        rows = [e.step(scripted_action(e))[-1] for _ in range(850)]
        held = [r for r in rows if 4.8 < r['time_s'] < 5.4]
        self.assertGreater(max(r['lift_m'] for r in rows), .09)
        self.assertTrue(all(r['grasped'] for r in held))
        self.assertLess(np.mean([r['slip'] for r in held]), .15)
        self.assertLess(rows[-1]['lift_m'], .003)
        self.assertEqual(rows[-1]['normal_force_n'], 0.)
        self.assertEqual(rows[-1]['passive_normal_force_n'], 0.)
        self.assertGreater(min(min(r['min_volume_ratio'], r['passive_min_volume_ratio']) for r in rows), .9)
        self.assertTrue(np.isfinite([r['magnetic_ut'] for r in rows]).all())
        np.testing.assert_array_equal(e.data.warning.number, 0)

class ReducedTaskChecks(unittest.TestCase):
    def config(self, task):
        return response_configuration(SimpleNamespace(task=task, skin_dof='trilinear', ideal=True,
            magnetic_fit=ROOT/'configs/magnetic_fit_pollen_nonslip.json'))

    def test_legacy_short_cable_layouts_grasp_lift_release(self):
        from wowskin_lab.envs.cable import PandaCableEnv
        from tests.physics.test_cable_stability import CableStabilityChecks
        for layout in ('routing', 'single_clip'):
            with self.subTest(layout=layout):
                cfg = self.config('cable')
                cfg['cable_layout'] = layout
                if layout == 'routing':
                    cfg['scene_design'] = 'ebim_task1_single_arm'  # Recorded one-ended 0.38 m benchmark.
                e = PandaCableEnv(cfg, pollen=False)
                self.assertIsNone(e.reskin_response)
                initial = e.data.xpos[e.cable_bodies, 2].copy()
                rows = []
                # Single clip occupies x=.45: approach the free cable end,
                # not the routing benchmark's mid-point under the fixture.
                x = .53 if layout == 'single_clip' else .45
                for _ in range(600):
                    t = e.data.time
                    if t < 2.4: target = [x, -.12, .185]
                    elif t < 3.3: target = [x, -.12, .123]
                    elif t < 4.2:
                        target = [x, -.12, .123]
                        e.command_gripper(True)
                    else:
                        target = [x, -.12, .17]
                        if t >= 5.4: e.command_gripper(False)
                    e.target_xyz[:] = target
                    rows.append(e.step(e.hold_action())[-1])
                CableStabilityChecks().check_rows(e, rows)
                lift = max(float(np.max(r['cable_centerline_m'][:, 2]-initial)) for r in rows)
                self.assertGreater(lift, .03)
                self.assertGreater(sum(r['grasped'] for r in rows), 50)
                self.assertLess(rows[-1]['normal_force_n'], .001)
                print(f'{layout}: lift {lift*1000:.1f} mm', flush=True)

    def test_feeding_loaded_move_release(self):
        import time
        from wowskin_lab.envs.feeding import PandaFeedingEnv
        e = PandaFeedingEnv(self.config('feeding'), pollen=False)
        self.assertIsNone(e.reskin_response)
        rows, elapsed = [], []
        for i in range(350):
            a = e.hold_action()
            if i < 100:
                a[2] = .0003
            elif i < 200:
                a[0] = .0002
            elif i == 250:
                e.command_gripper(False)
                a = e.hold_action()
            start = time.perf_counter()
            rows.append(e.step(a)[-1])
            elapsed.append(time.perf_counter()-start)
        self.assertGreater(sum(r['grasped'] for r in rows[:200]), 150)
        self.assertLess(rows[-1]['normal_force_n'], .001)
        self.assertGreater(min(min(r['min_volume_ratio'], r['passive_min_volume_ratio']) for r in rows), .15)
        self.assertTrue(np.isfinite([r['magnetic_ut'] for r in rows]).all())
        np.testing.assert_array_equal(e.data.warning.number, 0)
        print(f'Feeding fitted dipole: median {np.median(elapsed)*1000:.1f} ms/sample', flush=True)

if __name__ == '__main__':
    unittest.main()
