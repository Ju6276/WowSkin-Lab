"""Run using .venv/bin/python -m unittest tests.physics.test_reduced_skin."""
from types import SimpleNamespace
import unittest
import mujoco
import numpy as np
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
        from wowskin_lab.paths import ROOT
        return response_configuration(SimpleNamespace(task=task, skin_dof='trilinear', ideal=True,
            reskin_model=ROOT/'outputs/reskin_forward_mapped', reskin_extrapolate=True))

    def test_cable_both_layouts_grasp_lift_release(self):
        from wowskin_lab.envs.cable import PandaCableEnv
        from tests.physics.test_cable_stability import CableStabilityChecks
        for layout in ('routing', 'single_clip'):
            with self.subTest(layout=layout):
                cfg = self.config('cable')
                cfg['cable_layout'] = layout
                e = PandaCableEnv(cfg, pollen=False)
                initial = e.data.xpos[e.cable_bodies, 2].copy()
                rows = []
                # Single clip occupies x=.45: approach the free cable end,
                # not the routing benchmark's mid-point under the fixture.
                x = .53 if layout == 'single_clip' else .45
                for _ in range(600):
                    t = e.data.time
                    if t < 2.4: target = [x, -.12, .185]
                    elif t < 3.3: target = [x, -.12, .1275]
                    elif t < 4.2:
                        target = [x, -.12, .1275]
                        e.command_gripper(True)
                    else:
                        target = [x, -.12, .17]
                        if t >= 5.4: e.command_gripper(False)
                    e.target_xyz[:] = target
                    rows.append(e.step(e.hold_action())[-1])
                CableStabilityChecks().check_rows(e, rows, network_preview=True)
                lift = max(float(np.max(r['cable_centerline_m'][:, 2]-initial)) for r in rows)
                self.assertGreater(lift, .03)
                self.assertGreater(sum(r['grasped'] for r in rows), 50)
                self.assertLess(rows[-1]['normal_force_n'], .001)
                print(f'{layout}: lift {lift*1000:.1f} mm', flush=True)

    def test_feeding_loaded_move_release(self):
        import time
        from wowskin_lab.envs.feeding import PandaFeedingEnv
        e = PandaFeedingEnv(self.config('feeding'), pollen=False)
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
        self.check_network_rows(rows)
        np.testing.assert_array_equal(e.data.warning.number, 0)
        print(f'Feeding network: median {np.median(elapsed)*1000:.1f} ms/sample', flush=True)

    def check_network_rows(self, rows):
        for r in rows:
            self.assertTrue(np.isfinite(r['fitted_dipole_ideal_delta_ut']).all())
            if r['reskin_prediction_emitted']:
                self.assertTrue(np.isfinite(r['magnetic_ut']).all())
            else:
                self.assertTrue(np.isnan(r['magnetic_ut']).all())
                self.assertTrue(r['reskin_status'].startswith(('UNAVAILABLE:', 'BLOCKED:')))

if __name__ == '__main__':
    unittest.main()
