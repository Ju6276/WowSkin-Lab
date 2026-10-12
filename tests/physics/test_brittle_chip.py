import unittest

import mujoco
import numpy as np

from wowskin_lab.envs.brittle_chip import BrittleChipBench, configuration, demo_target, weld_wrench


class BrittleChipTests(unittest.TestCase):
    def test_rest_stays_intact_and_unloaded(self):
        b = BrittleChipBench()
        for _ in range(100):
            b.step()
        self.assertEqual(len(b.events), 0)
        self.assertEqual(b.components(), 1)
        self.assertEqual(b.peak_probe_force_n, 0)
        self.assertFalse(np.any(b.data.warning.number))
        residual = b.data.efc_pos[:len(b.bonds)*6].reshape(-1, 6)
        self.assertLess(np.max(np.linalg.norm(residual[:, :3], axis=1)), 1e-5)
        self.assertLess(np.max(np.linalg.norm(residual[:, 3:], axis=1)), .001)

    def test_physical_press_breaks_and_reset_restores(self):
        b = BrittleChipBench()
        for _ in range(900):
            b.step(demo_target(b.data.time, b.config))
        self.assertGreater(b.peak_probe_force_n, .001)
        self.assertGreater(b.components(), 1)
        self.assertTrue(all(e['load_ratio'] >= 1 for e in b.events))
        self.assertFalse(np.any(b.data.warning.number))
        # Separated fragments fall below their initial supported height.
        self.assertLess(b.data.xipos[b.body_ids, 2].min(), .01)
        b.reset()
        self.assertEqual(b.components(), 1)
        self.assertEqual(len(b.events), 0)
        np.testing.assert_array_equal(b.data.qpos, b.model.qpos0)
        self.assertTrue(np.all(b.data.eq_active[b.eq_ids]))

    def test_strength_controls_breakage_not_elapsed_time(self):
        c = configuration()
        c['bond_force_limit_n'] *= 1e4
        c['bond_moment_limit_nm'] *= 1e4
        b = BrittleChipBench(c)
        for _ in range(500):
            b.step(demo_target(b.data.time, c))
        self.assertGreater(b.peak_probe_force_n, .01)
        self.assertEqual(b.components(), 1)
        self.assertEqual(len(b.events), 0)

    def test_rupture_does_not_change_positions_or_velocities(self):
        b = BrittleChipBench()
        for _ in range(50):
            b.step()
        b.config['bond_force_limit_n'] = 1e-12
        b.config['bond_moment_limit_nm'] = 1e-12
        mujoco.mj_step(b.model, b.data)
        q, v = b.data.qpos.copy(), b.data.qvel.copy()
        self.assertTrue(b.break_overloaded_bonds())
        np.testing.assert_array_equal(b.data.qpos, q)
        np.testing.assert_array_equal(b.data.qvel, v)

    def test_batched_wrenches_match_spatial_jacobian(self):
        b = BrittleChipBench()
        # A globally rotated, airborne chip tests all free-joint axis conventions.
        quat = np.array([.9, .1, .2, .3]); quat /= np.linalg.norm(quat)
        rotation = np.zeros(9); mujoco.mju_quat2Mat(rotation, quat)
        for body in b.body_ids:
            q = b.model.jnt_qposadr[b.model.body_jntadr[body]]
            b.data.qpos[q:q+3] = rotation.reshape(3, 3) @ b.data.qpos[q:q+3] + [0, 0, .1]
            b.data.qpos[q+3:q+7] = quat
        b.data.xfrc_applied[b.body_ids[0]] = [.01, -.02, .03, .0001, .0002, -.0003]
        mujoco.mj_step(b.model, b.data)
        ids, wrenches = b.bond_wrenches()
        for i, w in zip(ids, wrenches):
            a, _, p = b.bonds[i]
            body = b.body_ids[a]
            point = b.data.xpos[body]+b.data.xmat[body].reshape(3, 3) @ (p-b.pieces[a][0])
            expected = weld_wrench(b.model, b.data, b.eq_ids[i], body, point)
            np.testing.assert_allclose(w, expected, atol=1e-10, rtol=1e-8)

    def test_weld_wrench_units_against_known_static_load(self):
        # COM offset, rotated body and different weld scales catch the tempting
        # but incorrect interpretation of efc_force's last three rows as Nm.
        for scale in (.01, .1):
            xml = f'''<mujoco><option timestep=".0002" gravity="0 0 0" jacobian="dense"
                integrator="implicitfast" solver="Newton" tolerance="1e-12"/>
              <worldbody><body name="test" pos="0 0 .2" euler="20 30 40">
                <freejoint/><geom type="box" size=".01 .02 .03" pos=".005 0 0" mass=".1"/>
              </body></worldbody>
              <equality><weld body1="test" anchor="0 0 .2" torquescale="{scale}"
                solref=".003 1" solimp=".99 .999 .0001"/></equality></mujoco>'''
            m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
            body = m.body('test').id
            load = np.array([.2, -.3, .4, .001, .003, -.002])
            d.xfrc_applied[body] = load
            for _ in range(10000):
                mujoco.mj_step(m, d)
            self.assertLess(np.max(abs(d.qvel)), 1e-4, 'Load fixture has not reached static equilibrium')
            point = d.xpos[body]+[.02, -.01, 0]
            wrench = weld_wrench(m, d, 0, body, point)
            expected = np.r_[-load[:3], -load[3:]-np.cross(d.xipos[body]-point, load[:3])]
            np.testing.assert_allclose(wrench, expected, atol=1e-6, rtol=1e-3)

    def test_separated_fragments_can_collide_and_reset_restores_masks(self):
        b = BrittleChipBench()
        a, neighbor = b.bonds[0][:2]
        # Make an overlapping geometry fixture. Internal faces are suppressed
        # only while part of the same unbroken cluster.
        qa = b.model.jnt_qposadr[b.model.body_jntadr[b.body_ids[a]]]
        qb = b.model.jnt_qposadr[b.model.body_jntadr[b.body_ids[neighbor]]]
        b.data.qpos[qa:qa+3] = b.data.qpos[qb:qb+3]
        ga = b.model.geom(f'piece_{a}').id
        def has_fragment_contact():
            mujoco.mj_forward(b.model, b.data)
            return any(ga in contact.geom and all(g in b.piece_geoms for g in contact.geom)
                       for contact in b.data.contact)
        self.assertFalse(has_fragment_contact())
        for eq, (i, j, _) in zip(b.eq_ids, b.bonds):
            if a in (i, j):
                b.data.eq_active[eq] = False
        b._update_fragment_collision_masks()
        self.assertTrue(has_fragment_contact())
        b.reset()
        b.data.qpos[qa:qa+3] = b.data.qpos[qb:qb+3]
        self.assertFalse(has_fragment_contact())


if __name__ == '__main__':
    unittest.main()
