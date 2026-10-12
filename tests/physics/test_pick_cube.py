"""Contact-only grasp/release, mount registration, and tactile display checks."""
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from wowskin_lab.envs.franka import FrankaTactileSim, mount_reference
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, rotation_matrix, scripted_action, task_camera
from wowskin_lab.ui.tactile import TactileHUD


class PickCubeChecks(unittest.TestCase):
    def test_hud_tolerates_window_closing_between_frames(self):
        from types import SimpleNamespace
        from contextlib import nullcontext
        hud = TactileHUD.__new__(TactileHUD)
        hud.show(SimpleNamespace(lock=nullcontext, is_running=lambda: False), {})
        hud.show(SimpleNamespace(lock=nullcontext, is_running=lambda: True, viewport=None), {})

    def test_removed_stock_pads_and_registered_upper_edge(self):
        sim = PandaPickCubeTactileEnv(pollen=False)
        root = ET.fromstring(sim.xml)
        reference = mount_reference(sim.cfg)
        self.assertAlmostEqual(sim.cfg['base_height_m'], .0115)
        self.assertAlmostEqual(sim.cfg['total_height_m'], .0135)
        self.assertAlmostEqual(sim.rest[:, 2].max(), .0135)
        self.assertAlmostEqual(sim.rest[:, 2].min(), .0095)
        np.testing.assert_allclose(sim.rest, sim.passive_rest, atol=1e-12)
        for finger_name, mount_name in (("left_finger", "wowskin_mount"), ("right_finger", "silicone_mount")):
            finger = root.find(f".//body[@name='{finger_name}']")
            self.assertIsNone(finger.find("geom[@mesh='finger_1']"))
            self.assertFalse(any(g.get("class", "").startswith("fingertip_pad_collision_") for g in finger.findall("geom")))
            bid = sim.model.body(mount_name).id
            # The proximal housing/skin outline edge is registered in finger Z.
            self.assertAlmostEqual(sim.model.body_pos[bid, 2]-sim.cfg["length_m"]/2,
                                   reference["removed_pad_proximal_z_m"], places=12)
            self.assertAlmostEqual(sim.model.body_pos[bid, 1], reference["exposed_seat_y_m"], places=12)

    def test_free_cube_pick_hold_tug_and_release(self):
        env = PandaPickCubeTactileEnv()
        m, d = env.model, env.data
        self.assertEqual(m.jnt_type[m.body_jntadr[env.probe_body]], mujoco.mjtJoint.mjJNT_FREE)
        self.assertFalse(np.any(m.eq_type == mujoco.mjtEq.mjEQ_WELD))
        rows = [env.step(scripted_action(env))[-1] for _ in range(850)]
        self.assertFalse(env.success)  # Lift/tug/release in place is not a two-box delivery.
        self.assertGreater(max(r["lift_m"] for r in rows), .08)
        held = [r for r in rows if 4.8 < r["time_s"] < 5.4]
        self.assertTrue(all(r["grasped"] for r in held))
        self.assertLess(np.mean([r["slip"] for r in held]), .15)
        tugged = [r for r in rows if 5.5 < r["time_s"] < 6.1]
        self.assertTrue(any(r["slip"] for r in tugged))
        self.assertLess(rows[-1]["lift_m"], .003)
        self.assertLess(rows[-1]["normal_force_n"], .001)
        self.assertLess(rows[-1]["passive_normal_force_n"], .001)
        self.assertGreater(min(r["min_volume_ratio"] for r in rows), .8)
        self.assertGreater(min(r["passive_min_volume_ratio"] for r in rows), .8)
        self.assertTrue(np.isfinite([r["magnetic_ut"] for r in rows]).all())
        self.assertTrue(np.isfinite([r["pollen_probability"] for r in rows]).all())
        np.testing.assert_array_equal(d.warning.number, 0)
        obs, _ = env.reset(seed=3)
        np.testing.assert_array_equal(obs["magnetic_ut"], np.zeros(15))
        self.assertEqual(env.data.time, 0.)
        with self.assertRaises(ValueError):
            env.step([0, 0, float("nan"), 0, 0, 0, 0])

    def test_seven_dimensional_action_rotation_and_gripper(self):
        env = PandaPickCubeTactileEnv(pollen=False)
        for invalid in (np.zeros(4), [0, 0, 0, float('nan'), 0, 0, 0]):
            with self.assertRaises(ValueError):
                env.step(invalid)
        start = env.target_rotation.copy()
        a = np.array([0., 0., 0., .02, 0., 0., .4])
        obs, _, _, _, row = env.step(a)
        np.testing.assert_allclose(env.target_rotation, start @ rotation_matrix(a[3:6]), atol=1e-12)
        np.testing.assert_allclose(row['action'], a)
        self.assertEqual(row['tcp_pose_before_m_rad'].shape, (6,))
        self.assertEqual(obs['tcp_rotvec_rad'].shape, (3,))
        self.assertAlmostEqual(row['time_s']-row['action_time_s'], .01)
        self.assertAlmostEqual(env.gripper_fraction(target=True), .4)
        b = np.array([0., 0., 0., 0., .02, 0., .4])
        env.step(b)
        np.testing.assert_allclose(env.target_rotation,
                                   start @ rotation_matrix(a[3:6]) @ rotation_matrix(b[3:6]), atol=1e-12)
        env.step(env.hold_action())
        self.assertAlmostEqual(env.gripper_fraction(target=True), .4)
        env.step(np.zeros(7))
        self.assertEqual(env.grip_target, 0.)
        env.reset()
        np.testing.assert_allclose(env.target_rotation, start, atol=1e-12)
        np.testing.assert_allclose(env.servo_rotation, start, atol=1e-12)

    def test_hud_z_sign_xy_rotation_and_display_only_scale(self):
        sim = PandaPickCubeTactileEnv(pollen=False)
        hud = TactileHUD(sim, full_scale_ut=10.)
        signal = np.zeros((5, 3))
        signal[0, 2], signal[1, 2] = 6., -6.
        row = dict(magnetic_ut=signal.reshape(15), time_s=0., slip=0)
        before = row["magnetic_ut"].copy()
        pixels = hud.render(row)
        _, _, centres = hud.glyphs(row["magnetic_ut"])
        c0, c1 = centres[:2].astype(int)
        np.testing.assert_array_equal(pixels[c0[1]+8, c0[0]+8], [255, 0, 0])
        np.testing.assert_array_equal(pixels[c1[1]+8, c1[0]+8], [21, 27, 34])
        hud.key(ord("."))
        hud.render(row)
        np.testing.assert_array_equal(row["magnetic_ut"], before)
        signal[1, :2] = [1., 0.]
        _, board, _ = hud.glyphs(signal)
        np.testing.assert_allclose(board[1, :2], [-1, 0], atol=1e-12)

    def test_mouse_drag_force_survives_control_and_releases(self):
        env = PandaPickCubeTactileEnv(pollen=False)
        perturb = mujoco.MjvPerturb()
        perturb.select = env.probe_body
        scene = mujoco.MjvScene(env.model, maxgeom=4000)
        mujoco.mjv_updateScene(env.model, env.data, mujoco.MjvOption(), None,
                              task_camera(env), mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjv_initPerturb(env.model, env.data, scene, perturb)
        perturb.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
        mujoco.mjv_movePerturb(env.model, env.data, mujoco.mjtMouse.mjMOUSE_MOVE_H,
                             .01, 0., scene, perturb)
        env.update_mouse_perturbation(perturb)
        env.control()
        self.assertGreater(np.linalg.norm(env.data.xfrc_applied[env.probe_body, :3]), .01)
        self.assertLessEqual(np.linalg.norm(env.data.xfrc_applied[env.probe_body, :3]), 1.)
        perturb.active = 0
        env.update_mouse_perturbation(perturb)
        env.control()
        np.testing.assert_array_equal(env.data.xfrc_applied, 0.)


if __name__ == "__main__":
    unittest.main()
