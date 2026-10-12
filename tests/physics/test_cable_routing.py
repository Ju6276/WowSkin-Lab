"""Two installed ends stay connected under load; no robot grasp/clip assist."""
from types import SimpleNamespace
import unittest
import tempfile
import json
import h5py
import numpy as np
import mujoco
from wowskin_lab.envs.cable import PandaCableEnv
from wowskin_lab.envs.pick_cube import response_configuration


class RoutingPhysicsTests(unittest.TestCase):
    def test_panda_physically_lifts_and_releases_connected_cable(self):
        c = response_configuration(SimpleNamespace(task='cable', cable_layout='routing', ideal=True))
        e = PandaCableEnv(c)
        for _ in range(300):
            e.step(e.hold_action())
        initial = e.data.xpos[e.cable_bodies, 2].copy()
        point = e.data.xpos[e.cable_bodies[29]].copy()
        held, lift = 0, 0.
        for i in range(650):
            t = i*.01
            # Bench trajectory, not a runtime target override or grasp helper.
            e.target_xyz[:] = [point[0], point[1], .20 if t < 2.4 else .123 if t < 4.2 else .17]
            if t >= 3.3:
                e.command_gripper(t < 5.8)
            row = e.step(e.hold_action())[-1]
            held += int(row['grasped'])
            lift = max(lift, float(np.max(row['cable_centerline_m'][:, 2]-initial)))
            self.assertLess(row['cable_end_anchor_error_m'], .001)
            self.assertGreater(min(row['min_volume_ratio'], row['passive_min_volume_ratio']), 0.)
        self.assertGreater(held, 100)
        self.assertGreater(lift, .03)
        self.assertLess(row['normal_force_n'], .001)
        self.assertLess(row['passive_normal_force_n'], .001)
        self.assertFalse(np.any(e.data.warning.number))

    def test_recording_includes_route_and_both_lips(self):
        from wowskin_lab.data.raw_recording import RawRobotRecorder
        from wowskin_lab.paths import ROOT
        c = response_configuration(SimpleNamespace(task='cable', cable_layout='routing',
            clip_model='elastic_beam', ideal=True, magnetic_fit=ROOT/'configs/magnetic_fit_pollen_nonslip.json'))
        e = PandaCableEnv(c)
        with tempfile.TemporaryDirectory() as tmp, RawRobotRecorder(e, tmp) as recorder:
            recorder.start()
            for _ in range(10):
                recorder.step(e.step(e.hold_action())[-1])
            path = recorder.finish('interrupted')
            meta = json.loads((path/'metadata.json').read_text())
            self.assertEqual(meta['simulation_config']['cable_routing']['capsules'], 48)
            with h5py.File(path/'measurements.h5') as stream:
                self.assertEqual(stream['sensor_states'].shape[1], 15)
                self.assertEqual(stream['camera_base_to_cv'].shape[1], 3)
                self.assertIn('evaluation/routing_gate_count', stream)
                self.assertIn('evaluation/clip_lower_tip_m', stream)
                self.assertIn('evaluation/clip_upper_tip_m', stream)

    def test_both_grippers_two_end_installation_and_elastic_lips(self):
        for gripper in ('panda', 'robotiq_2f85'):
            with self.subTest(gripper=gripper):
                c = response_configuration(SimpleNamespace(task='cable', cable_layout='routing',
                    gripper=gripper, skin_dof='trilinear', clip_model='elastic_beam', ideal=True))
                e = PandaCableEnv(c)
                self.assertEqual(e.cable_count, 49)
                self.assertGreater(e.cable_length, .7)
                self.assertEqual(e.model.body_dofnum[e.model.body('wire_B_first').id], 0)
                self.assertEqual(e.model.body_dofnum[e.model.body('adapter_1').id], 0)
                self.assertEqual(e.model.eq_type[e.model.equality('installed_cable_end').id], mujoco.mjtEq.mjEQ_CONNECT)
                self.assertGreater(e.model.joint('elastic_clip_lower_bend_0').stiffness, 0.)
                first = e.data.site_xpos[e.model.site('wire_S_first').id].copy()
                middle = e.cable_bodies[len(e.cable_bodies)//2]
                minimum_z = 1.
                control = e.control
                load = 0.
                def loaded_control():
                    control()
                    # Apply after the robot controller clears transient forces,
                    # on EVERY physics step, not just before env.step().
                    e.data.xfrc_applied[middle, 2] = load
                e.control = loaded_control
                loaded_steps = 0
                for i in range(180):
                    # External bench load, never used by the task or scorer.
                    load = .08 if 50 <= i < 100 else 0.
                    row = e.step(e.hold_action())[-1]
                    self.assertEqual(e.data.xfrc_applied[middle, 2], load)
                    loaded_steps += int(load > 0.)
                    minimum_z = min(minimum_z, e.cable_segments()[:, :, 2].min())
                    self.assertFalse(row['success'])
                    self.assertLess(row['cable_end_anchor_error_m'], .001)
                np.testing.assert_allclose(e.data.site_xpos[e.model.site('wire_S_first').id], first, atol=1e-12)
                self.assertGreater(minimum_z, .102)
                self.assertFalse(np.any(e.data.warning.number))
                self.assertFalse(np.any(e.data.xfrc_applied))
                self.assertEqual(loaded_steps, 50)
                self.assertFalse(e.cfg['grasp_assist'])
                self.assertFalse(e.cfg['clip_assist'])
                e.reset()
                self.assertFalse(e.success)


if __name__ == '__main__':
    unittest.main()
