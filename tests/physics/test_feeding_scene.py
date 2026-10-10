"""Upstream asset integrity, physical rest and raw acquisition for task 3."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import cv2
import h5py
import mujoco
import numpy as np

from wowskin_lab.core.physics import configuration
from wowskin_lab.envs.feeding import ASSETS, BEAN_COUNT, HEAD_POSITION, HEAD_SCALE, ROBOT_POSITION, TABLE_TOP, PandaFeedingEnv, selected_world
from wowskin_lab.envs.pick_cube import key_command, response_configuration
from wowskin_lab.data.learning_recording import LearningSimRecorder


class FeedingChecks(unittest.TestCase):
    def test_original_assets_station_dynamics_and_recorders(self):
        manifest = json.loads((ASSETS/'manifest.json').read_text())
        for name, entry in manifest['files'].items():
            self.assertEqual(hashlib.sha256((ASSETS/name).read_bytes()).hexdigest(), entry['sha256'])
        cfg = response_configuration(SimpleNamespace(task='feeding', ideal=True))
        env = PandaFeedingEnv(cfg)
        source = ET.parse(ASSETS/'scene_100.xml').getroot()
        built = ET.fromstring(env.xml)
        # Preserve every original prop collision/visual mesh, size and mass.
        for element in selected_world(source):
            if element.tag == 'body' and built.find(f".//body[@name='{element.get('name')}']") is None:
                continue  # User-requested delivery-only scene omits crockery/extra beans.
            for node in element.iter():
                if node.tag != 'geom':
                    continue  # Complete free-body masses are checked below.
                counterpart = built.find(f".//geom[@name='{node.get('name')}']")
                if counterpart is None and element.get('name', '').startswith('dynamic_coffee_bean_'):
                    continue  # Delivery-only starts with six original beans.
                for key, value in node.attrib.items():
                    if element.get('name') == 'dynamic_cup' and key in ('friction', 'priority'):
                        continue  # Upstream config.json runtime overrides.
                    self.assertEqual(counterpart.get(key), value, (node.get('name'), key))
        for name, mass in [('dynamic_spoon2', .035)]:
            self.assertAlmostEqual(env.model.body_mass[env.model.body(name).id], mass)
        for name in ('dynamic_bowl2', 'dynamic_plate2', 'dynamic_simple_tray', 'dynamic_cup'):
            self.assertIsNone(built.find(f".//body[@name='{name}']"))
        self.assertEqual(len(env.bean_bodies), BEAN_COUNT)
        self.assertTrue(all(env.model.body_mass[b] == .00018 for b in env.bean_bodies))
        self.assertFalse(np.any(env.model.body_mocapid >= 0))
        self.assertFalse(np.any(env.model.eq_type == mujoco.mjtEq.mjEQ_WELD))
        self.assertEqual(env.model.opt.solver, mujoco.mjtSolver.mjSOL_NEWTON)
        a = env.gripper_control
        self.assertEqual(env.model.actuator_biastype[a], mujoco.mjtBias.mjBIAS_AFFINE)
        np.testing.assert_allclose(env.model.actuator_forcerange[a], [-100, 100])
        np.testing.assert_allclose(env.model.actuator_biasprm[a, :3], [0, -100, -10])
        head = env.model.body('configurable_visual_head').id
        np.testing.assert_allclose(env.data.xpos[head], HEAD_POSITION)
        self.assertAlmostEqual(HEAD_POSITION[2]-.1*HEAD_SCALE, TABLE_TOP)
        self.assertGreater(np.linalg.norm((np.array(HEAD_POSITION)-ROBOT_POSITION)[:2]), .64)
        for name in ('visual_head_skin_mesh', 'visual_head_eye_left_mesh', 'visual_head_eye_right_mesh', 'local_head_back_mesh'):
            np.testing.assert_allclose(np.fromstring(built.find(f"asset/mesh[@name='{name}']").get('scale'), sep=' '), HEAD_SCALE)
        face = env.data.xmat[head].reshape(3, 3)@np.array([0., -1., 0.])
        toward_robot = np.array(ROBOT_POSITION)-HEAD_POSITION
        toward_robot[2] = 0; toward_robot /= np.linalg.norm(toward_robot)
        self.assertGreater(face@toward_robot, .999)
        env.reset()
        start = env.data.site_xpos[env.tcp_site].copy()
        lowest_volume = 1.
        with tempfile.TemporaryDirectory() as tmp, LearningSimRecorder(env, tmp, source='feeding_contract_test') as recorder:
            recorder.start()
            for _ in range(110):
                action = env.hold_action();action[2] = .0001
                row = env.step(action)[-1]
                recorder.step(row)
                lowest_volume = min(lowest_volume, row['min_volume_ratio'], row['passive_min_volume_ratio'])
                self.assertEqual(row['bean_positions_m'].shape, (BEAN_COUNT, 3))
                self.assertFalse(row['success'])  # No cube success leaking into feeding.
                self.assertTrue(row['grasped'])
                local = (row['bean_positions_m']-row['spoon_xyz_m'])@env.data.xmat[env.probe_body].reshape(3, 3)
                self.assertTrue(np.all((local[:, 0] > .015) & (local[:, 0] < .062)))
                self.assertTrue(np.all(np.abs(local[:, 1]) < .014))
                self.assertTrue(np.all(np.abs(local[:, 2]) < .008))
            result = recorder.finish('unreviewed')
            self.assertGreater(env.data.site_xpos[env.tcp_site, 2]-start[2], .008)
            self.assertGreater(lowest_volume, .15)  # Existing physical-validity cutoff, now under load.
            for kind, meta_name in [('visk', 'metadata.json'), ('ftf', 'simulation_metadata.json')]:
                meta = json.loads((result[kind]/meta_name).read_text())
                self.assertEqual(meta['task'], 'feeding')
                self.assertTrue(meta['simulation_config']['grasp_assist'] is False)
                self.assertEqual(meta['simulation_config']['bean_count'], BEAN_COUNT)
            with h5py.File(result['visk']/'measurements.h5') as v, h5py.File(result['ftf']/'reskin_sensor_values.h5') as f:
                self.assertEqual(v['sensor_states'].shape, (111, 15))
                self.assertEqual(v['command_7d'].shape, (111, 7))
                self.assertFalse(any('bean' in k or 'object' in k for k in v))
                times = v['timestamp'][1:]; indices = np.searchsorted(f['timestamp'][:], times)
                np.testing.assert_allclose(f['timestamp'][:][indices], times, atol=1e-10)
                np.testing.assert_allclose(f['sensor_values'][:][indices], v['sensor_states'][1:], atol=1e-4)
                for cid in (1, 2):
                    stream = cv2.VideoCapture(str(result['visk']/f'cam_{cid}.avi'))
                    ok, frame = stream.read();stream.release()
                    self.assertTrue(ok);self.assertEqual(frame.shape, (480, 640, 3))
        np.testing.assert_array_equal(env.data.warning.number, 0)
        # All initial beans still belong to the table station; no explosive
        # solver event disguised as a successful acquisition test.
        positions = env.data.xpos[env.bean_bodies]
        self.assertTrue(np.all((positions[:, 2] > 1.08) & (positions[:, 2] < 1.3)))
        # Impose a diagnostic bean/head contact and check every-step peak
        # persistence; remove contact again to verify current force returns 0.
        saved = env.data.qpos.copy()
        bid = env.bean_bodies[0]
        adr = env.model.jnt_qposadr[env.model.body_jntadr[bid]]
        head_rotation = env.data.xmat[head].reshape(3, 3)
        for point in (np.array(HEAD_POSITION)+.045*HEAD_SCALE*face,
                      np.array(HEAD_POSITION)+head_rotation@np.array([0., .055, .056])*HEAD_SCALE):
            env.data.qpos[adr:adr+3] = point
            mujoco.mj_forward(env.model, env.data)
            # Reference full scan guards the faster candidate filter: both
            # face and posterior hull contribute, with the same force signs.
            total, net, wrench = 0., np.zeros(3), np.zeros(6)
            for i, contact in enumerate(env.data.contact):
                if any(g in env.head_geoms for g in contact.geom):
                    mujoco.mj_contactForce(env.model, env.data, i, wrench)
                    total += np.linalg.norm(wrench[:3])
                    sign = 1. if contact.geom[1] in env.head_geoms else -1.
                    net += sign*contact.frame.reshape(3, 3).T@wrench[:3]
            env.after_physics_step()
            self.assertGreater(total, 0.)
            self.assertAlmostEqual(env.head_force_n, total)
            np.testing.assert_allclose(env.head_net_force, net)
        peak = env.head_peak_n
        env.data.qpos[:] = saved
        mujoco.mj_forward(env.model, env.data);env.after_physics_step()
        self.assertEqual(env.head_force_n, 0.)
        self.assertEqual(env.head_peak_n, peak)
        print(f'Feeding: six beans held/lifted, raw FTF/ViSk synchronized, min V/V0={lowest_volume:.4f}', flush=True)


if __name__ == '__main__':
    unittest.main()
