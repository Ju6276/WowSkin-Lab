"""Physical/source checks for the robot scene, separate from food calibration."""
import copy
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from wowskin_lab.envs.brittle_chip import BrittleChipBench, constraint_blocks
from wowskin_lab.envs.chip_tray import PandaChipEnv, ASSETS, chip_config


class ChipTrayTests(unittest.TestCase):
    def test_tray_source_preserved_and_saddle_has_opposite_curvatures(self):
        e = PandaChipEnv()
        root = ET.fromstring(e.xml)
        names = {b.get('name') for b in root.iter('body')}
        self.assertFalse(names & {'configurable_visual_head', 'dynamic_spoon2', 'care_seated_adult'})
        self.assertFalse(any('bean' in n for n in names))
        actual = root.find(".//body[@name='dynamic_simple_tray']")
        source = copy.deepcopy(ET.parse(ASSETS/'scene_100.xml').find(".//body[@name='dynamic_simple_tray']"))
        source.set('pos', actual.get('pos'))
        source.set('childclass', 'ebim_task3')
        self.assertEqual(ET.tostring(source), ET.tostring(actual))
        vertices = np.concatenate([center+v for center, v, _ in e.fracture.pieces])
        c = chip_config()
        self.assertAlmostEqual(np.ptp(vertices[:, 2]), 2*c['saddle_height_m']+c['thickness_m'])
        self.assertGreater(vertices[np.argmax(vertices[:, 0]), 2], 0)
        self.assertLess(vertices[np.argmax(vertices[:, 1]), 2], 0)
        for eq in e.fracture.eq_ids:
            self.assertIn(e.model.eq_obj1id[eq], e.fracture.body_ids)
            self.assertIn(e.model.eq_obj2id[eq], e.fracture.body_ids)
        self.assertEqual(len(e.contact_geoms), len(e.fracture.pieces))

    def test_rest_intact_then_physical_robot_contact_breaks(self):
        e = PandaChipEnv()
        for _ in range(100):
            *_, row = e.step(e.hold_action())
        self.assertEqual(len(e.fracture.events), 0)
        self.assertTrue(row['tactile_valid'])
        target = np.array([-1.7822603, .13180272, 1.115])
        peak = 0.
        for k in range(450):
            delta = target-e.target_xyz
            delta *= min(1., .0006/max(np.linalg.norm(delta), 1e-12))
            close = 0. if k < 300 else min(.85, (k-300)*.006)
            *_, row = e.step(np.r_[delta, np.zeros(3), close])
            peak = max(peak, row['normal_force_n'], row['passive_normal_force_n'])
            if e.fracture.events:
                break
        self.assertGreater(peak, .001)
        self.assertTrue(e.fracture.events, 'Physical contact must load the seams')
        self.assertTrue(all(event['load_ratio'] >= 1 for event in e.fracture.events))
        self.assertFalse(np.any(e.data.warning.number))
        e.reset()
        self.assertEqual(e.fracture.components(), 1)
        self.assertEqual(len(e.fracture.events), 0)

    def test_sparse_robot_path_matches_dense_jacobian(self):
        b = BrittleChipBench()
        b.data.xfrc_applied[b.body_ids[0]] = [.02, -.01, .03, .0001, 0, .0003]
        mujoco.mj_forward(b.model, b.data)
        ids, dense_w = b.bond_wrenches()
        _, rows = b._bond_cache
        dense_blocks = constraint_blocks(b.model, b.data, rows, b._seam_dofs[ids])
        b.model.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
        b._bond_cache = None
        mujoco.mj_forward(b.model, b.data)
        sparse_ids, sparse_w = b.bond_wrenches()
        sparse_blocks = constraint_blocks(b.model, b.data, b._bond_cache[1], b._seam_dofs[sparse_ids])
        np.testing.assert_array_equal(ids, sparse_ids)
        np.testing.assert_allclose(dense_blocks, sparse_blocks, atol=1e-12)
        np.testing.assert_allclose(dense_w, sparse_w, atol=1e-8)
        # Reusing cached sparse indices after stepping/contact changes is exact.
        mujoco.mj_step(b.model, b.data)
        _, cached = b.bond_wrenches()
        b._bond_cache = None
        _, rebuilt = b.bond_wrenches()
        np.testing.assert_array_equal(cached, rebuilt)
