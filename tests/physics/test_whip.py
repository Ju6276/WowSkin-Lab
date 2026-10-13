"""Published model parameters remain intact and its chain actually bends."""
import hashlib
import json
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from wowskin_lab.envs.whip import ASSETS, WhipBench


class WhipChecks(unittest.TestCase):
    def test_source_and_gravitational_bending(self):
        manifest = json.loads((ASSETS/'manifest.json').read_text())
        for name, entry in manifest['files'].items():
            self.assertEqual(hashlib.sha256((ASSETS/name).read_bytes()).hexdigest(), entry['sha256'])
        bench = WhipBench()
        original = ET.parse(ASSETS/'3D_model_w_whip.xml').find(".//body[@name='body_whip_node1']")
        derived = ET.fromstring(bench.xml).find(".//body[@name='body_whip_node1']")
        for a, b in zip(original.iter(), derived.iter(), strict=True):
            self.assertEqual(a.tag, b.tag)
            for key, value in a.attrib.items():
                if a is original and key == 'pos':
                    continue
                self.assertEqual(b.get(key), value)
        self.assertEqual(bench.model.nv, 50)
        self.assertAlmostEqual(bench.model.body_mass[2:].sum(), .3)
        self.assertAlmostEqual(bench.model.opt.timestep, .0001)
        initial = bench.data.geom_xpos[bench.nodes[-1]].copy()
        for _ in range(100):
            row = bench.step()
        self.assertLess(row['nodes_m'][-1, 2], initial[2]-.5)
        # More than root rotation: internal joints bend under gravity.
        self.assertGreater(np.linalg.norm(bench.data.qpos[2:]), .01)
        np.testing.assert_array_equal(bench.data.warning.number, 0)
        bench.reset(horizontal=False)
        for _ in range(30):
            row = bench.step()
        np.testing.assert_allclose(row['nodes_m'][-1], [0., 0., .3], atol=1e-5)


if __name__ == '__main__':
    unittest.main()
