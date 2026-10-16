"""Keep renderer-specific invalid depths lossless, with explicit masks."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import h5py
import mujoco
import numpy as np

from wowskin_lab.data.depth import DepthWriter


class DepthChecks(unittest.TestCase):
    def test_warp_no_hit_is_zero_and_masked_without_changing_source_values(self):
        model = mujoco.MjModel.from_xml_string('<mujoco/>')
        depth = np.ones((480, 640), dtype=np.float32)
        depth[0, 0] = 0
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'depth.h5'
            writer = DepthWriter(path, model, renderer='warp')
            try:
                writer.camera(1, depth, np.eye(3))
                for value in (-1., float('nan')):
                    invalid = depth.copy()
                    invalid[0, 1] = value
                    with self.assertRaisesRegex(ValueError, 'Invalid metric depth'):
                        writer.camera(1, invalid, np.eye(3))
            finally:
                writer.close()
            with h5py.File(path) as stream:
                np.testing.assert_array_equal(stream['camera1'][0], depth)
                self.assertFalse(stream['valid_camera1'][0, 0, 0])
                self.assertTrue(stream['valid_camera1'][0, 1, 1])
                self.assertEqual(stream.attrs['renderer'], 'warp')
            writer = DepthWriter(Path(tmp)/'opengl.h5', model)
            try:
                with self.assertRaisesRegex(ValueError, 'Invalid metric depth'):
                    writer.camera(1, depth, np.eye(3))
            finally:
                writer.close()
