import unittest
from dataclasses import replace

import numpy as np

from wowskin_lab.data.pointcloud import RGBDView, backproject, fuse_rgbd
from wowskin_lab.ui.pointcloud import OnlinePointCloud


class PointCloudTests(unittest.TestCase):
    def view(self, camera=1):
        return RGBDView(camera, 1., np.full((3,3,3), [100,150,200], dtype=np.uint8),
            np.ones((3,3)), np.array([[2.,0,1],[0,2.,1],[0,0,1.]]), np.eye(4), .01, 10.)

    def test_camera_to_base_direction_and_depth_units(self):
        # OpenCV axes rotated into base; centre ray lies at known base point.
        camera_to_base = np.array([[0.,0.,1.,.2], [1.,0.,0.,.3], [0.,1.,0.,.4], [0.,0.,0.,1.]])
        v = replace(self.view(), base_to_camera=np.linalg.inv(camera_to_base))
        xyz, rgb = backproject(v)
        np.testing.assert_allclose(xyz[4], [1.2,.3,.4])
        np.testing.assert_allclose(xyz[5]-xyz[4], [0,.5,0])
        np.testing.assert_allclose(xyz[7]-xyz[4], [0,0,.5])
        np.testing.assert_array_equal(rgb[4], [100,150,200])

    def test_far_invalid_and_mixed_times_are_rejected(self):
        v = self.view()
        depth = v.depth.copy(); depth[0] = [np.nan, 0, 10.]
        self.assertEqual(len(backproject(replace(v, depth=depth))[0]), 6)
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            fuse_rgbd([v, replace(v, camera_id=52, timestamp=1.01)])

    def test_sources_and_no_cross_time_history(self):
        cloud = fuse_rgbd([self.view(), self.view(2), self.view(52)], stride=1)
        self.assertEqual(set(cloud.camera_ids), {1,2,52})
        self.assertEqual(len(cloud.xyz), 27)
        later = fuse_rgbd([replace(self.view(), timestamp=2.)], stride=1)
        self.assertEqual(len(later.xyz), 9)
        self.assertEqual(later.timestamp, 2.)

    def test_mode_change_drops_old_worker_result(self):
        preview = OnlinePointCloud(enabled=True)
        try:
            preview.submit([self.view(), self.view(2), self.view(52)])
            preview.key(ord('M'))  # 3 -> off while work is in flight
            preview._future.result(timeout=5)
            preview.poll()
            self.assertEqual(preview.mode, 0)
            self.assertIsNone(preview.panel)
            preview.key(ord('M'))
            preview.submit([self.view(), self.view(2), self.view(52)])
            preview._future.result(timeout=5)
            preview.poll()
            self.assertEqual(set(preview.last_cloud.camera_ids), {1,2})
        finally:
            preview.close()


if __name__ == '__main__':
    unittest.main()
