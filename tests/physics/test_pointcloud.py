"""Validate reconstructed surfaces against a known plane, with a moving camera."""
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from wowskin_lab.data.cameras import add_camera
from wowskin_lab.data.pointcloud import RGBDView, backproject, fuse_rgbd
from wowskin_lab.data.robot_state import camera_from_world, world_from_base


class RenderedPointCloudTests(unittest.TestCase):
    def test_three_view_alignment_before_and_after_wrist_motion(self):
        xml = ET.fromstring('''<mujoco><worldbody>
          <geom name="table" type="plane" size="2 2 .01" rgba=".5 .6 .7 1"/>
          <body name="link0" pos=".12 -.2 .1" euler="0 0 25"/>
          <body name="wrist" pos=".2 0 .2"><joint type="hinge" axis="0 0 1"/>
            <geom type="sphere" size=".005" mass="1"/></body>
        </worldbody></mujoco>''')
        world = xml.find('worldbody')
        add_camera(world, 'c1', [.4,-.4,.6], [0,0,0])
        add_camera(world, 'c2', [-.4,.4,.6], [0,0,0])
        add_camera(world.find("body[@name='wrist']"), 'c52', [.1,.1,.4], [0,0,-.2])
        model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
        data = mujoco.MjData(model)
        env = SimpleNamespace(model=model, data=data)
        renderer = mujoco.Renderer(model, width=320, height=240)
        previous = None
        try:
            for angle in (0., .9):
                data.qpos[0] = angle
                mujoco.mj_forward(model, data)
                before = data.qpos.copy()
                views = []
                for camera in (1,2,52):
                    cid = model.camera(f'c{camera}').id
                    renderer.disable_depth_rendering()
                    renderer.update_scene(data, camera=cid)
                    rgb = renderer.render().copy()
                    renderer.enable_depth_rendering()
                    depth = renderer.render().copy()
                    f = 120/np.tan(np.deg2rad(model.cam_fovy[cid])/2)
                    k = np.array([[f,0,159.5],[0,f,119.5],[0,0,1.]])
                    ext = camera_from_world(env,cid) @ world_from_base(env)
                    views.append(RGBDView(camera, angle, rgb, depth, k, ext,
                        model.vis.map.znear*model.stat.extent, model.vis.map.zfar*model.stat.extent))
                    points, _ = backproject(views[-1], 3)
                    # Transform to world to compare the table surface, excluding
                    # the tiny camera-body sphere and far plane.
                    base = world_from_base(env)
                    world_points = np.einsum('ij,nj->ni', base[:3,:3], points)+base[:3,3]
                    table = np.linalg.norm(world_points[:,:2]-[.2,0], axis=1) > .03
                    self.assertLess(np.quantile(abs(world_points[table,2]), .95), .003)
                if previous is not None:
                    np.testing.assert_allclose(views[0].base_to_camera, previous[0].base_to_camera)
                    self.assertGreater(np.max(abs(views[2].base_to_camera-previous[2].base_to_camera)), .1)
                cloud = fuse_rgbd(views)
                self.assertEqual(set(cloud.camera_ids), {1,2,52})
                np.testing.assert_array_equal(before, data.qpos)
                previous = views
        finally:
            renderer.close()

    @unittest.skipUnless(mujoco.__version__ == '3.15.0', 'Use .venv for task scenes')
    def test_live_preview_all_tasks_preserves_physics_and_rgb(self):
        from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, response_configuration
        from wowskin_lab.envs.cable import PandaCableEnv
        from wowskin_lab.envs.feeding import PandaFeedingEnv
        from wowskin_lab.ui.camera_preview import CameraPreview
        from wowskin_lab.ui.physics_worker import PhysicsWorker
        for task, cls in (('pick_cube', PandaPickCubeTactileEnv), ('cable', PandaCableEnv), ('feeding', PandaFeedingEnv)):
            with self.subTest(task=task):
                env = cls(response_configuration(SimpleNamespace(task=task, skin_dof='trilinear', ideal=True)), pollen=False)
                model, data = PhysicsWorker.display(env)
                frames = {}
                hud = SimpleNamespace(show=lambda handle, row, **kw: frames.update(kw))
                handle = SimpleNamespace(is_running=lambda: True)
                state = data.qpos.copy(), data.qvel.copy(), data.time
                with CameraPreview(model, data, hud, pointcloud=dict(enabled=True)) as preview:
                    preview.show(handle, row={})
                    rgb_before = [frame.copy() for _, frame in frames['cameras']]
                    preview.cloud._future.result(timeout=10)
                    preview.cloud.poll()
                    self.assertEqual(set(preview.cloud.last_cloud.camera_ids), {1,2,52})
                    self.assertEqual(preview.cloud.last_cloud.timestamp, data.time)
                    self.assertEqual(preview.cloud.panel.shape, (460,720,3))
                    self.assertGreater(len(preview.cloud.last_cloud.xyz), 1000)
                    preview.key(ord('M'))  # turn off, confirm RGB is unchanged
                    preview.show(handle, row={})
                    for old, (_, new) in zip(rgb_before, frames['cameras']):
                        np.testing.assert_allclose(old, new, atol=1, rtol=0)
                np.testing.assert_array_equal(state[0], data.qpos)
                np.testing.assert_array_equal(state[1], data.qvel)
                self.assertEqual(state[2], data.time)


if __name__ == '__main__':
    unittest.main()
