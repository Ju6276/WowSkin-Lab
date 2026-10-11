"""Operator annotations must never enter policy RGB/depth or physics state."""
import unittest
from types import SimpleNamespace

import mujoco
import numpy as np

from wowskin_lab.data.cameras import active_cameras, capture_options
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, response_configuration
from wowskin_lab.envs.cable import PandaCableEnv
from wowskin_lab.envs.feeding import PandaFeedingEnv
from wowskin_lab.ui.task_guides import TaskGuides, goal_regions
from wowskin_lab.ui.teleop_hud import render_operator_panel


@unittest.skipUnless(mujoco.__version__ == '3.15.0', 'Run in .venv')
class OperatorGuideTests(unittest.TestCase):
    def test_guides_leave_all_three_rgb_depth_and_physics_unchanged(self):
        for task, cls in (('pick_cube', PandaPickCubeTactileEnv),
                          ('cable', PandaCableEnv), ('feeding', PandaFeedingEnv)):
            with self.subTest(task=task):
                cfg = response_configuration(SimpleNamespace(task=task, skin_dof='trilinear',
                    cable_layout='single_clip', ideal=True))
                env = cls(cfg, pollen=False)
                m, d = env.model, env.data
                scene = mujoco.MjvScene(m, maxgeom=100)
                guides = TaskGuides(env)
                counts = (m.ngeom, m.nsite)
                state = d.qpos.copy(), d.qvel.copy(), d.time
                renderer = mujoco.Renderer(m, width=160, height=120)
                try:
                    def capture():
                        frames = []
                        for camera in active_cameras(env).values():
                            renderer.disable_depth_rendering()
                            renderer.update_scene(d, camera=camera, scene_option=capture_options())
                            frames.append(renderer.render().copy())
                            renderer.enable_depth_rendering()
                            renderer.update_scene(d, camera=camera, scene_option=capture_options())
                            frames.append(renderer.render().copy())
                        renderer.disable_depth_rendering()
                        return frames
                    before = capture()
                    guides.append(scene, m, d, dict(success=False))
                    self.assertGreater(scene.ngeom, 10)
                    for old, new in zip(before, capture()):
                        if old.dtype == np.uint8:
                            # Native macOS OpenGL varies by one 8-bit level on
                            # a few antialiased pixels even without any guides.
                            np.testing.assert_allclose(old, new, atol=1, rtol=0)
                        else:
                            np.testing.assert_array_equal(old, new)
                    self.assertEqual(len(before), 6)
                    self.assertEqual(counts, (m.ngeom, m.nsite))
                    np.testing.assert_array_equal(state[0], d.qpos)
                    np.testing.assert_array_equal(state[1], d.qvel)
                    self.assertEqual(state[2], d.time)
                    # Exercise the actual renderer for the operator-only geoms.
                    renderer.update_scene(d)
                    initial = renderer.scene.ngeom
                    guides.append(renderer.scene, m, d, dict(success=True))
                    self.assertGreater(renderer.scene.ngeom, initial)
                    self.assertTrue(np.isfinite(renderer.render()).all())
                    if task == 'feeding':
                        self.assertEqual(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE,
                                                         'feeding_delivery_target'), -1)
                        region = goal_regions(env, m, d, {})[0]
                        self.assertTrue(region.ellipsoid)
                        np.testing.assert_allclose(region.half, env.delivery.config['target_ellipsoid_radii_m'])
                finally:
                    renderer.close()

    def test_success_text_is_bright_green(self):
        panel = render_operator_panel(['Task', 'Record'],
            ['SUCCESS: delivered', 'IDLE | B starts recording'], success=True)
        green = (panel[..., 1] > 230) & (panel[..., 0] < 110) & (panel[..., 2] < 165)
        self.assertGreater(green.sum(), 100)


if __name__ == '__main__':
    unittest.main()
