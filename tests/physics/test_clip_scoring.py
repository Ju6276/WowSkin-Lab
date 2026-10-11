"""Scoring transforms on actual clip geometry (synthetic poses, not a rollout)."""
import unittest
from types import SimpleNamespace
import mujoco
import numpy as np
from wowskin_lab.envs.pick_cube import response_configuration
from wowskin_lab.envs.cable import PandaCableEnv, CABLE_COUNT


@unittest.skipUnless(mujoco.__version__ == '3.15.0', 'Run in .venv')
class ClipScoringTests(unittest.TestCase):
    def test_both_clip_layouts_use_rotated_local_tunnel(self):
        for layout in ('single_clip', 'routing'):
            with self.subTest(layout=layout):
                cfg = response_configuration(SimpleNamespace(task='cable', cable_layout=layout,
                                                              skin_dof='trilinear', ideal=True))
                env = PandaCableEnv(cfg, pollen=False)
                self.assertFalse(env.evaluate_clip()['clip_success'])
                clip = env.model.body('cclip_0').id
                rotation = env.data.xmat[clip].reshape(3, 3)
                p = np.array([env.clip_score.x_bounds.mean(), env.clip_score.y_bounds.mean(),
                              env.clip_score.z_bounds.mean()])
                gid = env.model.geom(f'wire_G{CABLE_COUNT-2}').id
                # Test fixture only: put one capsule lengthwise under the roof.
                # Runtime scoring only reads positions; it never does this.
                env.data.geom_xpos[gid] = env.data.xpos[clip]+rotation@p
                env.data.geom_xmat[gid] = (rotation@np.array([[0.,0.,1.], [1.,0.,0.], [0.,1.,0.]])).ravel()
                self.assertTrue(env.evaluate_clip()['clip_inside'])
                env.data.time += .31
                self.assertTrue(env.evaluate_clip()['clip_success'])
                env.clip_score.reset()
                # The same capsule above the top surface must fail.
                env.data.geom_xpos[gid] += rotation@np.array([0.,0.,.02])
                self.assertFalse(env.evaluate_clip()['clip_inside'])
                env.reset()
                self.assertFalse(env.clip_score.success)
