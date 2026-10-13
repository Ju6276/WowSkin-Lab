from collections import deque
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from wowskin_lab.envs.feeding import PandaFeedingEnv
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv


class FeedingFallbackTests(unittest.TestCase):
    def test_failed_cg_sample_restores_state_and_observation_history(self):
        # Use a tiny real MjModel; no slow scene construction for retry logic.
        env = object.__new__(PandaFeedingEnv)
        env.is_optical = False
        env.cfg = {"skin_dof": "full"}
        env.model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom size=".01"/></body></worldbody></mujoco>')
        env.data = mujoco.MjData(env.model)
        env.model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        env.rng = np.random.default_rng(1)
        env.filtered = np.zeros((5,3));env.delay = deque([np.ones((5,3))])
        env.cg_fallbacks = 0
        for name in ('head_force_n','head_interval_peak_n','head_peak_n','head_contact_impulse_ns'):
            setattr(env,name,0.)
        env.head_net_force = np.zeros(3)
        saved = env.data.qpos.copy()
        calls = []
        def sample(controller):
            calls.append(1)
            if len(calls) == 1:
                env.data.qpos[:] = 123
                env.data.time = 9.
                env.filtered[:] = 9
                env.head_peak_n = 999
                env.delay.clear()
                raise RuntimeError('MuJoCo instability/warning; refusing to export invalid data')
            np.testing.assert_array_equal(env.data.qpos,saved)
            np.testing.assert_array_equal(env.filtered,0.)
            self.assertEqual(env.data.time,0.)
            self.assertEqual(env.head_peak_n,0.)
            self.assertEqual(len(env.delay),1)
            self.assertEqual(env.model.opt.solver,mujoco.mjtSolver.mjSOL_NEWTON)
            return {'valid':True}
        with patch.object(PandaPickCubeTactileEnv,'sample',side_effect=sample):
            self.assertTrue(env.sample()['valid'])
        self.assertEqual(env.cg_fallbacks,1)

    def test_optical_failure_does_not_switch_the_source_solver(self):
        env = object.__new__(PandaFeedingEnv)
        env.is_optical = True
        env.model = mujoco.MjModel.from_xml_string('<mujoco/>')
        env.model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        with patch.object(PandaPickCubeTactileEnv, 'sample', side_effect=RuntimeError('MuJoCo instability/warning')) as sample:
            with self.assertRaisesRegex(RuntimeError, 'MuJoCo instability'):
                env.sample()
        sample.assert_called_once_with(None)
        self.assertEqual(env.model.opt.solver, mujoco.mjtSolver.mjSOL_CG)
