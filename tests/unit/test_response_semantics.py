"""No hidden temporal effects in ideal mode; skin contact is not object-only."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from wowskin_lab.envs.pick_cube import response_configuration
from wowskin_lab.baselines.reskin_contact import ReskinContactResponse
from wowskin_lab.data.recording import require_training_tactile, MissingTactileError

class ResponseSemantics(unittest.TestCase):
    def test_held_display_is_never_a_training_sample(self):
        sim = NS(last_row={'tactile_valid': False, 'tactile_status': 'degraded mesh'},
                 observation=lambda: {'magnetic_ut': np.ones(15)})
        with self.assertRaisesRegex(MissingTactileError, 'Held display values'):
            require_training_tactile(sim)

    def test_positive_volume_warning_continues_fresh_readout(self):
        from collections import deque
        from wowskin_lab.core.physics import TactileSim
        env = object.__new__(TactileSim)
        env.data = NS(time=1.)
        env.cfg = dict(sample_hz=100, filter_tau_s=0., delay_samples=0, noise_ut=0.)
        env.rng = np.random.default_rng(0)
        env.baseline = np.zeros((5, 3))
        env.filtered = np.zeros((5, 3)); env.delay = deque()
        env.drift_rate = np.zeros((5, 3)); env.zero_time = 0.
        with patch.object(env, 'magnetic_field', side_effect=[np.full((5, 3), i) for i in (1., 2., 3., 4.)]) as field:
            initial = env.tactile_readout({'min_volume_ratio': 1., 'passive_min_volume_ratio': 1.})
            np.testing.assert_array_equal(initial['magnetic_ut'], 1.)
            env.data.time = 2.
            warning = env.tactile_readout({'min_volume_ratio': .1, 'passive_min_volume_ratio': 1.})
            self.assertTrue(warning['tactile_valid'])
            self.assertTrue(warning['mesh_quality_warning'])
            np.testing.assert_array_equal(warning['magnetic_ut'], 2.)
            sim = NS(last_row=warning, observation=lambda: warning)
            require_training_tactile(sim)
            right = env.tactile_readout({'min_volume_ratio': 1., 'passive_min_volume_ratio': .1})
            self.assertTrue(right['mesh_quality_warning'])
            np.testing.assert_array_equal(right['magnetic_ut'], 3.)
            recovered = env.tactile_readout({'min_volume_ratio': .9, 'passive_min_volume_ratio': .9})
            self.assertFalse(recovered['mesh_quality_warning'])
            np.testing.assert_array_equal(recovered['magnetic_ut'], 4.)
            self.assertEqual(field.call_count, 4)
            for ratio in (0., -.1, np.nan):
                with self.assertRaisesRegex(RuntimeError, 'inverted/nonfinite'):
                    env.tactile_readout({'min_volume_ratio': ratio})

    def test_ideal_removes_all_temporal_and_noise_effects(self):
        c=response_configuration(NS(ideal=True))
        for name in ('noise_ut','drift_ut_per_s','filter_tau_s','delay_samples'):
            self.assertEqual(c[name],0)

    def test_table_contact_is_not_reported_as_unloaded(self):
        backend=object.__new__(ReskinContactResponse)
        backend.q=np.eye(2);backend.origin=np.array([8.,8.])
        c=NS(flex=np.array([0,-1]),geom=np.array([-1,7]),pos=np.array([0.,0.,.001]))
        sim=NS(model=object(),data=NS(ncon=1,contact=[c]),mount_body=None,
            skin_flex=0,probe_geom=3,contact_geoms={3},rest=np.array([[0.,0.,.002]]),
            skin_vertices_local=lambda: np.array([[0.,0.,.001]]))
        def force(m,d,i,out): out[0]=1.
        with patch('wowskin_lab.baselines.reskin_contact.mujoco.mj_contactForce',side_effect=force):
            query,_,n=backend.query(sim)
        np.testing.assert_allclose(query,[8,8,1])
        self.assertEqual(n,1)

    def test_missing_observations_and_queries_cannot_be_training_samples(self):
        sim=NS(reskin_response=True, last_row={'reskin_query_mm':[8.,8.,1.5]},
               observation=lambda: {'magnetic_ut':np.full(15,np.nan)})
        with self.assertRaises(MissingTactileError):
            require_training_tactile(sim)
        sim.observation=lambda: {'magnetic_ut':np.ones(15)}
        require_training_tactile(sim)
        sim.last_row={'reskin_query_mm':np.full(3,np.nan)}
        with self.assertRaises(MissingTactileError):
            require_training_tactile(sim)

    def test_auto_collection_requires_explicit_backend_before_creating_output(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from wowskin_lab.data.auto_collect import collect
        with TemporaryDirectory() as tmp:
            output=Path(tmp)/'recording'
            with self.assertRaisesRegex(ValueError, 'Select exactly one'):
                collect(output)
            self.assertFalse(output.exists())
