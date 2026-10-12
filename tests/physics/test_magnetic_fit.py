"""Fitted readout must not alter dynamics, baseline consistency or noise units."""
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from wowskin_lab.core.physics import TactileSim, configuration


class MagneticFitChecks(unittest.TestCase):
    def test_field_scale_keeps_physics_zero_and_raw_delta_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'profile.json'
            gains = np.array([2., 3., 4., 5., 6.])
            path.write_text(json.dumps(dict(schema='wowskin_sim_magnetic_gain_v1', gains=gains.tolist(), fit_magnetization_a_per_m=4000.)))
            cfg = configuration(); cfg.update(noise_ut=0., drift_ut_per_s=0.)
            if tuple(map(int, mujoco.__version__.split('.')[:2])) >= (3, 15):
                cfg.update(skin_dof='trilinear', integrator='discrete', noslip_iterations=0)
            plain = TactileSim(cfg)
            fitted = TactileSim(dict(cfg, magnetic_fit=str(path)))
            # Impose the same small deformation, without changing any material.
            plain.data.qpos[0] += 1e-5
            fitted.data.qpos[:] = plain.data.qpos
            mujoco.mj_forward(plain.model, plain.data)
            mujoco.mj_forward(fitted.model, fitted.data)
            delta = plain.magnetic_field()-plain.baseline
            self.assertGreater(np.linalg.norm(delta), 1e-8)
            np.testing.assert_allclose(fitted.magnetic_field()-fitted.baseline, delta*gains[:, None], atol=1e-10)
            for _ in range(5):
                a, b = plain.sample(), fitted.sample()
                np.testing.assert_array_equal(plain.data.qpos, fitted.data.qpos)
                np.testing.assert_allclose(b['ideal_delta_ut'], a['ideal_delta_ut']*np.repeat(gains, 3), atol=1e-10)
                np.testing.assert_allclose(b['raw_field_ut']-fitted.baseline.ravel(), b['ideal_delta_ut'], atol=1e-10)
                np.testing.assert_allclose(b['unfitted_ideal_delta_ut'], a['ideal_delta_ut'], atol=1e-10)
                np.testing.assert_allclose(b['magnetic_ut'], a['magnetic_ut']*np.repeat(gains, 3), atol=1e-10)
            # Identical random noise is added AFTER the fitted field, not amplified.
            plain.cfg['noise_ut'] = fitted.cfg['noise_ut'] = .5
            # Fresh zero-delay pair makes the noise convention directly observable.
            plain.delay.clear(); fitted.delay.clear()
            a, b = plain.sample(), fitted.sample()
            np.testing.assert_allclose(a['magnetic_ut']-plain.filtered.ravel(), b['magnetic_ut']-fitted.filtered.ravel(), atol=1e-10)

    def test_ambiguous_backend_is_rejected(self):
        cfg = configuration(); cfg.update(magnetic_fit='unused', reskin_model='unused')
        with self.assertRaisesRegex(ValueError, 'not both'):
            TactileSim(cfg)


if __name__=='__main__':
    unittest.main()
