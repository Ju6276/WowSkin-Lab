"""Check saved forward response and refusal of unsupported contact queries."""
import json
import unittest
import numpy as np
from wowskin_lab.baselines.reskin_forward import ResponseModel, OUTPUT


@unittest.skipUnless((OUTPUT/'model.json').exists(), 'Train the optional 3D Cal model first')
class ReskinForwardChecks(unittest.TestCase):
    def test_saved_predictions_and_spatial_split(self):
        model = ResponseModel()
        with np.load(OUTPUT/'heldout_predictions.npz') as data:
            np.testing.assert_allclose(model.predict(data['input_mm'][0]), data['predicted_delta_ut'][0], atol=.001)
        report = json.loads((OUTPUT/'report.json').read_text())
        groups = [set(map(tuple, report['split_xy'][k])) for k in ('train','validation','test')]
        self.assertTrue(all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i)))
        self.assertEqual(sum(map(len, groups)), 189)

    def test_unsupported_input_is_not_silently_extrapolated(self):
        model = ResponseModel()
        for query in ([8,8,0], [8,8,3], [100,100,1.5], [8,8,1.75], [8,8,float('nan')]):
            with self.assertRaises(ValueError):
                model.predict(query)
        self.assertEqual(model.predict([8,8,1.5]).shape, (15,))
        self.assertTrue(np.isfinite(model.predict([8,8,1.75], allow_depth_interpolation=True)).all())


class MappedResponseChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from wowskin_lab.baselines.reskin_forward import MAPPED_OUTPUT
        if not (MAPPED_OUTPUT/'model.json').exists():
            raise unittest.SkipTest('Mapped checkpoint has not been trained')
        cls.path = MAPPED_OUTPUT

    def test_mapped_targets_are_original_measured_vectors(self):
        from wowskin_lab.baselines.reskin_forward import read_data
        model = ResponseModel(self.path)
        x, y, _ = read_data()
        permutation = model.meta['mapping']['source_groups_for_wowskin']
        data = np.load(self.path/'heldout_predictions.npz')
        query = data['input_mm'][0]
        candidates = y[np.all(x == query, axis=1)].reshape(-1,5,3)[:,permutation].reshape(-1,15)
        self.assertTrue(any(np.allclose(data['observed_delta_ut'][0], row) for row in candidates))
        np.testing.assert_allclose(model.predict(query), data['predicted_delta_ut'][0], atol=.002)

    def test_cube_proxy_is_blocked_unless_preview_is_explicit(self):
        from types import SimpleNamespace
        import mujoco
        from wowskin_lab.baselines.reskin_contact import ReskinContactResponse
        backend = ReskinContactResponse(self.path)
        backend.query = lambda sim: (np.array([8.,8.,.02]), 20., 5)
        sim = SimpleNamespace(model=SimpleNamespace(geom_type=[mujoco.mjtGeom.mjGEOM_BOX]), probe_geom=0)
        blocked = backend.evaluate(sim)
        self.assertTrue(np.isnan(blocked['magnetic_ut']).all())
        self.assertFalse(blocked['reskin_prediction_emitted'])
        self.assertIn('depth_outside', blocked['reskin_status'])
        self.assertIn('contact_shape_transfer_unverified', blocked['reskin_status'])
        backend.extrapolate = True
        preview = backend.evaluate(sim)
        np.testing.assert_allclose(preview['magnetic_ut'], backend.model.predict([8,8,.02], allow_extrapolation=True))
        self.assertFalse(preview['reskin_in_training_domain'])
        self.assertFalse(preview['reskin_transfer_validated'])
        backend.query = lambda sim: (None, 0., 0)
        unloaded = backend.evaluate(sim)
        self.assertTrue(np.isnan(unloaded['magnetic_ut']).all())
        self.assertTrue(unloaded['reskin_status'].startswith('UNAVAILABLE:'))

    def test_hud_marks_invalid_predictions_without_nan_crash(self):
        from types import SimpleNamespace
        from wowskin_lab.ui.tactile import TactileHUD
        sim = SimpleNamespace(cfg={'reskin_model':str(self.path)})
        hud = TactileHUD(sim)
        row = dict(magnetic_ut=np.full(15, np.nan), time_s=1., reskin_query_mm=[8,8,.02], reskin_status='BLOCKED:depth')
        pixels = hud.render(row)
        self.assertEqual(pixels.shape, (558,400,3))
        self.assertTrue(np.isnan(row['magnetic_ut']).all())

    def test_auto_collection_rejects_unloaded_network_before_creating_files(self):
        import mujoco
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from wowskin_lab.data.auto_collect import collect
        from wowskin_lab.data.recording import MissingTactileError
        dof = 'trilinear' if mujoco.__version__ == '3.15.0' else 'full'
        with TemporaryDirectory() as temporary:
            output = Path(temporary)/'episode'
            with self.assertRaises(MissingTactileError):
                collect(output, reskin_model=self.path, reskin_extrapolate=True, skin_dof=dof)
            self.assertFalse(output.exists())

    def test_recording_does_not_mix_dipole_raw_with_empirical_delta(self):
        from types import SimpleNamespace
        from wowskin_lab.baselines.reskin_contact import ReskinContactResponse
        from wowskin_lab.data.recording import magnetic_baseline, magnetic_metadata
        response = ReskinContactResponse(self.path, extrapolate=True)
        baseline = np.array(response.model.meta['baseline_ut'])
        delta = response.model.predict([8,8,1.5])
        row = dict(raw_field_ut=baseline+delta, normal_force_n=.35, reskin_query_mm=[8,8,1.5],
                   reskin_in_training_domain=True)
        env = SimpleNamespace(reskin_response=response, last_row=row, cfg={'sample_hz':100},
                              baseline=np.zeros(15), tcp_site=0, success=False,
                              data=SimpleNamespace(time=1.,site_xpos=np.zeros((1,3)),site_xmat=np.eye(3).reshape(1,9),qpos=np.zeros(7),qvel=np.zeros(7)),
                              gripper_fraction=lambda **kwargs: .5, observation=lambda: {'magnetic_ut':delta},
                              magnetic_field=lambda: np.full(15,99999.))
        from wowskin_lab.data.ftf_recording import FTFSimRecorder
        ftf = FTFSimRecorder.__new__(FTFSimRecorder)
        ftf.env = env
        np.testing.assert_allclose(ftf._field(), baseline+delta, atol=.0002)
        np.testing.assert_allclose(magnetic_baseline(env), baseline)
        self.assertEqual(magnetic_metadata(env)['magnetic_source'], '3dcal_local_mlp')
        self.assertFalse(magnetic_metadata(env)['real_transfer_validated'])
