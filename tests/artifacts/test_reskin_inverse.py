"""Cable scene isolation and cross-model channel contract regression checks."""
import json
import hashlib
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from wowskin_lab.paths import ROOT


@unittest.skipUnless((ROOT/'outputs/reskin_inverse_mapped/model.json').exists() and
                     (ROOT/'outputs/reskin_forward/model.json').exists() and
                     (ROOT/'outputs/reskin_forward_mapped/model.json').exists(),
                     'Optional inverse/forward checkpoints not present')
class InverseChecks(unittest.TestCase):
    @unittest.skipUnless((ROOT/'outputs/reskin_inverse_mapped/model.json').exists(), 'Optional inverse model not trained')
    def test_saved_prediction_and_original_checkpoint_permutation(self):
        from wowskin_lab.baselines.reskin_inverse import InverseModel, predict_batches
        from wowskin_lab.baselines.reskin_forward import read_data
        local, official = InverseModel(), InverseModel(published=True)
        with np.load(ROOT/'outputs/reskin_inverse_mapped/heldout_predictions.npz') as data:
            np.testing.assert_allclose(local.predict(data['input_delta_ut'][:8]),data['predicted_xyz_mm'][:8],atol=1e-5)
        _, source, _ = read_data()
        # Only group permutation; published prediction must be exactly the
        # same after undoing it, including signed XYZ values.
        expected = predict_batches(official.model, source[:10])
        actual = official.predict(source[:10,official.indices])
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_forward_original_and_mapped_archives_are_preserved(self):
        from wowskin_lab.baselines.reskin_forward import OUTPUT,MAPPED_OUTPUT
        self.assertNotEqual(OUTPUT,MAPPED_OUTPUT)
        self.assertIsNone(json.loads((OUTPUT/'model.json').read_text()).get('mapping'))
        self.assertEqual(json.loads((MAPPED_OUTPUT/'model.json').read_text())['mapping']['source_groups_for_wowskin'],[0,1,3,2,4])


@unittest.skipUnless((ROOT/'outputs/diagnostics/pollen_pseudolabel_augmentation/report.json').exists(),
                     'Optional pseudo-label experiment not run')
class PseudoLabelArtifactChecks(unittest.TestCase):
    def test_baseline_and_public_data_preserved_and_no_heldout_training(self):
        from wowskin_lab.baselines.reskin_forward import MAPPED_OUTPUT
        from wowskin_lab.baselines.pollen import RESOURCE_DIR
        output=ROOT/'outputs/diagnostics/pollen_pseudolabel_augmentation'
        report=json.loads((output/'report.json').read_text())
        for path,key in [(MAPPED_OUTPUT/'model.safetensors','baseline_weights_sha256'),
                         (RESOURCE_DIR/'data/cable.csv','pollen_file_sha256')]:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),report[key])
        for name,branch in report['branches'].items():
            with np.load(output/name/'pollen_predictions.npz',allow_pickle=False) as data:
                train,held=data['used_for_training'],data['heldout']
                self.assertFalse(np.any(train & held))
                self.assertEqual(int(train.sum()),branch['accepted_pseudo_count'])
                np.testing.assert_array_equal(np.flatnonzero(held),np.arange(3856,5011))
                self.assertEqual(int((held & (data['slip']==1)).sum()),74)
                q=data['estimated_xyz_mm'][train]
                self.assertTrue(np.isfinite(q).all())
                self.assertTrue(((q[:,:2]>=0)&(q[:,:2]<=16)).all())
                self.assertTrue(((q[:,2]>0)&(q[:,2]<=2)).all())

    def test_saved_models_reproduce_report_without_claiming_measured_pseudo_depths(self):
        from scripts.augment_pollen_response import predict
        from wowskin_lab.baselines.reskin_forward import MAPPED_OUTPUT,ResponseModel
        baseline=ResponseModel(MAPPED_OUTPUT)
        output=ROOT/'outputs/diagnostics/pollen_pseudolabel_augmentation'
        for name in ('official_teacher','train_split_teacher'):
            model=ResponseModel(output/name)
            for key in ('x_mean','x_scale','y_mean','y_scale','known_xy_mm','measured_depths_mm','mapping'):
                self.assertEqual(model.meta[key],baseline.meta[key])
            self.assertEqual(model.meta['augmentation']['independent_xyz_labels'],0)
            report=json.loads((output/name/'report.json').read_text())
            with np.load(output/name/'pollen_predictions.npz',allow_pickle=False) as data:
                predictions=predict(model.model,model.meta,data['estimated_xyz_mm'])
                np.testing.assert_allclose(predictions,data['after_delta_ut'],atol=1e-5)
                mask=data['heldout'] & (data['slip']==1)
                rmse=float(np.sqrt(np.mean((predictions[mask]-data['input_magnetic_ut'][mask])**2)))
                self.assertAlmostEqual(rmse,report['pollen_reconstruction']['heldout_slip']['after']['rmse_ut'],places=5)

    @unittest.skipUnless((ROOT/'outputs/diagnostics/pollen_pseudolabel_augmentation/source_only_control/model.json').exists(),
                         'Optional source-only control not run')
    def test_source_only_control_saved_metrics(self):
        from scripts.augment_pollen_response import predict
        from wowskin_lab.baselines.reskin_forward import ResponseModel
        output=ROOT/'outputs/diagnostics/pollen_pseudolabel_augmentation'
        model=ResponseModel(output/'source_only_control')
        report=json.loads((output/'source_only_control/report.json').read_text())
        self.assertEqual(model.meta['continued_training']['actual_pseudo_loss_weight'],0.)
        for name in ('official_teacher','train_split_teacher'):
            with np.load(output/name/'pollen_predictions.npz',allow_pickle=False) as data:
                mask=data['heldout'] & (data['slip']==1)
                # Keep full evaluation batch to reproduce float32 results.
                pred=predict(model.model,model.meta,data['estimated_xyz_mm'])
                rmse=float(np.sqrt(np.mean((pred[mask]-data['input_magnetic_ut'][mask])**2)))
                self.assertAlmostEqual(rmse,report['pollen_reconstruction'][name]['heldout_slip']['rmse_ut'],places=5)
