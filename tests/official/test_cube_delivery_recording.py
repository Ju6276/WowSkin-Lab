"""Real two-box pick/place, reviewed save and three-view ViSk conversion."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import mujoco

from wowskin_lab.data.auto_collect import collect
from wowskin_lab.data.dataset_export import export_dataset
from wowskin_lab.data.lerobot_recording import validate_dataset
from wowskin_lab.data.visk_processing import process_visk_sim
from wowskin_lab.learning.official import prepare
from wowskin_lab.paths import ROOT


@unittest.skipUnless(mujoco.__version__ == '3.15.0', 'Requires .venv-reduced')
class CubeDeliveryRecordingTests(unittest.TestCase):
    def test_two_box_success_recording_and_original_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = collect(root/'capture', episodes=1, ideal=True,
                             magnetic_fit=ROOT/'configs/magnetic_fit_pollen_nonslip.json')
            self.assertEqual(report['successes'], 1, report)
            episode = Path(report['episodes'][0]['dataset_episode'])
            meta = json.loads((episode/'metadata.json').read_text())
            self.assertTrue(meta['task_evaluation']['success'])
            self.assertTrue(meta['controller_contract']['cartesian_limits_enabled'])
            with h5py.File(episode/'measurements.h5') as f:
                self.assertTrue(f['evaluation/cube_success'][-1])
                self.assertEqual(f['sensor_states'].shape[1], 15)
                self.assertEqual(f['camera_base_to_cv'].shape[1], 3)
            checked = validate_dataset(root/'capture/pick_cube/lerobot')
            self.assertTrue(checked['official_loader_passed'])
            export_dataset(root/'capture/pick_cube', root/'export', 'visk')
            prepare('visk', root/'workspace', root/'export', 'pick_cube')
            processed = process_visk_sim(root/'workspace')
            self.assertTrue(processed['validation']['command_label_check']['next_command_matches'])
            self.assertEqual(json.loads((root/'workspace/manifest.json').read_text())['camera_ids'], [1, 2, 52])


if __name__ == '__main__':
    unittest.main()
