"""Published checkpoint fidelity and physically mounted sensor invariants."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
import torch

from wowskin_lab.envs.franka import FrankaTactileSim
from wowskin_lab.baselines.pollen import CHANNELS, RESOURCE_DIR, PollenPredictor, read_csv


class PollenChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = PollenPredictor()

    def test_identical_to_published_forward_and_scaler(self):
        import json
        cfg = json.loads((RESOURCE_DIR/"model/params.json").read_text())
        spec = importlib.util.spec_from_file_location("pollen_upstream", RESOURCE_DIR/"upstream/lstm.py")
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        params = {k: v for k, v in cfg["model"].items() if k not in ("epochs", "batch_size")}
        model = upstream.LSTMNet(**params)
        model.load_state_dict(self.predictor.model.state_dict())
        model.eval()
        values, _, _ = read_csv(RESOURCE_DIR/"data/apple.csv")
        values = values[::113]
        scaled = torch.tensor((values-self.predictor.mean)/self.predictor.scale, dtype=torch.float32).unsqueeze(1)
        with torch.inference_mode():
            expected = torch.sigmoid(model(scaled)).numpy().reshape(-1)
        np.testing.assert_allclose(self.predictor.predict(values), expected, atol=1e-7)
        np.testing.assert_allclose(self.predictor.predict(values[:3]), [self.predictor.update(x) for x in values[:3]], atol=1e-7)

    def test_named_csv_channels_not_file_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"reordered.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["slip", *reversed(CHANNELS)])
                writer.writeheader()
                writer.writerow(dict(slip=1, **dict(zip(CHANNELS, range(15)))))
            values, labels, _ = read_csv(path)
        np.testing.assert_array_equal(values[0], np.arange(15))
        self.assertEqual(labels.tolist(), [1])
