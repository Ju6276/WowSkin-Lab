"""Pinned Pollen resources and unmodified published-model inference on CPU.

Network structure follows pollen-robotics/anyskin-slip-detection (Apache-2.0).
The accompanying LICENSE is downloaded with the original resources.
Public-data replay is not an independent held-out evaluation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file

from wowskin_lab.paths import ROOT
RESOURCE_DIR = ROOT / "assets/pollen"
MODEL_REVISION = "775924649e22452b983ada6c4b59b6f928de0241"
DATA_REVISION = "d9f5e315006482635a2effa34488715871d55c13"
CODE_REVISION = "24a3728c35985def720ff292a8a45ef7e8662a64"
CHANNELS = [f"mag{i}_{axis}" for i in range(1, 6) for axis in "xyz"]
OBJECTS = "apple ball book bottle cable carambole cork cream cup elastic foam foam_filter gluestick grappe no_slip socks tissues".split()


def download(destination=RESOURCE_DIR):
    destination = Path(destination)
    model_base = f"https://huggingface.co/pollen-robotics/anyskin-slip-detection/resolve/{MODEL_REVISION}"
    data_base = f"https://huggingface.co/datasets/pollen-robotics/anyskin_slip_detection/resolve/{DATA_REVISION}"
    code_base = f"https://raw.githubusercontent.com/pollen-robotics/anyskin-slip-detection/{CODE_REVISION}"
    sources = {"model/model.safetensors": model_base+"/model.safetensors",
               "model/params.json": model_base+"/params.json",
               "model/README.md": model_base+"/README.md",
               "data/README.md": data_base+"/README.md",
               "LICENSE": code_base+"/LICENSE",
               "upstream/lstm.py": code_base+"/src/lstm.py",
               "upstream/pred.py": code_base+"/src/pred.py",
               "upstream/train.py": code_base+"/src/train.py"}
    sources.update({f"data/{name}.csv": f"{data_base}/data/{name}.csv" for name in OBJECTS})
    old_manifest = destination/"manifest.json"
    known = json.loads(old_manifest.read_text()).get("files", {}) if old_manifest.exists() else {}

    def fetch(item):
        relative, url = item
        path = destination/relative
        if path.exists() and relative in known:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if known[relative]["url"] == url and digest == known[relative]["sha256"]:
                return relative, known[relative]
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        temporary = path.with_name(path.name+".part")
        temporary.write_bytes(content)
        temporary.replace(path)
        print(f"downloaded {relative} ({len(content)} bytes)", flush=True)
        return relative, dict(url=url, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = dict(pool.map(fetch, sources.items()))
    manifest = dict(model_revision=MODEL_REVISION, dataset_revision=DATA_REVISION,
                    code_revision=CODE_REVISION, license="Apache-2.0", files=files)
    (destination/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


class PollenNet(nn.Module):
    """Same parameter names and forward pass as the published LSTMNet."""
    def __init__(self, cfg):
        super().__init__()
        self.lstm = nn.LSTM(cfg["input_size"], cfg["lstm_hidden_size"],
                            cfg["nlayers"], batch_first=True)
        self.fc = nn.Linear(cfg["lstm_hidden_size"], cfg["hidden_size"])
        self.out = nn.Linear(cfg["hidden_size"], cfg["output_size"])

    def forward(self, x):
        # Implicit zero initial state is equivalent to upstream h0/c0.
        values, _ = self.lstm(x)
        return self.out(self.fc(values[:, -1]))


class PollenPredictor:
    def __init__(self, resource_dir=RESOURCE_DIR):
        root = Path(resource_dir)
        if not (root/"model/model.safetensors").is_file():
            raise FileNotFoundError("Run .venv/bin/python pollen.py download first")
        cfg = json.loads((root/"model/params.json").read_text())
        if cfg["model"]["input_size"] != 15:
            raise ValueError("Expected the published 15-channel Pollen model")
        self.mean = np.asarray(cfg["scaler"]["mean"], dtype=np.float64)
        self.scale = np.asarray(cfg["scaler"]["scale"], dtype=np.float64)
        if self.mean.shape != (15,) or self.scale.shape != (15,) or np.any(self.scale <= 0):
            raise ValueError("Invalid paired scaler")
        torch.set_num_threads(4)
        self.model = PollenNet(cfg["model"])
        self.model.load_state_dict(load_file(str(root/"model/model.safetensors")), strict=True)
        self.model.eval()

    def predict(self, magnetic, batch_size=2048):
        values = np.asarray(magnetic, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 15 or not np.isfinite(values).all():
            raise ValueError("Expected finite magnetic data of shape (N, 15), in published channel order")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        result = []
        with torch.inference_mode():
            for start in range(0, len(values), batch_size):
                normalized = (values[start:start+batch_size]-self.mean)/self.scale
                tensor = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(1).to(next(self.model.parameters()).device)
                result.append(torch.sigmoid(self.model(tensor)).cpu().numpy().reshape(-1))
        return np.concatenate(result) if result else np.empty(0, dtype=np.float32)

    def update(self, magnetic):
        # Original published inference has a one-frame sequence; do not add a
        # 20-frame window or carry hidden state from the local synthetic model.
        return float(self.predict(np.asarray(magnetic).reshape(1, 15))[0])


def read_csv(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(CHANNELS+["slip"]).issubset(reader.fieldnames):
            raise ValueError(f"Missing named magnetic columns or slip label: {path}")
        rows, labels, timestamps = [], [], []
        for row in reader:
            rows.append([float(row[c]) for c in CHANNELS])
            label = int(row["slip"])
            if label not in (0, 1):
                raise ValueError("Slip labels must be 0 or 1")
            labels.append(label)
            timestamps.append(row.get("log_time", ""))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return np.asarray(rows), np.asarray(labels, dtype=np.int8), timestamps


def metrics(labels, probabilities):
    actual = np.asarray(labels).astype(bool)
    predicted = np.asarray(probabilities) > .5
    tp, tn = int((actual & predicted).sum()), int((~actual & ~predicted).sum())
    fp, fn = int((~actual & predicted).sum()), int((actual & ~predicted).sum())
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else None
    specificity = tn/(tn+fp) if tn+fp else None
    return dict(samples=len(actual), slip_samples=int(actual.sum()),
                accuracy=(tp+tn)/len(actual), precision=precision, recall=recall,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                balanced_accuracy=(recall+specificity)/2 if recall is not None and specificity is not None else None,
                confusion=dict(tn=tn, fp=fp, fn=fn, tp=tp))


def evaluate(resource_dir=RESOURCE_DIR, output=ROOT/"outputs/pollen_public"):
    root, output = Path(resource_dir), Path(output)
    predictor = PollenPredictor(root)
    output.mkdir(parents=True, exist_ok=True)
    reports, all_labels, all_probabilities = {}, [], []
    for name in OBJECTS:
        values, labels, timestamps = read_csv(root/f"data/{name}.csv")
        probabilities = predictor.predict(values)
        reports[name] = metrics(labels, probabilities)
        reports[name]["magnetic_min"] = values.min(axis=0).tolist()
        reports[name]["magnetic_max"] = values.max(axis=0).tolist()
        with (output/f"{name}_predictions.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["log_time", "slip_label", "pollen_probability", "prediction"])
            writer.writerows(zip(timestamps, labels, probabilities, (probabilities > .5).astype(int)))
        all_labels.extend(labels)
        all_probabilities.extend(probabilities)
        print(f"{name}: n={len(labels)} accuracy={reports[name]['accuracy']:.4f} F1={reports[name]['f1']:.4f}", flush=True)
    report = dict(status="PUBLIC DATA REPLAY; overlaps upstream training data; not independent test accuracy",
                  model_revision=MODEL_REVISION, dataset_revision=DATA_REVISION,
                  model="published Pollen safetensors, unchanged", channels=CHANNELS,
                  preprocessing="published mean/scale; sequence length 1; threshold > 0.5; no refit",
                  overall=metrics(all_labels, all_probabilities), files=reports)
    (output/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.bar(list(reports), [r["accuracy"] for r in reports.values()])
    ax.tick_params(axis="x", rotation=55)
    ax.set(ylim=(0, 1.05), ylabel="Accuracy on published records",
           title="Pollen published checkpoint / public data replay\nTraining overlap: not a held-out or WowSkin transfer result")
    fig.savefig(output/"replay_accuracy.png", dpi=150)
    plt.close(fig)
    print(json.dumps(report["overall"], indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "evaluate"])
    parser.add_argument("--resources", type=Path, default=RESOURCE_DIR)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/pollen_public")
    args = parser.parse_args()
    if args.command == "download":
        download(args.resources)
    else:
        evaluate(args.resources, args.output)


if __name__ == "__main__":
    main()
