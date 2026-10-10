"""Pollen-style LSTM; temporal windows and episode-disjoint evaluation.

Only the 15 magnetic channels are model inputs. Privileged simulation labels,
forces, velocities, timestamps and episode phase are excluded from features.
"""
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from wowskin_lab.core.physics import CHANNELS


class SlipLSTM(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(15, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # No state is carried between windows or episodes.
        values, _ = self.lstm(x)
        return self.out(self.fc(values[:, -1])).squeeze(-1)


def windows(episodes, length, mean, scale):
    features, labels, groups = [], [], []
    for path in episodes:
        with np.load(path, allow_pickle=False) as data:
            x = (data["magnetic_ut"]-mean)/scale
            y = data["slip"]
            for end in range(length-1, len(x)):
                features.append(x[end-length+1:end+1])
                labels.append(y[end])
                groups.append(path.stem)
    if not features:
        raise ValueError("No complete windows. Generate longer episodes or reduce --window.")
    return np.asarray(features, np.float32), np.asarray(labels, np.float32), groups


def scores(labels, probabilities):
    truth = np.asarray(labels).astype(bool)
    pred = np.asarray(probabilities) >= .5
    tp, tn = int((pred & truth).sum()), int((~pred & ~truth).sum())
    fp, fn = int((pred & ~truth).sum()), int((~pred & truth).sum())
    precision = tp/max(tp+fp, 1)
    recall = tp/max(tp+fn, 1)
    return dict(samples=len(truth), accuracy=(tp+tn)/len(truth), precision=precision,
                recall=recall, f1=2*precision*recall/max(precision+recall, 1e-12),
                balanced_accuracy=.5*(recall+tn/max(tn+fp, 1)),
                confusion={"tn": tn, "fp": fp, "fn": fn, "tp": tp})


def train(dataset, output, window=20, epochs=25, seed=0):
    if window < 1 or epochs < 1:
        raise ValueError("window and epochs must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    root, output = Path(dataset), Path(output)
    manifest = json.loads((root/"manifest.json").read_text())
    groups = {k: [root/p for p in manifest["splits"][k]] for k in ("train", "val", "test")}
    if any(not v for v in groups.values()):
        raise ValueError("Each split needs at least one full episode")
    sets = [set(v) for v in groups.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Episode leakage between splits")
    # Fit normalization on training episodes only, before constructing windows.
    train_rows = []
    for p in groups["train"]:
        with np.load(p, allow_pickle=False) as d:
            train_rows.append(d["magnetic_ut"])
    fit = np.concatenate(train_rows)
    mean, scale = fit.mean(axis=0), np.maximum(fit.std(axis=0), 1e-3)
    split = {k: windows(v, window, mean, scale) for k, v in groups.items()}
    x, y, _ = split["train"]
    if not 0 < y.sum() < len(y):
        raise ValueError("Training split must contain both slip and non-slip")
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                              batch_size=128, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    model = SlipLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((1-y).sum()/y.sum())))
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output/"slip_lstm.pt"
    best = float("inf")
    history = []

    def predict(key):
        xx, yy, _ = split[key]
        result, losses = [], []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(xx), 256):
                batch = torch.from_numpy(xx[start:start+256])
                target = torch.from_numpy(yy[start:start+256])
                logits = model(batch)
                result.extend(torch.sigmoid(logits).tolist())
                losses.append((float(criterion(logits, target)), len(batch)))
        return np.asarray(result), sum(a*b for a, b in losses)/sum(b for _, b in losses)

    for epoch in range(1, epochs+1):
        model.train()
        total = 0.
        for xx, yy in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xx), yy)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss)*len(xx)
        probabilities, val_loss = predict("val")
        metric = scores(split["val"][1], probabilities)
        history.append(dict(epoch=epoch, train_loss=total/len(x), val_loss=val_loss,
                            val_f1=metric["f1"]))
        print(f"epoch {epoch:02d}: train loss {total/len(x):.4f}, "
              f"val loss {val_loss:.4f}, val F1 {metric['f1']:.3f}", flush=True)
        if val_loss < best:
            best = val_loss
            torch.save(dict(state_dict=model.state_dict(), mean=mean.tolist(), scale=scale.tolist(),
                            channels=CHANNELS, window=window, hidden_size=128, threshold=.5,
                            sample_hz=manifest["sample_hz"], selected_epoch=epoch,
                            status="simulation only; unvalidated on hardware"), checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    # Test is evaluated once, after checkpoint selection by validation loss.
    test_prob, _ = predict("test")
    metric = scores(split["test"][1], test_prob)
    report = dict(status="SIMULATION ONLY; no real-world performance claim", model="LSTM 15→128→128→1",
                  window_samples=window, selected_epoch=saved["selected_epoch"],
                  parameters=sum(p.numel() for p in model.parameters()),
                  normalization="fit on training episodes only", splits=manifest["splits"],
                  test=metric, history=history)
    (output/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    print("Held-out simulation:", json.dumps(metric), flush=True)
    return report


class Predictor:
    def __init__(self, checkpoint):
        torch.set_num_threads(2)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["channels"] != CHANNELS:
            raise ValueError("Checkpoint channel order mismatch")
        self.model = SlipLSTM(saved["hidden_size"])
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        self.mean, self.scale = np.array(saved["mean"]), np.array(saved["scale"])
        self.history = deque(maxlen=saved["window"])
        self.sample_hz = saved["sample_hz"]

    def update(self, magnetic):
        self.history.append((np.asarray(magnetic)-self.mean)/self.scale)
        if len(self.history) < self.history.maxlen:
            return float("nan")
        x = torch.tensor(np.array(self.history)[None], dtype=torch.float32)
        with torch.no_grad():
            return float(torch.sigmoid(self.model(x))[0])
