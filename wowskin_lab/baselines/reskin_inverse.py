"""Official 3D Cal MagNet architecture with an explicit candidate channel adapter.

Published weights stay unchanged. Local retraining uses the same architecture,
raw baseline-subtracted uT, MSE/AdamW (lr/weight-decay 1e-4), batch 64, 60 epochs. The whole-XY
split is inherited from our forward-model experiment, not the official split.
No Pollen labels or test positions are used to fit either inverse model.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import urllib.request

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from .reskin_forward import ASSETS, MAPPED_OUTPUT, read_data
from wowskin_lab.paths import ROOT

SOURCE = ASSETS/'upstream'
OUTPUT = ROOT/'outputs/reskin_inverse_mapped'
REVISION = 'fd04a700eca70f3618a8b25ce8458c12276253c3'
WEIGHT_MD5 = 'aace218069cca20dbd012246037f3718'


def fetch():
    SOURCE.mkdir(parents=True, exist_ok=True)
    base = f'https://raw.githubusercontent.com/rohankotanu/py3DCal/{REVISION}/'
    urls = {name: base+path for name, path in {
        'magnet.py': 'py3DCal/model_training/models/magnet.py',
        'ReSkin_dataset.py': 'py3DCal/model_training/datasets/ReSkin_dataset.py',
        'reskin_prediction.py': 'py3DCal/model_training/lib/reskin_prediction.py',
        'train_model.py': 'py3DCal/model_training/lib/train_model.py',
        'LICENSE': 'LICENSE'}.items()}
    urls['reskin_pretrained_weights.pth'] = 'https://zenodo.org/records/18462608/files/reskin_pretrained_weights.pth?download=1'
    previous = json.loads((SOURCE/'manifest.json').read_text()).get('files', {}) if (SOURCE/'manifest.json').exists() else {}
    entries = {}
    for name, url in urls.items():
        path = SOURCE/name
        if not (path.exists() and previous.get(name, {}).get('url') == url and
                hashlib.sha256(path.read_bytes()).hexdigest() == previous.get(name, {}).get('sha256')):
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if name.endswith('.pth') and hashlib.md5(data).hexdigest() != WEIGHT_MD5:
                raise ValueError('Published MagNet checksum mismatch')
            path.write_bytes(data)
        entries[name] = dict(url=url, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (SOURCE/'manifest.json').write_text(json.dumps(dict(revision=REVISION, files=entries), indent=2)+'\n')


def architecture():
    spec = importlib.util.spec_from_file_location('pinned_3dcal_magnet', SOURCE/'magnet.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MagNet()


def channel_indices(profile):
    groups = profile['source_groups_for_wowskin']
    if sorted(groups) != list(range(5)):
        raise ValueError('Expected a complete five-chip permutation')
    return np.array([3*i+j for i in groups for j in range(3)])


def predict_batches(model, values):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 15 or not np.isfinite(values).all():
        raise ValueError('Expected finite (N, 15) delta-B, not xyz or raw unzeroed fields')
    with torch.inference_mode():
        return np.concatenate([model(torch.tensor(v, dtype=torch.float32)).numpy()
                               for v in np.array_split(values, max(1, int(np.ceil(len(values)/1024))))])


class InverseModel:
    def __init__(self, path=OUTPUT, *, published=False, forward_path=MAPPED_OUTPUT):
        self.model = architecture()
        self.profile = json.loads((Path(forward_path)/'model.json').read_text())['mapping']
        self.indices = channel_indices(self.profile)
        self.published = published
        if published:
            weight_path = SOURCE/'reskin_pretrained_weights.pth'
            if hashlib.md5(weight_path.read_bytes()).hexdigest() != WEIGHT_MD5:
                raise ValueError('Published MagNet checksum mismatch')
            self.model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True))
        else:
            self.meta = json.loads((Path(path)/'model.json').read_text())
            if self.meta['mapping'] != self.profile:
                raise ValueError('Forward and inverse channel mappings differ')
            self.model.load_state_dict(load_file(str(Path(path)/'model.safetensors')))
        self.model.eval()

    def predict(self, mapped_delta_ut):
        values = np.asarray(mapped_delta_ut)
        # B_mapped[:, j] = B_source[:, indices[j]]. Undo this only for the
        # original checkpoint. No guessed chip-axis rotations or amplitude fit.
        if self.published:
            values = values[:, np.argsort(self.indices)]
        return predict_batches(self.model, values)


def position_errors(true, pred):
    return dict(xyz_rmse_mm=np.sqrt(np.mean((pred-true)**2, axis=0)).tolist(),
                xy_euclidean_mean_mm=float(np.linalg.norm(pred[:,:2]-true[:,:2], axis=1).mean()),
                depth_mae_mm=float(np.abs(pred[:,2]-true[:,2]).mean()))


def train(output=OUTPUT, forward_path=MAPPED_OUTPUT, epochs=60):
    fetch()
    output, forward_path = Path(output), Path(forward_path)
    if (output/'model.safetensors').exists():
        raise FileExistsError(f'Preserving existing model: {output}')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    forward = json.loads((forward_path/'model.json').read_text())
    split = json.loads((forward_path/'report.json').read_text())['split_xy']
    xyz, magnetic, _ = read_data()
    magnetic = magnetic[:,channel_indices(forward['mapping'])]
    masks = {name: np.array([tuple(p) in set(map(tuple, locations)) for p in xyz[:,:2]])
             for name,locations in split.items()}
    x, y = torch.tensor(magnetic, dtype=torch.float32), torch.tensor(xyz, dtype=torch.float32)
    model = architecture()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    ids = torch.from_numpy(np.flatnonzero(masks['train']))
    best, best_epoch, history = float('inf'), 0, []
    for epoch in range(1, epochs+1):
        model.train()
        order = ids[torch.randperm(len(ids))]
        for batch in order.split(64):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        val = predict_batches(model, magnetic[masks['validation']])
        score = float(np.mean((val-xyz[masks['validation']])**2))
        history.append(dict(epoch=epoch, validation_xyz_mse_mm2=score))
        if score < best:
            best, best_epoch = score, epoch
            save_file(model.state_dict(), str(output/'model.safetensors'))
        if epoch % 10 == 0:
            print(f'MagNet local mapped retrain {epoch}/{epochs}: val MSE {score:.5f} mm^2', flush=True)
    model.load_state_dict(load_file(str(output/'model.safetensors')))
    prediction = predict_batches(model, magnetic)
    meta = dict(model='LOCAL retrain of unchanged official MagNet architecture, 15->200->200->40->200->200->3',
                parameters=sum(p.numel() for p in model.parameters()), mapping=forward['mapping'],
                input='15 baseline-subtracted uT in candidate WowSkin group order; source XYZ retained',
                output='original ReSkin dataset x/y/depth in mm', official_architecture_revision=REVISION,
                optimizer='AdamW', learning_rate=1e-4, weight_decay=1e-4,
                epochs=epochs, batch_size=64, seed=42, best_epoch=best_epoch,
                split_source=str(forward_path/'report.json'),
                changes_from_official_recipe='Whole-XY train/validation/test partition; validation checkpoint selection; candidate chip-group permutation.',
                transfer_validated=False)
    (output/'model.json').write_text(json.dumps(meta, indent=2)+'\n')
    report = dict(metrics={k:position_errors(xyz[v],prediction[v]) for k,v in masks.items()},
                  split_rows={k:int(v.sum()) for k,v in masks.items()}, history=history,
                  interpretation='Held-out spatial accuracy on author ReSkin; not WowSkin/Pollen accuracy.')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(output/'heldout_predictions.npz', observed_xyz_mm=xyz[masks['test']],
                        predicted_xyz_mm=prediction[masks['test']], input_delta_ut=magnetic[masks['test']])
    print(json.dumps(report['metrics'], indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['fetch','train'])
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--epochs', type=int, default=60)
    args = p.parse_args()
    if args.epochs < 1: p.error('--epochs must be positive')
    fetch() if args.command == 'fetch' else train(args.output, epochs=args.epochs)


if __name__ == '__main__':
    main()
