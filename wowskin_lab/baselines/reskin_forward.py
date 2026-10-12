"""Local 3D Cal response regression, NOT the official inverse MagNet model.

Position/depth (mm) -> baseline-subtracted 15-channel field (uT).
The data describes the author's ReSkin/probe, not an identified WowSkin.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from wowskin_lab.paths import ROOT

ASSETS = ROOT / 'assets/3dcal'
OUTPUT = ROOT / 'outputs/reskin_forward'
MAPPED_OUTPUT = ROOT / 'outputs/reskin_forward_mapped'
URL = 'https://zenodo.org/records/18462608/files/reskin_calibration_data.tar.gz?download=1'
MD5 = 'd610dd83df8ac29a6547b86823136c3e'
CHANNELS = [f'B{a}{i}' for i in range(5) for a in 'xyz']


def fetch_data():
    ASSETS.mkdir(parents=True, exist_ok=True)
    archive = ASSETS / 'reskin_calibration_data.tar.gz'
    raw = archive.read_bytes() if archive.exists() else urllib.request.urlopen(URL, timeout=45).read()
    if hashlib.md5(raw).hexdigest() != MD5:
        raise ValueError('Published dataset checksum mismatch')
    archive.write_bytes(raw)
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as pack:
        for name in ('probe_data.csv', 'no_contact_data.csv'):
            (ASSETS/name).write_bytes(pack.extractfile('reskin_calibration_data/'+name).read())
    manifest = dict(url=URL, md5=MD5, sha256=hashlib.sha256(raw).hexdigest(),
                    source='3D Cal Zenodo record 18462608, version 2.0.0',
                    attribution='Rohan Kota, Kaival Shah, J. Edward Colgate, Gregory Reardon',
                    rights='See original Zenodo record; no new license assigned to source data')
    (ASSETS/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


def read_data():
    with (ASSETS/'probe_data.csv').open() as f:
        rows = list(csv.DictReader(f))
    with (ASSETS/'no_contact_data.csv').open() as f:
        empty = list(csv.DictReader(f))
    baseline = np.array([[float(r[k]) for k in CHANNELS] for r in empty]).mean(0)
    x = np.array([[float(r[k]) for k in ('x_mm', 'y_mm', 'penetration_depth_mm')] for r in rows])
    y = np.array([[float(r[k]) for k in CHANNELS] for r in rows])-baseline
    return x, y, baseline


def network():
    return torch.nn.Sequential(torch.nn.Linear(3, 64), torch.nn.Tanh(),
                               torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 15))


def errors(actual, predicted):
    error = predicted-actual
    return dict(rmse_ut=float(np.sqrt(np.mean(error**2))),
                mae_ut=float(np.abs(error).mean()),
                per_channel_rmse_ut=np.sqrt(np.mean(error**2, axis=0)).tolist())


def train(output, epochs=2000, seed=42, mapping=None):
    fetch_data()
    output = Path(output)
    if (output/'model.safetensors').exists():
        raise FileExistsError(f'Choose a new output directory; preserving {output}')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    x, y, baseline = read_data()
    profile = json.loads(Path(mapping).read_text()) if mapping else None
    columns = CHANNELS
    if profile:
        groups_order = profile['source_groups_for_wowskin']
        if sorted(groups_order) != list(range(5)):
            raise ValueError('Mapping must be a permutation of source groups 0..4')
        indices = np.array([3*i+j for i in groups_order for j in range(3)])
        y, baseline = y[:, indices], baseline[indices]
        columns = [CHANNELS[i] for i in indices]
    locations, groups = np.unique(x[:, :2], axis=0, return_inverse=True)
    order = np.random.default_rng(seed).permutation(len(locations))
    n = len(order); a, b = int(.6*n), int(.8*n)
    splits = dict(train=order[:a], validation=order[a:b], test=order[b:])
    masks = {k: np.isin(groups, ids) for k, ids in splits.items()}
    xm, xs = x[masks['train']].mean(0), x[masks['train']].std(0)
    ym, ys = y[masks['train']].mean(0), y[masks['train']].std(0)
    xs, ys = np.maximum(xs, 1e-6), np.maximum(ys, 1e-6)
    tx = torch.tensor((x-xm)/xs, dtype=torch.float32)
    ty = torch.tensor((y-ym)/ys, dtype=torch.float32)
    model = network()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=1e-5)
    best, best_epoch = float('inf'), 0
    history = []
    for epoch in range(1, epochs+1):
        model.train(); optimizer.zero_grad()
        loss = (model(tx[masks['train']])-ty[masks['train']]).square().mean()
        loss.backward(); optimizer.step()
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                pred = model(tx[masks['validation']]).numpy()*ys+ym
            score = errors(y[masks['validation']], pred)['rmse_ut']
            history.append(dict(epoch=epoch, training_standardized_mse=float(loss.detach()), validation_rmse_ut=score))
            if score < best:
                best, best_epoch = score, epoch
                save_file(model.state_dict(), str(output/'model.safetensors'))
    model.load_state_dict(load_file(str(output/'model.safetensors')))
    model.eval()
    with torch.no_grad():
        pred = model(tx).numpy()*ys+ym
    # Nearest training position/depth is a reference; test data never tunes it.
    unique_train, inv = np.unique(x[masks['train']], axis=0, return_inverse=True)
    mean_train = np.array([y[masks['train']][inv == i].mean(0) for i in range(len(unique_train))])
    q = x[masks['test']]
    distances = (((q[:, None]-unique_train[None])/xs)**2).sum(-1)
    nearest = mean_train[distances.argmin(1)]
    metadata = dict(model='LOCAL forward MLP 3-64-64-15; not official MagNet',
                    input_columns=['x_mm', 'y_mm', 'penetration_depth_mm'], output_columns=columns,
                    mapping=profile,
                    output_units='microtesla, difference from published mean no-contact reading',
                    baseline_ut=baseline.tolist(), x_mean=xm.tolist(), x_scale=xs.tolist(),
                    y_mean=ym.tolist(), y_scale=ys.tolist(), known_xy_mm=locations.tolist(),
                    measured_depths_mm=np.unique(x[:, 2]).tolist(), max_xy_distance_mm=.75,
                    parameters=sum(p.numel() for p in model.parameters()), seed=seed, best_epoch=best_epoch,
                    provenance=json.loads((ASSETS/'manifest.json').read_text()),
                    limits='Author ReSkin and calibration probe only; no force, shear or WowSkin transfer validation. No artificial unloaded samples or force-to-depth mapping.')
    (output/'model.json').write_text(json.dumps(metadata, indent=2)+'\n')
    report = dict(model=metadata['model'], rows=len(x), unique_xy=len(locations),
                  split_unit='Whole XY location: all depths and repeats stay together',
                  split_xy={k: locations[v].tolist() for k, v in splits.items()},
                  split_rows={k: int(v.sum()) for k, v in masks.items()},
                  metrics={k: errors(y[v], pred[v]) for k, v in masks.items()},
                  nearest_training_reference_test=errors(y[masks['test']], nearest),
                  training_mean_reference_test=errors(y[masks['test']], np.broadcast_to(ym, y[masks['test']].shape)),
                  best_epoch=best_epoch, history=history,
                  conclusion='Tests measure held-out spatial prediction on this ReSkin dataset, not sim-to-real accuracy.')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(output/'heldout_predictions.npz', input_mm=q, observed_delta_ut=y[masks['test']], predicted_delta_ut=pred[masks['test']])
    print(json.dumps({k:v for k,v in report.items() if k not in ('split_xy','history')}, indent=2))


class ResponseModel:
    def __init__(self, path=OUTPUT):
        path = Path(path)
        self.meta = json.loads((path/'model.json').read_text())
        self.model = network()
        self.model.load_state_dict(load_file(str(path/'model.safetensors')))
        self.model.eval()

    def domain_issues(self, xyz_mm):
        q = np.asarray(xyz_mm, dtype=float)
        if q.shape != (3,) or not np.isfinite(q).all():
            raise ValueError('Expected finite [x_mm, y_mm, depth_mm]')
        depths = np.asarray(self.meta['measured_depths_mm'])
        issues = []
        if q[2] < depths.min()-1e-8 or q[2] > depths.max()+1e-8:
            issues.append('depth_outside_1.5_to_2mm')
        elif not np.isclose(q[2], depths).any():
            issues.append('unvalidated_depth_interpolation')
        distance = np.linalg.norm(np.asarray(self.meta['known_xy_mm'])-q[:2], axis=1).min()
        if distance > self.meta['max_xy_distance_mm']:
            issues.append('xy_outside_sampled_region')
        return issues

    def predict(self, xyz_mm, allow_depth_interpolation=False, allow_extrapolation=False):
        q = np.asarray(xyz_mm, dtype=float)
        if q.shape != (3,) or not np.isfinite(q).all():
            raise ValueError('Expected finite [x_mm, y_mm, depth_mm]')
        depths = np.asarray(self.meta['measured_depths_mm'])
        if not allow_extrapolation and (q[2] < depths.min()-1e-8 or q[2] > depths.max()+1e-8):
            raise ValueError('Depth outside observed 1.5–2.0 mm; extrapolation refused')
        if not (allow_depth_interpolation or allow_extrapolation) and not np.isclose(q[2], depths).any():
            raise ValueError('Only 1.5 and 2.0 mm were measured; explicitly allow unvalidated depth interpolation')
        distance = np.linalg.norm(np.asarray(self.meta['known_xy_mm'])-q[:2], axis=1).min()
        if not allow_extrapolation and distance > self.meta['max_xy_distance_mm']:
            raise ValueError('Position too far from the sampled XY region; extrapolation refused')
        t = torch.tensor((q-self.meta['x_mean'])/self.meta['x_scale'], dtype=torch.float32)
        with torch.no_grad():
            return self.model(t).numpy()*self.meta['y_scale']+self.meta['y_mean']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('fetch', help='Download and verify the reference dataset without training')
    t = sub.add_parser('train'); t.add_argument('--output', default=str(OUTPUT)); t.add_argument('--epochs', type=int, default=2000)
    t.add_argument('--mapping', type=Path, help='Explicit provisional source-group permutation profile')
    q = sub.add_parser('predict'); q.add_argument('--model', default=str(OUTPUT)); q.add_argument('--xyz-mm', type=float, nargs=3, required=True); q.add_argument('--allow-depth-interpolation', action='store_true')
    args = p.parse_args()
    if args.command == 'fetch':
        fetch_data()
        print(f'Reference data ready: {ASSETS}')
    elif args.command == 'train':
        if args.epochs < 1: p.error('--epochs must be positive')
        train(args.output, args.epochs, mapping=args.mapping)
    else:
        m = ResponseModel(args.model)
        result = m.predict(args.xyz_mm, args.allow_depth_interpolation)
        print(json.dumps(dict(source=m.meta['model'], input_mm=args.xyz_mm, delta_b_ut=result.reshape(5,3).tolist(), limitations=m.meta['limits']), indent=2))


if __name__ == '__main__':
    main()
