"""Fit a simulation-only dipole readout gain to a Pollen/network reference.

Unpaired amplitude distributions, NOT measured xyz or an identified magnetic
material. Zero-intercept positive gains retain dipole directions and zero.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from wowskin_lab.paths import ROOT
from wowskin_lab.baselines.pollen import read_csv, RESOURCE_DIR
from wowskin_lab.baselines.reskin_forward import ResponseModel, MAPPED_OUTPUT
from wowskin_lab.baselines.reskin_inverse import InverseModel, OUTPUT as INVERSE_OUTPUT

QUANTILES = [.1, .25, .5, .75, .9]
# Dipole PCB order C/U/R/D/L -> provisional network order C/L/R/U/D.
PERMUTATION = np.array([0, 4, 2, 1, 3])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantiles(values):
    return np.quantile(np.linalg.norm(np.asarray(values).reshape(-1, 5, 3), axis=2), QUANTILES, axis=0)


def error(x, y):
    return float(np.sqrt(np.mean((x-y)**2)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode', type=Path, default=ROOT/'outputs/cable/teleop_20261010_010635/episode.npz')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/diagnostics/pollen_nonslip_dipole_fit')
    args = p.parse_args()
    if (args.output/'profile.json').exists():
        raise FileExistsError('Choose a fresh --output to preserve the existing fit')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    b, labels, _ = read_csv(RESOURCE_DIR/'data/cable.csv')
    amplitude = np.linalg.norm(b.reshape(-1, 5, 3), axis=2).max(1)
    reset = np.flatnonzero((amplitude[:-1]>100) & (amplitude[1:]<50) & (np.diff(amplitude)<-100))+1
    if len(reset) != 3:
        raise ValueError('Pinned cable file no longer has the three documented reset boundaries')
    pulls = np.searchsorted(reset, np.arange(len(b)), side='right')
    source_split = np.minimum(np.maximum(pulls-1, 0), 2)  # pulls 0/1 train, 2 val, 3 test
    inverse, forward = InverseModel(), ResponseModel(MAPPED_OUTPUT)
    xyz = inverse.predict(b)
    gate = ((labels == 0) & (amplitude >= 50) & np.isfinite(xyz).all(1)
            & (xyz[:, :2] >= 0).all(1) & (xyz[:, :2] <= 16).all(1)
            & (xyz[:, 2] > 0) & (xyz[:, 2] <= 2))
    with torch.inference_mode():
        x = torch.tensor((xyz-forward.meta['x_mean'])/forward.meta['x_scale'], dtype=torch.float32)
        teacher = forward.model(x).numpy()*forward.meta['y_scale']+forward.meta['y_mean']
    with np.load(args.episode) as a:
        dipole = a['dipole_ideal_delta_ut'].reshape(-1, 5, 3)[:, PERMUTATION]
        sim_gate = (a['grasped'] & (a['slip'] == 0) & (a['action'][:, 6] > .99)
                    & (np.linalg.norm(dipole, axis=2).max(1)>1e-6))
        sim_split = np.zeros(len(dipole), int)
        sim_split[len(dipole)//2:] = 1
        sim_split[len(dipole)*3//4:] = 2
    xq = quantiles(dipole[sim_gate & (sim_split == 0)])
    yq = quantiles(teacher[gate & (source_split == 0)])
    uniform = float(np.sum(xq*yq)/np.sum(xq*xq))
    candidates = dict(uniform=np.full(5, uniform), per_chip=(xq*yq).sum(0)/(xq*xq).sum(0))
    splits = {}
    for i, name in enumerate(('train', 'validation', 'test')):
        sm, pm = sim_gate & (sim_split == i), gate & (source_split == i)
        if not sm.any() or not pm.any():
            raise ValueError('Empty independent split')
        xq, yq, measured_q = quantiles(dipole[sm]), quantiles(teacher[pm]), quantiles(b[pm])
        splits[name] = dict(sim_frames=int(sm.sum()), pollen_frames=int(pm.sum()),
            original_to_network_quantile_rmse_ut=error(xq, yq),
            fitted_to_network_quantile_rmse_ut={k:error(xq*g, yq) for k,g in candidates.items()},
            original_to_pollen_quantile_rmse_ut=error(xq, measured_q),
            fitted_to_pollen_quantile_rmse_ut={k:error(xq*g, measured_q) for k,g in candidates.items()},
            sim_original_chip_norm_quantiles_ut=xq.tolist(), network_chip_norm_quantiles_ut=yq.tolist(),
            measured_pollen_chip_norm_quantiles_ut=measured_q.tolist())
    choice = min(candidates, key=lambda k:splits['validation']['fitted_to_network_quantile_rmse_ut'][k])
    gains_candidate = candidates[choice]
    gains_pcb = gains_candidate[np.argsort(PERMUTATION)]
    source_paths = [RESOURCE_DIR/'data/cable.csv', MAPPED_OUTPUT/'model.safetensors',
                    INVERSE_OUTPUT/'model.safetensors', args.episode]
    profile = dict(schema='wowskin_sim_magnetic_gain_v1',
        status='SIMULATION AMPLITUDE FIT; not real sensor or material calibration',
        method='Positive zero-intercept chip gains fitted to unpaired norm quantiles of local dipole and Pollen->local MagNet->existing forward MLP.',
        selected=choice, gain_order=['U1_center','U2_up','U3_right','U4_down','U5_left'],
        gains=gains_pcb.tolist(), candidate_order_gains=gains_candidate.tolist(),
        fit_magnetization_a_per_m=4000., quantiles=QUANTILES,
        application='Scale dipole field and baseline before filtering and additive noise. No changes to forces, stiffness, friction, controller or timestep.',
        split='Pollen whole pulls 0/1 train, 2 validation, 3 test; manual simulation first 50% train, next 25% validation, final 25% test, no shuffled frames.',
        selection='Both candidates fit training only; choose by validation norm-quantile RMSE. No refit on held-out frames.',
        source_sha256={str(path):sha(path) for path in source_paths},
        limitations=['Pollen slip=0 is not contact/force ground truth; >=50 uT signal gate is a heuristic.',
                     'Inverse xyz is a pseudo-label; forward MLP is extrapolated at shallow depths.',
                     'Unpaired norm statistics cannot identify vector axes, contact-response timing or physical magnetization.',
                     'Fitted on cable only; spoon/cube are unvalidated cross-task transfers.'])
    report = dict(profile=profile, reset_boundaries=reset.tolist(), candidate_gains={k:v.tolist() for k,v in candidates.items()},
                  selection_counts=dict(pollen_all=len(b), pollen_non_slip=int((labels==0).sum()), pollen_accepted=int(gate.sum()),
                                        sim_all=len(dipole),sim_accepted=int(sim_gate.sum())), splits=splits)
    (args.output/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.output/'fit_data.npz', pollen_delta_ut=b, pollen_slip=labels,
                        pseudo_xyz_mm=xyz, network_reference_ut=teacher, pollen_accepted=gate,
                        pollen_split=source_split, sim_original_candidate_ut=dipole, sim_accepted=sim_gate,
                        sim_split=sim_split, sim_fitted_candidate_ut=dipole*gains_candidate[None,:,None])
    print(json.dumps(dict(selected=choice,gains_pcb=gains_pcb.tolist(),splits=splits),indent=2))


if __name__ == '__main__':
    main()
