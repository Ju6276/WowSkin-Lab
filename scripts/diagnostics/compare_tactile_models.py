"""Reproduce Pollen cable -> MagNet -> forward response and paired sim diagnostics.

Cycle error is model compatibility, not depth accuracy. Pollen has no xyz/depth
labels; no inverse output is injected into MuJoCo as measured indentation.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wowskin_lab.paths import ROOT
from wowskin_lab.baselines.pollen import PollenPredictor, read_csv, metrics, RESOURCE_DIR, CHANNELS
from wowskin_lab.baselines.reskin_forward import ResponseModel, MAPPED_OUTPUT, OUTPUT, read_data
from wowskin_lab.baselines.reskin_inverse import InverseModel, channel_indices, position_errors


def distribution(values):
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    return dict(min=float(v.min()), median=float(np.median(v)), p05=float(np.percentile(v,5)),
                p95=float(np.percentile(v,95)), max=float(v.max())) if len(v) else None


def signal_stats(values):
    v = np.asarray(values).reshape(-1,5,3)
    return dict(samples=len(v), component_rms_ut=float(np.sqrt(np.mean(v**2))), component_abs_ut=distribution(np.abs(v)),
                chip_norm_ut=distribution(np.linalg.norm(v,axis=2)))


def delta_metrics(actual, predicted):
    residual = predicted-actual
    rmse = float(np.sqrt(np.mean(residual**2)))
    return dict(component_rmse_ut=rmse, component_mae_ut=float(np.abs(residual).mean()),
                per_component_rmse_ut=np.sqrt(np.mean(residual**2, axis=0)).tolist(),
                rmse_over_signal_rms=rmse/max(float(np.sqrt(np.mean(actual**2))),1e-12))


def forward_batch(model, query):
    # Diagnostic explicitly permits unsupported depths; never silently clips.
    with torch.inference_mode():
        t = torch.tensor((query-model.meta['x_mean'])/model.meta['x_scale'], dtype=torch.float32)
        return model.model(t).numpy()*model.meta['y_scale']+model.meta['y_mean']


def domain_summary(forward, query):
    issues = [forward.domain_issues(q) for q in query]
    depth = query[:,2]
    return dict(depth_mm=distribution(depth), negative_depth_count=int((depth<0).sum()),
                outside_measured_depth_count=int(((depth<1.5)|(depth>2.)).sum()),
                outside_xy_count=sum('xy_outside_sampled_region' in s for s in issues),
                strictly_measured_input_count=sum(not s for s in issues))


def pollen_cycle(output):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    forward, old = ResponseModel(MAPPED_OUTPUT), ResponseModel(OUTPUT)
    local, published = InverseModel(), InverseModel(published=True)
    values, labels, stamps = read_csv(RESOURCE_DIR/'data/cable.csv')
    probability = PollenPredictor().predict(values)
    estimates = dict(local_mapped_mag=local.predict(values), official_pretrained=published.predict(values))
    indices = channel_indices(forward.meta['mapping'])
    groups = {'all':np.ones(len(values),dtype=bool), 'slip':labels==1, 'non_slip':labels==0}
    report = dict(input_source=str(RESOURCE_DIR/'data/cable.csv'), rows=len(values),
        available_columns=['log_time',*CHANNELS,'slip'], has_depth_ground_truth=False,
        mapping=forward.meta['mapping'],
        input_convention='Pollen CSV used as provided (reset/differential magnetic readings); no second ReSkin baseline subtraction or Pollen scaler for MagNet. Only candidate chip-group order, no axis or amplitude fitting.',
        interpretation='Out-of-domain cross-sensor probe-contact hypothesis. Estimated XYZ is not cable ground truth. Cycle residual is not calibration accuracy.',
        pollen_official_replay=metrics(labels,probability), groups={}, inverse={})
    for name,mask in groups.items(): report['groups'][name]=signal_stats(values[mask])
    arrays = dict(input_delta_ut=values, slip_label=labels, pollen_probability=probability)
    for name, xyz in estimates.items():
        reconstructed = forward_batch(forward, xyz)
        old_reconstructed = forward_batch(old, xyz)[:,indices]
        arrays[name+'_xyz_mm'] = xyz
        arrays[name+'_forward_delta_ut'] = reconstructed
        arrays[name+'_old_forward_delta_ut'] = old_reconstructed
        report['inverse'][name] = dict(input_domain=domain_summary(forward, xyz), groups={})
        for group,mask in groups.items():
            report['inverse'][name]['groups'][group] = dict(
                estimated_xyz_mm=[distribution(xyz[mask,i]) for i in range(3)],
                current_forward_cycle=delta_metrics(values[mask],reconstructed[mask]),
                original_forward_cycle=delta_metrics(values[mask],old_reconstructed[mask]),
                current_vs_original_forward=delta_metrics(old_reconstructed[mask],reconstructed[mask]))
    report['between_inverse_paths'] = {group: dict(
        reconstructed_field_difference=delta_metrics(arrays['local_mapped_mag_forward_delta_ut'][mask],
                                                     arrays['official_pretrained_forward_delta_ut'][mask]),
        estimated_xyz_difference=position_errors(estimates['local_mapped_mag'][mask],estimates['official_pretrained'][mask]))
        for group,mask in groups.items()}
    # Source-domain sanity checks; published checkpoint's training split is
    # unknown, so its full-source score is explicitly not a held-out metric.
    xyz, b, _ = read_data()
    mapped_b = b[:,indices]
    report['official_source_data_replay_not_heldout'] = position_errors(xyz,published.predict(mapped_b))
    report['official_source_cycle_not_heldout'] = delta_metrics(mapped_b,forward_batch(forward,published.predict(mapped_b)))
    report['local_inverse_heldout'] = json.loads((ROOT/'outputs/reskin_inverse_mapped/report.json').read_text())['metrics']['test']
    held = np.load(MAPPED_OUTPUT/'heldout_predictions.npz')
    hx = local.predict(held['observed_delta_ut'])
    report['source_heldout_cycle'] = delta_metrics(held['observed_delta_ut'],forward_batch(forward,hx))
    np.savez_compressed(output/'pollen_cable_cycle.npz', **arrays)
    with (output/'pollen_cable_cycle.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['log_time','slip_label','pollen_probability',
                         'local_x_mm','local_y_mm','local_depth_mm','official_x_mm','official_y_mm','official_depth_mm',
                         *CHANNELS,*['reconstructed_'+c for c in CHANNELS]])
        writer.writerows([stamps[i],int(labels[i]),float(probability[i]),*estimates['local_mapped_mag'][i],
                          *estimates['official_pretrained'][i],*values[i],*arrays['local_mapped_mag_forward_delta_ut'][i]] for i in range(len(values)))
    fig,axs=plt.subplots(4,1,figsize=(12,10),sharex=True,layout='constrained')
    for ax in axs:
        ax.fill_between(np.arange(len(labels)),0,1,where=labels.astype(bool),transform=ax.get_xaxis_transform(),
                        color='#e59b31',alpha=.16,label='Public slip label')
    norms=np.linalg.norm(values.reshape(-1,5,3),axis=2)
    for chip in range(5): axs[0].plot(norms[:,chip],lw=.7,label=f'CH{chip+1}')
    axs[0].set(ylabel='Measured norm (uT)',title='Pollen cable: magnetic input -> candidate-mapped MagNet -> forward model\nEstimated depth has NO ground truth; shaded regions are public slip labels')
    axs[0].legend(ncol=6,fontsize=8)
    for name,xyz in estimates.items(): axs[1].plot(xyz[:,2],lw=.8,label=name)
    axs[1].axhspan(1.5,2.,color='green',alpha=.12,label='ReSkin training depths span')
    axs[1].set_ylabel('Estimated depth (mm)');axs[1].legend(fontsize=8,ncol=3)
    residual=np.sqrt(np.mean((arrays['local_mapped_mag_forward_delta_ut']-values)**2,axis=1))
    axs[2].plot(residual,lw=.7);axs[2].set_ylabel('Cycle RMSE (uT)')
    axs[3].plot(probability,lw=.7);axs[3].set(xlabel='CSV row (no assumed sample rate)',ylabel='Pollen p(slip)')
    fig.savefig(output/'pollen_cable_cycle.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(5,3,figsize=(14,12),sharex=True,layout='constrained')
    for i,ax in enumerate(axs.flat):
        ax.plot(values[:,i],lw=.5,label='Pollen measurement')
        ax.plot(arrays['local_mapped_mag_forward_delta_ut'][:,i],lw=.5,label='Inverse -> forward')
        ax.set_title(CHANNELS[i],fontsize=9)
    axs[0,0].legend(fontsize=7);fig.suptitle('15 channels in uT; candidate grouping only, axes unverified')
    fig.savefig(output/'pollen_15_channels.png',dpi=140);plt.close(fig)
    return report


def paired_sim(output, episode, prefix='paired_sim'):
    a=np.load(episode)
    p=PollenPredictor()
    mlp=a['magnetic_ut']; dipole=a['dipole_magnetic_ut']
    # Dipole reference PCB order: centre, up, right, down, left. Reorder groups
    # to centre, left, right, up, down for diagnostic Pollen input only.
    dipole_mapped=dipole.reshape(-1,5,3)[:,[0,4,2,1,3]].reshape(-1,15)
    pm,pd=p.predict(mlp),p.predict(dipole_mapped)
    contact=a['normal_force_n']>=.01
    masks=dict(contact=contact, slip=contact & a['slip'].astype(bool),
               non_slip_contact=contact & ~a['slip'].astype(bool), stable_hold=a['phase']=='hold')
    scene = 'cube' if 'cube_xyz_m' in a.files else 'cable'
    report=dict(source=str(episode), provenance=f'{len(mlp)}-frame paired physical {scene} trajectory.',
                physics='Both response methods share exactly the same MuJoCo FEM trajectory. Neither is ground truth.',
                signal_convention='Actual emitted signals: dipole has configured low-pass/delay; current MLP directly evaluates. Axis correspondence is unverified; vector residual not treated as physical error.',
                pollen_evaluation='Cross-domain diagnostic only: unknown chip-axis alignment and different slip-label definitions.', groups={})
    for name,mask in masks.items():
        if not mask.any(): continue
        report['groups'][name]=dict(frames=int(mask.sum()), fem_depth_mm=distribution(a['reskin_query_mm'][mask,2]),
            dipole=signal_stats(dipole[mask]), forward_mlp=signal_stats(mlp[mask]),
            pollen_mlp_probability=distribution(pm[mask]),pollen_dipole_probability=distribution(pd[mask]))
    report['contact_classifier_diagnostic'] = dict(mlp=metrics(a['slip'][contact],pm[contact]),
                                                 dipole=metrics(a['slip'][contact],pd[contact]))
    np.savez_compressed(output/(prefix+'_comparison.npz'),time_s=a['time_s'], depth_mm=a['reskin_query_mm'][:,2],
        physics_slip=a['slip'],forward_delta_ut=mlp,dipole_delta_ut=dipole,pollen_forward=pm,pollen_dipole=pd)
    fig,axs=plt.subplots(5,1,figsize=(12,11),sharex=True,layout='constrained');t=a['time_s']
    axs[0].plot(t,a['normal_force_n']);axs[0].set_ylabel('Left normal force (N)');axs[0].set_title('Same MuJoCo trajectory, two magnetic output methods (uncalibrated)')
    axs[1].plot(t,a['slip_speed_m_per_s']*1000);axs[1].set_ylabel('Relative slip (mm/s)')
    axs[2].plot(t,a['reskin_query_mm'][:,2]);axs[2].set_ylabel('FEM compression (mm)')
    for signal,name in [(mlp,'MLP'),(dipole,'Dipole')]:
        axs[3].plot(t,np.linalg.norm(signal.reshape(-1,5,3),axis=2).mean(1),label=name)
    axs[3].set_ylabel('Mean chip norm (uT)');axs[3].set_yscale('symlog',linthresh=1.);axs[3].legend()
    axs[4].plot(t,pm,label='Pollen on MLP');axs[4].plot(t,pd,label='Pollen on dipole');axs[4].plot(t,a['slip'],label='Physics slip');axs[4].legend();axs[4].set(xlabel='Simulation time (s)',ylabel='Probability / label')
    fig.savefig(output/(prefix+'_comparison.png'),dpi=150);plt.close(fig)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/diagnostics/pollen_reskin_comparison')
    p.add_argument('--sim-episode',type=Path,default=ROOT/'outputs/diagnostics/reskin_cube/episode.npz')
    p.add_argument('--cable-episode',type=Path)
    args=p.parse_args()
    report=dict(pollen_cycle=pollen_cycle(args.output), paired_sim=paired_sim(args.output,args.sim_episode))
    if args.cable_episode:
        report['cable_sim'] = paired_sim(args.output,args.cable_episode,prefix='cable_sim')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(pollen_rows=report['pollen_cycle']['rows'],inverse=report['pollen_cycle']['inverse'],
                         sim=report['paired_sim']['groups']),indent=2))


if __name__=='__main__': main()
