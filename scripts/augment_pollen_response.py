"""Controlled pseudo-label augmentation, never a physical depth calibration.

Teacher: official pretrained MagNet, plus our train-split-only MagNet control.
Student: copy of the existing mapped 3->64->64->15 forward model.
The final Pollen cable pull is held out by acquisition-reset boundary.
Pollen evaluation is B->estimated xyz->B reconstruction (target-derived input),
not accuracy at measured contact locations. Never overwrite the original model.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wowskin_lab.paths import ROOT
from wowskin_lab.baselines.reskin_forward import MAPPED_OUTPUT, ResponseModel, read_data, errors
from wowskin_lab.baselines.reskin_inverse import InverseModel, channel_indices
from wowskin_lab.baselines.pollen import RESOURCE_DIR, read_csv

DEFAULT_OUTPUT = ROOT/'outputs/diagnostics/pollen_pseudolabel_augmentation'


def predict(model, meta, xyz):
    with torch.inference_mode():
        x=torch.tensor((xyz-meta['x_mean'])/meta['x_scale'],dtype=torch.float32)
        return model(x).numpy()*meta['y_scale']+meta['y_mean']


def error_report(actual,pred):
    result=errors(actual,pred)
    result['signal_component_rms_ut']=float(np.sqrt(np.mean(actual**2)))
    result['rmse_over_signal_rms']=result['rmse_ut']/max(result['signal_component_rms_ut'],1e-12)
    return result


def train_source_control(output):
    """Matched continued training without a pseudo-label loss."""
    dest=output/'source_only_control'
    if (dest/'model.safetensors').exists():
        raise ValueError('Preserving existing source-only control')
    experiment=json.loads((output/'report.json').read_text())
    settings=experiment['training']
    baseline=ResponseModel(MAPPED_OUTPUT);meta=baseline.meta
    xy,values,_=read_data();values=values[:,channel_indices(meta['mapping'])]
    split=json.loads((MAPPED_OUTPUT/'report.json').read_text())['split_xy']
    masks={name:np.array([tuple(q) in set(map(tuple,points)) for q in xy[:,:2]]) for name,points in split.items()}
    x=torch.tensor((xy[masks['train']]-meta['x_mean'])/meta['x_scale'],dtype=torch.float32)
    y=torch.tensor((values[masks['train']]-meta['y_mean'])/meta['y_scale'],dtype=torch.float32)
    torch.set_num_threads(2);torch.manual_seed(settings['seed'])
    model=copy.deepcopy(baseline.model)
    optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    pseudo_count=experiment['branches']['train_split_teacher']['accepted_pseudo_count']
    for _ in range(settings['steps']):
        model.train();optimizer.zero_grad()
        indices=torch.randint(len(x),(settings['batch_real'],))
        # Match the RNG draws of the train-split teacher experiment; these
        # pseudo indices are deliberately not used in the source-only loss.
        torch.randint(pseudo_count,(settings['batch_pseudo'],))
        loss=(model(x[indices])-y[indices]).square().mean()
        loss.backward();optimizer.step()
    model.eval()
    report=dict(status='Continued ReSkin training only; zero Pollen training examples',training=settings,
                actual_pseudo_loss_weight=0.,matched_sampling='same real batch RNG stream as train_split_teacher',
                reskin_after={k:error_report(values[v],predict(model,meta,xy[v])) for k,v in masks.items()},
                pollen_reconstruction={})
    for name in experiment['branches']:
        with np.load(output/name/'pollen_predictions.npz',allow_pickle=False) as data:
            pred=predict(model,meta,data['estimated_xyz_mm'])
            subsets=dict(heldout_all=data['heldout'],heldout_slip=data['heldout'] & (data['slip']==1))
            report['pollen_reconstruction'][name]={k:dict(frames=int(mask.sum()),**error_report(data['input_magnetic_ut'][mask],pred[mask])) for k,mask in subsets.items()}
    dest.mkdir(exist_ok=True)
    updated=copy.deepcopy(meta)
    updated.update(model='LOCAL forward MLP, continued source-only training control',
                   continued_training=dict(**settings,actual_pseudo_loss_weight=0.))
    (dest/'model.json').write_text(json.dumps(updated,indent=2)+'\n')
    save_file(model.state_dict(),str(dest/'model.safetensors'))
    (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def evaluate_sim(output):
    """Replay saved deformation queries; no physics rerun or model deployment."""
    baseline=ResponseModel(MAPPED_OUTPUT)
    models={name:ResponseModel(output/name) for name in ('official_teacher','train_split_teacher')}
    report=dict(method='Offline evaluation of identical saved MuJoCo queries; no physics or controller changes.',
                limitation='Every contact query is outside measured ReSkin calibration support. Pseudo-labels are not measured shallow-depth support. Dipole output retains its original filter/delay; MLP outputs are direct.',
                scenes={})
    for scene,folder in [('cube','reskin_cube'),('cable','cable_contact')]:
        source=ROOT/'outputs/diagnostics'/folder/'episode.npz'
        with np.load(source,allow_pickle=False) as saved:
            data={k:saved[k] for k in saved.files}
        xyz=data['reskin_query_mm']
        emitted=data['reskin_prediction_emitted'] & np.isfinite(xyz).all(1)
        contact=(data['normal_force_n']>=.01) & emitted
        signals=dict(dipole=data['dipole_magnetic_ut'],baseline=data['magnetic_ut'])
        # Batched evaluation may differ slightly from the recorded per-frame
        # float32 GEMV; catch unit/order/normalization mistakes, not roundoff.
        np.testing.assert_allclose(predict(baseline.model,baseline.meta,xyz[emitted]),
                                   signals['baseline'][emitted],rtol=1e-5,atol=.001)
        for name,model in models.items():
            values=np.zeros_like(signals['baseline'])
            values[emitted]=predict(model.model,model.meta,xyz[emitted])
            signals[name]=values
        masks=dict(contact=contact,slip=contact & data['slip'].astype(bool),
                   non_slip_contact=contact & ~data['slip'].astype(bool),
                   stable_hold=contact & (data['phase']=='hold'))
        groups={}
        for name,mask in masks.items():
            if not mask.any():continue
            groups[name]=dict(frames=int(mask.sum()),median_fem_depth_mm=float(np.median(xyz[mask,2])),
                mean_left_force_n=float(data['normal_force_n'][mask].mean()),
                mean_right_force_n=float(data['passive_normal_force_n'][mask].mean()),
                signals={})
            for signal_name,values in signals.items():
                norms=np.linalg.norm(values[mask].reshape(-1,5,3),axis=2)
                groups[name]['signals'][signal_name]=dict(
                    pooled_chip_norm_median_ut=float(np.median(norms)),
                    mean_per_chip_norm_ut=norms.mean(0).tolist(),
                    component_rms_ut=float(np.sqrt(np.mean(values[mask]**2))))
        report['scenes'][scene]=dict(source=str(source),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),groups=groups)
        np.savez_compressed(output/f'{scene}_query_replay.npz',time_s=data['time_s'],
            xyz_mm=xyz,contact=contact,slip=data['slip'],**signals)
        fig,axs=plt.subplots(3,1,figsize=(12,9),layout='constrained',sharex=True)
        t=data['time_s']
        axs[0].plot(t,data['normal_force_n'],label='Left pad')
        axs[0].plot(t,data['passive_normal_force_n'],label='Right pad')
        axs[0].set_ylabel('Normal force (N)');axs[0].legend()
        axs[1].plot(t,xyz[:,2]);axs[1].set_ylabel('FEM compression (mm)')
        for name,values in signals.items():
            axs[2].plot(t,np.linalg.norm(values.reshape(-1,5,3),axis=2).mean(1),label=name)
        axs[2].set_yscale('symlog',linthresh=1.)
        axs[2].set(ylabel='Mean chip norm (uT)',xlabel='Simulation time (s)');axs[2].legend()
        fig.suptitle(f'{scene}: identical physical trajectory, four uncalibrated magnetic outputs\nOffline query replay; no model deployed')
        fig.savefig(output/f'{scene}_query_replay.png',dpi=150);plt.close(fig)
    (output/'sim_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--pseudo-weight',type=float,default=.1)
    p.add_argument('--evaluate-sim',action='store_true',help='Evaluate saved models on existing cube/cable logs without retraining')
    p.add_argument('--source-only-control',action='store_true',help='Run matched continued ReSkin training with no pseudo-label loss')
    args=p.parse_args()
    if args.source_only_control:
        if args.evaluate_sim:p.error('Choose training control or offline evaluation, not both')
        train_source_control(args.output)
        return
    if args.evaluate_sim:
        evaluate_sim(args.output)
        return
    if args.steps<1 or not np.isfinite(args.pseudo_weight) or args.pseudo_weight<=0:
        p.error('steps and pseudo-weight must be positive')
    if any((args.output/name/'model.safetensors').exists() for name in ('official_teacher','train_split_teacher')):
        p.error('Choose a new output directory; preserving the previous experiment')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    baseline=ResponseModel(MAPPED_OUTPUT); meta=baseline.meta
    b,slip,timestamps=read_csv(RESOURCE_DIR/'data/cable.csv')
    amplitude=np.linalg.norm(b.reshape(-1,5,3),axis=2).max(1)
    reset_rows=np.flatnonzero((amplitude[:-1]>100)&(amplitude[1:]<50)&((amplitude[:-1]-amplitude[1:])>100))+1
    if len(reset_rows)!=3:
        raise ValueError('Expected the three documented reset drops in the pinned four-pull cable file')
    boundary=int(reset_rows[-1])
    train_rows=np.arange(len(b))<boundary
    held=~train_rows
    xy,b_source,_=read_data()
    b_source=b_source[:,channel_indices(meta['mapping'])]
    split=json.loads((MAPPED_OUTPUT/'report.json').read_text())['split_xy']
    masks={name:np.array([tuple(q) in set(map(tuple,positions)) for q in xy[:,:2]]) for name,positions in split.items()}
    train_x=torch.tensor((xy[masks['train']]-meta['x_mean'])/meta['x_scale'],dtype=torch.float32)
    train_y=torch.tensor((b_source[masks['train']]-meta['y_mean'])/meta['y_scale'],dtype=torch.float32)
    source_baseline={name:error_report(b_source[mask],predict(baseline.model,meta,xy[mask])) for name,mask in masks.items()}
    report=dict(status='PSEUDO-LABEL EXPERIMENT; not measured position/depth calibration',
        baseline=str(MAPPED_OUTPUT), baseline_weights_sha256=hashlib.sha256((MAPPED_OUTPUT/'model.safetensors').read_bytes()).hexdigest(),
        pollen_file_sha256=hashlib.sha256((RESOURCE_DIR/'data/cable.csv').read_bytes()).hexdigest(),
        pollen_split=dict(reset_rows=reset_rows.tolist(),train_rows=[0,boundary-1],heldout_rows=[boundary,len(b)-1],
                          train_frames=int(train_rows.sum()),heldout_frames=int(held.sum()),heldout_slip_frames=int((held&(slip==1)).sum()),
                          rule='Whole final pull held out after the third signal-reset discontinuity; no shuffle across time.'),
        pseudo_gate='finite xyz; xy within source bounding box 0..16 mm; 0 < estimated depth <= 2 mm; max chip norm >= 50 uT. Signal gate is heuristic, not a contact ground-truth label.',
        training=dict(steps=args.steps,batch_real=256,batch_pseudo=256,optimizer='Adam',learning_rate=.0003,
                      weight_decay=1e-5,pseudo_loss_weight=args.pseudo_weight,seed=42,
                      preprocessing='Frozen original forward-model normalization; final fixed-step checkpoint; no hyperparameter selection on held-out Pollen pull.'),
        evaluation_caveats=['Pollen input xyz is inferred from the very B being reconstructed, so held-out reconstruction is not a forward prediction test with independently measured xyz.',
                            'Official pretrained teacher may have seen ReSkin test positions. Its ReSkin score is diagnostic only; train_split_teacher avoids that teacher split contamination.',
                            'Pseudo samples add observations, not independently measured depth/position labels.'],
        reskin_baseline=source_baseline,branches={})
    results={}
    for name,official in [('official_teacher',True),('train_split_teacher',False)]:
        teacher=InverseModel(published=official)
        estimates=teacher.predict(b)
        valid=(np.isfinite(estimates).all(1)&(estimates[:,:2]>=0).all(1)&(estimates[:,:2]<=16).all(1)
               &(estimates[:,2]>0)&(estimates[:,2]<=2.)&(amplitude>=50))
        accepted=train_rows & valid
        if not accepted.any():raise ValueError('No accepted pseudo samples')
        px=torch.tensor((estimates[accepted]-meta['x_mean'])/meta['x_scale'],dtype=torch.float32)
        py=torch.tensor((b[accepted]-meta['y_mean'])/meta['y_scale'],dtype=torch.float32)
        torch.manual_seed(42)
        student=copy.deepcopy(baseline.model)
        optimizer=torch.optim.Adam(student.parameters(),lr=.0003,weight_decay=1e-5)
        history=[]
        for step in range(1,args.steps+1):
            student.train();optimizer.zero_grad()
            ri=torch.randint(len(train_x),(256,));pi=torch.randint(len(px),(256,))
            real_loss=(student(train_x[ri])-train_y[ri]).square().mean()
            pseudo_loss=(student(px[pi])-py[pi]).square().mean()
            loss=real_loss+args.pseudo_weight*pseudo_loss
            loss.backward();optimizer.step()
            if step==1 or step%100==0:
                student.eval()
                v=error_report(b_source[masks['validation']],predict(student,meta,xy[masks['validation']]))
                history.append(dict(step=step,real_standardized_mse=float(real_loss.detach()),pseudo_standardized_mse=float(pseudo_loss.detach()),reskin_validation_rmse_ut=v['rmse_ut']))
                print(name,step,history[-1],flush=True)
        student.eval()
        before=predict(baseline.model,meta,estimates)
        after=predict(student,meta,estimates)
        subsets={'train_accepted':accepted,'heldout_all':held,'heldout_slip':held&(slip==1),
                 'heldout_signal_gate':held&(amplitude>=50),'heldout_pseudo_gate':held&valid}
        branch=dict(teacher='Official published weights' if official else 'Official architecture trained on only original forward train XY locations',
                    accepted_pseudo_count=int(accepted.sum()),dropped_train_count=int((train_rows&~valid).sum()),
                    new_independent_xyz_measurements=0, pseudo_estimated_depth_range_mm=[float(estimates[accepted,2].min()),float(estimates[accepted,2].max())],
                    reskin_after={k:error_report(b_source[v],predict(student,meta,xy[v])) for k,v in masks.items()},
                    pollen_reconstruction={k:dict(frames=int(mask.sum()),before=error_report(b[mask],before[mask]),after=error_report(b[mask],after[mask])) for k,mask in subsets.items() if mask.any()},history=history)
        dest=args.output/name;dest.mkdir(exist_ok=True)
        updated=copy.deepcopy(meta)
        updated.update(model='LOCAL forward MLP augmented with MagNet pseudo-labelled Pollen observations; not physically calibrated',
                       augmentation=dict(teacher=branch['teacher'],split=report['pollen_split'],pseudo_gate=report['pseudo_gate'],training=report['training'],independent_xyz_labels=0),
                       pseudo_estimated_depth_range_mm=branch['pseudo_estimated_depth_range_mm'])
        # Measured depths/known XY remain the original real calibration support;
        # pseudo-labels must not turn into a claim of measured shallow coverage.
        (dest/'model.json').write_text(json.dumps(updated,indent=2)+'\n')
        save_file(student.state_dict(),str(dest/'model.safetensors'))
        (dest/'report.json').write_text(json.dumps(branch,indent=2)+'\n')
        np.savez_compressed(dest/'pollen_predictions.npz',row_index=np.arange(len(b)),input_magnetic_ut=b,
            estimated_xyz_mm=estimates,used_for_training=accepted,heldout=held,slip=slip,
            before_delta_ut=before,after_delta_ut=after)
        # Native units and unchanged signal scaling for fair downstream checks.
        results[name]=(before,after)
        report['branches'][name]=branch
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axs=plt.subplots(2,1,figsize=(12,8),layout='constrained',sharex=True)
    indices=np.flatnonzero(held)
    for ax,(name,(before,after)) in zip(axs,results.items()):
        ax.plot(indices,np.sqrt(np.mean((before[held]-b[held])**2,axis=1)),label='Before augmentation')
        ax.plot(indices,np.sqrt(np.mean((after[held]-b[held])**2,axis=1)),label='After augmentation')
        ax.fill_between(indices,0,1,where=slip[held].astype(bool),transform=ax.get_xaxis_transform(),color='orange',alpha=.15,label='Public slip label')
        ax.set(title=name,ylabel='Reconstruction RMSE (uT)');ax.legend()
    axs[-1].set_xlabel('Held-out final pull: original CSV row')
    fig.suptitle('Pseudo-label augmentation: B -> estimated xyz -> B\nReconstruction evaluation, NOT measured depth accuracy')
    fig.savefig(args.output/'heldout_reconstruction.png',dpi=150);plt.close(fig)
    print('Saved',args.output,flush=True)


if __name__=='__main__':main()
