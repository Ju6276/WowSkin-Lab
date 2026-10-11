"""Reproducible cube comparison. Run once per environment; results stay local."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, response_configuration, scripted_action
from wowskin_lab.paths import ROOT

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skin-dof', choices=['full', 'trilinear'], required=True)
    p.add_argument('--physics-step-us', type=int)
    p.add_argument('--response', choices=['fitted', 'network'], default='fitted')
    p.add_argument('--reskin-extrapolate', action='store_true',
                   help='Explicitly allow experimental single-probe contact predictions')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path, help='NPZ from the full benchmark')
    args = p.parse_args()
    args.task, args.ideal, args.seed = 'pick_cube', True, 0
    if args.response == 'network':
        args.reskin_model = ROOT/'outputs/reskin_forward_mapped'
    else:
        args.magnetic_fit = ROOT/'configs/magnetic_fit_pollen_nonslip.json'
    cfg = response_configuration(args)
    env = PandaPickCubeTactileEnv(cfg, pollen=False)
    rows, elapsed = [], []
    for i in range(850):
        start = time.perf_counter()
        rows.append(env.step(scripted_action(env))[-1])
        elapsed.append(time.perf_counter()-start)
        if (i+1) % 100 == 0:
            print(f'{i+1}/850: lift={rows[-1]["lift_m"]:.4f} m', flush=True)
    arrays = {key: np.array([r[key] for r in rows]) for key in
              ('time_s', 'lift_m', 'normal_force_n', 'passive_normal_force_n',
               'magnetic_ut', 'ideal_delta_ut', 'min_volume_ratio', 'passive_min_volume_ratio',
               'slip', 'slip_speed_m_per_s', 'grasped')}
    held = (arrays['time_s'] > 4.8) & (arrays['time_s'] < 5.4)
    finite_held = held & np.isfinite(arrays['ideal_delta_ut']).all(axis=1)
    def rms(values):
        return float(np.sqrt(np.mean(values**2))) if values.size else None
    report = dict(engine=mujoco.__version__, skin_dof=args.skin_dof, timestep_s=env.model.opt.timestep,
                  nv=env.model.nv, held_sample_ms=float(np.median(np.array(elapsed)[held])*1000),
                  max_lift_m=float(arrays['lift_m'].max()), held_grasp=bool(arrays['grasped'][held].all()),
                  held_slip_fraction=float(arrays['slip'][held].mean()), final_lift_m=float(arrays['lift_m'][-1]),
                  min_volume_ratio=float(min(arrays['min_volume_ratio'].min(), arrays['passive_min_volume_ratio'].min())),
                  held_force_n=float(arrays['normal_force_n'][held].mean()),
                  held_field_rms_ut=rms(arrays['ideal_delta_ut'][finite_held]),
                  held_field_samples=int(finite_held.sum()), held_total_samples=int(held.sum()),
                  warnings=env.data.warning.number.tolist(), calibrated=False,
                  comparison_note='Combined engine/integrator/DOF/boundary change; not an isolated DOF ablation')
    if args.reference:
        with np.load(args.reference) as reference:
            np.testing.assert_allclose(arrays['time_s'], reference['time_s'], atol=1e-8)
            paired = finite_held & np.isfinite(reference['ideal_delta_ut']).all(axis=1)
            report['held_field_difference_rmse_ut'] = rms(
                arrays['ideal_delta_ut'][paired]-reference['ideal_delta_ut'][paired])
            report['held_difference_samples'] = int(paired.sum())
    if args.response == 'network':
        for key in ('fitted_dipole_ideal_delta_ut', 'reskin_query_mm', 'reskin_in_training_domain',
                    'reskin_status', 'reskin_prediction_emitted'):
            arrays[key] = np.array([r[key] for r in rows])
        fit = arrays['fitted_dipole_ideal_delta_ut'].reshape(-1, 5, 3)[:, [0, 4, 2, 1, 3], :].reshape(-1, 15)
        arrays['fitted_in_network_candidate_order_ut'] = fit
        report.update(comparison_note='Network and fitted dipole use identical physical states; position groups aligned, magnetic axes unverified',
                      fitted_held_rms_ut=float(np.sqrt(np.mean(fit[held]**2))),
                      network_vs_fitted_rmse_ut=rms(arrays['ideal_delta_ut'][finite_held]-fit[finite_held]),
                      experimental_extrapolation=bool(args.reskin_extrapolate),
                      missing_field_policy='NaN, no zero filling; reported metrics include only finite held frames',
                      held_input_min_mm=arrays['reskin_query_mm'][held].min(0).tolist(),
                      held_input_max_mm=arrays['reskin_query_mm'][held].max(0).tolist(),
                      held_in_training_domain_fraction=float(arrays['reskin_in_training_domain'][held].mean()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'), **arrays)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
