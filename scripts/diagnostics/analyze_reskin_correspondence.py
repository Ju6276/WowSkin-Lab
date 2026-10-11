"""Exploratory channel matching; this does not identify magnetic axis rotations."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wowskin_lab.baselines.reskin_forward import read_data
from wowskin_lab.paths import ROOT


def main():
    out = ROOT/'outputs/diagnostics/reskin_correspondence'
    out.mkdir(parents=True, exist_ok=True)
    x, y, _ = read_data()
    source = ROOT/'outputs/diagnostics/five_point_20261009'
    summary = json.loads((source/'summary.json').read_text())
    snapshot = np.load(source/'analysis_snapshot.npz')
    t, raw, baseline = (snapshot[k] for k in ('host_elapsed_s', 'raw_magnetic_ut', 'reference_ut'))
    repeats, real_ut = [], []
    for group in summary['groups']:
        values = np.array([np.linalg.norm((raw[abs(t-peak)<.05].mean(0)-baseline).reshape(5, 3), axis=1)
                           for peak in group['exploratory_peak_times_s']])
        real_ut.append(values.mean(0))
        repeats.append(values/np.linalg.norm(values, axis=1, keepdims=True))
    def real_signature(rows):
        a = np.array([r.mean(0) for r in rows])
        return a/np.linalg.norm(a, axis=1, keepdims=True)
    real = real_signature(repeats)
    locations = np.unique(x[:, :2], axis=0)
    maps = {}
    for depth in (1.5, 2.):
        maps[depth] = np.array([np.linalg.norm(y[(x[:, 2]==depth)&np.all(x[:, :2]==xy, axis=1)].mean(0).reshape(5, 3), axis=1) for xy in locations])
    # Response lobes support these *regions*, not an exact localization of chips.
    source_regions = np.array([[0, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])
    target_regions = np.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1]])
    candidates = []
    for flip in (1, -1):
        for k in range(4):
            angle = k*np.pi/2
            q = np.round([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])@np.diag([flip, 1])
            perm = [int(np.flatnonzero(np.all(source_regions@q.T==v, axis=1))[0]) for v in target_regions]
            trials = []
            signatures = []
            for radius in (1., 2., 3.):
                for depth in (1.5, 2.):
                    # Mean of vector norms: do NOT cancel opposite vectors across different press positions.
                    r = np.array([maps[depth][np.linalg.norm(locations-(8+7*pt), axis=1)<=radius].mean(0) for pt in source_regions])
                    r = r[perm][:, perm]
                    r /= np.linalg.norm(r, axis=1, keepdims=True)
                    signatures.append(r)
                    trials.append(dict(radius_mm=radius, depth_mm=depth, cosine=float((r*real).sum(1).mean())))
            candidates.append(dict(source_groups_for_wowskin=perm, dataset_xy_to_board=q.tolist(),
                                   score=float(np.mean([v['cosine'] for v in trials])), trials=trials,
                                   signatures=np.array(signatures)))
    candidates.sort(key=lambda c: c['score'], reverse=True)
    rng = np.random.default_rng(42)
    wins = np.zeros(len(candidates), dtype=int)
    for _ in range(1000):
        r = real_signature([v[rng.integers(len(v), size=len(v))] for v in repeats])
        scores = [float((c['signatures']*r).sum(-1).mean()) for c in candidates]
        wins[np.argmax(scores)] += 1
    for c, count in zip(candidates, wins):
        c['bootstrap_win_fraction'] = float(count/1000)
        del c['signatures']
    selected = candidates[0]
    profile = dict(status='PROVISIONAL response-pattern candidate, not identified hardware mapping',
                   source_groups_for_wowskin=selected['source_groups_for_wowskin'],
                   output_positions=['center', 'left', 'right', 'up', 'down'],
                   dataset_xy_to_board=selected['dataset_xy_to_board'], dataset_origin_mm=[8., 8.],
                   xy_scale=1., axes='Source ReSkin XYZ retained; no WowSkin magnetic-axis calibration',
                   evidence='outputs/diagnostics/reskin_correspondence/report.json',
                   limitations='Uncontrolled hand depth/contact shape; near-symmetric layout; ranking is not a mapping probability. The XY origin/7mm regions are analysis assumptions. No footprint scaling.')
    (out/'mapping_candidate.json').write_text(json.dumps(profile, indent=2)+'\n')
    report = dict(real_source=str(source), dataset_source='assets/3dcal/probe_data.csv',
                  method='100ms mean XYZ at previously detected real peaks, baseline subtraction, per-chip norms, per-press L2 normalization; compare whole five-channel response patterns across eight D4 hypotheses',
                  real_mean_peak_norm_ut=np.array(real_ut).tolist(),
                  score_gap=candidates[0]['score']-candidates[1]['score'],
                  candidates=candidates, bootstrap_note='Only resamples 2-3 exploratory hand-press peaks per location. Not statistical identification confidence.',
                  region_assumptions='Dataset origin (8,8), 7mm offsets; radii 1,2,3mm; same sensor spacing; x/y reflection hypotheses allowed, magnetic axes left untouched.')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), layout='constrained')
    for row, depth in enumerate((1.5, 2.)):
        for group in range(5):
            ax = axes[row, group]
            a = ax.scatter(*locations.T, c=maps[depth][:, group], s=22, cmap='viridis', vmin=0)
            ax.set(xlabel='Dataset X (mm)', ylabel='Dataset Y (mm)', aspect='equal')
            ax.set_title(f'Group {group} | depth {depth} mm', fontsize=10)
            ax.tick_params(labelsize=8)
            bar = fig.colorbar(a, ax=ax, shrink=.7)
            bar.set_label('Delta B norm (uT)', size=9)
            bar.ax.tick_params(labelsize=8)
    fig.suptitle('Measured ReSkin spatial response; mean of 10 readings minus published no-contact baseline')
    fig.savefig(out/'spatial_response.png', dpi=150)
    plt.close(fig)
    print(json.dumps(dict(profile=profile, ranking=candidates, real_mean_peak_norm_ut=real_ut), default=lambda a:a.tolist(), indent=2))


if __name__ == '__main__':
    main()
