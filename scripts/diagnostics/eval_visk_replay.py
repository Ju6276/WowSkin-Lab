"""Evaluate a trusted local ViSk checkpoint on recorded observations, on CPU.

Uses the prepared upstream BCDataset, BCAgent.act, checkpoint normalization,
and temporal aggregation. This is teacher-forced replay on the training data,
not a robot rollout, success-rate measurement, or held-out validation.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    training = args.training.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    status = json.loads((training/'status.json').read_text())
    source = Path(status['source'])
    manifest = json.loads((source.parent/'manifest.json').read_text())
    for name in ('agent/bc.py', 'agent/networks/gpt.py', 'agent/networks/policy_head.py',
                 'agent/networks/rgb_modules.py'):
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != manifest['original_sha256'][name]:
            raise ValueError(f'Policy source differs from pinned upstream: {name}')
    if hashlib.sha256(Path(status['dataset']).read_bytes()).hexdigest() != status['dataset_sha256']:
        raise ValueError('Training dataset has changed')
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    sys.path.insert(0, str(source))
    import hydra
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    import utils

    torch.set_num_threads(2)
    cfg = OmegaConf.load(training/'config.yaml')
    cfg.device = 'cpu'
    utils.set_seed_everywhere(cfg.seed)
    dataset = hydra.utils.instantiate(cfg.expert_dataset)
    episode = dataset._episodes[0][args.episode]
    obs = episode['observation']
    # These checkpoints were generated locally; upstream includes Python
    # optimizer/timer objects, so weights_only loading is not compatible.
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    stats = payload['stats']
    for key in stats:
        for stat in stats[key]:
            np.testing.assert_allclose(stats[key][stat], dataset.stats[key][stat], rtol=0, atol=0)
    shapes = {key: (3, cfg.img_size, cfg.img_size) for key in cfg.suite.pixel_keys}
    shapes.update({key: obs[f'{key}_states'].shape[1:] for key in cfg.suite.aux_keys})
    shapes['proprioceptive'] = (8,)
    cfg.suite.task_make_fn.max_episode_len = int(payload['max_episode_len'])
    agent = hydra.utils.instantiate(cfg.agent, obs_shape=shapes, action_shape=(7,))
    agent.load_snapshot(payload, eval=True)
    agent.buffer_reset()

    # Upstream _sample() pairs observation[j] with action[j+1] when action
    # chunking is enabled. Keep that exact target alignment, including the
    # initial duplicated observation, instead of inventing new action labels.
    count = len(obs[cfg.suite.pixel_keys[0]]) - 1
    predictions, first_queries, targets, timings = [], [], [], []
    with torch.inference_mode():
        for step in range(count):
            observation = {key: obs[key][step].transpose(2, 0, 1).copy()
                           for key in cfg.suite.pixel_keys}
            observation.update({key: obs[f'{key}_states'][step].copy()
                                for key in cfg.suite.aux_keys})
            started = time.perf_counter()
            action = agent.act(observation, None, stats, step,
                               int(payload['_global_step']), eval_mode=True)
            timings.append((time.perf_counter()-started)*1000)
            if np.shape(action) != (7,) or not np.isfinite(action).all():
                raise ValueError(f'Invalid action at replay step {step}: {action}')
            predictions.append(action)
            chunk0 = agent.all_time_actions[step, step].cpu().numpy()
            first_queries.append(chunk0 * (stats['actions']['max']-stats['actions']['min'])
                                 + stats['actions']['min'])
            targets.append(episode['action'][step+1])
    pred, target, first = map(np.asarray, (predictions, targets, first_queries))

    def errors(values):
        difference = values-target
        return dict(translation_component_mae_mm=(np.abs(difference[:, :3]).mean(axis=0)*1000).tolist(),
                    translation_vector_rmse_mm=float(np.sqrt(np.mean(np.sum(difference[:, :3]**2, axis=1)))*1000),
                    rotation_vector_component_mae_deg=np.rad2deg(np.abs(difference[:, 3:6]).mean(axis=0)).tolist(),
                    gripper_closure_mae=float(np.abs(difference[:, 6]).mean()))

    warm_timings = np.asarray(timings[min(5, count-1):])
    report = dict(scope='teacher-forced replay of training demonstration; NOT closed-loop or held-out evaluation',
                  checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  checkpoint_global_step=int(payload['_global_step']), device='cpu', cpu_threads=2,
                  source=str(source), upstream_revision=manifest['upstream']['revision'],
                  dataset=status['dataset'], dataset_sha256=status['dataset_sha256'], episode=args.episode,
                  inference_steps=count, subsample=int(cfg.dataloader.bc_dataset.subsample),
                  input_cameras=list(cfg.suite.pixel_keys), input_tactile=list(cfg.suite.aux_keys),
                  raw_action_chunk_shape=[int(cfg.num_queries), 7], output_action_shape=list(pred.shape),
                  normalization='checkpoint stats, verified identical to original loader',
                  target_alignment='original BCDataset._sample(): observation[j] -> action[j+1]',
                  image_channels='unchanged OpenCV BGR order in upstream converted PKL',
                  action_units=['m', 'm', 'm', 'rad', 'rad', 'rad', 'closure fraction'],
                  temporal_aggregation=True, all_actions_finite=bool(np.isfinite(pred).all()),
                  latency_ms=dict(first=timings[0], median=float(np.median(warm_timings)),
                                  p95=float(np.percentile(warm_timings, 95))),
                  aggregated_action_errors=errors(pred), first_query_errors=errors(first),
                  first_action=pred[0].tolist(), first_target=target[0].tolist(),
                  policy_rollout_validated=False)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    labels = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'gripper']
    with (output/'actions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['replay_step', 'inference_ms']+
                        [f'{prefix}_{key}' for prefix in ('predicted', 'target', 'first_query') for key in labels])
        for i in range(count):
            writer.writerow([i, timings[i], *pred[i], *target[i], *first[i]])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(11, 10), constrained_layout=True)
    for k, ax in enumerate(axes.flat):
        if k == 7:
            ax.axis('off')
            ax.text(.02, .8, 'Recorded observations only.\nTraining episode, not held out.\n'
                    'No actions applied to MuJoCo.\n'
                    f'CPU median: {report["latency_ms"]["median"]:.1f} ms', va='top')
            continue
        scale, unit = (1000, 'mm') if k < 3 else ((180/np.pi, 'deg') if k < 6 else (1, 'closure'))
        ax.plot(target[:, k]*scale, label='Official loader target', color='#333333')
        ax.plot(pred[:, k]*scale, label='Policy + temporal aggregation', color='#0077bb')
        ax.set_ylabel(f'{labels[k]} ({unit})'); ax.set_xlabel('Replay step')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'ViSk checkpoint {checkpoint.stem}: recorded-input inference')
    fig.savefig(output/'actions.png', dpi=140)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
