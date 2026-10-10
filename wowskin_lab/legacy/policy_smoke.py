"""ARCHIVED simplified experiment; NOT an official reproduction entry point.

One method per process: upstream repositories use conflicting top-level module
names (agent, utils, read_data). No real-robot environment is imported here.
"""
import argparse
import importlib
import json
from pathlib import Path
import pickle
import random
import sys

import numpy as np

from wowskin_lab.paths import ROOT


def build(method, path, device='cpu', seed=0, agent_needed=True):
    import torch
    from omegaconf import OmegaConf
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    path = Path(path).resolve()
    with path.open('rb') as stream:
        data = pickle.load(stream)
    expected = 'visk' if method == 'visk' else 'ftf_robot_p3po'
    if data.get('metadata', {}).get('schema') != expected:
        raise ValueError(f'Expected {expected} export, not an arbitrary PKL')
    module_root = ROOT/'third_party'/('visuoskin' if method == 'visk' else 'feel-the-force/point_policy')
    if not module_root.exists():
        raise FileNotFoundError('Run scripts/setup_upstream.py first')
    if 'agent' in sys.modules or 'read_data' in sys.modules:
        raise RuntimeError('Use a fresh process per official method to avoid upstream import-name collisions')
    sys.path.insert(0, str(module_root))
    keys = ['pixels1', 'pixels2', 'pixels51', 'pixels52'] if method == 'visk' else ['pixels1', 'pixels2']
    first = data['observations'][0]
    queries = 10
    if method == 'visk':
        Dataset = importlib.import_module('read_data.xarm_env_aa').BCDataset
        dataset = Dataset(path=path.parent, tasks=[path.stem], num_demos_per_task=5000,
                          temporal_agg=True, num_queries=queries, img_size=128,
                          action_after_steps=1, store_actions=True, pixel_keys=keys,
                          aux_keys=['sensor0'], subsample=1, skip_first_n=0, relative_actions=True,
                          random_mask_proprio=False,
                          sensor_params=OmegaConf.create(dict(sensor_type='reskin',
                              subtract_sensor_baseline=True, num_sensors=1, scale_floor=1e-6)))
    else:
        Dataset = importlib.import_module('read_data.p3po').BCDataset
        dataset = Dataset(path=path.parent, tasks=path.stem, num_demos_per_task=5000,
                          history=True, history_len=10, temporal_agg=True, num_queries=queries,
                          img_size=[128, 128], action_after_steps=1, use_robot_points=True,
                          num_robot_points=9, use_object_points=True, num_object_points=8,
                          point_dim=3, pixel_keys=keys, action_type='absolute', subsample=1,
                          skip_first_n=0, gt_depth=False, predict_force=True, binarize_gripper_data=False)
    if not agent_needed:
        return dataset, None, data['metadata']
    obs_shape = {key: (3, first[key].shape[1], first[key].shape[2]) for key in keys}
    common = dict(obs_shape=obs_shape, action_shape=(7,), device=device, lr=1e-4,
                  hidden_dim=256, stddev_schedule=.1, use_tb=True, policy_head='deterministic',
                  pixel_keys=keys, temporal_agg=True, max_episode_len=dataset._max_episode_len+1,
                  num_queries=queries)
    if method == 'visk':
        obs_shape.update(sensor0=(15,), proprioceptive=(8,))
        Agent = importlib.import_module('agent.bc').BCAgent
        agent = Agent(**common, stddev_clip=.3, augment=True, encoder_type='resnet',
                      policy_type='gpt', aux_keys=['sensor0'], use_aux_inputs=True,
                      train_encoder=True, norm=False, separate_encoders=False, use_actions=False)
    else:
        Agent = importlib.import_module('agent.p3po').BCAgent
        agent = Agent(**common, history=True, history_len=10, eval_history_len=10,
                      use_robot_points=True, num_robot_points=9, use_object_points=True,
                      num_object_points=8, point_dim=3, predict_force=True, mask_force=False)
    return dataset, agent, data['metadata']


def run(method, path, steps=0, batch_size=2, device='cpu', output=None, seed=0):
    import torch
    dataset, agent, metadata = build(method, path, device, seed, agent_needed=steps > 0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0)
    iterator = iter(loader)
    first_batch = next(iterator)
    for key, value in first_batch.items():
        if not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite training tensor: {key}')
    report = dict(method=method, dataset=str(Path(path).resolve()), steps=steps, batch_size=batch_size,
                  device=device, seed=seed, upstream=json.loads((ROOT/'configs/upstream.json').read_text())[method],
                  dataset_metadata=metadata, batch_shapes={k: list(v.shape) for k, v in first_batch.items()},
                  status='loader_checked' if not steps else 'training_completed', losses=[])
    print(json.dumps(report['batch_shapes'], indent=2), flush=True)
    if steps:
        if output is None:
            raise ValueError('Training requires --output')
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        for step in range(steps):
            metrics = agent.update(iterator, step)
            metrics = {k: float(v) for k, v in metrics.items()}
            if not all(np.isfinite(v) for v in metrics.values()):
                raise RuntimeError(f'Nonfinite loss at step {step}')
            report['losses'].append(dict(step=step, **metrics))
            if step % 100 == 0 or step == steps-1:
                print(f'{method} update {step+1}/{steps}: {metrics}', flush=True)
        payload = agent.save_snapshot()
        # Keep checkpoints readable without serializing live optimizer objects.
        for key, value in list(payload.items()):
            if isinstance(value, torch.optim.Optimizer):
                payload[key] = value.state_dict()
        payload.update(stats=dataset.stats, metadata=report, completed_steps=steps)
        torch.save(payload, output/'checkpoint.pt')
    elif output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
    if output is not None:
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main(method, argv=None, check_only=False):
    parser = argparse.ArgumentParser(description=f'Archived {method} experiment, NOT official reproduction')
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--steps', type=int, default=0 if check_only else 10000)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    if args.steps < 0 or args.batch_size < 1:
        parser.error('steps >= 0 and batch-size >= 1 are required')
    if args.steps and args.output is None:
        parser.error('--output is required for training')
    return run(method, args.dataset, args.steps, args.batch_size, args.device, args.output, args.seed)
