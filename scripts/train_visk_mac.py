"""Run prepared, original ViSk WorkspaceIL with explicit Mac runtime changes.

No policy, loss, normalizer, sampler, batch, or horizon replacement. On MPS,
float64 batches are cast BEFORE transfer (upstream immediately uses .float()).
Unsupported MPS operators use PyTorch's CPU fallback. DataLoader workers are
configurable and explicitly joined, avoiding Python 3.12 exit-time hangs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    parser.add_argument('--workers', type=int, default=0,
                        help='0 avoids Python 3.12 multiprocessing resource-tracker shutdown hangs on this Mac')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--save-every', type=int, default=3000,
                        help='Periodic checkpoint interval in official loop step numbers; always save the final update too')
    parser.add_argument('--task', default='pick_cube')
    args = parser.parse_args()
    if args.steps < 1 or args.save_every < 1 or args.workers < 0:
        parser.error('steps and save-every must be positive; workers must be nonnegative')
    root = Path(__file__).resolve().parents[1]
    source = args.workspace.resolve()/'source'
    manifest = json.loads((args.workspace.resolve()/'manifest.json').read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(name, '4')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    if args.device == 'mps':
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    for path in (root/'third_party/dependencies/Open-Teach', source/'envs/xarm-env', source):
        sys.path.insert(0, str(path))
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import train_bc
    import replay_buffer
    import utils

    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; explicitly select --device cpu')
    torch.set_num_threads(4)
    with initialize_config_dir(config_dir=str(source/'cfgs'), version_base=None):
        cfg = compose(config_name='config', overrides=[f'suite.task.tasks=[{args.task}]',
                          f'device={args.device}', f'suite.num_train_steps={args.steps}',
                          f'suite.save_every_steps={args.save_every}'])
    if cfg.eval or cfg.load_bc:
        raise ValueError('This runner supports offline training from scratch, no robot services')
    report = dict(status='initializing', pid=os.getpid(), source=str(source), device=args.device,
        action_profile=manifest.get('action_profile', 'published-state'),
        requested_updates=args.steps, completed_updates=0, dataset=str(Path(cfg.root_dir)/'processed_data_pkl_aa'/f'{args.task}.pkl'),
        batch_size=int(cfg.batch_size), num_queries=int(cfg.num_queries), subsample=int(cfg.dataloader.bc_dataset.subsample),
        upstream_source_sha256={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source/'train_bc.py', source/'agent/bc.py', source/'utils.py', source/'read_data/xarm_env_aa.py')},
        runtime_changes=dict(dataloader_workers=args.workers, upstream_dataloader_workers=16,
            checkpoint_interval_steps=args.save_every, upstream_checkpoint_interval_steps=500,
            float64_cast_before_mps_transfer=args.device == 'mps',
            unsupported_mps_operators_on_cpu=args.device == 'mps', explicit_worker_shutdown=True,
            final_checkpoint=True), policy_rollout_validated=False,
        scope='Official training code on local simulated cube demonstration; not paper task reproduction')
    dataset = Path(report['dataset'])
    report['dataset_sha256'] = hashlib.sha256(dataset.read_bytes()).hexdigest()
    def save_report():
        tmp = output/'status.tmp'
        tmp.write_text(json.dumps(report, indent=2)+'\n')
        tmp.replace(output/'status.json')
    save_report()
    OmegaConf.save(cfg, output/'config.yaml')

    def make_loader(iterable, batch_size):
        return torch.utils.data.DataLoader(iterable, batch_size=batch_size,
            num_workers=args.workers, pin_memory=args.device != 'mps',
            worker_init_fn=replay_buffer._worker_init_fn)
    train_bc.make_expert_replay_loader = make_loader
    original_to_torch = utils.to_torch
    if args.device == 'mps':
        def to_mps(xs, device):
            # All floating features used by upstream update are float32; doing
            # that same conversion on CPU avoids MPS's unsupported float64.
            prepared = {}
            for key, value in xs.items():
                tensor = torch.as_tensor(value)
                prepared[key] = tensor.float() if tensor.dtype == torch.float64 else tensor
            return original_to_torch(prepared, device)
        utils.to_torch = to_mps

    os.chdir(output)
    workspace = None
    started = time.monotonic()
    try:
        workspace = train_bc.WorkspaceIL(cfg)
        update = workspace.agent.update
        def checked_update(iterator, step):
            metrics = update(iterator, step)
            loss = float(metrics['actor_loss'])
            if not __import__('math').isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at update {step+1}')
            report.update(status='running', completed_updates=step+1, last_actor_loss=loss,
                          elapsed_s=time.monotonic()-started)
            report.setdefault('first_actor_loss', loss)
            if step < 3 or (step+1) % 100 == 0 or step+1 == args.steps:
                save_report()
                print(f"ViSk {step+1}/{args.steps}: loss={loss:.6f}, elapsed={report['elapsed_s']:.1f}s", flush=True)
            return metrics
        workspace.agent.update = checked_update
        workspace.train()  # Original loop, optimizer updates and periodic saves.
        workspace.save_snapshot()  # Upstream omits a checkpoint AFTER last update.
        snapshot = output/'snapshot'/f'{workspace.global_step}.pt'
        payload = torch.load(snapshot, map_location='cpu', weights_only=False)
        def all_finite(value):
            if torch.is_tensor(value):
                return bool(torch.isfinite(value).all())
            if isinstance(value, dict):
                return all(all_finite(v) for v in value.values())
            if isinstance(value, (tuple, list)):
                return all(all_finite(v) for v in value)
            if isinstance(value, torch.optim.Optimizer):
                return all_finite(value.state_dict())
            return True
        if not all_finite(payload):
            raise RuntimeError('Nonfinite values in saved model/optimizer state')
        report.update(status='completed', checkpoint=str(snapshot), all_checkpoint_tensors_finite=True,
                      checkpoint_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if workspace is not None:
            iterator = workspace.expert_replay_iter
            if hasattr(iterator, '_shutdown_workers'):
                iterator._shutdown_workers()
            if workspace.logger._sw is not None:
                workspace.logger._sw.close()
            for group in (workspace.logger._train_mg, workspace.logger._eval_mg):
                if group._csv_file is not None:
                    group._csv_file.close()
            for env in workspace.env:
                if hasattr(env, 'close'):
                    env.close()
        report['elapsed_s'] = time.monotonic()-started
        report['cleanup_completed'] = True
        save_report()


if __name__ == '__main__':
    main()
