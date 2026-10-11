"""Run prepared, original ViSk WorkspaceIL with explicit Mac runtime changes.

No policy, loss, normalizer, sampler, batch, or horizon replacement. On MPS,
float64 batches are cast BEFORE transfer (upstream immediately uses .float()).
Unsupported MPS operators use PyTorch's CPU fallback. DataLoader workers are
configurable and explicitly joined, avoiding Python 3.12 exit-time hangs.
Metrics go directly to W&B; checkpoints use completed optimizer updates.
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
                        help='Save every N completed updates, and after the final update')
    parser.add_argument('--wandb-project', default='WowSkin-Lab')
    parser.add_argument('--wandb-entity', help='Default: workspace from your W&B login')
    parser.add_argument('--wandb-name', help='Default: output directory name')
    parser.add_argument('--task', help='Default: task recorded in the prepared workspace manifest')
    args = parser.parse_args()
    if args.steps < 1 or args.save_every < 1 or args.workers < 0:
        parser.error('steps and save-every must be positive; workers must be nonnegative')
    root = Path(__file__).resolve().parents[1]
    source = args.workspace.resolve()/'source'
    manifest = json.loads((args.workspace.resolve()/'manifest.json').read_text())
    from wowskin_lab.learning.official import verify_workspace
    verified = verify_workspace(args.workspace)
    if not verified['source_matches_prepared_manifest']:
        raise ValueError(f'Workspace has unrecorded source edits: {verified["unexpected_changes"]}')
    if manifest['route'] != 'visk':
        raise ValueError('Expected a prepared ViSk workspace')
    task = args.task or manifest['dataset_task']
    if task != manifest['dataset_task']:
        raise ValueError('Task differs from the workspace; prepare a separate workspace for this task')
    if not Path(manifest['dataset']).is_file():
        raise FileNotFoundError('Run official process-visk-sim before training')
    if not (root/'third_party/dependencies/Open-Teach/openteach/utils/network.py').is_file():
        raise FileNotFoundError('Run scripts/setup_upstream.py --only visk to fetch Open-Teach')
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
    import wandb
    from wowskin_lab.learning.training_runtime import WandbLogger, CompletedUpdateCheckpoints

    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; explicitly select --device cpu')
    torch.set_num_threads(4)
    with initialize_config_dir(config_dir=str(source/'cfgs'), version_base=None):
        cfg = compose(config_name='config', overrides=[f'suite.task.tasks=[{task}]',
                          f'device={args.device}', f'suite.num_train_steps={args.steps}',
                          f'suite.save_every_steps={args.save_every}'])
    if cfg.eval or cfg.load_bc:
        raise ValueError('This runner supports offline training from scratch, no robot services')
    if cfg.suite.action_repeat != 1:
        raise ValueError('W&B update numbering requires the ViSk action_repeat=1 configuration')
    # The upstream agent gates returning loss metrics on use_tb, not just the
    # writer. Keep that gate enabled; replace Logger before constructing it.
    cfg.use_tb = True
    # Original saves are zero-indexed AFTER update; the runtime scheduler below
    # instead calls the same save_snapshot BEFORE the next update (exact count).
    cfg.suite.save_every_steps = None
    report = dict(status='initializing', pid=os.getpid(), source=str(source), device=args.device,
        action_profile=manifest.get('action_profile', 'published-state'),
        requested_updates=args.steps, completed_updates=0, dataset=str(Path(cfg.root_dir)/'processed_data_pkl_aa'/f'{task}.pkl'),
        batch_size=int(cfg.batch_size), num_queries=int(cfg.num_queries), subsample=int(cfg.dataloader.bc_dataset.subsample),
        upstream_source_sha256={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source/'train_bc.py', source/'agent/bc.py', source/'utils.py', source/'read_data/xarm_env_aa.py')},
        runtime_changes=dict(dataloader_workers=args.workers, upstream_dataloader_workers=16,
            checkpoint_interval_steps=args.save_every, upstream_checkpoint_interval_steps=500,
            float64_cast_before_mps_transfer=args.device == 'mps',
            unsupported_mps_operators_on_cpu=args.device == 'mps', explicit_worker_shutdown=True,
            final_checkpoint=True, checkpoint_step_unit='completed_optimizer_updates',
            metrics_sink='wandb_only', upstream_use_tb_retained_for_metrics_return=True,
            tensorboard_writer=False, csv_writer=False), policy_rollout_validated=False,
        scope=f'Official training code on local simulated {task} demonstrations; not paper task reproduction')
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
    run = None
    started = time.monotonic()
    try:
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
            name=args.wandb_name or output.name, mode='online', dir=str(output),
            # Dataset-dependent shapes are mandatory placeholders until the
            # original WorkspaceIL fills them. Resolve only after its init.
            config={**OmegaConf.to_container(cfg, resolve=False),
                    'runtime_changes': report['runtime_changes'],
                    'dataset_sha256': report['dataset_sha256'],
                    'source_sha256': report['upstream_source_sha256']},
            settings=wandb.Settings(disable_code=True, disable_git=True,
                                   console='off', save_code=False, x_disable_meta=True))
        run.define_metric('train/*', step_metric='train/completed_updates')
        report.update(wandb_url=run.url, wandb_run_id=run.id, wandb_project=run.project,
                      wandb_entity=run.entity)
        save_report()
        train_bc.Logger = lambda *unused_args, **unused_kwargs: WandbLogger(run)
        workspace = train_bc.WorkspaceIL(cfg)
        run.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)
        OmegaConf.save(cfg, output/'config.yaml')
        checkpoints = CompletedUpdateCheckpoints(workspace, args.save_every)
        update = workspace.agent.update
        def checked_update(iterator, step):
            checkpoints.before_update(step)
            update_started = time.monotonic()
            metrics = update(iterator, step)
            loss = float(metrics['actor_loss'])
            if not __import__('math').isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at update {step+1}')
            report.update(status='running', completed_updates=step+1, last_actor_loss=loss,
                          elapsed_s=time.monotonic()-started,
                          saved_checkpoint_steps=list(checkpoints.saved_steps))
            report.setdefault('first_actor_loss', loss)
            metrics['update_seconds'] = time.monotonic()-update_started
            if step < 3 or (step+1) % 100 == 0 or step+1 == args.steps:
                save_report()
                print(f"ViSk {step+1}/{args.steps}: loss={loss:.6f}, elapsed={report['elapsed_s']:.1f}s", flush=True)
            return metrics
        workspace.agent.update = checked_update
        workspace.train()  # Original loop and optimizer; only logging/saving adapted.
        checkpoints.finish()
        report['saved_checkpoint_steps'] = list(checkpoints.saved_steps)
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
            workspace.logger.close()
            for env in workspace.env:
                if hasattr(env, 'close'):
                    env.close()
        report['elapsed_s'] = time.monotonic()-started
        report['cleanup_completed'] = True
        save_report()
        if run is not None:
            run.summary.update(dict(training_status=report['status'],
                completed_updates=report['completed_updates'], elapsed_s=report['elapsed_s'],
                saved_checkpoint_steps=report.get('saved_checkpoint_steps', []),
                policy_rollout_validated=False))
            run.finish(exit_code=0 if report['status'] == 'completed' else 1)


if __name__ == '__main__':
    main()
