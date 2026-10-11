"""Monitor an existing ViSk run without modifying/restarting its trainer.

Read TensorBoard loss events and the local status/config. Publish only scalar
metrics and selected hyperparameters; no dataset, media, checkpoint or source
upload. Default offline; --online explicitly enables cloud synchronization.
"""
import argparse
import json
from pathlib import Path
import time
import uuid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('training', type=Path)
    p.add_argument('--project', default='WowSkin-Lab')
    p.add_argument('--entity', help='Your W&B username or team; default from login')
    p.add_argument('--online', action='store_true')
    p.add_argument('--once', action='store_true', help='Import currently available events and exit')
    args = p.parse_args()
    training = args.training.resolve()
    import wandb
    import yaml
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    status = json.loads((training/'status.json').read_text())
    cfg = yaml.safe_load((training/'config.yaml').read_text())
    config = {k: status[k] for k in ('batch_size', 'num_queries', 'subsample', 'device', 'requested_updates', 'dataset_sha256', 'runtime_changes')}
    config.update({k: cfg[k] for k in ('encoder_type', 'policy_type', 'policy_head', 'seed', 'use_aux_inputs')})
    config.update(learning_rate=cfg['agent']['lr'], camera_keys=cfg['suite']['pixel_keys'],
                  tactile_keys=cfg['suite']['aux_keys'], policy_rollout_validated=False)
    mode = 'online' if args.online else 'offline'
    binding_file = training/f'wandb-monitor-{mode}.json'
    old = json.loads(binding_file.read_text()) if binding_file.exists() and args.online else {}
    if old and (old['project'] != args.project or old['entity'] != args.entity):
        raise ValueError('Existing monitor uses another project/entity; choose the same destination')
    run_id = old.get('run_id', uuid.uuid4().hex[:12])
    settings = wandb.Settings(disable_code=True, disable_git=True, console='off',
                              x_disable_stats=True, x_disable_meta=True, save_code=False)
    accumulator = EventAccumulator(str(training/'tb'), size_guidance={'scalars': 0})
    with wandb.init(project=args.project, entity=args.entity, name=training.name,
                    id=run_id, resume='allow' if args.online else None,
                    mode=mode, dir=str(training), config=config, settings=settings) as run:
        last_step = run.step-1 if args.online else -1
        while True:
            status = json.loads((training/'status.json').read_text())
            accumulator.Reload()
            events = accumulator.Scalars('train/actor_loss') if 'train/actor_loss' in accumulator.Tags()['scalars'] else []
            # Upstream logs the same scalar twice on a console-log step.
            unique = {event.step: event for event in events if event.step > last_step}
            for step, event in sorted(unique.items()):
                run.log({'train/actor_loss': event.value, 'train/completed_updates': step+1}, step=step)
                last_step = step
            run.summary.update(dict(training_status=status['status'],
                completed_updates=status['completed_updates'], elapsed_s=status.get('elapsed_s', 0.),
                policy_rollout_validated=False))
            record = dict(mode=mode, project=args.project, entity=args.entity, run_id=run.id,
                          url=run.url if args.online else None, last_logged_step=last_step,
                          training_status=status['status'], run_directory=run.dir)
            temp = binding_file.with_suffix('.tmp')
            temp.write_text(json.dumps(record, indent=2)+'\n'); temp.replace(binding_file)
            if args.once or status['status'] in ('completed', 'failed'):
                break
            time.sleep(5)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
