"""Local optical baseline using the installed, unmodified LeRobot ACT policy.

This is a WowSkin data/control adapter, not the TacEx, UniVTAC, ViSk or FTF
training pipeline. A small smoke run validates interfaces, not task mastery.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time

import av
import mujoco
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from wowskin_lab.data.cameras import active_cameras, camera_configuration, capture_options
from wowskin_lab.data.episode_review import resolve_episode
from wowskin_lab.data.robot_state import robot_sample, world_from_base
from .optical_act_data import OpticalEpisodes, absolute_command_delta, control_deadline, image_tensor, relative_rotation_pose


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)


def _batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def _versions():
    return {key: importlib.metadata.version(key) for key in ('lerobot', 'torch', 'mujoco')}


def _fingerprint():
    """Detect local environment/renderer/config changes before execution."""
    from wowskin_lab.paths import ROOT
    paths = [*ROOT.glob('wowskin_lab/envs/*.py'), *ROOT.glob('wowskin_lab/core/*.py'),
             *ROOT.glob('configs/*.json'), *ROOT.glob('assets/reference/*/manifest.json'),
             ROOT/'wowskin_lab/data/cameras.py']
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


@torch.no_grad()
def validation_error(policy, loader, device):
    policy.eval()
    total, count = torch.zeros(7, device=device), 0
    for batch in loader:
        batch = _batch(batch, device)
        prediction = policy.predict_action_chunk(batch)
        mask = ~batch['action_is_pad']
        total += ((prediction-batch['action']).abs()*mask.unsqueeze(-1)).sum((0, 1))
        count += int(mask.sum())
    return (total/count).cpu().tolist()


def train(args):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    if args.steps < 1 or args.batch_size < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('steps, batch-size and learning rate must be positive and finite')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new training directory: {output}')
    _seed(args.seed)
    output.mkdir(parents=True)
    image_size, chunk = (96, 10) if args.smoke else (224, 30)
    dataset = OpticalEpisodes(args.dataset, image_size=image_size, chunk_size=chunk,
                              cache_dir=output/'image_cache')
    validation_episode = max(dataset.episodes) if args.validation_episode is None else args.validation_episode
    dataset.split(validation_episode)  # Validate the split before choosing its reference.
    first_train = next(ep for ep in dataset.episodes if ep != validation_episode)
    rotation_reference = dataset.episodes[first_train]['observation.state'][0, 3:6].tolist()
    dataset.relative_rotations(rotation_reference)
    train_indices, validation_indices, stats = dataset.split(validation_episode)
    # A fixed spread across the held-out episode bounds evaluation cost.
    validation_indices = [validation_indices[i] for i in np.linspace(0, len(validation_indices)-1,
                          min(64, len(validation_indices)), dtype=int)]
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(Subset(dataset, validation_indices), batch_size=args.batch_size, num_workers=0)
    features = {key: PolicyFeature(FeatureType.VISUAL, (3, image_size, image_size)) for key in dataset.image_keys}
    features['observation.state'] = PolicyFeature(FeatureType.STATE, (7,))
    kwargs = dict(input_features=features, output_features={'action': PolicyFeature(FeatureType.ACTION, (7,))},
        device=args.device, push_to_hub=False, chunk_size=chunk, n_action_steps=1,
        optimizer_lr=args.lr, optimizer_lr_backbone=args.lr)
    if args.smoke:
        kwargs.update(dim_model=128, n_heads=4, dim_feedforward=512, n_encoder_layers=1,
                      n_vae_encoder_layers=1, pretrained_backbone_weights=None)
    config = ACTConfig(**kwargs)
    policy = ACTPolicy(config, dataset_stats=stats).to(args.device)
    optimizer = config.get_optimizer_preset().build(policy.get_optim_params())
    raw, _ = resolve_episode(args.dataset, next(ep for ep in dataset.episodes if ep != validation_episode))
    metadata = json.loads((raw/'metadata.json').read_text())
    report = dict(schema='wowskin_optical_act_v2', status='training',
        implementation='Unmodified lerobot.policies.act.modeling_act.ACTPolicy',
        origin='https://github.com/huggingface/lerobot/tree/v0.3.3', versions=_versions(),
        environment_fingerprint=_fingerprint(), seed=args.seed, smoke=args.smoke,
        scope='Local optical baseline; interface smoke test' if args.smoke else 'Local optical ACT baseline',
        dataset=str(dataset.root), source_provenance=list(dataset.sources.values()),
        train_episodes=[ep for ep in dataset.episodes if ep != validation_episode], validation_episode=validation_episode,
        train_frames=len(train_indices), validation_sampled_frames=len(validation_indices),
        split='Whole episodes; phase=1 only; normalization fitted to training episodes only',
        input_keys=list(features), image_size=image_size, image_preprocessing='RGB uint8, OpenCV INTER_AREA letterbox, gray 127 padding, /255, ImageNet normalization',
        training_images='Encoded LeRobot RGB videos; live RGB bypasses video encoding (compression domain difference)',
        action_contract=dataset.contract['action'], action_horizon=dataset.contract['action_horizon'],
        rotation_reference_base_rotvec=rotation_reference,
        rotation_preprocessing='Both state and action rotvec are relative to first training observation orientation; XYZ/grip unchanged; inverted before control',
        policy_hz=30, controller_hz=100, action_execution='Absolute base TCP/rotvec goal held across 4/3/3 control ticks; pending target subtraction, existing servo limits',
        observation_contract=dataset.contract, scene_metadata=metadata,
        training_device=args.device, steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        checkpoint_selection='Final fixed training step; validation is reported, not used to select this checkpoint',
        validation_metric='Absolute chunk error per channel: XYZ metres, rotvec radians, closure fraction; padding excluded',
        closed_loop_task_success_validated=False)
    started = time.perf_counter()
    report['validation_before'] = validation_error(policy, validation_loader, args.device)
    _write(output/'run.json', report)
    step, iterator = 0, iter(train_loader)
    with (output/'loss.jsonl').open('w') as log:
        try:
            while step < args.steps:
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    batch = next(iterator)
                policy.train()
                loss, metrics = policy(_batch(batch, args.device))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite ACT training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
                step += 1
                row = dict(step=step, loss=float(loss.detach()), grad_norm=float(grad_norm), **metrics)
                log.write(json.dumps(row)+'\n')
                log.flush()
                if step == 1 or step % 10 == 0 or step == args.steps:
                    print(f'ACT step {step}/{args.steps}: loss={row["loss"]:.5f}', flush=True)
            report['validation_after'] = validation_error(policy, validation_loader, args.device)
            policy.save_pretrained(output/'pretrained_model')
            report['checkpoint_sha256'] = hashlib.sha256((output/'pretrained_model/model.safetensors').read_bytes()).hexdigest()
            report['status'] = 'complete'
        except BaseException as exc:
            report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            report.update(completed_steps=step, wall_time_s=time.perf_counter()-started)
            _write(output/'run.json', report)
    return report


def create_env(metadata, seed):
    from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv
    from wowskin_lab.envs.cable import PandaCableEnv
    from wowskin_lab.envs.feeding import PandaFeedingEnv
    from wowskin_lab.envs.peg_insert import PandaPegInsertEnv
    classes = dict(pick_cube=PandaPickCubeTactileEnv, cable=PandaCableEnv, cable_insert=PandaCableEnv,
                   feeding=PandaFeedingEnv, peg_insert=PandaPegInsertEnv)
    cfg = dict(metadata['simulation_config'], seed=seed)
    if metadata['task'] not in classes or cfg.get('sensor') not in ('tacto', 'digit_fots', 'gelsight'):
        raise ValueError('Unsupported optical task/backend in checkpoint metadata')
    env = classes[metadata['task']](cfg, pollen=False)
    if camera_configuration(env) != metadata['camera_configuration']:
        raise ValueError('Live camera configuration differs from the recorded dataset')
    return env


class LiveOptical:
    def __init__(self, env, size):
        from wowskin_lab.core.optical_backends import make_optics
        self.env, self.size = env, size
        self.renderer = mujoco.Renderer(env.model, width=640, height=480)
        self.optics = make_optics(env, env.cfg['sensor'], ideal=env.cfg.get('optical_ideal', False),
                                  shadow=env.cfg.get('optical_shadow', True))
        self.frames = {}

    def reset(self):
        if hasattr(self.optics, 'reset'):
            self.optics.reset()

    def observation(self):
        frames = {}
        for key, camera in active_cameras(self.env).items():
            self.renderer.update_scene(self.env.data, camera=camera, scene_option=capture_options())
            frames['observation.images.camera'+key.removeprefix('pixels')] = self.renderer.render().copy()
        optical = self.optics.render(self.env.data)
        for side in ('left', 'right'):
            if abs(optical[side].time_s-self.env.data.time) > 1e-8:
                raise ValueError('Stale live optical observation')
            frames['observation.images.tactile_'+side] = optical[side].rgb
        self.frames = {key: image_tensor(rgb, self.size) for key, rgb in frames.items()}
        observation = {key: frame.float().unsqueeze(0)/255 for key, frame in self.frames.items()}
        observation['observation.state'] = torch.tensor(np.r_[robot_sample(self.env)['cartesian_states'],
                    self.env.gripper_fraction()], dtype=torch.float32).unsqueeze(0)
        return observation

    def close(self):
        self.renderer.close()
        self.optics.close()


def evaluate(args):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    if args.episodes < 1 or args.max_steps < 1:
        raise ValueError('episodes and max-steps must be positive')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new evaluation directory: {output}')
    run = json.loads((args.checkpoint/'run.json').read_text())
    if run['status'] != 'complete' or run['schema'] != 'wowskin_optical_act_v2':
        raise ValueError('Expected a completed optical ACT training run')
    if run['versions']['lerobot'] != _versions()['lerobot'] or run['versions']['mujoco'] != mujoco.__version__:
        raise ValueError('LeRobot/MuJoCo version differs from checkpoint environment')
    if run['environment_fingerprint'] != _fingerprint():
        raise ValueError('Physics/renderer/camera/config source changed since training; use the checkpoint environment')
    model_path = args.checkpoint/'pretrained_model'
    if hashlib.sha256((model_path/'model.safetensors').read_bytes()).hexdigest() != run['checkpoint_sha256']:
        raise ValueError('Checkpoint digest mismatch')
    config = PreTrainedConfig.from_pretrained(model_path)
    if not isinstance(config, ACTConfig):
        raise ValueError('Checkpoint is not an ACT policy')
    config.device = args.device
    config.pretrained_backbone_weights = None  # All backbone weights come from the checkpoint.
    policy = ACTPolicy.from_pretrained(model_path, config=config, local_files_only=True, strict=True).to(args.device)
    _seed(args.seed)
    env = create_env(run['scene_metadata'], args.seed)
    live = LiveOptical(env, run['image_size'])
    output.mkdir(parents=True)
    report = dict(schema='wowskin_optical_act_evaluation_v1', status='running', checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=run['checkpoint_sha256'], versions=_versions(), device=args.device,
        scope='Fresh live MuJoCo observations and policy actions; training dataset not read during rollout',
        policy_receives_object_ground_truth=False, policy_hz=30, controller_hz=100,
        input_keys=run['input_keys'], backend=env.cfg['sensor'], task=run['scene_metadata']['task'],
        max_policy_steps=args.max_steps, smoke_checkpoint=run['smoke'], episodes=[])
    started = time.perf_counter()
    try:
        for episode in range(args.episodes):
            env.reset(seed=args.seed+episode)
            policy.reset()
            live.reset()
            # Match acquisition's real 0.6 s pre-roll, without replaying state.
            live.optics.render(env.data)
            camera_index = 1
            for tick in range(1, run['scene_metadata'].get('pre_roll_samples', 61)):
                env.step(env.hold_action())
                if tick >= control_deadline(camera_index):
                    live.optics.render(env.data)
                    camera_index += 1
            ticks, rows, success = 0, [], False
            with av.open(str(output/f'episode_{episode:03d}.mp4'), mode='w') as movie:
                stream = movie.add_stream('libx264', rate=30)
                stream.width = len(config.image_features)*run['image_size']
                stream.height, stream.pix_fmt = run['image_size'], 'yuv420p'
                for step in range(args.max_steps):
                    observation = live.observation()
                    if set(observation) != set(config.input_features):
                        raise ValueError('Live policy keys differ from training inputs')
                    before = float(env.data.time)
                    measured = observation['observation.state'][0].numpy().copy()
                    observation['observation.state'] = torch.from_numpy(relative_rotation_pose(
                        measured, run['rotation_reference_base_rotvec'])).unsqueeze(0)
                    with torch.no_grad():
                        predicted = policy.select_action(_batch(observation, args.device))[0].cpu().numpy()
                    action = relative_rotation_pose(predicted, run['rotation_reference_base_rotvec'], inverse=True)
                    image = np.concatenate([live.frames[key].permute(1, 2, 0).numpy() for key in config.image_features], axis=1)
                    for packet in stream.encode(av.VideoFrame.from_ndarray(image, format='rgb24')):
                        movie.mux(packet)
                    deadline = control_deadline(step+1)
                    executed = []
                    while ticks < deadline:
                        command = absolute_command_delta(action, world_from_base(env), env.target_xyz, env.target_rotation)
                        row = env.step(command)[-1]
                        ticks += 1
                        executed.append(np.asarray(row['action']).tolist())
                        if np.any(env.data.warning.number):
                            raise RuntimeError('MuJoCo warning during live rollout')
                        if env.success:
                            success = True
                            break
                    rows.append(dict(step=step, observation_time_s=before, end_time_s=float(env.data.time),
                        measured_state=measured.tolist(), absolute_policy_action=action.tolist(),
                        executed_100hz_commands=executed))
                    if success:
                        break
                for packet in stream.encode():
                    movie.mux(packet)
            _write(output/f'episode_{episode:03d}.json', rows)
            report['episodes'].append(dict(seed=args.seed+episode, success=success, policy_steps=len(rows),
                control_ticks=ticks, task_evaluation=env.task_evaluation(),
                termination='success' if success else 'max_policy_steps'))
            _write(output/'report.json', report)
            print(f'ACT rollout {episode+1}/{args.episodes}: success={success}, steps={len(rows)}', flush=True)
        report.update(status='complete', successes=sum(ep['success'] for ep in report['episodes']))
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        live.close()
        report['wall_time_s'] = time.perf_counter()-started
        _write(output/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('train')
    fit.add_argument('dataset', type=Path)
    fit.add_argument('--steps', type=int, default=1000)
    fit.add_argument('--batch-size', type=int, default=8)
    fit.add_argument('--lr', type=float, default=1e-5)
    fit.add_argument('--validation-episode', type=int, help='Default: last episode; all other episodes train')
    fit.add_argument('--smoke', action='store_true', help='Compact random-init 96px ACT for interface checks, not paper defaults')
    rollout = commands.add_parser('eval')
    rollout.add_argument('checkpoint', type=Path, help='Training output directory containing run.json')
    rollout.add_argument('--episodes', type=int, default=1)
    rollout.add_argument('--max-steps', type=int, default=450)
    for command in (fit, rollout):
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
        command.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    result = train(args) if args.command == 'train' else evaluate(args)
    print(json.dumps({key: result[key] for key in ('status', 'wall_time_s')}, indent=2))


if __name__ == '__main__':
    main()
