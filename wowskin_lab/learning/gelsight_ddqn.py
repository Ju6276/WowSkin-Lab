"""Run the pinned GelSightMujoco DDQN learner on WowSkin Panda pick/place.

The upstream learner, ResNet branches, replay sampler and Bellman update are
loaded from an auditable prepared workspace. Robot control, observation capture,
termination, output handling and the bounded run loop are local adapters.
"""
import argparse
import difflib
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from wowskin_lab.paths import ROOT

REVISION = 'a060b0ec384b189e39d9576df753d381857bcc56'
SOURCE = ROOT/'third_party/GelSightMujoco'
SOURCE_FILES = ('rl_algos/ddqn_model.py', 'rl_algos/neuralnetwork.py', 'rl_algos/replaybuffer.py')
RUNTIME = ROOT/'.tools/gelsight_ddqn_runtime'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def fingerprint():
    paths = [*ROOT.glob('wowskin_lab/envs/*.py'), *ROOT.glob('wowskin_lab/core/*.py'),
             *ROOT.glob('configs/*.json'), *ROOT.glob('assets/reference/*/manifest.json'),
             *ROOT.glob('wowskin_lab/learning/gelsight_ddqn*.py'), ROOT/'wowskin_lab/data/cameras.py']
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def prepare(workspace):
    workspace = workspace.resolve()
    if workspace.exists():
        raise FileExistsError(f'Choose a new workspace: {workspace}')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    if revision != REVISION:
        raise ValueError('Run scripts/setup_upstream.py --only gelsight_mujoco for the pinned source')
    changes = {
        'rl_algos/neuralnetwork.py': [
            ('from    tensorflow.keras.optimizers                         import Adam',
             'from    tensorflow.keras.optimizers.legacy                  import Adam'),
            ('activation="softmax", dtype=', 'activation="linear", dtype=')],
        'rl_algos/replaybuffer.py': [
            ('self.next_state_memory      = self.state_memory.copy()',
             'self.next_state_memory      = ([v.copy() for v in self.state_memory] if isinstance(self.state_memory, list) else self.state_memory.copy())')],
    }
    prepared, patch, records = {}, [], {}
    for name in SOURCE_FILES:
        path = SOURCE/'Test_bed'/name
        original = path.read_text()
        pinned = subprocess.check_output(['git', 'show', f'{REVISION}:Test_bed/{name}'], cwd=SOURCE)
        if path.read_bytes() != pinned:
            raise ValueError(f'Upstream source changed: {path}')
        modified = original
        for old, new in changes.get(name, []):
            if modified.count(old) != 1:
                raise ValueError(f'Unexpected upstream patch context: {name}')
            modified = modified.replace(old, new)
        prepared[name] = modified
        patch += list(difflib.unified_diff(original.splitlines(True), modified.splitlines(True),
                                         fromfile='a/'+name, tofile='b/'+name))
        records[name] = dict(original_sha256=digest(path), prepared_sha256=hashlib.sha256(modified.encode()).hexdigest())
    workspace.mkdir(parents=True)
    for name, source in prepared.items():
        destination = workspace/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source)
    patch_path = workspace/'UPSTREAM_FIXES.patch'
    patch_path.write_text(''.join(patch))
    manifest = dict(schema='wowskin_gelsight_ddqn_source_v1',
        repository='https://github.com/rlamprell/GelSightMujoco', revision=REVISION, files=records,
        patch_file=patch_path.name, patch_sha256=digest(patch_path),
        changes=['Keras 2 graph-mode Adam compatibility', 'Unbounded linear Q output for negative returns',
                 'Independent next-state arrays in multimodal replay'],
        preserved=['Original DDQN class and Bellman update', 'Original ResNet18 ImageNet branches and dense layers',
                   'Original Huber/Adam settings, epsilon-greedy and replay sampling'],
        local_adapter=['Panda Cartesian discrete actions replace UR5 actuator actions',
            'One scene RGB, two tactile RGB, measured TCP xyz (source vision_tactile_raw mode)',
            'Source negative-distance reward with WowSkin full pick/place success and explicit time limit',
            'Synchronize target weights before first update; bootstrap time-limit truncation',
            'Bounded train/eval loop and provenance; no scripted demonstrations or policy warm start'])
    write_json(workspace/'manifest.json', manifest)
    return manifest


def verify_workspace(workspace):
    manifest = json.loads((workspace/'manifest.json').read_text())
    if manifest['revision'] != REVISION or manifest['schema'] != 'wowskin_gelsight_ddqn_source_v1':
        raise ValueError('Unknown source workspace')
    for name, record in manifest['files'].items():
        if digest(workspace/'source'/name) != record['prepared_sha256']:
            raise ValueError(f'Prepared upstream source was modified: {name}')
    if digest(workspace/manifest['patch_file']) != manifest['patch_sha256']:
        raise ValueError('Prepared patch hash changed')
    return manifest


def load_sources(workspace, runtime, seed):
    manifest = verify_workspace(workspace)
    if not (runtime/'tensorflow').is_dir():
        raise FileNotFoundError('Install requirements/gelsight_ddqn.txt with --target .tools/gelsight_ddqn_runtime')
    sys.path.insert(0, str(runtime.resolve()))
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    import numpy as np
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    modules = {}
    for name in SOURCE_FILES:
        path = workspace/'source'/name
        spec = importlib.util.spec_from_file_location('wowskin_upstream_'+path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[path.stem] = module
    # The upstream module assumes a 10 GB GPU. Reserve space for MuJoCo's EGL
    # renderer and the desktop on smaller GPUs; this does not change learning.
    if os.environ.get('WOWSKIN_TF_GPU_MEMORY_MB'):
        limit = int(os.environ['WOWSKIN_TF_GPU_MEMORY_MB'])
        if limit <= 0:
            raise ValueError('WOWSKIN_TF_GPU_MEMORY_MB must be positive')
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=limit)])
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    return modules, manifest, tf


def make_learner(modules, args):
    dimensions = (args.image_size, args.image_size, 3)
    types = [True, True, False, True]
    networks = []
    for name in ('q_evaluation', 'q_target'):
        network = modules['neuralnetwork'].NeuralNetwork(lr=args.lr, n_actions=8,
            vision_dims=dimensions, tact_dims=dimensions, num_inp=3, nn_type=types,
            net_name=name, concate_dense=[1024, 512, 256], numeric_dense=[256, 256, 256], transfer_mod='resnet18')
        networks.append(network.build())
    buffer = modules['replaybuffer'].ReplayBuffer(n_actions=8, vision_dims=dimensions,
        tact_dims=dimensions, num_inp=3, nn_type=types, max_size=args.capacity)
    learner = modules['ddqn_model'].ddqn(gamma=.99, n_actions=8, epsilon=1., batch_size=args.batch_size,
        epsilon_decay=.99996, epsilon_minim=.1, eval_neuralnetwork=networks[0], targ_neuralnetwork=networks[1],
        replaybuffer=buffer, target_update=1000)
    learner.update_network()
    return learner


def environment_arguments(args):
    return dict(image_size=args.image_size, action_ticks=args.action_ticks, max_steps=args.max_steps,
                randomize=True, translation=.005, grip_step=.1)


def train(args):
    if args.steps < 1 or args.batch_size < 2 or args.capacity <= args.batch_size or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('Positive steps/lr, batch-size >=2, and capacity > batch-size required')
    if args.output.exists():
        raise FileExistsError(f'Choose a new run directory: {args.output}')
    modules, manifest, tf = load_sources(args.workspace, args.runtime, args.seed)
    import numpy as np
    from .gelsight_ddqn_env import PandaGelSightDDQN, ACTION_NAMES
    args.output.mkdir(parents=True)
    snapshot = args.output/'adapter_source'
    snapshot.mkdir()
    for name in ('gelsight_ddqn.py', 'gelsight_ddqn_env.py'):
        (snapshot/name).write_bytes((Path(__file__).parent/name).read_bytes())
    report = dict(schema='wowskin_gelsight_ddqn_run_v1', status='initializing', source=manifest,
        environment_fingerprint=fingerprint(), environment=environment_arguments(args), seed=args.seed,
        learning=dict(steps=args.steps, batch_size=args.batch_size, capacity=args.capacity, lr=args.lr,
            gamma=.99, epsilon_start=1., epsilon_decay=.99996, epsilon_min=.1, target_update=1000),
        tensorflow_version=tf.__version__, algorithm='Pinned upstream DDQN with recorded fixes',
        runtime_versions={name: importlib.metadata.version(name) for name in
                          ('tensorflow', 'tf-keras', 'image-classifiers', 'numpy', 'mujoco')},
        image_preprocessing='Source RGB 0..255, OpenCV INTER_AREA resize to square, no ImageNet input normalization',
        physics_control_hz=100, policy_hz=100/args.action_ticks,
        observations=['front RGB', 'left GelSight RGB', 'right GelSight RGB', 'measured TCP xyz'],
        privileged_policy_inputs=False, pretrained_backbone='Upstream ImageNet ResNet18',
        pretrained_robot_policy=False, action_names=list(ACTION_NAMES), episodes=[],
        reward='-(Euclidean TCP/object distance + Euclidean object/target distance), metres',
        completion='Existing full pick/place evaluator; lift, target containment, table, release, stability and dwell',
        replay_terminal='True task/fall termination only; time limits bootstrap',
        checkpoint_use='Inference; online replay/optimizer state is not saved for training resume')
    write_json(args.output/'run.json', report)
    environment = None
    completed, updates = 0, 0
    started = time.perf_counter()
    try:
        learner = make_learner(modules, args)
        environment = PandaGelSightDDQN(seed=args.seed, **report['environment'])
        report['simulation_config'] = json.loads(json.dumps(environment.env.cfg))
        report['network_parameters'] = int(learner.q_evaluation.count_params())
        report['status'] = 'training'
        counts = np.zeros(8, dtype=int)
        with (args.output/'steps.jsonl').open('w') as log:
            while completed < args.steps:
                seed = args.seed+len(report['episodes'])
                state = environment.reset(seed)
                score, max_lift = 0., 0.
                while completed < args.steps:
                    action = int(learner.choose_action(state))
                    next_state, reward, terminated, truncated, info = environment.step(action)
                    learner.store_memory(state, action, reward, next_state, int(terminated))
                    history = learner.learn()
                    loss = None if history is None else float(history.history['loss'][0])
                    if loss is not None:
                        if not np.isfinite(loss):
                            raise FloatingPointError('Nonfinite DDQN loss')
                        updates += 1
                    state = next_state
                    completed += 1
                    counts[action] += 1
                    score += reward
                    max_lift = max(max_lift, info['lift_m'])
                    row = dict(step=completed, episode=len(report['episodes']), seed=seed, action=action,
                        reward=reward, loss=loss, epsilon=float(learner.get_epsilon()), **info)
                    log.write(json.dumps(row, allow_nan=False)+'\n')
                    log.flush()
                    if completed == 1 or completed % 100 == 0:
                        print(f'DDQN step {completed}/{args.steps}, updates={updates}, loss={loss}, epsilon={learner.get_epsilon():.4f}', flush=True)
                    if terminated or truncated:
                        break
                report['episodes'].append(dict(seed=seed, reward=score, max_lift_m=max_lift,
                    budget_ended=not (terminated or truncated), **info))
                report.update(completed_steps=completed, updates=updates, action_counts=counts.tolist(),
                              epsilon=float(learner.get_epsilon()), wall_time_s=time.perf_counter()-started)
                write_json(args.output/'run.json', report)
        checkpoint = args.output/'policy.h5'
        learner.q_evaluation.save(checkpoint, include_optimizer=False)
        prediction = learner.q_evaluation.predict([v[None] for v in state], verbose=0)
        if not np.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite checkpoint prediction')
        np.savez_compressed(args.output/'reload_observation.npz', **{str(i): v for i, v in enumerate(state)})
        report.update(status='complete', checkpoint_sha256=digest(checkpoint),
                      checkpoint_probe_q=prediction.tolist(), successes=sum(e['success'] for e in report['episodes']))
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report.update(completed_steps=completed, updates=updates, wall_time_s=time.perf_counter()-started)
        write_json(args.output/'run.json', report)
        if environment is not None:
            environment.close()
    return report


def evaluate(args):
    if args.output.exists() or args.episodes < 1:
        raise ValueError('Use a new output directory and positive episodes')
    run = json.loads((args.checkpoint/'run.json').read_text())
    if run['status'] != 'complete' or run['environment_fingerprint'] != fingerprint():
        raise ValueError('Training incomplete or environment/adapter changed since training')
    if digest(args.checkpoint/'policy.h5') != run['checkpoint_sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    training_seeds = {e['seed'] for e in run['episodes']}
    if training_seeds.intersection(range(args.seed, args.seed+args.episodes)):
        raise ValueError('Evaluation must use seeds absent from training')
    _, manifest, tf = load_sources(args.workspace, args.runtime, args.seed)
    if manifest != run['source'] or tf.__version__ != run['tensorflow_version']:
        raise ValueError('Upstream source/runtime differs from checkpoint')
    import av
    import numpy as np
    from .gelsight_ddqn_env import PandaGelSightDDQN
    policy = tf.keras.models.load_model(args.checkpoint/'policy.h5', compile=False)
    with np.load(args.checkpoint/'reload_observation.npz') as saved:
        probe = [saved[str(i)][None] for i in range(4)]
    np.testing.assert_allclose(policy.predict(probe, verbose=0), run['checkpoint_probe_q'], atol=1e-5, rtol=1e-5)
    args.output.mkdir(parents=True)
    report = dict(schema='wowskin_gelsight_ddqn_eval_v1', checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=run['checkpoint_sha256'], reload_predictions_match=True,
        policy='uniform_random' if args.random else 'greedy_upstream_ddqn', episodes=[], status='evaluating')
    write_json(args.output/'evaluation.json', report)
    environment = PandaGelSightDDQN(seed=args.seed, **run['environment'])
    try:
        for ep in range(args.episodes):
            seed = args.seed+ep
            rng = np.random.default_rng(seed)
            state = environment.reset(seed)
            score, max_lift = 0., 0.
            counts = np.zeros(8, dtype=int)
            container = av.open(str(args.output/f'episode_{ep:03d}.mp4'), 'w')
            stream = container.add_stream('libx264', rate=100//run['environment']['action_ticks'])
            stream.width, stream.height, stream.pix_fmt = 720, 240, 'yuv420p'
            stream.options = {'crf': '24', 'preset': 'fast'}
            try:
                while True:
                    q = policy.predict([v[None] for v in state], verbose=0)
                    if not np.isfinite(q).all():
                        raise FloatingPointError('Nonfinite greedy action values')
                    action = int(rng.integers(8)) if args.random else int(np.argmax(q))
                    state, reward, terminated, truncated, info = environment.step(action)
                    score += reward
                    max_lift = max(max_lift, info['lift_m'])
                    counts[action] += 1
                    for packet in stream.encode(av.VideoFrame.from_ndarray(environment.preview(), format='rgb24')):
                        container.mux(packet)
                    if terminated or truncated:
                        break
                for packet in stream.encode():
                    container.mux(packet)
            finally:
                container.close()
            report['episodes'].append(dict(seed=seed, reward=score, max_lift_m=max_lift,
                action_counts=counts.tolist(), video=f'episode_{ep:03d}.mp4', **info))
            write_json(args.output/'evaluation.json', report)
            print(f'Evaluation {ep+1}/{args.episodes}: success={info["success"]}, reward={score:.3f}, lift={max_lift:.4f}', flush=True)
        report.update(status='complete', successes=sum(e['success'] for e in report['episodes']))
        report['success_rate'] = report['successes']/args.episodes
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(args.output/'evaluation.json', report)
        environment.close()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare', help='Copy pinned learner into a new workspace and record patches')
    prepare_parser.add_argument('--workspace', type=Path, required=True)
    for name in ('train', 'eval', 'experiment'):
        command = sub.add_parser(name)
        command.add_argument('--workspace', type=Path, required=True)
        command.add_argument('--runtime', type=Path, default=RUNTIME)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--seed', type=int, default=1000 if name == 'eval' else 0)
        if name in ('train', 'experiment'):
            command.add_argument('--steps', type=int, default=100000 if name == 'experiment' else 10000)
            command.add_argument('--batch-size', type=int, default=128)
            command.add_argument('--capacity', type=int, default=10000)
            command.add_argument('--lr', type=float, default=1e-4)
            command.add_argument('--image-size', type=int, default=32)
            command.add_argument('--action-ticks', type=int, choices=(1, 2, 4, 5, 10, 20, 25, 50, 100), default=10)
            command.add_argument('--max-steps', type=int, default=300)
            if name == 'experiment':
                command.add_argument('--checkpoint-every', type=int, default=5000)
                command.add_argument('--eval-every', type=int, default=10000)
                command.add_argument('--eval-episodes', type=int, default=10)
                command.add_argument('--test-episodes', type=int, default=20)
                command.add_argument('--resume', action='store_true')
        else:
            command.add_argument('--checkpoint', type=Path, required=True)
            command.add_argument('--episodes', type=int, default=5)
            command.add_argument('--random', action='store_true', help='Uniform random action reference on held-out seeds')
    args = parser.parse_args(argv)
    if args.command == 'experiment':
        from .gelsight_ddqn_experiment import experiment
        result = experiment(args)
    else:
        result = prepare(args.workspace) if args.command == 'prepare' else train(args) if args.command == 'train' else evaluate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
