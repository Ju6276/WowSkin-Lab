"""Long-running upstream DDQN experiment with recoverable episode boundaries."""
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import time

from .gelsight_ddqn import (digest, environment_arguments, fingerprint,
                            load_sources, make_learner, write_json)


def save_resume(path, learner, report, tf):
    """Atomically replace one local checkpoint: both Q nets, Adam, replay, RNG."""
    import h5py
    import numpy as np
    temporary = path.with_suffix('.pending.h5')
    learner.q_evaluation._make_train_function()
    with h5py.File(temporary, 'w') as saved:
        saved.attrs['schema'] = 'wowskin_ddqn_resume_v1'
        saved.attrs['report'] = json.dumps(report, allow_nan=False)
        for name, values in (
            ('online', learner.q_evaluation.get_weights()),
            ('target', learner.q_target.get_weights()),
            ('adam', tf.keras.backend.batch_get_value(learner.q_evaluation.optimizer.weights)),
        ):
            group = saved.create_group(name)
            for index, value in enumerate(values):
                group.create_dataset(str(index), data=value)
        memory = learner.memory
        count = min(memory.memory_count, memory.memory_size)
        saved.attrs['memory_count'] = memory.memory_count
        saved.attrs['epsilon'] = learner.get_epsilon()
        for name in ('state_memory', 'next_state_memory'):
            group = saved.create_group(name)
            for index, value in enumerate(getattr(memory, name)):
                group.create_dataset(str(index), data=value[:count], compression='lzf', fletcher32=True)
        for name in ('action_memory', 'reward_memory', 'terminal_memory'):
            saved.create_dataset(name, data=getattr(memory, name)[:count], compression='lzf', fletcher32=True)
        rng = np.random.get_state()
        saved.create_dataset('rng_keys', data=rng[1])
        saved.attrs['rng_other'] = json.dumps([rng[0], rng[2], rng[3], rng[4]])
        saved.flush()
    os.replace(temporary, path)


def load_resume(path, learner, expected, tf):
    import h5py
    import numpy as np
    with h5py.File(path, 'r') as saved:
        if saved.attrs['schema'] != 'wowskin_ddqn_resume_v1':
            raise ValueError('Unknown resume checkpoint')
        report = json.loads(saved.attrs['report'])
        for key in ('source', 'environment_fingerprint', 'environment', 'learning', 'seed', 'runtime_versions',
                    'validation_seed', 'test_seed', 'eval_episodes', 'test_episodes', 'checkpoint_every', 'eval_every'):
            if report[key] != expected[key]:
                raise ValueError(f'Resume configuration changed: {key}')
        def values(name):
            group = saved[name]
            return [group[str(i)][()] for i in range(len(group))]
        learner.q_evaluation.set_weights(values('online'))
        learner.q_target.set_weights(values('target'))
        learner.q_evaluation._make_train_function()
        learner.q_evaluation.optimizer.set_weights(values('adam'))
        memory = learner.memory
        memory.memory_count = int(saved.attrs['memory_count'])
        count = min(memory.memory_count, memory.memory_size)
        for name in ('state_memory', 'next_state_memory'):
            for index, value in enumerate(getattr(memory, name)):
                value[:count] = saved[name][str(index)][()]
        for name in ('action_memory', 'reward_memory', 'terminal_memory'):
            getattr(memory, name)[:count] = saved[name][()]
        learner.set_epsilon(float(saved.attrs['epsilon']))
        kind, pos, has_gauss, cached_gauss = json.loads(saved.attrs['rng_other'])
        np.random.set_state((kind, saved['rng_keys'][()], pos, has_gauss, cached_gauss))
    return report


def evaluate_policy(environment, policy, destination, *, seeds, random=False, videos=1):
    """Evaluation never inserts transitions or updates the learner."""
    import av
    import numpy as np
    destination.mkdir(parents=True, exist_ok=True)
    rng_state = np.random.get_state()
    report = dict(policy='uniform_random' if random else 'greedy_ddqn', episodes=[], status='evaluating')
    try:
        for index, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            state = environment.reset(seed)
            counts = np.zeros(8, dtype=int)
            score, lift, grasped = 0., 0., False
            video = None
            if index < videos:
                video = av.open(str(destination/f'episode_{index:03d}.mp4'), 'w')
                stream = video.add_stream('libx264', rate=100//environment.action_ticks)
                stream.width, stream.height, stream.pix_fmt = 720, 240, 'yuv420p'
                stream.options = {'crf': '24', 'preset': 'fast'}
            try:
                while True:
                    if random:
                        action = int(rng.integers(8))
                    else:
                        q = policy.predict([v[None] for v in state], verbose=0)
                        if not np.isfinite(q).all():
                            raise FloatingPointError('Nonfinite evaluation Q values')
                        action = int(np.argmax(q))
                    state, reward, terminated, truncated, info = environment.step(action)
                    counts[action] += 1
                    score += reward
                    lift = max(lift, info['lift_m'])
                    grasped |= info['grasped']
                    if video:
                        for packet in stream.encode(av.VideoFrame.from_ndarray(environment.preview(), format='rgb24')):
                            video.mux(packet)
                    if terminated or truncated:
                        break
                if video:
                    for packet in stream.encode():
                        video.mux(packet)
            finally:
                if video:
                    video.close()
            report['episodes'].append(dict(seed=seed, reward=score, max_lift_m=lift,
                ever_grasped=bool(grasped), action_counts=counts.tolist(), **info))
            write_json(destination/'evaluation.json', report)
        report.update(status='complete', successes=sum(e['success'] for e in report['episodes']),
                      mean_reward=float(np.mean([e['reward'] for e in report['episodes']])))
        report['success_rate'] = report['successes']/len(report['episodes'])
        write_json(destination/'evaluation.json', report)
    finally:
        np.random.set_state(rng_state)
    return report


def plot_progress(output, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), constrained_layout=True)
    episodes = report['episodes']
    if episodes:
        x = [e['end_step'] for e in episodes]
        y = [e['reward']/e['steps'] for e in episodes]
        axes[0].plot(x, y, alpha=.35, label='Episode mean reward')
        if len(y) >= 10:
            axes[0].plot(x[9:], np.convolve(y, np.ones(10)/10, mode='valid'), label='10-episode mean')
        axes[0].legend()
        axes[1].plot(x, [e['epsilon'] for e in episodes], label='Exploration probability')
    axes[0].set(title='Panda + GelSight / upstream DDQN', ylabel='Reward per action (negative metres)')
    axes[1].set(ylabel='Epsilon (probability)', ylim=(0, 1.05))
    evaluations = report.get('evaluations', [])
    if evaluations:
        axes[2].plot([e['step'] for e in evaluations], [e['success_rate'] for e in evaluations],
                     'o-', label='Held-out full pick/place success')
        axes[2].legend()
    axes[2].set(ylabel='Success fraction', ylim=(-.03, 1.03))
    for ax in axes:
        ax.set_xlabel('Training environment decisions (steps)')
        ax.grid(alpha=.2)
    fig.savefig(output/'progress.png', dpi=130)
    plt.close(fig)


def experiment(args):
    if (args.steps < 1 or args.batch_size < 2 or args.capacity <= args.batch_size
            or args.checkpoint_every < 1 or args.eval_every < 1 or args.eval_episodes < 1
            or args.test_episodes < 1 or not math.isfinite(args.lr) or args.lr <= 0):
        raise ValueError('Positive budgets/intervals/lr and capacity > batch-size >= 2 required')
    if args.output.exists() != args.resume:
        raise ValueError('Use a new output directory, or --resume with an existing experiment')
    # Even a run ending every step cannot exhaust this separation between seeds.
    validation_seed = args.seed+args.steps+1000000
    test_seed = validation_seed+args.eval_episodes+1000000
    modules, manifest, tf = load_sources(args.workspace, args.runtime, args.seed)
    import numpy as np
    from .gelsight_ddqn_env import PandaGelSightDDQN
    args.output.mkdir(parents=True, exist_ok=args.resume)
    report = dict(schema='wowskin_ddqn_experiment_v1', status='initializing', source=manifest,
        environment_fingerprint=fingerprint(), environment=environment_arguments(args), seed=args.seed,
        learning=dict(steps=args.steps, batch_size=args.batch_size, capacity=args.capacity, lr=args.lr,
                      gamma=.99, epsilon_decay=.99996, epsilon_min=.1, target_update=1000),
        runtime_versions={name: importlib.metadata.version(name) for name in
                          ('tensorflow', 'tf-keras', 'image-classifiers', 'numpy', 'mujoco')},
        parallel_environments=1, policy_hz=100/args.action_ticks,
        observations=['front RGB', 'left GelSight RGB', 'right GelSight RGB', 'measured TCP xyz'],
        privileged_policy_inputs=False, pretrained_robot_policy=False,
        reward='Source negative TCP/object + object/target distance; no added shaping or demonstrations',
        checkpoint_use='Episode-boundary resume: both Q networks, Adam, full replay, epsilon and NumPy RNG',
        devices=[device.name for device in tf.config.list_physical_devices()],
        completed_steps=0, updates=0, wall_time_s=0., episodes=[], evaluations=[],
        validation_seed=validation_seed, test_seed=test_seed,
        eval_episodes=args.eval_episodes, test_episodes=args.test_episodes,
        checkpoint_every=args.checkpoint_every, eval_every=args.eval_every)
    learner = make_learner(modules, args)
    environment = PandaGelSightDDQN(seed=args.seed, **report['environment'])
    report['simulation_config'] = copy.deepcopy(environment.env.cfg)
    report['network_parameters'] = int(learner.q_evaluation.count_params())
    if args.resume:
        report = load_resume(args.output/'resume.h5', learner, report, tf)
        # Discard only uncheckpointed log tail, then replay those steps from the
        # last complete episode with restored learner/RNG state.
        log_path = args.output/'steps.jsonl'
        if log_path.exists():
            retained = []
            for line in log_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break
                if row['step'] <= report['completed_steps']:
                    retained.append(line)
            log_path.write_text('\n'.join(retained)+'\n')
    else:
        snapshot = args.output/'adapter_source'
        snapshot.mkdir()
        for path in Path(__file__).parent.glob('gelsight_ddqn*.py'):
            (snapshot/path.name).write_bytes(path.read_bytes())
    report['status'] = 'running'
    report['pid'] = os.getpid()
    write_json(args.output/'run.json', report)
    old_wall, started = report['wall_time_s'], time.perf_counter()
    stop_requested = False
    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print('Stop requested; finishing this episode and saving resumable state.', flush=True)
    previous_handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}

    def checkpoint():
        report['wall_time_s'] = old_wall+time.perf_counter()-started
        save_resume(args.output/'resume.h5', learner, report, tf)
        write_json(args.output/'run.json', report)
        plot_progress(args.output, report)

    def assessment(kind, seed, count, random=False):
        destination = args.output/'evaluations'/kind
        result = evaluate_policy(environment, learner.q_evaluation, destination,
                                 seeds=range(seed, seed+count), random=random)
        print(f'{kind}: success={result["successes"]}/{count}, mean_reward={result["mean_reward"]:.3f}', flush=True)
        return dict(step=report['completed_steps'], directory=str(destination.relative_to(args.output)),
                    **{key: result[key] for key in ('successes', 'success_rate', 'mean_reward')})

    try:
        if 'random_validation' not in report:
            report['random_validation'] = assessment('random_validation', validation_seed, args.eval_episodes, True)
            report['evaluations'].append(assessment('step_000000', validation_seed, args.eval_episodes))
            checkpoint()
        last_checkpoint = report['completed_steps']
        next_eval = (report['completed_steps']//args.eval_every+1)*args.eval_every
        with (args.output/'steps.jsonl').open('a') as log:
            while report['completed_steps'] < args.steps and not stop_requested:
                seed = args.seed+len(report['episodes'])
                state = environment.reset(seed)
                score, lift, grasped = 0., 0., False
                losses = []
                counts = np.zeros(8, dtype=int)
                while report['completed_steps'] < args.steps:
                    action = int(learner.choose_action(state))
                    state_next, reward, terminated, truncated, info = environment.step(action)
                    learner.store_memory(state, action, reward, state_next, int(terminated))
                    history = learner.learn()
                    loss = None if history is None else float(history.history['loss'][0])
                    if loss is not None:
                        if not np.isfinite(loss):
                            raise FloatingPointError('Nonfinite DDQN loss')
                        losses.append(loss)
                        report['updates'] += 1
                    state = state_next
                    report['completed_steps'] += 1
                    counts[action] += 1
                    score += reward
                    lift = max(lift, info['lift_m'])
                    grasped |= info['grasped']
                    log.write(json.dumps(dict(step=report['completed_steps'], episode=len(report['episodes']),
                        action=action, loss=loss, reward=reward, epsilon=learner.get_epsilon(),
                        lift_m=info['lift_m'], grasped=info['grasped'], success=info['success']), allow_nan=False)+'\n')
                    log.flush()
                    if report['completed_steps'] % 100 == 0:
                        elapsed = old_wall+time.perf_counter()-started
                        report.update(wall_time_s=elapsed, epsilon=float(learner.get_epsilon()), last_loss=loss)
                        write_json(args.output/'run.json', report)
                        print(f'Step {report["completed_steps"]}/{args.steps}, updates={report["updates"]}, '
                              f'epsilon={learner.get_epsilon():.4f}, loss={loss}, elapsed_s={elapsed:.1f}', flush=True)
                    if terminated or truncated:
                        break
                report['episodes'].append(dict(seed=seed, end_step=report['completed_steps'], reward=score,
                    max_lift_m=lift, ever_grasped=bool(grasped), epsilon=float(learner.get_epsilon()),
                    mean_loss=float(np.mean(losses)) if losses else None, action_counts=counts.tolist(),
                    budget_ended=not (terminated or truncated), **info))
                report['successes'] = sum(ep['success'] for ep in report['episodes'])
                if report['completed_steps'] >= next_eval:
                    # Keep policy snapshots small enough for local use; full
                    # replay/Adam state is retained only in the latest resume.h5.
                    name = f'step_{report["completed_steps"]:06d}'
                    result = assessment(name, validation_seed, args.eval_episodes)
                    learner.q_evaluation.save(args.output/'evaluations'/name/'policy.h5', include_optimizer=False)
                    report['evaluations'].append(result)
                    next_eval = (report['completed_steps']//args.eval_every+1)*args.eval_every
                    checkpoint()
                    last_checkpoint = report['completed_steps']
                elif report['completed_steps']-last_checkpoint >= args.checkpoint_every:
                    checkpoint()
                    last_checkpoint = report['completed_steps']
                write_json(args.output/'run.json', report)
        if stop_requested:
            report['status'] = 'stopped_at_episode_boundary'
            checkpoint()
            return report
        checkpoint()
        report['status'] = 'final_evaluation'
        write_json(args.output/'run.json', report)
        policy_path = args.output/'policy.h5'
        learner.q_evaluation.save(policy_path, include_optimizer=False)
        report['checkpoint_sha256'] = digest(policy_path)
        probe = [v[None] for v in environment.reset(test_seed)]
        before = learner.q_evaluation.predict(probe, verbose=0)
        reloaded = tf.keras.models.load_model(policy_path, compile=False)
        np.testing.assert_allclose(reloaded.predict(probe, verbose=0), before, atol=1e-5, rtol=1e-5)
        report['reload_predictions_match'] = True
        report['final_test'] = assessment('final_test', test_seed, args.test_episodes)
        report['random_test'] = assessment('random_test', test_seed, args.test_episodes, True)
        report['status'] = 'complete'
        checkpoint()
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['wall_time_s'] = old_wall+time.perf_counter()-started
        write_json(args.output/'run.json', report)
        environment.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return report
