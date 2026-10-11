"""MuJoCo transport for the pinned ViSk evaluator, not another policy.

The upstream WorkspaceIL.eval loop, BCAgent.act, checkpoint normalization and
temporal aggregation run unchanged. Only the real XArm/ZMQ environment is
replaced. Observations are newly rendered/read after applying each action.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import cv2
import dm_env
from dm_env import specs
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from wowskin_lab.data.cameras import CAMERAS, capture_options
from wowskin_lab.data.recording import magnetic_baseline, require_training_tactile
from wowskin_lab.data.robot_state import robot_sample, world_from_base
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, hold_current_pose, task_camera


def policy_increment_in_world(action, base_rotation, ticks):
    """Distribute ONE base dXYZ/local rotation command across control ticks."""
    action = np.asarray(action, dtype=float)
    if action.shape != (7,) or not np.isfinite(action).all() or ticks < 1:
        raise ValueError('Expected one finite seven-dimensional action and positive tick count')
    return np.r_[base_rotation@action[:3]/ticks, action[3:6]/ticks, action[6]]


class MuJoCoViSkEnv(dm_env.Environment):
    def __init__(self, env, cfg, output, max_steps, seed=0, viewer=False, reset_hold_steps=60):
        self.env, self.cfg, self.output = env, cfg, Path(output)
        self.max_steps, self.seed, self.episode = max_steps, seed, -1
        self.reset_hold_steps = int(reset_hold_steps)
        if self.reset_hold_steps < 5:
            raise ValueError('Reset must include at least five real baseline measurements')
        if cfg.suite.frame_stack != 1 or cfg.suite.action_repeat != 1:
            raise ValueError('This transport expects the existing checkpoint frame_stack/action_repeat = 1')
        if not cfg.dataloader.bc_dataset.relative_actions or cfg.suite.action_after_steps != 1:
            raise ValueError('This transport requires the upstream one-step relative-action checkpoint')
        if list(cfg.suite.pixel_keys) not in (['pixels1', 'pixels2'], ['pixels1', 'pixels2', 'pixels52']) or list(cfg.suite.aux_keys) != ['sensor0']:
            raise ValueError('Expected the configured two/three camera views and one 15-channel skin')
        self.policy_hz = 30./int(cfg.dataloader.bc_dataset.subsample)
        self.sample_hz = env.cfg['sample_hz']
        self.renderer = mujoco.Renderer(env.model, width=640, height=480)
        self.options = capture_options()
        self.frames = {}
        self.episodes, self.rows = [], []
        self.handle = None
        if viewer:
            from mujoco import viewer as mj_viewer
            from wowskin_lab.ui.physics_worker import PhysicsWorker
            self.display_model, self.display_data = PhysicsWorker.display(env)
            self.handle = mj_viewer.launch_passive(self.display_model, self.display_data, show_left_ui=False, show_right_ui=False)
            self.handle.cam.lookat[:] = task_camera(env).lookat
            for field in ('distance', 'azimuth', 'elevation'):
                setattr(self.handle.cam, field, getattr(task_camera(env), field))
        self._obs_spec = {key: specs.Array((3, cfg.img_size, cfg.img_size), np.uint8, key)
                          for key in cfg.suite.pixel_keys}
        self._obs_spec.update(sensor0=specs.Array((15,), np.float32, 'sensor0'),
                              proprioceptive=specs.Array((8,), np.float32, 'proprioceptive'))

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return specs.Array((7,), np.float32, 'action')

    def _sync_viewer(self):
        # Viewer reset/control/perturbation events affect only a display copy.
        # They must never rewrite the robot state mid-policy rollout.
        if self.handle is not None:
            with self.handle.lock():
                mujoco.mj_copyData(self.display_data, self.display_model, self.env.data)
            if hasattr(self.env, 'update_viewer_options'):
                self.env.update_viewer_options(self.handle)
            self.handle.sync()

    def _raw_touch(self):
        require_training_tactile(self.env)
        return np.asarray(self.env.observation()['magnetic_ut'])+magnetic_baseline(self.env)

    def _observation(self):
        observation = {}
        for key in self.cfg.suite.pixel_keys:
            self.renderer.update_scene(self.env.data, camera=CAMERAS[key], scene_option=self.options)
            rgb = self.renderer.render().copy()
            self.frames[key] = rgb
            # Upstream converter uses OpenCV BGR without swapping channels.
            if key == 'pixels52':
                rgb = rgb[:, :int(rgb.shape[1]*.8)]
            bgr = cv2.resize(rgb[..., ::-1], (self.cfg.img_size, self.cfg.img_size))
            observation[key] = bgr.transpose(2, 0, 1).copy()
        observation['sensor0'] = (self._raw_touch()-self.baseline).astype(np.float32)
        robot = robot_sample(self.env)
        observation['proprioceptive'] = np.r_[robot['cartesian_states'][:3],
            Rotation.from_rotvec(robot['cartesian_states'][3:]).as_quat(), self.env.gripper_fraction()].astype(np.float32)
        # Privileged object state is only used for this evaluator flag, not an
        # input key to BCAgent (which consumes pixels1/pixels2/sensor0).
        observation['goal_achieved'] = bool(self.env.success)
        return observation

    def reset(self):
        self.episode += 1
        self.rows = []
        self.step_index = self.tick_count = 0
        self.env.reset(seed=self.seed+self.episode)
        # Match acquisition's actual pre-roll before the first demonstration
        # image, using its recorded duration rather than an invented delay.
        for _ in range(self.reset_hold_steps-5):
            self.env.step(self.env.hold_action())
        baseline = []
        # Five actual current measurements, as in official RobotEnv.reset.
        for _ in range(5):
            self.env.step(self.env.hold_action())
            baseline.append(self._raw_touch())
        self.baseline = np.median(baseline, axis=0)
        # Initialize once. Repeating this at each policy query discards the
        # servo's remaining motion and attenuates the demonstration increments.
        hold_current_pose(self.env)
        self.start_time = float(self.env.data.time)
        self.initial_tcp = self.env.data.site_xpos[self.env.tcp_site].copy()
        self.max_lift, self.clipped_ticks = 0., 0
        self._sync_viewer()
        return dm_env.restart(self._observation())

    def step(self, action):
        if self.handle is not None and not self.handle.is_running():
            raise KeyboardInterrupt('MuJoCo rollout window closed')
        e = self.env
        before = robot_sample(e)['cartesian_states']
        raw_action = np.asarray(action, dtype=float).copy()
        # 30 Hz images / training subsample=5 -> 6 Hz policy for this dataset.
        # Deadlines give 17/17/16 actual 100 Hz updates, without clock drift.
        deadline = int(np.ceil((self.step_index+1)*self.sample_hz/self.policy_hz-1e-9))
        ticks = deadline-self.tick_count
        command = policy_increment_in_world(raw_action, world_from_base(e)[:3, :3], ticks)
        # Integrate each predicted increment into the persistent commanded
        # trajectory. The unchanged native servo follows that trajectory;
        # zero predicted motion means hold its target, not erase tracking error.
        applied_world = np.zeros(3)
        applied_rotation = np.eye(3)
        for _ in range(ticks):
            row = e.step(command)[-1]
            self.tick_count += 1
            applied_world += row['action'][:3]
            applied_rotation = applied_rotation@Rotation.from_rotvec(row['action'][3:6]).as_matrix()
            if not np.allclose(row['action'], command, atol=1e-10, rtol=1e-7):
                self.clipped_ticks += 1
            self.max_lift = max(self.max_lift, row.get('lift_m', 0.))
            self._sync_viewer()
        self.step_index += 1
        observation = self._observation()
        after = robot_sample(e)['cartesian_states']
        applied_base = world_from_base(e)[:3, :3].T@applied_world
        self.rows.append(dict(policy_step=self.step_index, sim_time_s=float(e.data.time), control_ticks=ticks,
            action=raw_action.tolist(), applied_base_dxyz=applied_base.tolist(),
            applied_local_drotvec=Rotation.from_matrix(applied_rotation).as_rotvec().tolist(),
            measured_base_pose_before=before.tolist(), measured_base_pose_after=after.tolist(),
            commanded_base_pose=robot_sample(e)['commanded_pose'].tolist(),
            commanded_gripper=float(e.gripper_fraction(target=True)),
            measured_gripper=float(e.gripper_fraction()),
            tactile_delta_ut=(self._raw_touch()-self.baseline).tolist(),
            goal_achieved=observation['goal_achieved'], max_lift_m=self.max_lift,
            clipped_control_ticks=self.clipped_ticks))
        # Same bounded episode as the official evaluator, with actual task
        # success separate from reaching the time limit. No demonstration actions.
        last = self.step_index >= self.max_steps
        reward = float(observation['goal_achieved'])
        if last:
            self.save_episode('complete')
        return dm_env.TimeStep(dm_env.StepType.LAST if last else dm_env.StepType.MID,
                               reward, 1., observation)

    def save_episode(self, status):
        if not self.rows:
            return
        path = self.output/f'episode_{self.episode:03d}.json'
        evaluation = self.env.task_evaluation() if hasattr(self.env, 'task_evaluation') else dict(
            criterion='existing cube lift: simultaneous two-finger load > 0.01 N and lift > 0.06 m',
            success=bool(self.env.success), full_pick_and_place_scored=False)
        result = dict(episode=self.episode, status=status, steps=len(self.rows),
            simulated_duration_s=float(self.env.data.time-self.start_time),
            tcp_displacement_m=float(np.linalg.norm(self.env.data.site_xpos[self.env.tcp_site]-self.initial_tcp)),
            max_lift_m=self.max_lift, task_evaluation=evaluation, clipped_control_ticks=self.clipped_ticks,
            trajectory=self.rows)
        path.write_text(json.dumps(result, indent=2)+'\n')
        self.episodes = [e for e in self.episodes if e['episode'] != self.episode]
        self.episodes.append({k: v for k, v in result.items() if k != 'trajectory'})

    def render(self):
        return np.concatenate([self.frames[k] for k in self.cfg.suite.pixel_keys], axis=1)

    def close(self):
        self.renderer.close()
        if self.handle is not None:
            self.handle.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene-metadata', type=Path, help='Recorded scene configuration; default: first training capture metadata')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=400, help='Official eval environment horizon, in policy steps')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--viewer', action='store_true', help='macOS: launch this with mjpython')
    parser.add_argument('--allow-legacy-action-labels', action='store_true',
                        help='Diagnostic only: run a legacy measured-gripper checkpoint; does not repair its action labels')
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.max_steps < 1:
        parser.error('episodes and max-steps must be positive')
    training, checkpoint, output = args.training.resolve(), args.checkpoint.resolve(), args.output.resolve()
    status = json.loads((training/'status.json').read_text())
    action_profile = status.get('action_profile', 'published-state')
    if action_profile != 'mujoco-command' and not args.allow_legacy_action_labels:
        raise ValueError('This checkpoint learned measured gripper opening, not MuJoCo commands. '
                         'Re-export/retrain with mujoco-command; legacy diagnostic runs require --allow-legacy-action-labels.')
    source = Path(status['source'])
    manifest = json.loads((source.parent/'manifest.json').read_text())
    names = ('eval.py', 'agent/bc.py', 'agent/networks/gpt.py', 'agent/networks/policy_head.py',
             'agent/networks/rgb_modules.py', 'video.py', 'logger.py')
    for name in names:
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != manifest['original_sha256'][name]:
            raise ValueError(f'Changed upstream evaluation/model source: {name}')
    sys.path.insert(0, str(source))
    import torch
    from omegaconf import OmegaConf
    import utils
    from logger import Logger
    from video import VideoRecorder

    torch.set_num_threads(2)
    cfg = OmegaConf.load(training/'config.yaml')
    cfg.device, cfg.eval = 'cpu', True
    cfg.suite.num_eval_episodes = args.episodes
    cfg.suite.task_make_fn.max_episode_len = args.max_steps
    cfg.suite.task_make_fn.max_state_dim = 8
    task = str(cfg.suite.task.tasks[0])
    metadata_path = args.scene_metadata
    if metadata_path is None:
        candidates = sorted((Path(cfg.root_dir)/'raw'/task).glob('*/metadata.json'))
        if not candidates:
            raise FileNotFoundError('Provide --scene-metadata for the recorded task configuration')
        metadata_path = candidates[0]
    metadata = json.loads(metadata_path.read_text())
    if metadata['task'] != task or metadata.get('pose_frame') != 'robot_base':
        raise ValueError('Checkpoint task and recorded scene/base-frame contract differ')
    if task not in ('pick_cube', 'feeding'):
        raise ValueError('This first closed-loop transport supports pick_cube and feeding checkpoints')
    environment_type = PandaPickCubeTactileEnv
    if task == 'feeding':
        from wowskin_lab.envs.feeding import PandaFeedingEnv
        environment_type = PandaFeedingEnv
    utils.set_seed_everywhere(args.seed)
    env = environment_type(metadata['simulation_config'])
    output.mkdir(parents=True, exist_ok=False)
    reset_hold_steps = int(metadata.get('pre_roll_samples', 61))-1
    transport = MuJoCoViSkEnv(env, cfg, output, args.max_steps, args.seed, args.viewer, reset_hold_steps)
    # Import the ORIGINAL evaluator; restore its Linux EGL environment setting
    # after import (MuJoCo's native macOS context was already initialized).
    gl = os.environ.get('MUJOCO_GL')
    spec = importlib.util.spec_from_file_location('visk_upstream_eval', source/'eval.py')
    official = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(official)
    finally:
        if gl is None:
            os.environ.pop('MUJOCO_GL', None)
        else:
            os.environ['MUJOCO_GL'] = gl
    # Checkpoints are trusted files produced locally by the official trainer.
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    agent = official.make_agent(transport.observation_spec(), transport.action_spec(), cfg)
    agent.load_snapshot(payload, eval=True)
    workspace = official.WorkspaceIL.__new__(official.WorkspaceIL)
    workspace.cfg, workspace.agent, workspace.stats = cfg, agent, payload['stats']
    workspace.env, workspace.envs_till_idx = [transport], 1
    workspace._global_step = int(payload['_global_step'])
    workspace._global_episode = 0
    workspace.logger = Logger(output, use_tb=False)
    workspace.video_recorder = VideoRecorder(output, fps=transport.policy_hz)
    report = dict(status='running', scope='LIVE MuJoCo closed loop, no recorded observations/actions replayed',
        action_profile=action_profile, legacy_action_labels=action_profile != 'mujoco-command',
        checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        upstream_revision=manifest['upstream']['revision'], unchanged_upstream_files=list(names),
        evaluation_loop='Original WorkspaceIL.eval', model='Original BCAgent.act and temporal aggregation',
        device='cpu', scene_metadata=str(metadata_path.resolve()), policy_hz=transport.policy_hz,
        rate_source='30 Hz local capture / original training subsample; original XArm evaluator itself is ZMQ-blocking',
        action='base dXYZ metres + local TCP relative rotvec radians + absolute closure fraction',
        execution='Initialize target once at reset; accumulate each base/local increment once across 100 Hz controller ticks without erasing pending servo motion',
        viewer='Detached display model/data; viewer reset or perturbations do not enter policy physics',
        controller='Existing MuJoCo IK + joint servos and command limits, not XArm hardware control',
        image_channels='BGR to match original converter PKL; live MuJoCo RGB converted once',
        input_keys=list(cfg.suite.pixel_keys)+list(cfg.suite.aux_keys), checkpoint_stats_used=True,
        baseline='Median of five actual reset measurements, as in upstream RobotEnv.reset',
        reset_hold_steps=reset_hold_steps,
        reset_source='Recorded acquisition pre-roll sample count minus initial observation; no recorded state/action replay',
        max_policy_steps=args.max_steps, requested_episodes=args.episodes,
        training_dataset_read_during_rollout=False, policy_receives_object_ground_truth=False,
        limitations='Local Franka simulation and task scores; success must be measured in closed-loop evaluation; no paper success-rate claim')
    path = output/'report.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    started = time.perf_counter()
    try:
        workspace.eval()  # Entire policy loop is upstream, including buffer reset.
        report['status'] = 'complete'
    except BaseException as error:
        transport.save_episode('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed')
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report.update(wall_time_s=time.perf_counter()-started, episodes=transport.episodes)
        path.write_text(json.dumps(report, indent=2)+'\n')
        transport.close()
    print(json.dumps({k: v for k, v in report.items() if k in ('status', 'episodes', 'wall_time_s')}, indent=2))


if __name__ == '__main__':
    main()
