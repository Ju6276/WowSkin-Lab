"""Privileged scripted demonstrations for optical feeding and single clip.

These controllers read simulator object poses to generate reference actions.
They act only through env.step: no object repositioning, grasp weld or score
override. Success uses the task's existing evaluator, not controller arrival.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from .recording import require_training_tactile
from .lerobot_recording import LeRobotSimRecorder
from .learning_recording import LearningSimRecorder
from wowskin_lab.envs.pick_cube import response_configuration


def bounded(vector, limit):
    vector = np.asarray(vector, dtype=float)
    return vector*min(1., limit/max(float(np.linalg.norm(vector)), 1e-12))


class FeedingController:
    source = 'scripted_feeding_delivery'
    max_steps = 1400
    parameters = dict(level_samples=350, rotation_step_rad=.001, translation_step_m=.0004,
                      success_hold_samples=50)

    def __init__(self, env):
        self.env, self.arrived = env, False
        self.head = env.model.body('configurable_visual_head').id
        long = env.data.xmat[env.probe_body].reshape(3, 3)[:, 0].copy()
        long[2] = 0.
        long /= np.linalg.norm(long)
        self.level_rotation = np.column_stack((long, np.cross([0., 0., 1.], long), [0., 0., 1.]))

    def action(self, step):
        env, c, p = self.env, self.env.delivery.config, self.parameters
        action = env.hold_action()
        if step < p['level_samples']:
            env.phase = 'level_loaded_spoon'
            error = Rotation.from_matrix(self.level_rotation @ env.data.xmat[env.probe_body].reshape(3, 3).T).as_rotvec()
            action[3:6] = env.target_rotation.T @ bounded(error, p['rotation_step_rad'])
        elif not self.arrived:
            env.phase = 'deliver_to_mouth_region'
            goal = env.data.xpos[self.head]+env.data.xmat[self.head].reshape(3, 3) @ (
                np.asarray(c['mouth_center_head_m'])+[0., -c['target_offset_outside_lips_m'], 0.])
            front = env.data.xpos[env.probe_body]+env.data.xmat[env.probe_body].reshape(3, 3) @ np.mean(c['delivery_part_bounds_spoon_m'], axis=0)
            target = env.data.site_xpos[env.tcp_site]+goal-front
            action[:3] = bounded(target-env.target_xyz, p['translation_step_m'])
        else:
            env.phase = 'hold_delivery'
        return action

    def observe(self, row):
        self.arrived |= bool(row['delivery_eligible'])

    def retained(self, row):
        return bool(row['delivery_eligible'])


class SingleClipController:
    source = 'scripted_cable_single_clip'
    max_steps = 1400
    parameters = dict(grasp_world_m=[.35, -.12, .125], approach_height_m=.19,
                      lift_m=.008, approach_step_m=.0015, insertion_step_m=.00025,
                      success_hold_samples=300)

    def __init__(self, env):
        self.env = env
        clip = env.model.body('cclip_0').id
        point = [env.clip_score.x_bounds.mean(), env.clip_score.y_bounds.mean(), env.clip_score.z_bounds.mean()]
        self.goal = env.data.xpos[clip]+env.data.xmat[clip].reshape(3, 3) @ point

    def cable_at_clip(self):
        points = []
        for a, b in self.env.cable_segments():
            if abs(b[0]-a[0]) < 1e-10:
                continue
            fraction = (self.goal[0]-a[0])/(b[0]-a[0])
            if 0 <= fraction <= 1:
                points.append(a+fraction*(b-a))
        if not points:
            raise RuntimeError('Cable no longer spans the clip cross-section; cannot continue scripted insertion')
        return min(points, key=lambda point: np.linalg.norm(point-self.goal))

    def action(self, step):
        env, p = self.env, self.parameters
        target = np.asarray(p['grasp_world_m']).copy()
        grip = 0.
        speed = p['approach_step_m']
        if step < 200:
            env.phase = 'align_with_cable'
            target[2] = p['approach_height_m']
        elif step < 400:
            env.phase = 'approach_cable'
        elif step < 500:
            env.phase = 'grasp_cable'
            grip = 1.
        elif step < 600:
            env.phase = 'lift_cable'
            target[2] += p['lift_m']
            grip = 1.
        else:
            env.phase = 'hold_in_clip' if env.success else 'insert_into_clip'
            # Success is latched by the evaluator. Continue centering the
            # cable while it settles, rather than freezing at first success.
            target = env.data.site_xpos[env.tcp_site]+self.goal-self.cable_at_clip()
            target[0] = p['grasp_world_m'][0]
            speed, grip = p['insertion_step_m'], 1.
        return np.r_[bounded(target-env.target_xyz, speed), np.zeros(3), grip]

    def observe(self, row):
        pass

    def retained(self, row):
        return bool(row['clip_inside'] and
                    row['clip_dwell_s']+1e-9 >= self.env.clip_score.config['continuous_dwell_s'])


class ElasticSingleClipController(SingleClipController):
    """Backend-specific approach poses for the existing two passive lips.

    These are scripted acquisition choices, not changes to clip stiffness,
    contact parameters or the task score. Keep them in episode provenance.
    """
    source = 'scripted_elastic_single_clip'
    max_steps = 1800
    presets = {
        'tacto': dict(grasp_world_m=[.35, -.12, .125], insertion_step_m=.0002,
                      goal_clip_m=[.0065, .01425, .01085], settle_at_success=False),
        'digit_fots': dict(grasp_world_m=[.50, -.12, .118], insertion_step_m=.00025,
                           goal_clip_m=[.0065, .012, .010], settle_at_success=False),
        'gelsight': dict(grasp_world_m=[.50, -.12, .118], insertion_step_m=.00025,
                         goal_clip_m=[.0065, .012, .010], settle_at_success=True),
    }

    def __init__(self, env):
        super().__init__(env)
        self.parameters = dict(SingleClipController.parameters, **self.presets[env.cfg['sensor']])
        clip = env.model.body('cclip_0').id
        self.goal = env.data.xpos[clip]+env.data.xmat[clip].reshape(3, 3) @ self.parameters['goal_clip_m']

    def action(self, step):
        if self.parameters['settle_at_success'] and self.env.success:
            self.env.phase = 'settle_elastic_clip'
            return self.env.hold_action()
        return super().action(step)


def collect_task(output, episodes, seed, *, task, cable_layout, sensor_backend,
                 ideal, record_format, instruction, record_depth, clip_model='rigid',
                 physics_backend='cpu', camera_backend='opengl', gpu_disable_multiccd=False, gpu_disable_nativeccd=False,
                 gpu_disable_noslip=False, gpu_solver_iterations=None, gpu_solver_tolerance=None):
    if clip_model not in ('rigid', 'elastic_beam'):
        raise ValueError('Unknown clip model')
    if clip_model != 'rigid' and task != 'cable':
        raise ValueError('--clip-model requires --task cable')
    if task == 'feeding':
        from wowskin_lab.envs.feeding import PandaFeedingEnv as Environment
        Controller = FeedingController
        if cable_layout is not None:
            raise ValueError('--cable-layout requires --task cable')
    elif task == 'cable' and cable_layout == 'single_clip':
        from wowskin_lab.envs.cable import PandaCableEnv as Environment
        Controller = ElasticSingleClipController if clip_model == 'elastic_beam' else SingleClipController
    else:
        raise ValueError('Scripted cable collection requires --task cable --cable-layout single_clip')
    if sensor_backend not in ('tacto', 'digit_fots', 'gelsight'):
        raise ValueError('These task demonstrations currently require an optical --sensor-backend')
    if record_format not in ('dataset', 'lerobot'):
        raise ValueError(record_format)
    cfg = response_configuration(SimpleNamespace(task=task, cable_layout=cable_layout,
        sensor_backend=sensor_backend, ideal=ideal, seed=seed, clip_model=clip_model,
        physics_backend=physics_backend, camera_backend=camera_backend, gpu_disable_multiccd=gpu_disable_multiccd,
        gpu_disable_nativeccd=gpu_disable_nativeccd, gpu_disable_noslip=gpu_disable_noslip,
        gpu_solver_iterations=gpu_solver_iterations, gpu_solver_tolerance=gpu_solver_tolerance))
    if task == 'cable':
        cfg['clip_model'] = clip_model
    env = Environment(cfg, pollen=False)
    require_training_tactile(env)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    controller_source = Path(__file__).read_bytes()
    (output/'controller_source.py').write_bytes(controller_source)
    report = dict(source='scripted_simulation_acquisition', task=env.task_name,
        sensor_backend=sensor_backend, human_demonstrations=False, calibrated=False,
        clip_model=clip_model if task == 'cable' else None,
        controller_uses_simulator_object_poses=True, object_ground_truth_exported_to_policy=False,
        controller=Controller.source, controller_parameters={},
        controller_sha256=hashlib.sha256(controller_source).hexdigest(),
        controller_source_file='controller_source.py',
        caveat=('Delivery to the local mouth approach region, not swallowing' if task == 'feeding'
                else ('Passive elastic upper/lower lips, uncalibrated planar beam approximation; '
                      'local centreline/dwell score, no release-retention evaluation' if clip_model == 'elastic_beam'
                      else 'Default rigid clip; local centreline/dwell score, not snap-fit or release retention')),
        episodes=[], successes=0)
    Recorder = LeRobotSimRecorder if record_format == 'lerobot' else LearningSimRecorder
    with Recorder(env, output, source=Controller.source, instruction=instruction, record_depth=record_depth) as recorder:
        for episode in range(episodes):
            env.cfg['seed'] = seed+episode
            env.reset(seed=seed+episode)
            recorder.start()
            controller = Controller(env)
            report['controller_parameters'] = controller.parameters
            recorder.metadata['scripted_acquisition'] = {key: report[key] for key in (
                'controller', 'controller_parameters', 'controller_sha256', 'clip_model',
                'controller_uses_simulator_object_poses', 'object_ground_truth_exported_to_policy', 'caveat')}
            recorder._metadata()
            first_success, last_phase, phases = None, None, []
            start = float(env.data.time)
            for step in range(controller.max_steps):
                row = env.step(controller.action(step))[-1]
                recorder.step(row)
                controller.observe(row)
                if env.phase != last_phase:
                    phases.append(dict(phase=env.phase, time_s=float(env.data.time)))
                    print(f'{sensor_backend}/{env.task_name} episode {episode+1}: {env.phase}', flush=True)
                    last_phase = env.phase
                if env.success and first_success is None:
                    first_success = step
                if (first_success is not None and
                        step-first_success >= controller.parameters['success_hold_samples'] and
                        controller.retained(row)):
                    break
            success = bool(env.success and controller.retained(row) and not np.any(env.data.warning.number))
            path = recorder.finish('success' if success else 'failure')
            report['episodes'].append(dict(seed=seed+episode, success=success,
                duration_s=float(env.data.time-start), phases=phases, dataset_episode=str(path),
                record_format=record_format, lerobot_episode=getattr(recorder, 'last_lerobot_episode', None),
                task_evaluation=env.task_evaluation(), warnings=env.data.warning.number.tolist()))
            report['successes'] = sum(item['success'] for item in report['episodes'])
            (output/'collection_report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(f'Saved {sensor_backend}/{env.task_name}: success={success}; {path}', flush=True)
    return report
