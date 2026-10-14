"""Panda pick/place bridge for the upstream GelSightMujoco discrete DDQN.

Robot actions and task completion belong to WowSkin; the reward is the source
environment's negative sum of TCP/object and object/goal distances. Simulator
object state is used only for reward, termination and diagnostic metrics.
"""
from types import SimpleNamespace

import cv2
import mujoco
import numpy as np

from wowskin_lab.data.cameras import capture_options
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, response_configuration, TABLE_Z, CUBE_HALF
from wowskin_lab.core.optical_backends import make_optics


ACTION_NAMES = ('x+', 'x-', 'y+', 'y-', 'z+', 'z-', 'close', 'open')


def discrete_command(action, closure, translation=.005, grip_step=.1):
    if isinstance(action, bool) or int(action) != action or not 0 <= action < len(ACTION_NAMES):
        raise ValueError('Expected a discrete action in [0, 7]')
    command = np.r_[np.zeros(6), closure]
    if action < 6:
        command[int(action)//2] = translation*(1 if action % 2 == 0 else -1)
    else:
        command[6] = np.clip(closure+(grip_step if action == 6 else -grip_step), 0., 1.)
    return command


def source_reward(tcp, cube, goal):
    return -float(np.linalg.norm(np.asarray(tcp)-cube)+np.linalg.norm(np.asarray(cube)-goal))


class PandaGelSightDDQN:
    def __init__(self, *, seed=0, image_size=32, action_ticks=10, max_steps=300,
                 randomize=True, translation=.005, grip_step=.1):
        if image_size < 32 or action_ticks < 1 or max_steps < 1:
            raise ValueError('image-size >=32 and positive action-ticks/max-steps required')
        if not 0 < translation <= .005 or not 0 < grip_step <= 1:
            raise ValueError('Translation must be in (0, .005], grip-step in (0, 1]')
        cfg = response_configuration(SimpleNamespace(task='pick_cube', sensor_backend='gelsight', seed=seed))
        cfg['randomize_cube'] = randomize
        self.env = PandaPickCubeTactileEnv(cfg, pollen=False)
        self.renderer = mujoco.Renderer(self.env.model, width=640, height=480)
        self.optics = make_optics(self.env, 'gelsight')
        self.image_size, self.action_ticks, self.max_steps = image_size, action_ticks, max_steps
        self.translation, self.grip_step = translation, grip_step
        self.goal = np.r_[self.env.pick_place.config['target_center_xy_m'], TABLE_Z+CUBE_HALF]
        self.frames = []
        self.steps = 0
        self.ended = True

    def observation(self):
        self.renderer.update_scene(self.env.data, camera='lab_front', scene_option=capture_options())
        scene = self.renderer.render().copy()
        optics = self.optics.render(self.env.data)
        if any(abs(optics[side].time_s-self.env.data.time) > 1e-8 for side in ('left', 'right')):
            raise ValueError('Stale tactile observation')
        self.frames = [scene, optics['left'].rgb, optics['right'].rgb]
        # Match source type_=vision_tactile_raw, RGB with no normalization.
        # Its raw numeric observation is only the measured gripper position.
        state = [cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA).astype(np.float32)
                 for rgb in self.frames]
        state.append(self.env.data.site_xpos[self.env.tcp_site].copy().astype(np.float32))
        if not all(np.isfinite(v).all() for v in state):
            raise FloatingPointError('Nonfinite policy observation')
        return state

    def reset(self, seed):
        self.env.cfg['seed'] = int(seed)
        self.env.reset(seed=int(seed))
        self.optics.reset()
        self.steps, self.ended = 0, False
        return self.observation()

    def step(self, action):
        if self.ended:
            raise RuntimeError('Reset before stepping a completed episode')
        command = discrete_command(action, self.env.gripper_fraction(target=True), self.translation, self.grip_step)
        row = None
        start = float(self.env.data.time)
        terminated = False
        for tick in range(self.action_ticks):
            row = self.env.step(command)[-1]
            # Translation changes the pending target only once. Subsequent
            # 100 Hz ticks hold it through the normal IK/servo/physics path.
            command[:6] = 0.
            if np.any(self.env.data.warning.number) or not np.isfinite(self.env.data.qpos).all():
                raise FloatingPointError('MuJoCo warning or nonfinite simulation state')
            terminated = bool(self.env.success or row['cube_xyz_m'][2] < TABLE_Z-CUBE_HALF)
            if terminated:
                break
        elapsed = float(self.env.data.time)-start
        if not np.isclose(elapsed, (tick+1)/self.env.cfg['sample_hz'], atol=1e-8):
            raise ValueError('Control/physics time mismatch')
        self.steps += 1
        truncated = self.steps >= self.max_steps and not terminated
        self.ended = terminated or truncated
        cube = self.env.data.xpos[self.env.probe_body].copy()
        tcp = self.env.data.site_xpos[self.env.tcp_site].copy()
        reward = source_reward(tcp, cube, self.goal)
        info = dict(success=bool(self.env.success), steps=self.steps, time_s=float(self.env.data.time),
                    terminated=terminated, truncated=truncated, lift_m=float(row['lift_m']),
                    grasped=bool(row['grasped']), target_distance_m=float(np.linalg.norm(cube-self.goal)),
                    evaluation=self.env.task_evaluation())
        return self.observation(), reward, terminated, truncated, info

    def preview(self, size=240):
        images = [cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA) for frame in self.frames]
        return np.concatenate(images, axis=1)

    def close(self):
        self.renderer.close()
        self.optics.close()
