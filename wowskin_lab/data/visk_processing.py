"""Acquisition-format bridge; ViSk's PKL conversion and BC loader run upstream.

The XArm processor detects pauses by bitwise-equal hardware pose samples.
MuJoCo has continuously measured servo motion instead. Episode boundaries
come from explicit record/save events, without fabricating repeated poses.
"""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import h5py
import numpy as np

from wowskin_lab.data.cameras import recorded_camera_ids
from wowskin_lab.learning.official import verify_workspace


def _pose_frame(metadata):
    schema = metadata.get('schema')
    if schema == 'visk_mujoco_raw_v2':
        return 'world'
    if schema in ('visk_mujoco_raw_v3', 'wowskin_robot_dataset_v1') and metadata.get('pose_frame') == 'robot_base':
        return 'robot_base'
    raise ValueError('Unsupported ViSk recording schema or missing pose_frame')


def _csv(path, headers, rows):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def _align_episode(raw, target, camera_ids=None, command_actions=False):
    metadata = json.loads((raw/'metadata.json').read_text())
    pose_frame = _pose_frame(metadata)
    if metadata.get('status') != 'success':
        raise ValueError(f'{raw}: only reviewed successful ViSk raw demonstrations are eligible')
    camera_ids = tuple(camera_ids or recorded_camera_ids(metadata))
    if not set(camera_ids).issubset(recorded_camera_ids(metadata)):
        raise ValueError('Selected camera views are missing from this episode')
    if metadata['tactile_channels'] != 15 or metadata['sensor_baseline_subtracted']:
        raise ValueError('Expected one raw 15-channel sensor; baseline subtraction belongs to upstream loader')
    with h5py.File(raw/'measurements.h5') as f:
        state_t = f['timestamp'][:]
        camera_t = f['camera_timestamp'][:]
        pose, grip, tactile = f['cartesian_states'][:], f['gripper_states'][:], f['sensor_states'][:]
        commanded = f['commanded_gripper'][:] if command_actions else None
        camera_keep = np.ones(len(camera_t), dtype=bool)
        if metadata['schema'] == 'wowskin_robot_dataset_v1':
            # Explicit demonstration boundaries, not guessed from motion.
            indices = np.searchsorted(state_t, camera_t).clip(0, len(state_t)-1)
            camera_keep = f['phase'][:][indices] == 1
            camera_t = camera_t[camera_keep]
    if len(camera_t) < 60:
        raise ValueError(f'{raw}: record at least 2 seconds (60 frames) for the unchanged subsample=5/action-chunk=10 settings')
    for name, time in [('state', state_t), ('camera', camera_t)]:
        if not np.isfinite(time).all() or not np.all(np.diff(time) > 0):
            raise ValueError(f'{raw}: invalid {name} timestamps')
    if not np.allclose(np.diff(state_t), .01, atol=1e-8):
        raise ValueError(f'{raw}: missing 100 Hz state/tactile samples')
    if pose.shape != (len(state_t), 6) or grip.shape != (len(state_t),) or tactile.shape != (len(state_t), 15):
        raise ValueError(f'{raw}: raw state/tactile shape mismatch')
    if any(not np.isfinite(x).all() for x in (pose, grip, tactile)):
        raise ValueError(f'{raw}: nonfinite measurements')
    if command_actions and (commanded.shape != grip.shape or not np.isfinite(commanded).all() or
                            np.any((commanded < 0) | (commanded > 1))):
        raise ValueError(f'{raw}: invalid recorded commanded gripper; no inferred replacement allowed')
    # Upstream alignment uses nearest timestamps. Sim cameras share a clock,
    # so the retained acquisition indices are normally exact state matches.
    right = np.searchsorted(state_t, camera_t).clip(0, len(state_t)-1)
    left = (right-1).clip(0, len(state_t)-1)
    index = np.where(abs(state_t[left]-camera_t) <= abs(state_t[right]-camera_t), left, right)
    alignment_error = abs(state_t[index]-camera_t)
    if np.max(alignment_error) > .01000001:
        raise ValueError(f'{raw}: camera and state clocks do not overlap')
    target.mkdir(parents=True)
    (target/'videos').mkdir()
    action_grip = commanded if command_actions else grip
    desired_grip = np.r_[action_grip[index][1:], action_grip[index][-1]]
    headers = ['created timestamp', 'pose_aa', 'gripper_state', 'desired_gripper_state']
    rows = [[float(state_t[k]), json.dumps(pose[k].tolist()), float(grip[k]), float(g)]
            for k, g in zip(index, desired_grip)]
    if command_actions:
        headers.append('commanded_gripper')
        for row, k in zip(rows, index):
            row.append(float(commanded[k]))
    _csv(target/'states.csv', headers, rows)
    _csv(target/'sensor.csv', ['created timestamp', 'sensor_values'],
         ((float(state_t[k]), json.dumps(tactile[k].tolist())) for k in index))
    for i in camera_ids:
        from .media import video_path
        cap = cv2.VideoCapture(str(video_path(raw, i, metadata)))
        writer = cv2.VideoWriter(str(target/f'videos/camera{i}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
        count = 0
        source_index = 0
        try:
            if not cap.isOpened() or not writer.isOpened():
                raise RuntimeError(f'{raw}: camera {i} video cannot be opened')
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape != (480, 640, 3):
                    raise ValueError('Camera resolution changed within demonstration')
                if source_index >= len(camera_keep):
                    raise ValueError('Video has more frames than recorded camera timestamps')
                if camera_keep[source_index]:
                    writer.write(frame)
                    count += 1
                source_index += 1
        finally:
            cap.release()
            writer.release()
        if source_index != len(camera_keep) or count != len(camera_t):
            raise ValueError(f'{raw}: camera {i} frames={count}, timestamps={len(camera_t)}')
    np.savez(target/'acquisition_alignment.npz', camera_timestamp=camera_t,
             state_timestamp=state_t[index], source_state_index=index)
    report = dict(source=str(raw.resolve()), frames=len(camera_t),
                  max_time_error_s=float(np.max(alignment_error)),
                  episode_boundary='explicit B/Enter recording events; no XArm equality-based pause inference',
                  pose_unit='metres, radians', pose_frame=pose_frame,
                  robot_state_schema=metadata.get('robot_state_schema', 'legacy_world'),
                  gripper_unit='measured closure fraction 0..1',
                  action_gripper=('recorded commanded_gripper 0..1; measured state remains separate'
                                  if command_actions else 'legacy measured closure'),
                  no_object_points=True, uncalibrated=True)
    (target/'acquisition.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def process_visk_sim(workspace, raw_root=None):
    workspace = Path(workspace).resolve()
    verified = verify_workspace(workspace)
    if not verified['source_matches_prepared_manifest']:
        raise ValueError(f'Workspace has unexpected edits: {verified["unexpected_changes"]}')
    manifest = json.loads((workspace/'manifest.json').read_text())
    if manifest['route'] != 'visk' or manifest.get('visk_profile') != 'paper-single' or manifest.get('camera_ids') not in ([1, 2], [1, 2, 52]):
        raise ValueError('Prepare a ViSk paper-single workspace with the selected two/three camera views first')
    task = manifest['dataset_task']
    data_root = Path(manifest['data_root'])
    raw_root = Path(raw_root).resolve() if raw_root else data_root/'raw'/task
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    # Current captures use numeric names even for failures. Status is stored
    # in metadata; media staging directories are not demonstrations.
    raw_episodes = sorted(p.parent for p in raw_root.glob('*/metadata.json')
                          if not p.parent.name.startswith('.')
                          and json.loads(p.read_text()).get('status') == 'success')
    if not raw_episodes:
        raise ValueError('No successful recorded demonstrations; save with Enter after completing the task')
    frames = {_pose_frame(json.loads((p/'metadata.json').read_text())) for p in raw_episodes}
    if len(frames) != 1:
        raise ValueError('Mixed world/base recordings: use separate data roots; old data are not converted implicitly')
    processed = data_root/'processed_data'/task
    pkl_path = Path(manifest['dataset'])
    if processed.exists() or pkl_path.exists():
        raise FileExistsError('Processed data already exist. Prepare a new data-root/workspace for a new dataset version; raw data are preserved.')
    reports = []
    for episode in raw_episodes:
        reports.append(_align_episode(episode, processed/episode.name, manifest['camera_ids'],
                                      manifest.get('action_profile') == 'mujoco-command'))
    # All image cropping/resizing and PKL keys/statistics are generated by
    # the original converter, not a locally reimplemented policy exporter.
    subprocess.run([sys.executable, 'convert_to_pkl.py', '-t', task], cwd=workspace/'source',
                   stdin=subprocess.DEVNULL, check=True)
    env = os.environ.copy()
    # Run in a fresh process to prevent another upstream repo's `utils` or
    # Hydra registrations from being reused through Python's import cache.
    script = r'''
import json
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from pathlib import Path
import torch
torch.set_num_threads(1)
with initialize_config_dir(config_dir=str(Path('cfgs').resolve()), version_base=None):
    cfg = compose(config_name='config', overrides=[TASK_OVERRIDE])
dataset = instantiate(cfg.expert_dataset)
sample = next(iter(dataset))
assert sample['sensor0'].shape == (1, 15)
assert 'sensor1' not in sample
assert sample['actions'].shape == (1, 10, 7)
assert all(np.isfinite(np.asarray(v)).all() for v in sample.values())
assert list(cfg.suite.pixel_keys) == SELECTED_PIXEL_KEYS
assert {key for key in sample if key.startswith('pixels')} == set(SELECTED_PIXEL_KEYS)
for key in SELECTED_PIXEL_KEYS:
    assert tuple(sample[key].shape) == (1, 3, 128, 128)
command_label_check = None
if ACTION_PROFILE == 'mujoco-command':
    gaps = []
    for episodes in dataset._episodes.values():
        for episode in episodes:
            obs, actions = episode['observation'], episode['action']
            # Upstream prepends one observation/action and labels each current
            # observation with the NEXT subsampled absolute gripper command.
            commands = obs['commanded_gripper'][1:]
            expected = np.r_[commands[1:], commands[-1]]
            np.testing.assert_allclose(actions[1:, -1], expected, atol=1e-7)
            gaps.append(float(np.max(np.abs(obs['gripper_states']-obs['commanded_gripper']))))
    command_label_check = dict(next_command_matches=True, max_measured_command_gap=max(gaps))
report = dict(sample_shapes={k:list(v.shape) for k,v in sample.items()},
              action_profile=ACTION_PROFILE, command_label_check=command_label_check,
              batch_size=cfg.batch_size, subsample=cfg.dataloader.bc_dataset.subsample,
              num_queries=cfg.num_queries, num_train_steps=cfg.suite.num_train_steps,
              sensor_scale_min=float(dataset.stats['sensor0']['max'].min()),
              algorithm_training_completed=False, dataset_loader_validated=True)
Path(REPORT_PATH).write_text(json.dumps(report, indent=2)+'\n')
'''
    report_path = workspace/'visk_loader_report.json'
    script = script.replace('SELECTED_PIXEL_KEYS', repr([f'pixels{i}' for i in manifest['camera_ids']]))
    script = script.replace('ACTION_PROFILE', repr(manifest.get('action_profile', 'published-state')))
    script = script.replace('TASK_OVERRIDE', repr('suite.task.tasks=['+task+']'))
    script = script.replace('REPORT_PATH', repr(str(report_path)))
    subprocess.run([sys.executable, '-c', script], cwd=workspace/'source', env=env, check=True)
    result = dict(stage='MuJoCo acquisition alignment -> original convert_to_pkl -> paper single-sensor BC loader',
                  dataset=str(pkl_path), demonstrations=len(reports), episodes=reports,
                  validation=json.loads(report_path.read_text()),
                  uncalibrated=True, full_training_completed=False, policy_rollout_validated=False)
    (workspace/'acquisition_report.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
