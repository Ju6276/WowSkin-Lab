"""Select shared raw episodes; export to the original algorithm data routes.

FTF serialization uses the unchanged Franka-Teach DataCollector with finite
recorded inputs. It does not synthesize tracks, pad tactile history or run a
policy. The canonical files are never modified.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from .media import video_path, depth_path
from .cameras import recorded_camera_ids


class _RecordedSubscriber:
    def __init__(self, values, count, event):
        self.values, self.remaining, self.event = iter(values), count, event

    def receive(self):
        value = next(self.values)
        self.remaining -= 1
        if self.remaining == 0:
            self.event.clear()  # Original writer still writes this last sample.
        return value

    recv_rgb_image = recv_keypoints = get_sensor_state = receive

    def stop(self):
        pass


def _link(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _validate(path):
    meta = json.loads((path/'metadata.json').read_text())
    if (meta.get('schema') != 'wowskin_robot_dataset_v1' or
            meta.get('pose_frame') != 'robot_base' or meta.get('status') != 'success'):
        raise ValueError(f'{path}: expected a reviewed successful shared base-frame episode')
    camera_ids = recorded_camera_ids(meta)
    with h5py.File(path/'measurements.h5') as f:
        times, phase = f['timestamp'][:], f['phase'][:]
        if len(times) < 82 or not np.allclose(np.diff(times), .01, atol=1e-8):
            raise ValueError('Incomplete 100 Hz sequence')
        if len(phase) != len(times) or not np.all(np.isin(phase, [0, 1, 2])) or not np.all(np.diff(phase) >= 0):
            raise ValueError('Invalid acquisition phases')
        if np.sum(phase == 0) < 60 or not np.any(phase == 1) or np.sum(phase == 2) < 20:
            raise ValueError('Missing actual pre-roll, demonstration or post-roll; cannot fabricate boundaries')
        for key, shape in [('cartesian_states', (6,)), ('commanded_pose', (6,)),
                           ('arm_qpos', (7,)), ('arm_qvel', (7,)),
                           *(([('optical_contact_force_n', (2,))] if meta.get('tactile_modality') == 'optical' else [('sensor_states', (15,))]))]:
            if f[key].shape != (len(times), *shape) or not np.isfinite(f[key][:]).all():
                raise ValueError(f'Invalid {key}')
        if not np.all(np.diff(f['camera_timestamp'][:]) > 0):
            raise ValueError('Invalid camera timestamps')
        for key in ('camera_base_to_cv', 'camera_world_to_cv'):
            if f[key].shape != (len(f['camera_timestamp']), len(camera_ids), 4, 4):
                raise ValueError(f'Invalid {key}')
        if meta.get('depth', {}).get('enabled'):
            with h5py.File(depth_path(path, meta)) as depth:
                if not np.array_equal(depth['timestamp'][:], f['camera_timestamp'][:]):
                    raise ValueError('Depth and RGB timestamps differ')
                for camera in recorded_camera_ids(meta):
                    for key in (f'camera{camera}', f'valid_camera{camera}'):
                        if depth[key].shape != (len(f['camera_timestamp']), 480, 640):
                            raise ValueError(f'Invalid depth stream {key}')
        if meta.get('tactile_modality') == 'optical':
            from .optical_recording import optical_path, validate_optical
            validate_optical(optical_path(path, meta), f['camera_timestamp'][:])
    return meta


def _video_frames(path, times, keep):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open {path}')
    try:
        for index, timestamp in enumerate(times):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'{path}: missing frame {index}')
            if keep[index]:
                yield frame, float(timestamp)*1000
        if cap.read()[0]:
            raise ValueError(f'{path}: extra frames beyond camera timestamps')
    finally:
        cap.release()


def _export_ftf(source, target_root, demo, meta):
    if meta.get('tactile_modality') == 'optical':
        raise ValueError('FTF magnetic adapter does not consume optical images; use the native optical dataset')
    from omegaconf import OmegaConf
    from wowskin_lab.learning.official import verify_reference
    reference, spec, _ = verify_reference('ftf-robot')
    sys.path.insert(0, str(reference/'Franka-Teach'))
    from frankateach.data_collector import DataCollector
    from frankateach.messages import FrankaState, FrankaAction
    collector = DataCollector(str(target_root), demo)
    target = collector.storage_path
    with h5py.File(source/'measurements.h5') as f:
        times = f['timestamp'][:]
        first = 40  # All histories contain 40 actual measured samples.
        indices = range(first, len(times))
        count = len(times)-first
        def messages(command=False):
            for i in indices:
                pose = f['commanded_pose' if command else 'cartesian_states'][i]
                quat = Rotation.from_rotvec(pose[3:]).as_quat()
                grip = float(2*f['commanded_gripper'][i]-1)
                yield (FrankaAction(pose[:3], quat, grip, False, float(times[i])) if command else
                       FrankaState(pose[:3], quat, grip, float(times[i]), bool(f['phase'][i] == 1)))
        collector.run_event.set()
        collector.state_socket = _RecordedSubscriber(messages(), count, collector.run_event)
        collector.commanded_state_socket = _RecordedSubscriber(messages(True), count, collector.run_event)
        collector.save_states()
        def tactile():
            for i in indices:
                yield dict(timestamp=float(times[i]), sensor_values=f['sensor_states'][i],
                           sensor_history=f['sensor_states'][i-39:i+1])
        collector.run_event.set()
        collector.reskin_subscriber = _RecordedSubscriber(tactile(), count, collector.run_event)
        collector.save_reskin()
        camera_times = f['camera_timestamp'][:]
        keep = camera_times >= times[first]-1e-9
        calibration = {}
        for camera in (1, 2):
            # A single official calibration file requires fixed camera/base extrinsics.
            transforms = f['camera_base_to_cv'][:, recorded_camera_ids(meta).index(camera)]
            if not np.allclose(transforms, transforms[0], atol=1e-9):
                raise ValueError('FTF fixed-camera pipeline cannot silently use time-varying calibration')
            calibration[f'cam_{camera}'] = dict(ext=transforms[0], int=f[f'camera_intrinsic_{camera}'][:],
                                               dist_coeff=np.zeros(5))
            collector.run_event.set()
            frames = _video_frames(video_path(source, camera, meta), camera_times, keep)
            collector.image_subscribers[camera] = _RecordedSubscriber(frames, int(keep.sum()), collector.run_event)
            try:
                collector.save_rgb(camera, OmegaConf.create(dict(width=640, height=480, fps=30)))
                if next(frames, None) is not None:
                    raise ValueError('Unconsumed video frames')
            finally:
                frames.close()
        np.save(target/'sim_camera_calibration.npy', calibration)
    # Joints stay in the single source dataset; an export records its provenance.
    metadata = dict(meta, format='official_frankateach_raw', upstream_revision=spec['revision'],
                    source_episode=str(source.resolve()), source_measurements=str((source/'measurements.h5').resolve()),
                    source_state_first_index=first, training_ready=False,
                    exported_camera_ids=[1, 2], omitted_camera_ids=[i for i in recorded_camera_ids(meta) if i not in (1, 2)],
                    camera_selection_reason='Official FTF triangulation uses two fixed calibrated cameras; wrist retained in canonical dataset',
                    point_source='none; run official visual annotation/tracking after export')
    (target/'simulation_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return target


def export_dataset(directory, output, algorithm, episodes=None):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if algorithm not in ('visk', 'ftf', 'lerobot'):
        raise ValueError(algorithm)
    if output.exists():
        raise FileExistsError(f'Choose a new export directory: {output}')
    # A LeRobot collection keeps its lossless numeric source alongside the
    # standard training container. Old raw-only roots remain readable.
    if (directory/'raw').is_dir():
        directory = directory/'raw'
    candidates = sorted(p.parent for pattern in ('capture_*/metadata.json', 'demonstration_*/metadata.json')
                        for p in directory.glob(pattern))
    if episodes is not None:
        if len(set(episodes)) != len(episodes):
            raise ValueError('Duplicate episode selection')
        candidates = [p for p in candidates if p.name in episodes]
        if len(candidates) != len(episodes):
            raise ValueError('Selected episode not found')
    else:
        candidates = [p for p in candidates if json.loads((p/'metadata.json').read_text()).get('status') == 'success']
    if not candidates:
        raise ValueError('No reviewed successful episodes selected')
    metadata = [_validate(p) for p in candidates]
    if algorithm in ('visk', 'ftf') and any(m.get('tactile_modality') == 'optical' for m in metadata):
        raise ValueError('ViSk/FTF magnetic adapters need 15 magnetic channels; optical datasets require an image-tactile policy adapter')
    if len({m['task'] for m in metadata}) != 1:
        raise ValueError('Select one task per algorithm export')
    # Different sensor/physics profiles are explicit dataset variants, not an
    # accidental mixture inside one algorithm's normalization statistics.
    signatures = {json.dumps(dict(config={k: v for k, v in m['simulation_config'].items() if k != 'seed'},
                                  joints=m['arm_joint_names']), sort_keys=True) for m in metadata}
    if len(signatures) > 1:
        raise ValueError('Different physics/sensor configurations: select matching episodes in separate exports')
    task = metadata[0]['task']
    if algorithm == 'lerobot':
        from .lerobot_recording import append_raw_episode, validate_dataset
        packed = [append_raw_episode(source, output) for source in candidates]
        report = dict(algorithm='lerobot', source_dataset=str(directory), episodes=packed,
                      validation=validate_dataset(output))
        (output/'export_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
        return report
    root = output/('raw' if algorithm == 'visk' else 'extracted_data')/task
    paths = []
    try:
        for index, (source, meta) in enumerate(zip(candidates, metadata)):
            if algorithm == 'ftf':
                target = _export_ftf(source, root, index, meta)
            else:
                target = root/f'demonstration_{index}'
                target.mkdir(parents=True)
                _link(source/'measurements.h5', target/'measurements.h5')
                videos = {}
                for camera in recorded_camera_ids(meta):
                    video = video_path(source, camera, meta)
                    name = f'cam_{camera}{video.suffix}'
                    _link(video, target/name)
                    videos[str(camera)] = name
                export_meta = dict(meta, source_episode=str(source), video_files=videos)
                if meta.get('depth', {}).get('enabled'):
                    _link(depth_path(source, meta), target/'depth.h5')
                    export_meta['depth'] = dict(meta['depth'], path='depth.h5')
                (target/'metadata.json').write_text(json.dumps(export_meta, indent=2)+'\n')
            paths.append(str(target))
        report = dict(algorithm=algorithm, task=task, source_dataset=str(directory),
                      selected_episodes=[p.name for p in candidates], exported_episodes=paths,
                      pose_frame='robot_base', algorithm_training_completed=False,
                      ftf_visual_tracking_complete=False if algorithm == 'ftf' else None)
        (output/'export_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
        return report
    except BaseException:
        if output.exists():
            (output/'EXPORT_FAILED.txt').write_text('Export incomplete. Do not train from this directory. Source dataset unchanged.\n')
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path, help='Task collection directory or raw directory (capture_*/old demonstration_*)')
    parser.add_argument('--format', choices=('visk', 'ftf', 'lerobot'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', nargs='+', help='Specific successful capture directory names')
    args = parser.parse_args(argv)
    print(json.dumps(export_dataset(args.dataset, args.output, args.format, args.episodes), indent=2))
