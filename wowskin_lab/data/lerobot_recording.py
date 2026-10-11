"""LeRobot v2.1 acquisition, using the pinned upstream writer and reader.

Capture once at the existing clocks, then pack reviewed episodes with the
official API. Keep full-rate raw data for FTF in the same collection directory.
No policy, synthetic labels, duplicate images, or filled tactile observations.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np

from .learning_recording import LearningSimRecorder
from .instructions import episode_instruction
from .depth import DEPTH_PATH, load_depth_frame
from .media import video_path, depth_path
from .cameras import active_cameras, recorded_camera_ids, camera_configuration

VERSION = '0.3.3'
FORMAT = 'v2.1'
REPO_ID = 'local/wowskin-lab'
ANNOTATION = 'annotation.human.action.task_description'
GR00T_REFERENCE = 'https://github.com/NVIDIA/Isaac-GR00T/blob/755876a9afdb41ca6eb6383b36f4a0adb085c73f/getting_started/LeRobot_compatible_data_schema.md'
POSE_NAMES = ['tcp_x_m', 'tcp_y_m', 'tcp_z_m', 'tcp_rx_rad', 'tcp_ry_rad', 'tcp_rz_rad', 'gripper_closure']


def upstream():
    try:
        version = importlib.metadata.version('lerobot')
        from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    except ImportError as exc:
        raise RuntimeError('Install the recording environment: python3.12 scripts/setup_capture.py') from exc
    if version != VERSION or CODEBASE_VERSION != FORMAT:
        raise RuntimeError(f'Expected lerobot {VERSION} / {FORMAT}, found {version} / {CODEBASE_VERSION}')
    return LeRobotDataset


def features(camera_ids=(1, 2, 52)):
    def vector(size, names=None, dtype='float32'):
        return dict(dtype=dtype, shape=(size,), names=names)
    result = {
        'observation.state': vector(7, POSE_NAMES),
        'action': vector(7, POSE_NAMES),
        'observation.joint_position': vector(7, [f'joint{i}_rad' for i in range(1, 8)]),
        'observation.joint_velocity': vector(7, [f'joint{i}_rad_s' for i in range(1, 8)]),
        'observation.tcp_velocity': vector(6, ['vx_m_s', 'vy_m_s', 'vz_m_s', 'wx_rad_s', 'wy_rad_s', 'wz_rad_s']),
        'observation.tactile': vector(15, [f'U{i}_{a}_uT' for i in range(1, 6) for a in 'xyz']),
        'aux.sim_timestamp': vector(1, dtype='float64'),
        'aux.action_timestamp': vector(1, dtype='float64'),
        'aux.raw_sample_index': vector(1, dtype='int64'),
        'aux.raw_camera_index': vector(1, dtype='int64'),
        'aux.next_command_delta': vector(7),
        'aux.acquisition_phase': vector(1, dtype='int64'),
        ANNOTATION: vector(1, dtype='int64'),
    }
    for camera in camera_ids:
        result[f'observation.images.camera{camera}'] = dict(
            dtype='video', shape=(480, 640, 3), names=['height', 'width', 'channels'])
    return result


def signature(meta):
    return dict(config={k: v for k, v in meta['simulation_config'].items() if k != 'seed'},
                pose_frame=meta['pose_frame'], joints=meta['arm_joint_names'], source=meta['source'],
                depth=bool(meta.get('depth', {}).get('enabled', False)), camera_ids=list(recorded_camera_ids(meta)),
                camera_configuration=meta.get('camera_configuration'))


def gr00t_modalities(camera_ids=(1, 2, 52)):
    """Original N1 schema. This is a field map, not a trained VLA adapter."""
    pose = dict(eef_position=dict(start=0, end=3, dtype='float32', absolute=True),
                eef_rotation=dict(start=3, end=6, dtype='float32', absolute=True, rotation_type='axis_angle'),
                gripper=dict(start=6, end=7, dtype='float32', absolute=True, range=[0., 1.]))
    state = dict(pose)
    for name, end in [('joint_position', 7), ('joint_velocity', 7), ('tcp_velocity', 6), ('tactile', 15)]:
        state[name] = dict(start=0, end=end, dtype='float32', absolute=True,
                           original_key=f'observation.{name}')
    return dict(state=state, action=pose,
        video={f'camera{i}': dict(original_key=f'observation.images.camera{i}') for i in camera_ids},
        annotation={'human.action.task_description': dict(original_key=ANNOTATION)})


def append_raw_episode(source, destination, *, progress=None, finalize=None):
    """Pack a successful episode. Raw data remain immutable and authoritative."""
    from .dataset_export import _validate, _link
    progress = progress or (lambda stage, detail: None)
    progress('SAVING', 'Checking original timestamps, states and camera files')
    source, destination = Path(source).resolve(), Path(destination).resolve()
    meta = _validate(source)
    camera_ids = recorded_camera_ids(meta)
    if meta.get('magnetic_source') != 'fitted_dipole':
        raise ValueError('LeRobot U1..U5 schema currently requires fitted_dipole; network candidate channels need a separate schema')
    with h5py.File(source/'measurements.h5') as raw:
        if 'tcp_velocity' not in raw:
            raise ValueError('Older capture has no measured TCP velocity; keep it as a legacy dataset rather than fabricate this field')
    Dataset = upstream()
    contract_file = destination/'meta/wowskin.json'
    pending = destination/'INCOMPLETE'
    if pending.exists():
        raise RuntimeError(f'Previous LeRobot write failed: {pending}. Raw data are preserved; export to a new directory.')
    if destination.exists():
        contract = json.loads(contract_file.read_text())
        if contract['schema'] not in ('wowskin_lerobot_v3', 'wowskin_lerobot_v4'):
            raise ValueError('Earlier LeRobot recording schema; choose a new collection directory (old data are preserved)')
        if {'camera_ids': [1, 2], 'camera_configuration': None, **contract['signature']} != signature(meta):
            raise ValueError('Different control/physics/sensor configuration; choose a separate collection directory')
        ds = Dataset(REPO_ID, root=destination, video_backend='pyav')
    else:
        ds = Dataset.create(REPO_ID, fps=30, features=features(camera_ids), root=destination,
                            robot_type='franka_panda_wowskin', video_backend='pyav', image_writer_threads=2)
        contract = dict(schema='wowskin_lerobot_v4', lerobot_version=VERSION, format=FORMAT,
            signature=signature(meta), pose_frame='robot_base', pose_rotation='rotation vector, not RPY',
            state='measured TCP xyz metres + rotvec radians + measured gripper closure',
            action='absolute commanded TCP xyz + rotvec + gripper target issued immediately after the observation',
            action_horizon='first following 100 Hz control update; not a summed 30 Hz delta',
            tactile='raw magnetic uT including baseline; U1..U5 XYZ; baseline in raw metadata',
            image_clock='one original camera frame per row, 30 Hz nominal; no frame duplication',
            timestamp='LeRobot frame_index/30, for video indexing; actual simulator time is aux.sim_timestamp',
            raw='100 Hz state/tactile, velocities and camera calibration only in raw/; RGB/depth stored once in this dataset',
            phases='aux.acquisition_phase: 0 real pre-roll, 1 task demonstration, 2 real post-roll; all camera frames retained for FTF export',
            instruction='Episode instruction in meta/tasks.jsonl; per-row task_index and '+ANNOTATION,
            gr00t_schema_reference=GR00T_REFERENCE,
            gr00t_scope='Original N1 modality map only; training/config/rollout not validated. Annotation namespace does not imply human teleoperation.',
            depth=dict(enabled=bool(meta.get('depth', {}).get('enabled', False)), path=DEPTH_PATH,
                index='aux.raw_camera_index', timestamp='same as aux.sim_timestamp',
                units='metres, float32 optical +Z', encoding='lossless LZF HDF5',
                camera_ids=list(camera_ids), includes='One map per camera + masks, timestamps, intrinsics and per-frame base/world extrinsics',
                compatibility='Extension: not automatically consumed by upstream LeRobot v2.1 or GR00T N1'),
            scope='scripted or teleoperated simulation; uncalibrated tactile; no paper task validation')
        contract_file.write_text(json.dumps(contract, indent=2)+'\n')
        (destination/'meta/modality.json').write_text(json.dumps(gr00t_modalities(camera_ids), indent=2)+'\n')
    sources = destination/'meta/wowskin_sources.jsonl'
    previous = [json.loads(line) for line in sources.read_text().splitlines()] if sources.exists() else []
    digest = hashlib.sha256((source/'measurements.h5').read_bytes()).hexdigest()
    # Deterministic episodes may have identical measurements. Prevent a
    # repeated import of one source, not a new trial with identical values.
    if any(item['source'] == str(source) for item in previous):
        ds.stop_image_writer()
        raise ValueError('This raw episode is already in the LeRobot dataset')
    pending.write_text(f'Packing {source}\n')
    source_videos = [video_path(source, i, meta) for i in camera_ids]
    caps = [cv2.VideoCapture(str(path)) for path in source_videos]
    count, max_clock_offset = 0, 0.
    episode_index = ds.num_episodes
    try:
        instruction = episode_instruction(meta['task'], meta.get('instruction'))
        if ds.meta.get_task_index(instruction) is None:
            ds.meta.add_task(instruction)
        instruction_index = ds.meta.get_task_index(instruction)
        if not all(c.isOpened() for c in caps):
            raise ValueError('Cannot decode all source cameras')
        with h5py.File(source/'measurements.h5') as f:
            times, camera_times = f['timestamp'][:], f['camera_timestamp'][:]
            indices = np.searchsorted(times, camera_times-1e-9)
            if not np.allclose(times[indices], camera_times, atol=1e-8):
                raise ValueError('Camera frame is not synchronous with a raw observation')
            # Preserve every actual RGB frame, including pre/post-roll, so
            # downstream FTF can reuse this single video without padding.
            keep = np.ones(len(indices), dtype=bool)
            selected = np.flatnonzero(keep)
            if len(selected) < 2 or not np.all(np.diff(selected) == 1):
                raise ValueError('Expected a contiguous demonstration with at least two camera frames')
            first_time = float(camera_times[selected[0]])
            for j, i in enumerate(indices):
                decoded = [c.read() for c in caps]
                if not all(ok for ok, _ in decoded):
                    raise ValueError(f'Missing original camera frame {j}')
                if not keep[j]:
                    continue
                next_i = int(i)+1
                if next_i >= len(times) or not f['command_valid'][next_i]:
                    raise ValueError('No actual following control sample; do not pad the last action')
                if abs(float(f['command_timestamp'][next_i])-times[i]) > 1e-8:
                    raise ValueError('Observation/action timestamps are misaligned')
                frame = {
                    'observation.state': np.r_[f['cartesian_states'][i], f['gripper_states'][i]].astype('float32'),
                    'action': np.r_[f['commanded_pose'][next_i], f['commanded_gripper'][next_i]].astype('float32'),
                    'observation.joint_position': f['arm_qpos'][i].astype('float32'),
                    'observation.joint_velocity': f['arm_qvel'][i].astype('float32'),
                    'observation.tcp_velocity': f['tcp_velocity'][i].astype('float32'),
                    'observation.tactile': f['sensor_states'][i].astype('float32'),
                    'aux.sim_timestamp': np.array([times[i]], dtype='float64'),
                    'aux.action_timestamp': np.array([f['command_timestamp'][next_i]], dtype='float64'),
                    'aux.raw_sample_index': np.array([i], dtype='int64'),
                    'aux.raw_camera_index': np.array([j], dtype='int64'),
                    'aux.next_command_delta': f['command_7d'][next_i].astype('float32'),
                    'aux.acquisition_phase': np.array([f['phase'][i]], dtype='int64'),
                    ANNOTATION: np.array([instruction_index], dtype='int64'),
                }
                for camera, (_, bgr) in zip(camera_ids, decoded):
                    frame[f'observation.images.camera{camera}'] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if not all(np.isfinite(v).all() for v in frame.values()):
                    raise ValueError('Nonfinite LeRobot frame')
                ds.add_frame(frame, task=instruction)
                max_clock_offset = max(max_clock_offset, abs(float(times[i])-first_time-count/30))
                count += 1
                if count == 1 or count % 30 == 0 or count == len(indices):
                    progress('SAVING', f'Preparing LeRobot frames {count}/{len(indices)}')
            if any(c.read()[0] for c in caps):
                raise ValueError('Extra video frames without source timestamps')
        progress('SAVING', f'Encoding {len(camera_ids)} videos and writing Parquet ({count} frames)')
        ds.save_episode()
        depth_relative = None
        if meta.get('depth', {}).get('enabled'):
            progress('SAVING', 'Storing synchronized lossless depth')
            depth_relative = DEPTH_PATH.format(episode_chunk=ds.meta.get_episode_chunk(episode_index),
                                          episode_index=episode_index)
            target = destination/depth_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            _link(depth_path(source, meta), target)
        provenance = dict(episode_index=episode_index, source=str(source),
                          source_relative=os.path.relpath(source, destination), sha256=digest,
                          frames=count, max_nominal_clock_offset_s=max_clock_offset,
                          source_method=meta['source'], status=meta['status'],
                          instruction=instruction, depth_path=depth_relative)
        if 'task_evaluation' in meta:
            provenance['task_evaluation'] = meta['task_evaluation']
        with sources.open('a') as stream:
            stream.write(json.dumps(provenance)+'\n')
        # A packed file is not committed until decoding/depth verification and
        # the caller's metadata/media finalization succeed. Keep the marker on
        # any failure so the next recording cannot append to an unverified set.
        progress('VERIFYING', 'Decoding saved videos and verifying lossless depth')
        if finalize is not None:
            finalize(provenance)
        else:
            verified_media(source, destination, provenance)
        pending.unlink()
        print(f'LeRobot {FORMAT}: {destination}, episode {episode_index}, {count} frames', flush=True)
        return provenance
    finally:
        for c in caps:
            c.release()
        ds.stop_image_writer()


def verified_media(source, native, provenance):
    """Verify the new episode only; return metadata and disposable originals."""
    import av
    meta = json.loads((source/'metadata.json').read_text())
    camera_ids = recorded_camera_ids(meta)
    ds = upstream()(REPO_ID, root=native, video_backend='pyav')
    ep = provenance['episode_index']
    old_files = [video_path(source, i, meta) for i in camera_ids]
    new_files = [native/ds.meta.get_video_file_path(ep, f'observation.images.camera{i}') for i in camera_ids]
    for video in new_files:
        with av.open(str(video)) as stream:
            if sum(1 for _ in stream.decode(video=0)) != meta['camera_frames']:
                raise ValueError('Do not remove capture media: encoded frame count differs')
    meta['video_files'] = {str(i): os.path.relpath(video, source) for i, video in zip(camera_ids, new_files)}
    if meta.get('depth', {}).get('enabled'):
        old = depth_path(source, meta)
        target = native/provenance['depth_path']
        with old.open('rb') as old_stream, target.open('rb') as new_stream:
            equal = hashlib.file_digest(old_stream, 'sha256').digest() == hashlib.file_digest(new_stream, 'sha256').digest()
        if not equal:
            raise ValueError('Do not remove original depth: lossless copy check failed')
        old_files.append(old)
        meta['depth']['path'] = os.path.relpath(target, source)
    meta['lerobot_episode_index'] = ep
    return meta, old_files


class LeRobotSimRecorder(LearningSimRecorder):
    """Manual B/Enter recording plus official packing on successful save."""
    def __init__(self, env, directory, source='keyboard', *, instruction=None, record_depth=True):
        # Fail before recording if the selected format is not installed.
        upstream()
        super().__init__(env, directory, source, instruction=instruction, record_depth=record_depth)
        self.collection_root = self.directory
        self.directory = self.collection_root/'raw'
        self.last_lerobot_episode = None
        self._writer_lock = None

    def _release_lock(self):
        if self._writer_lock is not None:
            self._writer_lock.close()  # OS releases flock, including on crashes.
            self._writer_lock = None

    def start(self):
        import fcntl
        if self.active:
            raise RuntimeError('Finish the current demonstration first')
        self.last_lerobot_episode = None
        pending = self.collection_root/'lerobot/INCOMPLETE'
        if pending.exists():
            raise RuntimeError(f'Resolve incomplete export or choose another --record-dir: {pending}')
        self.directory.mkdir(parents=True, exist_ok=True)
        self._writer_lock = (self.directory/'.writer.lock').open('a+')
        try:
            try:
                fcntl.flock(self._writer_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Another window is recording to this task directory; finish that recording or choose another --record-dir') from exc
            contract_path = self.collection_root/'lerobot/meta/wowskin.json'
            if contract_path.exists():
                contract = json.loads(contract_path.read_text())
                if contract['schema'] not in ('wowskin_lerobot_v3', 'wowskin_lerobot_v4'):
                    raise ValueError('Older collection schema; choose a new --record-dir, without overwriting old data')
                from .robot_state import robot_metadata
                current = dict(simulation_config=self.env.cfg, source=self.source,
                               camera_ids=[int(key.removeprefix('pixels')) for key in active_cameras(self.env)],
                               camera_configuration=camera_configuration(self.env),
                               depth=dict(enabled=self.record_depth), **robot_metadata(self.env))
                if json.dumps({'camera_ids': [1, 2], 'camera_configuration': None, **contract['signature']}, sort_keys=True) != json.dumps(signature(current), sort_keys=True):
                    raise ValueError('Changed recording/physics configuration; choose a separate --record-dir')
            super().start()
            self.metadata['task'] = self.collection_root.name
            self.metadata['training_container'] = 'LeRobot v2.1; packed with upstream API after successful save'
            self._metadata()
        except BaseException:
            self._release_lock()
            raise

    def finish(self, status='unreviewed'):
        try:
            path = super().finish(status)
            if path is not None and status == 'success':
                print('Packing reviewed episode into LeRobot v2.1; raw capture is already saved.', flush=True)
                def pack():
                    def finalize(provenance):
                        self.last_lerobot_episode = provenance
                        self._finalize_media(path)
                    self.last_lerobot_episode = append_raw_episode(path, self.collection_root/'lerobot',
                                                                  progress=self.set_progress, finalize=finalize)
                # Rendering and the actual pre/post-roll stay on their original
                # threads. Only closed-file packing/verification moves off UI.
                run = getattr(self.env, '_viewer_run_io', None)
                pack() if run is None else run(pack)
            if path is not None:
                self.set_progress('SAVED', path.name)
            return path
        finally:
            self._release_lock()

    def abort(self, reason):
        try:
            return super().abort(reason)
        finally:
            self._release_lock()

    def _finalize_media(self, source):
        """After encoding is verified, retain one final RGB/depth copy.

        Metadata is updated before removing temporary media. A failed write
        leaves those media intact for recovery. Existing old datasets are not
        retroactively altered.
        """
        native = self.collection_root/'lerobot'
        meta, old_files = verified_media(source, native, self.last_lerobot_episode)
        temporary = source/'metadata.tmp'
        temporary.write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
        temporary.replace(source/'metadata.json')
        for old in old_files:
            old.unlink()
        for parent in {p.parent for p in old_files}:
            if not any(parent.iterdir()):
                parent.rmdir()
                if not any(parent.parent.iterdir()):
                    parent.parent.rmdir()


def validate_dataset(directory):
    """Read every numeric row and video boundary/midpoint through upstream."""
    import av
    import torch
    Dataset = upstream()
    root = Path(directory).resolve()
    if (root/'INCOMPLETE').exists():
        raise ValueError('Incomplete LeRobot write')
    ds = Dataset(REPO_ID, root=root, video_backend='pyav')
    numeric = ds.hf_dataset.with_format('numpy')
    for name, feature in ds.features.items():
        if feature['dtype'] != 'video' and not np.isfinite(np.asarray(numeric[name])).all():
            raise ValueError(f'Nonfinite {name}')
    checked, video_counts, depth_counts = [], {}, {}
    contract = json.loads((root/'meta/wowskin.json').read_text())
    has_depth = contract.get('depth', {}).get('enabled', False)
    camera_ids = tuple(int(key.rsplit('camera', 1)[1]) for key in ds.meta.video_keys)
    for episode in ds.meta.episodes:
        lo, hi = int(ds.episode_data_index['from'][episode]), int(ds.episode_data_index['to'][episode])
        for index in sorted({lo, (lo+hi)//2, hi-1}):
            sample = ds[index]
            for camera in camera_ids:
                image = sample[f'observation.images.camera{camera}']
                if image.shape != (3, 480, 640) or not torch.isfinite(image).all():
                    raise ValueError('Invalid official decoded image')
            checked.append(index)
        for camera in ds.meta.video_keys:
            path = root / ds.meta.get_video_file_path(episode, camera)
            with av.open(str(path)) as container:
                count = sum(1 for _ in container.decode(video=0))
            if count != hi-lo:
                raise ValueError(f'{path}: decoded {count}, expected {hi-lo} frames')
            video_counts[f'{episode}/{camera}'] = count
        if ANNOTATION in ds.features:
            for index in range(lo, hi):
                row = numeric[index]
                if int(row[ANNOTATION]) != int(row['task_index']):
                    raise ValueError('Language annotation does not match task_index')
                if not ds.meta.tasks[int(row['task_index'])].strip():
                    raise ValueError('Empty task instruction')
        if has_depth:
            for index in range(lo, hi):
                row = numeric[index]
                depth = load_depth_frame(root, episode, int(row['aux.raw_camera_index']))
                if abs(depth['timestamp']-float(row['aux.sim_timestamp'])) > 1e-8:
                    raise ValueError('RGB/depth/robot clock mismatch')
                for camera in camera_ids:
                    values = depth[f'camera{camera}']
                    if values.shape != (480, 640) or values.dtype != np.float32 or not np.isfinite(values).all():
                        raise ValueError('Invalid metric depth')
            depth_counts[str(episode)] = hi-lo
    return dict(dataset=str(root), lerobot_version=VERSION, format=FORMAT, episodes=ds.num_episodes,
                frames=len(ds), fps=ds.fps, cameras=ds.meta.video_keys, decoded_rows=checked,
                video_frame_counts=video_counts,
                depth_frame_pairs=depth_counts, instruction_validated=ANNOTATION in ds.features,
                official_loader_passed=True, all_numeric_rows_finite=True)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(validate_dataset(args.dataset), indent=2))


if __name__ == '__main__':
    main()
