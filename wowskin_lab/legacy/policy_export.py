"""ARCHIVED local exports with simplified signals/ground-truth points; not official data."""
import json
from pathlib import Path
import pickle

import numpy as np


def load_episode(path):
    path = Path(path)
    metadata = json.loads((path/'metadata.json').read_text())
    files = sorted((path/'frames').glob('*.npz'))
    if len(files) != metadata['frame_count'] or len(files) < 12:
        raise ValueError(f'{path}: need at least 12 complete synchronized frames')
    frames = []
    for file in files:
        with np.load(file, allow_pickle=False) as frame:
            frames.append({key: frame[key] for key in frame.files})
    if any(set(f) != set(frames[0]) for f in frames):
        raise ValueError(f'{path}: inconsistent frame fields')
    arrays = {key: np.stack([f[key] for f in frames]) for key in frames[0]}
    if not (np.diff(arrays['time_s']) > 0).all():
        raise ValueError(f'{path}: non-increasing episode time')
    return arrays, metadata


def export_dataset(directory, destination, method, statuses=('success',), episode_ids=None):
    if method not in ('visk', 'ftf'):
        raise ValueError(method)
    directory, destination = Path(directory), Path(destination)
    if destination.exists():
        raise FileExistsError(f'Choose a new output path instead of overwriting {destination}')
    if episode_ids is not None and len(set(episode_ids)) != len(episode_ids):
        raise ValueError('Duplicate episode IDs')
    paths = sorted(p.parent for p in directory.glob('episode_*/metadata.json'))
    if episode_ids is not None:
        paths = [p for p in paths if p.name in episode_ids]
        if len(paths) != len(episode_ids):
            raise ValueError('One or more selected episode IDs were not found')
    observations, used, signatures = [], [], []
    for path in paths:
        metadata = json.loads((path/'metadata.json').read_text())
        if metadata['status'] not in statuses:
            continue
        arrays, metadata = load_episode(path)
        signature = {k: metadata[k] for k in ('record_hz', 'camera_keys', 'color_order',
                                              'sensor_baseline_subtracted', 'point_source')}
        if signatures and signature != signatures[0]:
            raise ValueError('Mixed acquisition configurations must be exported separately')
        signatures.append(signature)
        obs = {k: arrays[k].astype(np.float32) for k in ('cartesian_states', 'gripper_states')}
        keys = metadata['camera_keys'] if method == 'visk' else ['pixels1', 'pixels2']
        for key in keys:
            obs[key] = arrays[key].astype(np.uint8)
        if method == 'visk':
            obs['sensor_states'] = arrays['sensor_states'].astype(np.float32)
        else:
            center = metadata['center_sensor_index']
            obs['sensor_states'] = np.linalg.norm(arrays['sensor_states'][:, center*3:center*3+3], axis=1).astype(np.float32)
            # Original robot branch uses -1=open, +1=closed. Keep continuous
            # measured aperture here; binarization is a separate training option.
            obs['gripper_states'] = 2*obs['gripper_states']-1
            for key in keys:
                for label in ('robot', 'object'):
                    obs[f'{label}_tracks_{key}'] = arrays[f'{label}_tracks_{key}'].astype(np.float32)
                    obs[f'{label}_tracks_3d_{key}'] = arrays[f'{label}_points_world_m'].astype(np.float32)
        observations.append(obs)
        used.append(dict(id=path.name, status=metadata['status'], source=metadata['source'],
                         frames=len(arrays['time_s']), start_s=float(arrays['time_s'][0]),
                         end_s=float(arrays['time_s'][-1])))
    if not observations:
        raise ValueError('No eligible episodes; mark success or explicitly choose --include-status')
    result = dict(observations=observations)
    for name, key in [('cartesian', 'cartesian_states'), ('gripper', 'gripper_states'), ('sensor', 'sensor_states')]:
        all_values = np.concatenate([o[key] for o in observations], axis=0)
        result[f'min_{name}'], result[f'max_{name}'] = all_values.min(axis=0), all_values.max(axis=0)
    result['metadata'] = dict(schema='visk' if method == 'visk' else 'ftf_robot_p3po',
                              schema_version=1, episodes=used, **signatures[0],
                              uncalibrated=True, tactile='15 delta-B channels' if method == 'visk' else 'center delta-B norm; not newtons')
    result['metadata']['reproduction_status'] = 'simulation_adapter_not_paper_reproduction'
    if method == 'ftf':
        result['metadata']['robot_point_definition'] = 'legacy local pad-centered landmarks; not official FTF geometry'
        result['metadata']['object_point_definition'] = 'local task: eight cube corners from simulation ground truth'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix('.tmp').open('wb') as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
    destination.with_suffix('.tmp').replace(destination)
    destination.with_suffix('.json').write_text(json.dumps(result['metadata'], indent=2)+'\n')
    return dict(path=str(destination), episodes=len(observations), frames=sum(len(o['sensor_states']) for o in observations), method=method)
