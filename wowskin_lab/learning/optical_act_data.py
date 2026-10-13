"""Optical LeRobot observations and absolute TCP actions for upstream ACT.

This adapter changes neither the upstream policy nor recorded action labels.
Only demonstration rows enter training; splits are by complete episode.
"""
import json
from pathlib import Path

import av
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset

from wowskin_lab.data.lerobot_recording import REPO_ID, upstream


def image_tensor(rgb, size):
    """RGB uint8 -> square letterbox CHW uint8, identical offline and live."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or size < 32:
        raise ValueError('Expected uint8 RGB and image size >= 32')
    h, w = rgb.shape[:2]
    scale = size/max(h, w)
    image = cv2.resize(rgb, (max(1, round(w*scale)), max(1, round(h*scale))), interpolation=cv2.INTER_AREA)
    out = np.full((size, size, 3), 127, dtype=np.uint8)
    y, x = (size-image.shape[0])//2, (size-image.shape[1])//2
    out[y:y+image.shape[0], x:x+image.shape[1]] = image
    return torch.from_numpy(out.transpose(2, 0, 1).copy())


def absolute_command_delta(action, transform, target_xyz, target_rotation):
    """Absolute robot-base TCP goal -> world translation/local rotation delta.

    Subtract the pending commanded target, not measured TCP feedback. Reusing
    a goal on subsequent 100 Hz ticks therefore never accumulates it twice.
    Normal control limits remain enforced inside env.step.
    """
    action = np.asarray(action, dtype=float)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError('Expected one finite seven-dimensional absolute action')
    world_position = transform[:3, :3]@action[:3]+transform[:3, 3]
    world_rotation = transform[:3, :3]@Rotation.from_rotvec(action[3:6]).as_matrix()
    local_rotation = Rotation.from_matrix(target_rotation.T@world_rotation).as_rotvec()
    return np.r_[world_position-target_xyz, local_rotation, np.clip(action[6], 0., 1.)]


def relative_rotation_pose(pose, reference, *, inverse=False):
    """Express rotvec relative to a fixed training pose, preserving XYZ/grip.

    Mounted TCP poses are near pi in the base frame. Directly normalizing
    their canonical rotvec makes tiny rotations across pi a huge input jump.
    The stored reference moves that discontinuity away from the task poses.
    """
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape[-1] != 7 or not np.isfinite(pose).all():
        raise ValueError('Expected finite seven-dimensional poses')
    result = pose.copy()
    orientation = Rotation.from_rotvec(pose[..., 3:6].reshape(-1, 3))
    base = Rotation.from_rotvec(reference)
    result[..., 3:6] = ((base*orientation) if inverse else (base.inv()*orientation)).as_rotvec().reshape(pose.shape[:-1]+(3,))
    return result.astype('float32')


def control_deadline(policy_step):
    """100 Hz tick at/after the Nth 30 Hz camera deadline, without drift."""
    if policy_step < 0 or int(policy_step) != policy_step:
        raise ValueError('Policy step must be a nonnegative integer')
    return (int(policy_step)*100+29)//30


def chunk_indices(start, length, chunk_size):
    indices = np.arange(start, start+chunk_size)
    return np.minimum(indices, length-1), indices >= length


class OpticalEpisodes(Dataset):
    def __init__(self, root, *, image_size=224, chunk_size=30, cache_dir=None):
        self.root = Path(root).resolve()
        if (self.root/'INCOMPLETE').exists():
            raise ValueError('Dataset has an INCOMPLETE marker; finish/repair recording before training')
        self.contract = json.loads((self.root/'meta/wowskin.json').read_text())
        if self.contract['schema'] != 'wowskin_lerobot_optical_v1' or self.contract['pose_frame'] != 'robot_base':
            raise ValueError('Optical ACT requires the optical robot-base absolute-command dataset contract')
        self.ds = upstream()(REPO_ID, root=self.root, video_backend='pyav')
        if self.ds.fps != 30 or self.contract['signature']['config']['sample_hz'] != 100:
            raise ValueError('Optical ACT requires nominal 30 Hz images and 100 Hz robot acquisition')
        self.image_keys = list(self.ds.meta.video_keys)
        expected = [f'observation.images.camera{i}' for i in self.contract['signature']['camera_ids']]
        expected += ['observation.images.tactile_left', 'observation.images.tactile_right']
        if set(self.image_keys) != set(expected):
            raise ValueError('Expected every recorded scene camera and both optical fingers')
        self.image_size, self.chunk_size = image_size, chunk_size
        cache_dir = Path(cache_dir) if cache_dir is not None else None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=False)
        self.episodes, self.rows = {}, []
        sources = [json.loads(line) for line in (self.root/'meta/wowskin_sources.jsonl').read_text().splitlines()]
        self.sources = {s['episode_index']: s for s in sources}
        if set(self.sources) != set(range(self.ds.num_episodes)) or len(self.sources) != len(sources):
            raise ValueError('Missing or duplicate episode provenance')
        values = self.ds.hf_dataset.with_format('numpy')
        for ep in range(self.ds.num_episodes):
            if self.sources[ep]['status'] != 'success':
                raise ValueError('Training input contains an episode not marked successful')
            lo, hi = int(self.ds.episode_data_index['from'][ep]), int(self.ds.episode_data_index['to'][ep])
            rows = values[lo:hi]
            selected = np.flatnonzero(np.asarray(rows['aux.acquisition_phase']).reshape(-1) == 1)
            if len(selected) < 2 or np.any(np.diff(selected) != 1):
                raise ValueError('Need a contiguous demonstration phase with at least two camera frames')
            state = np.asarray(rows['observation.state'])[selected].astype('float32')
            action = np.asarray(rows['action'])[selected].astype('float32')
            if state.shape[1:] != (7,) or action.shape != state.shape or not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError('Expected finite seven-dimensional measured states and issued actions')
            episode = {'observation.state': torch.from_numpy(state), 'action': torch.from_numpy(action)}
            for key in self.image_keys:
                shape = (len(selected), 3, image_size, image_size)
                cache_file = cache_dir/f'{ep:06d}_{key}.npy' if cache_dir else None
                images = (np.lib.format.open_memmap(cache_file, mode='w+', dtype='uint8', shape=shape)
                          if cache_file else np.empty(shape, dtype='uint8'))
                # Decode sequentially once; training random access never seeks
                # to the wrong keyframe or repeatedly decodes full videos.
                with av.open(str(self.root/self.ds.meta.get_video_file_path(ep, key))) as stream:
                    count = 0
                    for i, frame in enumerate(stream.decode(video=0)):
                        count += 1
                        if selected[0] <= i <= selected[-1]:
                            images[i-selected[0]] = image_tensor(frame.to_ndarray(format='rgb24'), image_size).numpy()
                if count != hi-lo:
                    raise ValueError(f'Video/row count mismatch: episode {ep}, {key}')
                if cache_file:
                    images.flush()
                    del images
                    images = np.load(cache_file, mmap_mode='r')
                episode[key] = images
            self.episodes[ep] = episode
            self.rows.extend((ep, i) for i in range(len(selected)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        ep, offset = self.rows[index]
        episode = self.episodes[ep]
        selected, padded = chunk_indices(offset, len(episode['action']), self.chunk_size)
        result = {key: torch.from_numpy(episode[key][offset].copy()).float()/255 for key in self.image_keys}
        result.update({'observation.state': episode['observation.state'][offset],
                       'action': episode['action'][selected], 'action_is_pad': torch.from_numpy(padded)})
        return result

    def split(self, validation_episode):
        if len(self.episodes) < 2 or validation_episode not in self.episodes:
            raise ValueError('Need at least two episodes and an existing held-out validation episode')
        train = [i for i, (ep, _) in enumerate(self.rows) if ep != validation_episode]
        valid = [i for i, (ep, _) in enumerate(self.rows) if ep == validation_episode]
        stats = {}
        for key in ('observation.state', 'action'):
            values = torch.cat([episode[key] for ep, episode in self.episodes.items() if ep != validation_episode])
            stats[key] = {'mean': values.mean(0), 'std': values.std(0, unbiased=False).clamp_min(1e-3)}
        # The public ResNet input convention is fixed, not fit to held-out data.
        for key in self.image_keys:
            stats[key] = {'mean': torch.tensor([.485, .456, .406]).reshape(3, 1, 1),
                          'std': torch.tensor([.229, .224, .225]).reshape(3, 1, 1)}
        return train, valid, stats

    def relative_rotations(self, reference):
        for episode in self.episodes.values():
            for key in ('observation.state', 'action'):
                episode[key] = torch.from_numpy(relative_rotation_pose(episode[key].numpy(), reference))
                if torch.linalg.vector_norm(episode[key][:, 3:6], dim=-1).max() >= np.pi-.1:
                    raise ValueError('Episode approaches the relative-rotation branch cut; this 7D adapter needs a different orientation representation')
