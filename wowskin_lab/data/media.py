"""Resolve a capture's single RGB/depth copy, including older inline AVI data."""
import json
from pathlib import Path


def video_path(episode, camera, metadata=None):
    episode = Path(episode)
    metadata = metadata or json.loads((episode/'metadata.json').read_text())
    relative = metadata.get('video_files', {}).get(str(camera), f'cam_{camera}.avi')
    return (episode/relative).resolve()


def depth_path(episode, metadata=None):
    episode = Path(episode)
    metadata = metadata or json.loads((episode/'metadata.json').read_text())
    return (episode/metadata.get('depth', {}).get('path', 'depth.h5')).resolve()
