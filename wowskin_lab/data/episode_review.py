"""Build an offline, synchronized review of a recorded optical episode.

The MP4 is a display derivative. Lossless optical arrays and 100 Hz robot
measurements remain in the source files; no simulation is run during replay.
"""
import argparse
import base64
from contextlib import ExitStack
import json
from pathlib import Path

import av
import cv2
import h5py
import numpy as np

from .media import video_path


def resolve_episode(path, episode=0):
    path = Path(path).resolve()
    if (path/'metadata.json').is_file():
        return path, None
    info = json.loads((path/'meta/info.json').read_text())
    sources = [json.loads(line) for line in (path/'meta/wowskin_sources.jsonl').read_text().splitlines()]
    source = next((s for s in sources if s['episode_index'] == episode), None)
    if source is None:
        raise ValueError(f'No episode {episode} in {path}')
    candidates = [path/source['source_relative']] if source.get('source_relative') else []
    candidates.append(Path(source['source']))
    raw = next((p.resolve() for p in candidates if (p/'measurements.h5').is_file()), None)
    if raw is None:
        raise FileNotFoundError('Full review requires the raw 100 Hz measurements; keep the raw folder with LeRobot')
    return raw, (path, info, source)


def synchronized_indices(times, camera_times):
    times, camera_times = np.asarray(times), np.asarray(camera_times)
    if not len(times) or not len(camera_times) or not np.isfinite(times).all() or not np.isfinite(camera_times).all():
        raise ValueError('Missing or nonfinite acquisition timestamps')
    if np.any(np.diff(times) <= 0) or np.any(np.diff(camera_times) <= 0):
        raise ValueError('Acquisition timestamps must strictly increase')
    indices = np.searchsorted(times, camera_times-1e-8)
    if np.any(indices >= len(times)) or not np.allclose(times[indices], camera_times, atol=1e-8, rtol=0):
        raise ValueError('Camera and robot timestamps do not align')
    return indices


def _tile(rgb, label, width=320, height=268):
    canvas = np.full((height, width, 3), (238, 239, 232), dtype=np.uint8)
    cv2.putText(canvas, label, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (41, 58, 49), 1, cv2.LINE_AA)
    if rgb is None:
        cv2.putText(canvas, 'Not provided', (20, height//2), cv2.FONT_HERSHEY_SIMPLEX, .6, (76, 89, 80), 1, cv2.LINE_AA)
        return canvas
    h, w = rgb.shape[:2]
    scale = min(width/w, (height-36)/h)
    image = cv2.resize(rgb, (max(1, round(w*scale)), max(1, round(h*scale))), interpolation=cv2.INTER_AREA)
    y, x = 34+(height-36-image.shape[0])//2, (width-image.shape[1])//2
    canvas[y:y+image.shape[0], x:x+image.shape[1]] = image
    return canvas


def _depth_image(group, limit):
    depth = group['indentation_m'][:]
    valid = group['valid_mask'][:].astype(bool) if 'valid_mask' in group else np.isfinite(depth)
    scaled = np.clip(depth/limit*255, 0, 255).astype('uint8')
    rgb = cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS)[..., ::-1].copy()
    rgb[~valid] = (200, 200, 200)
    return rgb


def review(path, output, *, episode=0):
    raw, packed = resolve_episode(path, episode)
    meta = json.loads((raw/'metadata.json').read_text())
    if meta.get('tactile_modality') != 'optical':
        raise ValueError('dataset-review currently requires an optical episode')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new review directory: {output}')
    cameras = meta['camera_ids']
    optical_path = raw/meta['optical']['path']
    videos = [video_path(raw, camera, meta) for camera in cameras]
    if packed:
        root, info, source = packed
        optical_path = root/source['optical_path']
        videos = [root/info['video_path'].format(episode_index=episode,
                  episode_chunk=episode//info['chunks_size'], video_key=f'observation.images.camera{camera}')
                  for camera in cameras]
    with ExitStack() as stack:
        samples = stack.enter_context(h5py.File(raw/'measurements.h5'))
        optics = stack.enter_context(h5py.File(optical_path))
        times, camera_times = samples['timestamp'][:], samples['camera_timestamp'][:]
        indices = synchronized_indices(times, camera_times)
        if not np.array_equal(camera_times, optics['timestamp'][:]):
            raise ValueError('Optical timestamps differ from the recorded camera clock')
        frames = len(camera_times)
        if packed and packed[2]['frames'] != frames:
            raise ValueError('LeRobot provenance and raw camera count differ')
        # LeRobot may encode AV1; OpenCV builds do not consistently decode it.
        # Use the same PyAV decoder family as the pinned dataset reader.
        containers = [stack.enter_context(av.open(str(video))) for video in videos]
        decoders = [iter(container.decode(video=0)) for container in containers]
        # One shared scale across both fingers and the entire episode.
        limit = max(1e-6, max(float(np.max(optics[f'frames/{i:06d}/{s}/indentation_m'][:]))
                             for i in range(frames) for s in ('left', 'right')))
        state = np.column_stack((samples['cartesian_states'][:], samples['gripper_states'][:]))
        command = np.column_stack((samples['commanded_pose'][:], samples['commanded_gripper'][:]))
        if not np.isfinite(state).all() or not np.isfinite(command).all():
            raise ValueError('Nonfinite robot data')
        payload = dict(schema='wowskin_optical_review_v1', source=str(raw), task=meta['task'],
            backend=meta['optical']['backend'], status=meta['status'], instruction=meta.get('instruction', ''),
            task_evaluation=meta.get('task_evaluation'), fps=30, frames=frames,
            time_s=times.tolist(), camera_time_s=camera_times.tolist(), camera_sample_index=indices.tolist(),
            state=state.tolist(), command=command.tolist(),
            phase=samples['phase'][:].astype(int).tolist() if 'phase' in samples else None,
            depth_limit_mm=limit*1000, optical=[],
            display='RGB resized without brightness adjustment; H.264 preview only. Source H5 remains lossless.',
            clock='Video uses nominal 30 Hz; every frame maps to its actual acquisition timestamp.',
            robot='100 Hz, measured state and issued absolute target, in robot-base XYZ metres / rotation vector radians / closure 0..1. Not the next-command training label.')
        cfg = meta['simulation_config']
        payload['clip_model'] = cfg.get('clip_model', 'rigid') if meta['task'] in ('cable', 'cable_insert') else None
        if payload['clip_model'] == 'elastic_beam':
            evaluation = samples['evaluation']
            signals = {'opening_mm': evaluation['clip_tip_distance_m'][:]*1000}
            for side in ('upper', 'lower'):
                tip = evaluation[f'clip_{side}_tip_m'][:]
                if tip.shape != (len(times), 3):
                    raise ValueError('Elastic clip tip trajectory does not match the robot clock')
                signals[f'{side}_displacement_mm'] = np.linalg.norm(tip-tip[0], axis=1)*1000
            if any(values.shape != times.shape or not np.isfinite(values).all() for values in signals.values()):
                raise ValueError('Invalid recorded elastic clip diagnostics')
            payload['clip_signals'] = {key: values.tolist() for key, values in signals.items()}
            payload['clip_model_schema'] = cfg['elastic_clip']['schema']
        output.mkdir(parents=True)
        movie = stack.enter_context(av.open(str(output/'review.mp4'), mode='w', options={'movflags': '+faststart'}))
        stream = movie.add_stream('libx264', rate=30)
        stream.width, stream.height, stream.pix_fmt = 1280, 804, 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'fast', 'g': '30'}
        for i in range(frames):
            scene_tiles = []
            for camera, decoder in zip(cameras, decoders):
                frame = next(decoder, None)
                if frame is None:
                    raise ValueError(f'Missing camera {camera} frame {i}')
                scene_tiles.append(_tile(frame.to_ndarray(format='rgb24'), f'Camera {camera}', width=1280//len(cameras)))
            scene = np.concatenate(scene_tiles, axis=1)
            scene = cv2.copyMakeBorder(scene, 0, 0, 0, 1280-scene.shape[1], cv2.BORDER_REPLICATE)
            rows, details = [scene], {}
            for side in ('left', 'right'):
                group = optics[f'frames/{i:06d}/{side}']
                mask = group['contact_mask'][:].astype('uint8')*255 if 'contact_mask' in group else None
                if mask is not None:
                    mask = np.repeat(mask[..., None], 3, axis=2)
                marker = group['marker_flow_rgb'][:] if 'marker_flow_rgb' in group else None
                marker_status = str(group.attrs.get('marker_status', 'not_provided'))
                rows.append(np.concatenate([
                    _tile(group['rgb'][:], f'{side} | native RGB'),
                    _tile(_depth_image(group, limit), f'Indentation 0 - {limit*1000:.3f} mm'),
                    _tile(mask, 'Contact mask'), _tile(marker, 'Marker flow')], axis=1))
                details[side] = dict(normal_force_n=float(group.attrs['normal_force_n']),
                    max_indentation_mm=float(np.max(group['indentation_m'][:]))*1000,
                    marker_status=marker_status)
            payload['optical'].append(details)
            frame = av.VideoFrame.from_ndarray(np.concatenate(rows), format='rgb24')
            for packet in stream.encode(frame):
                movie.mux(packet)
        if any(next(decoder, None) is not None for decoder in decoders):
            raise ValueError('Video contains extra frames without timestamps')
        for packet in stream.encode():
            movie.mux(packet)
    # Escaping '<' prevents user-supplied instructions from closing the JSON
    # script element. All text in the UI is inserted with textContent.
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    (output/'review.json').write_text(serialized+'\n')
    template = Path(__file__).with_name('episode_review.html').read_text()
    page = template.replace('__REVIEW_DATA__', serialized.replace('<', '\\u003c'))
    # A Blob-backed video works offline and with simple local servers that
    # lack HTTP Range support (Safari cannot seek those external MP4 URLs).
    page = page.replace('__REVIEW_VIDEO__', base64.b64encode((output/'review.mp4').read_bytes()).decode('ascii'))
    (output/'index.html').write_text(page)
    return dict(page=str(output/'index.html'), video=str(output/'review.mp4'), frames=frames,
                raw_samples=len(times), backend=payload['backend'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path, help='LeRobot root or one raw capture directory')
    parser.add_argument('--episode', type=int, default=0, help='LeRobot episode index')
    parser.add_argument('--output', type=Path, required=True, help='New directory for offline HTML/MP4/JSON')
    args = parser.parse_args(argv)
    print(json.dumps(review(args.dataset, args.output, episode=args.episode), indent=2))


if __name__ == '__main__':
    main()
