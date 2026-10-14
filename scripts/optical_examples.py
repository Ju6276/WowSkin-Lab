"""Pack, verify, unpack and review the complete optical example recordings.

Archives preserve original bytes, including historical acquisition provenance.
Only raw recordings, LeRobot data, controller snapshots and reports are packed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import zstandard as zstd
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory

from wowskin_lab.paths import ROOT

BUNDLE = ROOT/'examples/optical'
BLOCK = 8*1024*1024
PART_SIZE = 512*1024*1024


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def source_files(source, task):
    roots = [source/task, source/'collection_report.json']
    roots += [source/name for name in ('controller_source.py', 'verification.json') if (source/name).is_file()]
    files = sorted(p for root in roots for p in (root.rglob('*') if root.is_dir() else [root])
                   if p.is_file() and not p.name.startswith('.'))
    if not files or any(p.name == 'INCOMPLETE' or p.is_symlink() for p in files):
        raise ValueError('Expected complete regular-file recordings')
    return files


def pack_one(item, output):
    source = ROOT/item['source']
    report = json.loads((source/'collection_report.json').read_text())
    if report['successes'] != item['episodes'] or len(report['episodes']) != item['episodes']:
        raise ValueError(f'Unexpected episode outcomes: {source}')
    files = source_files(source, item['task'])
    entry = dict(item, files={str(p.relative_to(source)): dict(bytes=p.stat().st_size, sha256=digest(p)) for p in files})
    entry['frames'] = json.loads((source/item['task']/'lerobot/meta/info.json').read_text())['total_frames']
    entry['parts'] = []
    with TemporaryDirectory(prefix='optical_pack_', dir=output) as temporary:
        archive = Path(temporary)/'data.tar.zst'
        # A large window also compresses repeated frame baselines without
        # modifying H5 layout, numeric precision, video encoding or timestamps.
        params = zstd.ZstdCompressionParameters.from_level(8, window_log=27, threads=2, write_checksum=1)
        compressor = zstd.ZstdCompressor(compression_params=params)
        with archive.open('wb') as raw, compressor.stream_writer(raw) as compressed, tarfile.open(fileobj=compressed, mode='w|') as packed:
            for path in files:
                info = packed.gettarinfo(path, arcname=str(Path(item['id'])/path.relative_to(source)))
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                info.mtime, info.mode = 0, 0o644
                with path.open('rb') as stream:
                    packed.addfile(info, stream)
        with archive.open('rb') as stream:
            index = 0
            while block := stream.read(BLOCK):
                path = output/'archives'/f'{item["id"]}.tar.zst.part{index:03d}'
                with path.open('xb') as part:
                    part.write(block)
                    remaining = PART_SIZE-len(block)
                    while remaining:
                        block = stream.read(min(BLOCK, remaining))
                        if not block:
                            break
                        part.write(block)
                        remaining -= len(block)
                entry['parts'].append(dict(path=str(path.relative_to(output)), bytes=path.stat().st_size, sha256=digest(path)))
                index += 1
    print(f'Packed {item["id"]}: {sum(p["bytes"] for p in entry["parts"])/1024**2:.1f} MiB', flush=True)
    return entry


def build(sources, output):
    if (output/'manifest.json').exists() or (output/'archives').exists():
        raise FileExistsError('Choose a new bundle directory; never overwrite existing archives')
    output.mkdir(parents=True, exist_ok=True)
    (output/'archives').mkdir()
    items = json.loads(sources.read_text())
    with ThreadPoolExecutor(max_workers=2) as executor:
        entries = list(executor.map(lambda item: pack_one(item, output), items))
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest = dict(schema='wowskin_optical_examples_v1', packaging_base_commit=revision,
                    source_code_sha256={name: digest(ROOT/name) for name in (
                        'wowskin_lab/data/auto_collect.py', 'wowskin_lab/data/scripted_tasks.py',
                        'wowskin_lab/envs/digit_sources.py', 'assets/reference/tacto/manifest.json')},
                    archive_format='tar.zst, ordered 512 MiB parts', entries=entries)
    write_json(output/'manifest.json', manifest)
    return manifest


def entries(bundle, ids):
    manifest = json.loads((bundle/'manifest.json').read_text())
    if manifest['schema'] != 'wowskin_optical_examples_v1':
        raise ValueError('Unknown bundle schema')
    result = manifest['entries']
    if ids:
        missing = set(ids)-{item['id'] for item in result}
        if missing:
            raise ValueError(f'Unknown examples: {sorted(missing)}')
        result = [item for item in result if item['id'] in ids]
    return result


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts or '\\' in name:
        raise ValueError(f'Unsafe archive path: {name}')
    return path


def verify_part(bundle, part):
    path = bundle/str(safe_relative(part['path']))
    if not path.is_file() or path.stat().st_size != part['bytes']:
        raise ValueError(f'Missing/incomplete local archive: {path}; obtain the recording bundle first')
    if digest(path) != part['sha256']:
        raise ValueError(f'Archive checksum mismatch: {path}')
    return path


def unpack_one(bundle, entry, output):
    name = str(safe_relative(entry['id']))
    if '/' in name:
        raise ValueError('Example id must be one directory name')
    destination = output/name
    if destination.exists():
        raise FileExistsError(f'Choose a new extraction directory: {destination}')
    parts = [verify_part(bundle, part) for part in entry['parts']]
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.optical_unpack_', dir=output) as temporary:
        root = Path(temporary)
        archive = root/'data.tar.zst'
        with archive.open('wb') as target:
            for path in parts:
                with path.open('rb') as stream:
                    shutil.copyfileobj(stream, target, BLOCK)
        seen = set()
        with archive.open('rb') as compressed, zstd.ZstdDecompressor().stream_reader(compressed) as stream, tarfile.open(fileobj=stream, mode='r|') as packed:
            for member in packed:
                path = safe_relative(member.name)
                if path.parts[0] != name or not member.isfile():
                    raise ValueError('Archive must contain only regular files under its example id')
                relative = str(PurePosixPath(*path.parts[1:]))
                if relative in seen or relative not in entry['files']:
                    raise ValueError(f'Unexpected/duplicate archive member: {member.name}')
                expected = entry['files'][relative]
                if member.size != expected['bytes']:
                    raise ValueError('Unpacked size differs from manifest')
                target = root/'unpacked'/str(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                with packed.extractfile(member) as stream, target.open('xb') as out:
                    shutil.copyfileobj(stream, out, BLOCK)
                if digest(target) != expected['sha256']:
                    raise ValueError(f'File checksum mismatch: {member.name}')
                seen.add(relative)
        if seen != set(entry['files']):
            raise ValueError('Archive is missing files')
        (root/'unpacked'/name).rename(destination)
    print(f'Unpacked and verified: {destination}', flush=True)
    return destination


def gallery(items, data, output):
    from wowskin_lab.data.episode_review import review
    if output.exists():
        raise FileExistsError(f'Choose a new gallery directory: {output}')
    output.mkdir(parents=True)
    catalog = []
    for item in items:
        source = data/item['id']
        report = json.loads((source/'collection_report.json').read_text())
        for episode in range(item['episodes']):
            target = output/item['id']/f'episode_{episode:06d}'
            result = review(source/item['task']/'lerobot', target, episode=episode)
            measured = json.loads((target/'review.json').read_text())
            key = (f'{item["backend"]}-{episode+1}' if item['task'] == 'pick_cube' else
                   f'{item["task"]}-'+('elastic_beam-' if item['clip_model'] == 'elastic_beam' else '')+f'{item["backend"]}-{episode+1}')
            catalog.append(dict(id=key, task=item['task'], backend=item['backend'], clip_model=item['clip_model'],
                sensor_label=item['sensor_label'], episode=episode, note=item['note'], frames=result['frames'],
                status=measured['status'], success=measured['task_evaluation']['success'],
                seed=report['episodes'][episode]['seed'], source='scripted_simulation_acquisition',
                url=str(target.relative_to(output)/'index.html')))
            print(f'Reviewed: {item["id"]}/{episode}', flush=True)
    write_json(output/'catalog.json', catalog)
    template = (ROOT/'wowskin_lab/data/episode_gallery.html').read_text()
    (output/'index.html').write_text(template.replace('__CATALOG_DATA__', json.dumps(catalog, ensure_ascii=False).replace('<', '\\u003c')))


def validate_data(items, data, output):
    """Check relocated examples through the real readers and action contract."""
    import h5py
    import numpy as np
    import torch
    from wowskin_lab.data.episode_review import resolve_episode
    from wowskin_lab.data.lerobot_recording import validate_dataset
    from wowskin_lab.learning.optical_act_data import OpticalEpisodes
    results = []
    for item in items:
        source = data/item['id']
        report = json.loads((source/'collection_report.json').read_text())
        root = source/item['task']/'lerobot'
        result = dict(id=item['id'], **validate_dataset(root))
        learning = OpticalEpisodes(root, image_size=32, chunk_size=5)
        rows = learning.ds.hf_dataset.with_format('numpy')
        result['episodes_checked'] = []
        for ep in range(item['episodes']):
            raw, _ = resolve_episode(root, ep)
            if not raw.is_relative_to(source.resolve()):
                raise ValueError('Readback escaped the relocated example into its original acquisition directory')
            meta = json.loads((raw/'metadata.json').read_text())
            expected_seed = report['episodes'][ep]['seed']
            if meta['simulation_config']['seed'] != expected_seed:
                raise ValueError(f'Raw metadata/reset seed mismatch: {item["id"]}/{ep}')
            if item['backend'] == 'tacto':
                with h5py.File(root/f'optical/chunk-000/episode_{ep:06d}.h5') as optics:
                    if 'black RGB [0.02, 0.02, 0.02]' not in str(optics.attrs['metadata']):
                        raise ValueError('TACTO example does not record the black appearance provenance')
            lo, hi = (int(learning.ds.episode_data_index[key][ep]) for key in ('from', 'to'))
            checked = dict(episode=ep, seed=expected_seed, frames=hi-lo)
            with h5py.File(raw/'measurements.h5') as original:
                for i in range(lo, hi):
                    row = rows[i]
                    k = int(np.asarray(row['aux.raw_sample_index']).item())
                    np.testing.assert_allclose(row['action'], np.r_[original['commanded_pose'][k+1], original['commanded_gripper'][k+1]], atol=1e-6)
                    np.testing.assert_allclose(row['observation.state'], np.r_[original['cartesian_states'][k], original['gripper_states'][k]], atol=1e-6)
                if item['clip_model'] == 'elastic_beam':
                    evaluation = original['evaluation']
                    for side in ('upper', 'lower'):
                        tips = evaluation[f'clip_{side}_tip_m'][:]
                        checked[side+'_max_tip_displacement_mm'] = float(np.linalg.norm(tips-tips[0], axis=1).max())*1000
                        if checked[side+'_max_tip_displacement_mm'] <= .01:
                            raise ValueError('Elastic lip did not move in recorded example')
            evaluation = meta['task_evaluation']
            if not evaluation['success']:
                raise ValueError('Example task score is not successful')
            if item['task'] == 'feeding' and not evaluation['final_metrics']['delivery_eligible']:
                raise ValueError('Feeding did not retain final eligibility')
            if item['task'] == 'cable_insert':
                metrics = evaluation['metrics']
                if not metrics['clip_inside'] or metrics['clip_span_count'] != 3 or metrics['clip_dwell_s'] < .3:
                    raise ValueError('Cable was not retained at the final recorded state')
            result['episodes_checked'].append(checked)
        for i in range(len(learning)):
            if not all(torch.isfinite(v).all() for v in learning[i].values()):
                raise ValueError('Nonfinite ACT input')
        if item['episodes'] > 1:
            train, valid, _ = learning.split(item['episodes']-1)
            result.update(act_train_rows=len(train), act_validation_rows=len(valid))
        else:
            try:
                learning.split(0)
            except ValueError:
                result['single_episode_split_rejected'] = True
            else:
                raise ValueError('Single-episode training split was incorrectly accepted')
        result['act_samples_finite'] = True
        results.append(result)
        if output:
            write_json(output, results)
        print(f'Validated {item["id"]}: {result["frames"]} frames', flush=True)
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['list', 'build', 'verify', 'unpack', 'validate', 'gallery'])
    p.add_argument('--bundle', type=Path, default=BUNDLE)
    p.add_argument('--ids', nargs='+', help='Only selected example ids (default: all)')
    p.add_argument('--output', type=Path)
    p.add_argument('--data', type=Path, default=ROOT/'data/examples_optical')
    p.add_argument('--sources', type=Path, default=BUNDLE/'sources.json')
    args = p.parse_args(argv)
    if args.command == 'build':
        return build(args.sources, args.output or args.bundle)
    selected = entries(args.bundle, args.ids)
    if args.command == 'list':
        for item in selected:
            print(item['id'], item['episodes'], 'episodes', sum(p['bytes'] for p in item['parts']), 'archive bytes')
    elif args.command == 'verify':
        for item in selected:
            for part in item['parts']:
                verify_part(args.bundle, part)
            print(f'Archive checksums OK: {item["id"]}', flush=True)
    elif args.command == 'unpack':
        for item in selected:
            unpack_one(args.bundle, item, args.output or args.data)
    elif args.command == 'validate':
        validate_data(selected, args.data, args.output)
    else:
        if args.output is None:
            p.error('gallery requires --output')
        gallery(selected, args.data, args.output)


if __name__ == '__main__':
    main()
