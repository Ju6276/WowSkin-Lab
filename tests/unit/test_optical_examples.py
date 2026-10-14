"""Distributed examples must survive relocation and reject unsafe archives."""
import io
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest
import zstandard as zstd
from unittest.mock import patch

from scripts.optical_examples import digest, pack_one, unpack_one


class OpticalExamplesTests(unittest.TestCase):
    def test_chunked_archive_roundtrip_preserves_every_byte(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'source'
            (source/'feeding/lerobot/meta').mkdir(parents=True)
            (source/'feeding/lerobot/meta/info.json').write_text('{"total_frames": 4}')
            (source/'feeding/raw/capture_000000').mkdir(parents=True)
            raw = source/'feeding/raw/capture_000000/measurements.h5'
            raw.write_bytes(bytes(range(256))*30)
            (source/'collection_report.json').write_text(json.dumps({'successes': 1, 'episodes': [{}]}))
            bundle = root/'bundle'
            (bundle/'archives').mkdir(parents=True)
            item = dict(id='example', source=str(source), task='feeding', episodes=1)
            with patch('scripts.optical_examples.PART_SIZE', 128), patch('scripts.optical_examples.BLOCK', 64):
                entry = pack_one(item, bundle)
            self.assertGreater(len(entry['parts']), 1)
            target = unpack_one(bundle, entry, root/'relocated')
            self.assertEqual((target/'feeding/raw/capture_000000/measurements.h5').read_bytes(), raw.read_bytes())
            self.assertEqual({str(p.relative_to(target)) for p in target.rglob('*') if p.is_file()}, set(entry['files']))
            with self.assertRaises(FileExistsError):
                unpack_one(bundle, entry, root/'relocated')
            part = bundle/entry['parts'][0]['path']
            part.write_bytes(b'x'*part.stat().st_size)
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                unpack_one(bundle, entry, root/'corrupt')
            self.assertFalse((root/'corrupt/example').exists())

    def test_traversal_and_links_are_rejected_before_installing(self):
        for name, kind in [('example/../../escape', tarfile.REGTYPE), ('example/link', tarfile.SYMTYPE)]:
            with self.subTest(name=name), TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = root/'bad.tar.zst'
                with archive.open('wb') as raw, zstd.ZstdCompressor().stream_writer(raw) as compressed, tarfile.open(fileobj=compressed, mode='w|') as packed:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    member.size = 1 if kind == tarfile.REGTYPE else 0
                    member.linkname = '/tmp'
                    packed.addfile(member, io.BytesIO(b'x') if member.size else None)
                entry = dict(id='example', files={}, parts=[dict(path=archive.name,
                    bytes=archive.stat().st_size, sha256=digest(archive))])
                with self.assertRaises(ValueError):
                    unpack_one(root, entry, root/'output')
                self.assertFalse((root/'output/example').exists())
                self.assertFalse((root/'escape').exists())
