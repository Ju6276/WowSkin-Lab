"""Fetch the unchanged feeding station subset from pinned EBiM task 3."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
REVISION = '161db49e9eafd34be31e5bdada72d8e2f796c16a'
BASE = f'https://raw.githubusercontent.com/EBiM-Benchmark/benchmark/{REVISION}/'
DEST = ROOT/'assets/ebim_feeding'
from wowskin_lab.envs.feeding import BODIES, selected_world, asset_subset


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    old = json.loads((DEST/'manifest.json').read_text()) if (DEST/'manifest.json').exists() else {}
    def fetch(item):
        local, upstream = item
        path, url = DEST/local, BASE+upstream
        previous = old.get('files', {}).get(local, {})
        if not (path.exists() and previous.get('url') == url and
                hashlib.sha256(path.read_bytes()).hexdigest() == previous.get('sha256')):
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix+'.part')
            temporary.write_bytes(content)
            temporary.replace(path)
        content = path.read_bytes()
        return local, dict(url=url, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    entries = dict([fetch(('scene_100.xml', 'task3_mujoco/scene_100.xml'))])
    source = ET.parse(DEST/'scene_100.xml').getroot()
    files = {'config.json': 'task3_mujoco/config.json', 'upstream_README.md': 'task3_mujoco/README.md',
             'upstream_teleop.py': 'task3_mujoco/teleop.py',
             'LICENSE': 'LICENSE', 'NOTICE': 'NOTICE', 'license_notes.md': 'LICENSES/README.md'}
    for e in asset_subset(source, selected_world(source)):
        if e.get('file'):
            path = str(Path('assets')/e.get('file')) if e.tag == 'mesh' else e.get('file')
            files[path] = 'task3_mujoco/'+path
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        entries.update(pool.map(fetch, files.items()))
    (DEST/'manifest.json').write_text(json.dumps(dict(
        repository='https://github.com/EBiM-Benchmark/benchmark', revision=REVISION, files=entries,
        selected_bodies=BODIES, bean_count=100,
        status='Unchanged upstream bytes. Local scene construction is in wowskin_lab/envs/feeding.py.',
        omitted='Mobile dual arm, apartment, distant A-I stations, sink and weighing/knock-box station.'
    ), indent=2)+'\n')
    print(f'Fetched {len(entries)} unchanged files ({sum(e["bytes"] for e in entries.values())} bytes) to {DEST}')


if __name__ == '__main__':
    main()
