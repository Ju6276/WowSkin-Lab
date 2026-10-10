"""Fetch only the pinned EBiM task-1 fixtures and scene reference, with provenance."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = '161db49e9eafd34be31e5bdada72d8e2f796c16a'
BASE = 'https://raw.githubusercontent.com/EBiM-Benchmark/benchmark/' + REVISION + '/'
CORE = 'task1_mujoco/robotiq_duo_full_scene_minimal_core/'
FILES = {
    'source_scene.xml': CORE + 'duo_full_scene_grasp.xml',
    'LICENSE': 'task1_mujoco/LICENSE',
    'repository_LICENSE': 'LICENSE',
    'NOTICE': 'NOTICE',
    'license_notes.md': 'LICENSES/README.md',
    **{name: CORE + 'assets/' + name for name in
       ('round_peg.stl', 'base_connector.stl', 'wire_harness.stl', 'C-Clip.stl', 'board_top_texture.png')},
}


def main():
    dest = ROOT / 'assets/ebim'
    dest.mkdir(parents=True, exist_ok=True)
    old = json.loads((dest/'manifest.json').read_text()) if (dest/'manifest.json').exists() else {}
    entries = {}
    for name, relative in FILES.items():
        path, url = dest/name, BASE+relative
        previous = old.get('files', {}).get(name, {})
        if not (path.exists() and previous.get('url') == url and
                hashlib.sha256(path.read_bytes()).hexdigest() == previous.get('sha256')):
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
            temporary = path.with_suffix(path.suffix+'.part')
            temporary.write_bytes(content)
            temporary.replace(path)
        content = path.read_bytes()
        entries[name] = dict(url=url, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        print(name, len(content), flush=True)
    (dest/'manifest.json').write_text(json.dumps(dict(
        repository='https://github.com/EBiM-Benchmark/benchmark', revision=REVISION,
        files=entries, status='Pinned upstream originals; local single-arm scene is a separate derivative.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
