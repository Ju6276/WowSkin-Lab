"""Reproduce per-component STL size comparisons. STL coordinates assumed metres."""
from pathlib import Path
import json
import struct

import numpy as np

from wowskin_lab.paths import ROOT


def read_parts(path):
    content = path.read_bytes()
    count = struct.unpack_from('<I', content, 80)[0]
    if len(content) != 84+50*count:
        raise ValueError(f"Expected binary STL: {path}")
    dtype = np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    triangles = np.frombuffer(content, offset=84, dtype=dtype)['vertices'].astype(float)*1000
    vertices, inverse = np.unique(np.round(triangles.reshape(-1, 3), 5), axis=0, return_inverse=True)
    parents = np.arange(len(vertices))
    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    for triangle in inverse.reshape(-1, 3):
        for other in triangle[1:]:
            parents[root(other)] = root(triangle[0])
    labels = np.array([root(t[0]) for t in inverse.reshape(-1, 3)])
    parts = sorted([triangles[labels == i] for i in np.unique(labels)], key=len, reverse=True)
    return triangles, parts


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    paths = sorted((ROOT/'assets/cad/anyskin').rglob('*.stl'))
    target = np.array([43.6, 11.4, 42.2])  # manufacturer image X,Y,Z
    report = {'units': 'mm, from metre interpretation of unitless STL coordinates',
              'target_xyz_mm': target.tolist(), 'files': []}
    fig, axes = plt.subplots(len(paths), 2, figsize=(12, 3*len(paths)), layout='constrained')
    for row, path in enumerate(paths):
        triangles, parts = read_parts(path)
        component_records = []
        for part in parts:
            low, high = part.min((0, 1)), part.max((0, 1))
            sizes = high-low
            component_records.append(dict(triangles=len(part), extents_xyz_mm=sizes.tolist(),
                                           delta_to_wowskin_xyz_mm=(sizes-target).tolist()))
        item = dict(file=str(path.relative_to(ROOT)), components=component_records,
                    whole_extents_xyz_mm=np.ptp(triangles.reshape(-1, 3), axis=0).tolist())
        report['files'].append(item)
        # Show a single connected component. The other paired finger is mirrored.
        part = parts[0]
        part = part-(part.min((0, 1))+part.max((0, 1)))/2
        color = '#276477' if path.name == 'OnRobot.stl' else '#777e8b'
        for col, (u, v, label) in enumerate([(0, 2, 'X-Z face'), (0, 1, 'X-Y side')]):
            ax = axes[row, col]
            ax.add_collection(PolyCollection(part[:, :, [u, v]], facecolors=color, edgecolors='none', alpha=.17))
            box = target[[u, v]]
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle(-box/2, *box, fill=False, edgecolor='#bd602d', ls='--', lw=1.5))
            ax.set_xlim(-38, 38); ax.set_ylim(-25, 25)
            ax.set_aspect('equal'); ax.grid(alpha=.12)
            ax.set_xlabel('XYZ'[u]+' (mm)'); ax.set_ylabel('XYZ'[v]+' (mm)')
            dims = component_records[0]['extents_xyz_mm']
            ax.set_title(f'{path.name} | {label} | {dims[u]:.2f} x {dims[v]:.2f} mm')
    fig.suptitle('AnySkin CAD: single-component comparison\nDashed orange: WowSkin product bounding box (not its exact outline)', fontsize=15)
    out = ROOT/'assets/cad'
    fig.savefig(out/'comparison.png', dpi=135)
    plt.close(fig)
    (out/'dimensions.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
