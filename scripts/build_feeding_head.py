"""Close the missing posterior skull of the EBiM visual head, on user request.

Original OBJ/texture remain byte-identical. This local geometric completion
is not official EBiM anatomy or a validated tissue/contact model.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'assets/ebim_feeding/assets/head/head_skin.obj'
DEST = ROOT/'assets/generated/feeding'


def build():
    vertices, uv, faces, face_uv = [], [], [], []
    for line in SOURCE.read_text().splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == 'v':
            vertices.append(list(map(float, p[1:4])))
        elif p[0] == 'vt':
            uv.append(list(map(float, p[1:3])))
        elif p[0] == 'f':
            face = [x.split('/') for x in p[1:]]
            faces.append([int(x[0])-1 for x in face])
            face_uv.append([int(x[1])-1 for x in face])
    original = np.asarray(vertices)
    vertices, inverse = np.unique(np.round(original, 9), axis=0, return_inverse=True)
    edges = Counter()
    for f in faces:
        f = inverse[f]
        edges.update(tuple(sorted((int(a), int(b)))) for a, b in zip(f, np.roll(f, -1)))
    graph = defaultdict(list)
    for (a, b), n in edges.items():
        if n == 1:
            graph[a].append(b);graph[b].append(a)
    if any(len(n) != 2 for n in graph.values()):
        raise ValueError('Expected closed boundary loops in pinned source')
    seen, loops = set(), []
    for start in graph:
        if start in seen:
            continue
        loop, current, previous = [], start, None
        while current not in seen:
            loop.append(current);seen.add(current)
            nxt = next(v for v in graph[current] if v != previous)
            previous, current = current, nxt
        loops.append(loop)
    # Posterior skull loop reaches the top of the head. Mouth/eyes and neck
    # base remain untouched, rather than indiscriminately filling every hole.
    loop = max(loops, key=lambda ids: vertices[ids, 2].max())
    rim = vertices[loop]
    assert len(loop) == 72 and rim[:, 2].min() > 0
    centre = (rim.min(0)+rim.max(0))/2
    depth = np.ptp(rim[:, 0])/2  # Rounded occiput proportional to original skull width.
    rings, count = 24, len(rim)
    patch = []
    for k in range(rings):
        angle = k/rings*np.pi/2
        ring = centre+(rim-centre)*np.cos(angle)
        # Fade the non-planar rim's Y offsets toward an elliptical crown: a first-order
        # fade makes an artificial cone/star at the posterior pole.
        ring[:, 1] = centre[1]+(rim[:, 1]-centre[1])*np.cos(angle)**6+depth*np.sin(angle)
        patch.extend(ring)
    patch.append(centre+[0, depth, 0])
    patch = np.asarray(patch)
    triangles = []
    for k in range(rings-1):
        for i in range(count):
            a, b = k*count+i, k*count+(i+1)%count
            triangles.extend([[a, b, a+count], [b, b+count, a+count]])
    for i in range(count):
        triangles.append([(rings-1)*count+i, (rings-1)*count+(i+1)%count, len(patch)-1])
    for f in triangles:
        v = patch[f]
        # Consistent exterior winding for backface-culling renderers.
        if np.dot(np.cross(v[1]-v[0], v[2]-v[0]), v.mean(0)-centre) < 0:
            f[1], f[2] = f[2], f[1]
    normals = np.zeros_like(patch)
    for f in triangles:
        v = patch[f]
        normals[f] += np.cross(v[1]-v[0], v[2]-v[0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    image = np.asarray(Image.open(SOURCE.with_name('bake_albedo2.png')).convert('RGB'))
    texcoords = np.asarray(uv)
    boundary_uv = [texcoords[t] for f, tf in zip(faces, face_uv) for v, t in zip(f, tf)
                   if inverse[v] in loop]
    colors = [image[int(np.clip((1-t[1])*(image.shape[0]-1), 0, image.shape[0]-1)),
                    int(np.clip(t[0]*(image.shape[1]-1), 0, image.shape[1]-1))] for t in boundary_uv]
    color = np.median(colors, axis=0)/255
    DEST.mkdir(parents=True, exist_ok=True)
    path = DEST/'head_back.obj'
    with path.open('w') as stream:
        stream.write('# LOCAL posterior skull completion; original EBiM front unchanged.\n')
        for v in patch:
            stream.write('v '+' '.join(f'{x:.12g}' for x in v)+'\n')
        for n in normals:
            stream.write('vn '+' '.join(f'{x:.12g}' for x in n)+'\n')
        for f in triangles:
            stream.write('f '+' '.join(f'{x+1}//{x+1}' for x in f)+'\n')
    meta = dict(source=str(SOURCE.relative_to(ROOT)), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                description='Local elliptical posterior cap lofted from the 72-vertex source skull rim; not official anatomy',
                boundary_vertices=count, rings=rings, centre_m=centre.tolist(), posterior_depth_m=float(depth),
                skin_rgba=[*color.tolist(), 1.], boundary_match_max_m=float(np.max(np.abs(patch[:count]-rim))))
    (DEST/'head_back.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    build()
