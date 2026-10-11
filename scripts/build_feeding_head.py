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


def build_seated_face():
    """Local visual-only neck fit; leave the pinned OBJ unchanged.

    Preserve source UVs; normals are recomputed by MuJoCo after deformation.
    Keep the original mouth; tuck the source shoulder flaps into the shirt collar.
    """
    lines, vertices = [], []
    for line in SOURCE.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == 'v':
            x, y, z = map(float, fields[1:4])
            neck = np.clip((-z-.033)/.023, 0, 1)
            x = (1-neck)*x+neck*np.clip(x, -.027, .027)
            y = (1-neck)*y+neck*np.clip(y, -.010, .043)
            vertices.append([x, y, z])
            lines.append('v '+' '.join(f'{v:.10g}' for v in (x, y, z)))
        elif fields[0] == 'vn':
            continue
        elif fields[0] == 'f':
            lines.append('f '+' '.join('/'.join(f.split('/')[:2]) for f in fields[1:]))
        else:
            lines.append(line)
    path = DEST/'head_seated_visual.obj'
    path.write_text('# Local neck fit; original face expression intact; source intact.\n'+'\n'.join(lines)+'\n')
    (DEST/'head_seated_visual.json').write_text(json.dumps(dict(
        source=str(SOURCE.relative_to(ROOT)), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        purpose='Visual tapered neck; original face expression unchanged; collision remains the original rigid hull',
        swallowing_or_oral_cavity_physics=False), indent=2)+'\n')
    print(path)

def build_hair():
    """Thin short-hair surface fitted to the source scalp and completed occiput.

    Local appearance asset only. No hat primitive, collision or anatomy change.
    """
    arrays, triangles = [], []
    for path in (SOURCE, DEST/'head_back.obj'):
        points, faces = [], []
        for line in path.read_text().splitlines():
            p = line.split()
            if p and p[0] == 'v':
                points.append(list(map(float, p[1:4])))
            elif p and p[0] == 'f':
                face = [int(v.split('/')[0])-1 for v in p[1:]]
                faces.extend([[face[0], face[i], face[i+1]] for i in range(1,len(face)-1)])
        triangles.extend(np.asarray(faces)+len(arrays))
        arrays.extend(points)
    original = np.asarray(arrays)
    vertices, inverse = np.unique(np.round(original, 9), axis=0, return_inverse=True)
    faces = inverse[np.asarray(triangles)]
    center = np.array([0., .014, .052])
    normals = np.zeros_like(vertices)
    for f in faces:
        a,b,c = vertices[f]
        n = np.cross(b-a,c-a)
        if np.dot(n,(a+b+c)/3-center)<0:
            n *= -1
        normals[f] += n
    normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    # Higher forehead, tapered temples and a lower nape. This follows the
    # actual head instead of placing a thick ellipsoid cap over it.
    theta = np.arctan2(vertices[:,1]-.014, vertices[:,0])
    front = (1-np.sin(theta))/2
    ears = np.abs(np.cos(theta))**8
    hairline = .009+.055*front**1.7+.018*ears
    hairline += .0015*np.sin(3*theta)+.0006*np.sin(11*theta)
    distance = vertices[:,2]-hairline
    out, out_normals, out_faces = [], [], []
    for face in faces:
        polygon = [(vertices[i],normals[i],distance[i]) for i in face]
        clipped=[]
        for a,b in zip(polygon,polygon[1:]+polygon[:1]):
            if a[2]>=0: clipped.append(a)
            if (a[2]>=0)!=(b[2]>=0):
                t=a[2]/(a[2]-b[2])
                clipped.append((a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1]),0.))
        if len(clipped)<3: continue
        start=len(out)
        for p,n,d in clipped:
            n=n/max(np.linalg.norm(n),1e-12)
            thickness=.00035+.00035*min(1.,max(0.,d)/.005)
            out.append(p+n*thickness);out_normals.append(n)
        for i in range(1,len(clipped)-1):
            tri=[start,start+i,start+i+1]
            a,b,c=[out[j] for j in tri]
            if np.dot(np.cross(b-a,c-a),sum(out_normals[j] for j in tri))<0:
                tri[1],tri[2]=tri[2],tri[1]
            out_faces.append(tri)
    out, ns = np.asarray(out),np.asarray(out_normals)
    uv=np.c_[(np.arctan2(out[:,1]-.014,out[:,0])/(2*np.pi)+.5),out[:,2]/.12]
    path=DEST/'head_short_hair.obj'
    with path.open('w') as f:
        f.write('# Local fitted short hair; visual only, source head unchanged.\n')
        for v in out:f.write('v '+' '.join(f'{x:.10g}' for x in v)+'\n')
        for v in uv:f.write('vt '+' '.join(f'{x:.10g}' for x in v)+'\n')
        for v in ns:f.write('vn '+' '.join(f'{x:.10g}' for x in v)+'\n')
        for tri in out_faces:f.write('f '+' '.join(f'{i+1}/{i+1}/{i+1}' for i in tri)+'\n')
    rng=np.random.default_rng(17)
    y,x=np.mgrid[0:768,0:768]
    grain=rng.normal(0,4.5,x.shape)
    strands=3*np.sin(x*1.9+8*np.sin(y/130))+2*np.sin(x*.56+y*.4)
    texture=np.clip(np.array([43.,37.,32.])+grain[...,None]+strands[...,None],0,255).astype('uint8')
    Image.fromarray(texture).save(DEST/'head_short_hair.png')
    (DEST/'head_short_hair.json').write_text(json.dumps(dict(
        source=str(SOURCE.relative_to(ROOT)), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        purpose='Local scalp-fitted short hair covering source/completion color seam; visual only',
        thickness_m=[.00035,.0007],collision=False,seed=17,
        mesh_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+'\n')
    print(path)


if __name__ == '__main__':
    build()
    build_seated_face()
    build_hair()
