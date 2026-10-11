"""Bounded asynchronous point-cloud preview, rendered as an operator-only panel."""
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wowskin_lab.data.pointcloud import fuse_rgbd

SOURCE_COLORS = {1: (66, 195, 255), 2: (255, 177, 68), 52: (213, 110, 255)}
VIEWS = ((135, 32, 'OBLIQUE'), (45, 32, 'REVERSE'), (135, 89.9, 'TOP'))


def render_cloud(cloud, center, *, view=0, span=.85, source_colors=False,
                 stride=4, voxel_m=.002, fusion_ms=0., capture_ms=0.):
    """Orthographic point rasterization with a nearest-point z buffer."""
    width, height = 720, 460
    azimuth, elevation, view_name = VIEWS[view]
    a, e = np.deg2rad([azimuth, elevation])
    toward_eye = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    right = np.array([-np.sin(a), np.cos(a), 0.])
    up = np.cross(toward_eye, right)
    rotation = np.stack((right, up, toward_eye), axis=1)
    projected = np.einsum('ni,ij->nj', cloud.xyz-center, rotation)
    scale = width/span
    x = np.rint(width/2+projected[:, 0]*scale).astype(int)
    y = np.rint(245-projected[:, 1]*scale).astype(int)
    good = (x >= 0) & (x < width-1) & (y >= 72) & (y < height-48)
    xyz, x, y = projected[good], x[good], y[good]
    colors = (np.array([SOURCE_COLORS[int(i)] for i in cloud.camera_ids[good]], dtype=np.uint8)
              if source_colors else cloud.rgb[good])
    canvas = np.full((height, width, 3), (17, 25, 34), dtype=np.uint8)
    # Small splats aid visibility; select the nearest point for each covered pixel.
    if len(x):
        addresses = np.concatenate([(y+dy)*width+x+dx for dx, dy in ((0,0), (1,0), (0,1), (1,1))])
        z = np.tile(xyz[:, 2], 4)
        order = np.lexsort((-z, addresses))
        addresses_sorted = addresses[order]
        first = np.r_[True, np.diff(addresses_sorted) != 0]
        chosen = order[first]
        canvas.reshape(-1, 3)[addresses[chosen]] = np.tile(colors, (4,1))[chosen]
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=14)
    ids = sorted(set(cloud.camera_ids.tolist()))
    draw.text((12, 10), f'RGB-D POINT CLOUD | BASE [m] | {view_name}', font=font, fill=(235,243,249))
    draw.text((12, 34), f'cams {"+".join(map(str, ids))} | t={cloud.timestamp:.2f}s | {len(cloud.xyz):,} points', font=small, fill=(167,194,211))
    if source_colors:
        for index, (key, color) in enumerate(SOURCE_COLORS.items()):
            draw.text((12+index*145, 54), f'{key}: {"wrist" if key == 52 else "fixed"}', font=small, fill=color)
    else:
        draw.text((12, 54), 'RGB colors | current instant only', font=small, fill=(167,194,211))
    origin = np.array([width-64, height-88])
    for index, (label, color) in enumerate(zip('XYZ', ((255,101,101), (97,238,155), (96,171,255)))):
        axis = rotation[index]
        end = origin+np.array([axis[0], -axis[1]])*32
        draw.line((tuple(origin), tuple(end)), fill=color, width=2)
        draw.text(tuple(end+2), label, font=small, fill=color)
    draw.text((12,height-44), f'M: 2/3/off | Y: view | 0: color | 9: zoom | span {span:.2f} m', font=small, fill=(237,225,183))
    draw.text((12,height-23), f'PREVIEW stride {stride}, voxel {voxel_m*1000:g} mm | fuse {fusion_ms:.1f} ms | RGB-D {capture_ms:.1f} ms', font=small, fill=(167,194,211))
    return np.asarray(img)


class OnlinePointCloud:
    def __init__(self, *, enabled=False, hz=5., stride=4, voxel_m=.002, center=(.4, 0, .17)):
        if not np.isfinite([hz, voxel_m]).all() or hz <= 0 or stride < 1 or voxel_m < 0:
            raise ValueError('Point-cloud preview needs positive Hz/stride and nonnegative voxel size')
        self.mode = 3 if enabled else 0
        self.hz, self.stride, self.voxel_m = hz, stride, voxel_m
        self.center = np.asarray(center, dtype=float)
        self.view, self.zoom, self.source_colors = 0, 0, False
        self.next_capture = 0.
        self._generation = 0
        self._future = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rgbd-fusion')
        self.panel = None
        self.error = ''
        self.last_cloud = None

    def close(self):
        self._pool.shutdown(wait=True, cancel_futures=True)

    def clear(self):
        self._generation += 1
        self.panel = self.last_cloud = None
        self.error = ''
        self.next_capture = 0.

    def key(self, code):
        if code == ord('M'):
            self.mode = {0: 2, 2: 3, 3: 0}[self.mode]
        elif code == ord('Y'):
            self.view = (self.view+1) % len(VIEWS)
        elif code == ord('0'):
            self.source_colors = not self.source_colors
        elif code == ord('9'):
            self.zoom = (self.zoom+1) % 3
        else:
            return False
        self.clear()
        return True

    def poll(self):
        if self._future is not None and self._future.done():
            generation, future = self._future_generation, self._future
            self._future = None
            try:
                cloud, panel = future.result()
                if generation == self._generation:
                    self.last_cloud, self.panel, self.error = cloud, panel, ''
            except Exception as exc:
                if generation == self._generation:
                    self.error = f'Point cloud unavailable: {exc}'
                    img = Image.new('RGB', (720, 140), (35, 22, 26))
                    ImageDraw.Draw(img).text((12, 15), self.error, font=ImageFont.load_default(size=16), fill=(255, 125, 116))
                    self.panel, self.last_cloud = np.asarray(img), None

    def ready(self):
        self.poll()
        return bool(self.mode and self._future is None and time.monotonic() >= self.next_capture)

    def submit(self, views, capture_ms=0.):
        if self._future is not None:
            raise RuntimeError('Point-cloud queue is bounded to one snapshot')
        selected = tuple(v for v in views if self.mode == 3 or v.camera_id != 52)
        settings = dict(view=self.view, span=(.85, .55, 1.3)[self.zoom], source_colors=self.source_colors,
                        stride=self.stride, voxel_m=self.voxel_m, capture_ms=capture_ms)
        def work():
            start = time.perf_counter()
            cloud = fuse_rgbd(selected, stride=settings['stride'], voxel_m=settings['voxel_m'])
            panel = render_cloud(cloud, self.center, fusion_ms=(time.perf_counter()-start)*1000, **settings)
            return cloud, panel
        self._future_generation = self._generation
        self._future = self._pool.submit(work)
        self.next_capture = time.monotonic()+1/self.hz
