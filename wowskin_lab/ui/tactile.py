"""Live magnetic glyphs inside the native MuJoCo window.

Matches WowSkin's meaning: green XY vector, red positive-filled / negative-
outlined Z circle. The drawing scale affects pixels only, never Pollen inputs.
Locations and XY rotations follow this simulation's PCB layout, not an assumed
mapping of the unverified WowSkin firmware channel order.
"""
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .manufacturer import DEFAULT_FULL_SCALE, draw as manufacturer_draw, pygame


class TactileHUD:
    width, height = 400, 558

    def __init__(self, sim, full_scale_ut=DEFAULT_FULL_SCALE, title="LEFT WowSkin | 5 x XYZ"):
        self.sim = sim
        self.panel_title = title
        self.full_scale_ut = float(full_scale_ut)
        self.operator_panel = None
        path = Path("/System/Library/Fonts/SFNSMono.ttf")
        if not path.exists():
            import matplotlib
            path = Path(matplotlib.get_data_path())/"fonts/ttf/DejaVuSansMono.ttf"
        self.font = ImageFont.truetype(str(path), 13)
        self.small = ImageFont.truetype(str(path), 11)
        self.title = ImageFont.truetype(str(path), 17)
        self.reskin_mode = bool(getattr(sim, 'cfg', {}).get('reskin_model'))
        if getattr(sim, 'magnetic_fit', None):
            self.panel_title = 'Fitted dipole | U1-5'
            self.subtitle = 'Pollen/network amplitude match [uT]'
            self.footer = 'SIM FIT | physical parameters uncalibrated'
        if self.reskin_mode:
            geometry = getattr(sim, 'contact_geometry', 'cube')
            self.panel_title = 'Local ReSkin MLP | CH1-5'
            self.subtitle = f'Experimental {geometry} proxy [uT]'
            self.axes_label = 'Source XYZ; WowSkin axes NOT identified'
            self.vector_label = 'Green: source dBx/dBy, not PCB axes'
            self.channel_prefix = 'CH'
            self.footer = f'NOT calibrated | {geometry}/probe mismatch'

    def glyphs(self, magnetic_ut):
        values = np.asarray(magnetic_ut, dtype=float).reshape(5, 3)
        if not np.isfinite(values).all():
            raise ValueError("HUD received non-finite magnetic data")
        if self.reskin_mode:
            # Do not silently apply the dipole model's PCB rotations to source
            # ReSkin chip axes. Only the experimental group permutation is known.
            centres = np.array([[200,183],[123,183],[277,183],[200,106],[200,260]])
            return values, values.copy(), centres
        board = np.einsum("nij,nj->ni", self.sim.rotations, values)
        xy = self.sim.sensor_xyz[:, :2]
        centres = np.c_[200+xy[:, 0]*11000, 183-xy[:, 1]*11000]
        return values, board, centres

    def render(self, row, probability=None):
        blocked = self.reskin_mode and not np.isfinite(row['magnetic_ut']).all()
        values, board, centres = self.glyphs(np.zeros(15) if blocked else row["magnetic_ut"])
        img = Image.new("RGB", (self.width, self.height), (21, 27, 34))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((1, 1, 398, 556), radius=12, outline=(68, 82, 97), width=2)
        draw.text((16, 13), self.panel_title, font=self.title, fill=(242, 244, 247))
        draw.text((16, 40), getattr(self, "subtitle", "Baseline-subtracted magnetic field [uT]"), font=self.small, fill=(171, 187, 204))
        draw.text((16, 61), getattr(self, "axes_label", "PCB +X ->   +Y up"), font=self.small, fill=(171, 187, 204))
        draw.line((75, 183, 325, 183), fill=(47, 57, 68), width=1)
        draw.line((200, 80, 200, 286), fill=(47, 57, 68), width=1)
        # Keep the simulation PCB reference axes, but use the exact upstream
        # circle/line rasterization and scale. No invented arrowheads or caps.
        surface = pygame.image.fromstring(img.tobytes(), img.size, 'RGB')
        display_values = np.c_[board[:, :2]*[1., -1.], values[:, 2]]
        manufacturer_draw(surface, display_values, scaling=self.full_scale_ut/29,
                          locations=centres, rotations=np.zeros(5))
        img = Image.fromarray(np.transpose(pygame.surfarray.array3d(surface),(1,0,2)))
        draw = ImageDraw.Draw(img)
        for i, (x, y) in enumerate(centres):
            draw.text((x-9, y+32), f"{getattr(self, 'channel_prefix', 'U')}{i+1}", font=self.small, fill=(223,230,240))
        draw.text((16, 302), getattr(self, "vector_label", "Green line: dBx/dBy (PCB frame)"), font=self.small, fill=(83, 233, 146))
        draw.text((16, 321), "Red Z: filled >= 0 | hollow < 0", font=self.small, fill=(241, 119, 109))
        draw.text((16, 340), f"{self.full_scale_ut/29:g} uT / px | , / . scale",
                  font=self.small, fill=(171, 187, 204))
        draw.text((16, 362), "Chip       dBx       dBy       dBz", font=self.font, fill=(171, 187, 204))
        for i, value in enumerate(values):
            numbers = '     ---       ---       ---' if blocked else f'{value[0]:+8.2f}  {value[1]:+8.2f}  {value[2]:+8.2f}'
            draw.text((16, 385+i*20), f"{getattr(self, 'channel_prefix', 'U')}{i+1}    {numbers}",
                      font=self.font, fill=(231, 236, 241))
        p = row.get("pollen_probability", float("nan")) if probability is None else probability
        status = f"Pollen {p:.3f} | " if np.isfinite(p) else ""
        diagnostics = (f"{status}physics slip {row['slip']}" if 'slip' in row else
                       f"Sim. contact load {row.get('normal_force_n', 0.):.4f} N")
        if self.reskin_mode:
            query = row.get('reskin_query_mm', [float('nan')]*3)
            diagnostics = f"{'BLOCKED' if blocked else 'PREVIEW'} | depth {query[2]:.4f} mm"
        if row.get("mesh_quality_warning"):
            diagnostics = "LIVE | mesh quality warning"
        draw.text((16, 493), row.get("hud_status", diagnostics), font=self.font, fill=(231, 236, 241))
        draw.text((16, 515), getattr(self, "footer", "uT, not force | uncalibrated simulation"), font=self.small, fill=(242, 196, 113))
        timing = row.get('reskin_status', '') if self.reskin_mode else f"t={row['time_s']:.2f}s | channel mapping is provisional"
        if self.reskin_mode and row.get('reskin_status', '').startswith('UNAVAILABLE:'):
            timing = 'Unavailable: no unloaded/recovery model'
        elif self.reskin_mode:
            timing = 'Outside depth/XY data' if not row.get('reskin_in_training_domain') else 'XY/depth in range; contact shape unvalidated'
        draw.text((16, 534), row.get("hud_timing", timing), font=self.small, fill=(171, 187, 204))
        return np.asarray(img)

    def show(self, handle, row, probability=None, cameras=None, pointcloud=None):
        # A camera capture can outlive the viewer's final frame. The native
        # viewport/set_images accessors do not check exitrequest themselves.
        # Do not hold the viewer lock: set_images waits for its render thread.
        if not handle.is_running():
            return
        viewport = handle.viewport
        if viewport is None:  # Window may close between is_running() and drawing.
            return
        def scaled(pixels, max_width, max_height):
            h, w = pixels.shape[:2]
            scale = min(1., max_width/w, max_height/h)
            width, height = max(1, round(w*scale)), max(1, round(h*scale))
            if (width, height) != (w, h):
                pixels = np.asarray(Image.fromarray(pixels).resize((width, height), Image.Resampling.LANCZOS))
            return pixels, width, height

        pixels = self.render(row, probability)
        pixels, width, height = scaled(pixels, viewport.width*.28, viewport.height*.48)
        # Tactile stays at bottom left; RGB and fused cloud share one right column.
        rectangle = mujoco.MjrRect(viewport.left+10, viewport.bottom+10, width, height)
        images = [(rectangle, pixels)]
        if self.operator_panel is not None:
            panel, panel_w, panel_h = scaled(self.operator_panel, viewport.width*.65, viewport.height*.46)
            images.append((mujoco.MjrRect(viewport.left+10,
                viewport.bottom+viewport.height-panel_h-10, panel_w, panel_h), panel))
        if cameras or pointcloud is not None:
            thumb_height = self.width*3//4
            camera_height = len(cameras or [])*(thumb_height+24)
            cloud_height = (round(pointcloud.shape[0]*self.width/pointcloud.shape[1])
                            if pointcloud is not None else 0)
            panel = Image.new('RGB', (self.width, camera_height+cloud_height), (21,27,34))
            draw = ImageDraw.Draw(panel)
            for index, (label, frame) in enumerate(cameras or []):
                y = index*(thumb_height+24)
                draw.text((10,y+4), label, font=self.small, fill=(223,230,240))
                panel.paste(Image.fromarray(frame).resize((self.width,thumb_height), Image.Resampling.LANCZOS), (0,y+24))
            if pointcloud is not None:
                panel.paste(Image.fromarray(pointcloud).resize((self.width,cloud_height),
                            Image.Resampling.LANCZOS), (0,camera_height))
            camera_pixels, width, height = scaled(np.asarray(panel), viewport.width*.30, viewport.height-20)
            rectangle = mujoco.MjrRect(viewport.left+viewport.width-width-10,
                                       viewport.bottom+viewport.height-height-10, width, height)
            images.append((rectangle, camera_pixels))
        if handle.is_running():
            handle.set_images(images)

    def key(self, code):
        if code in (ord(","), ord(".")):
            self.full_scale_ut = float(np.clip(self.full_scale_ut*(2 if code == ord(",") else .5), .125, 8192.))

    def composite(self, frame, row):
        """Use the same panel in headless evidence images as in the live window."""
        scene = Image.fromarray(frame)
        panel = Image.fromarray(self.render(row))
        image = Image.new("RGB", (scene.width+self.width, max(scene.height, self.height)), (21, 27, 34))
        image.paste(scene, (0, 0))
        image.paste(panel, (scene.width, 0))
        return np.asarray(image)
