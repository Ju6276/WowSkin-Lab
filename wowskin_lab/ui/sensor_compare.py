"""Side-by-side displays; firmware channel identities are not silently remapped."""
import numpy as np
from PIL import Image, ImageDraw
import mujoco
from .tactile import TactileHUD
from .channel_alignment import official_values
from .manufacturer import DEFAULT_FULL_SCALE, render as manufacturer_render, upstream


class RealSensorHUD(TactileHUD):
    height = 650
    heading = 'REAL WowSkin'
    reset_key = 'B'

    def background(self, values):
        return manufacturer_render(values, self.full_scale_ut/29), upstream()['chip_locations']

    def __init__(self, sim, full_scale_ut=DEFAULT_FULL_SCALE):
        super().__init__(sim, full_scale_ut, title='REAL WowSkin | R1-R5')
        self.channel_prefix = 'R'
        self.axes_label = 'Manufacturer display axes; mapping unverified'
        self.vector_label = 'Green arrow: XY (manufacturer display)'
        self.footer = 'Real magnetic readings, not force'

    def render(self, row, probability=None):
        values = row.get('magnetic_ut')
        if values is not None and not np.isfinite(values).all():
            raise ValueError('Non-finite real readings')
        background, locations = self.background(values)
        source = Image.fromarray(background)
        image = Image.new('RGB',(400,self.height),(21,27,34))
        image.paste(source,(0,0))
        d = ImageDraw.Draw(image)
        d.rectangle((0,0,399,59),fill=(21,27,34))
        d.rectangle((10,10,100,50),fill=(70,130,180))
        d.text((26,20),'Reset',font=self.title,fill='white')
        d.text((112,8),self.heading,font=self.title,fill='white')
        d.text((112,33),f'{self.full_scale_ut/29:g} uT/px | {self.reset_key}: 5-frame zero',font=self.small,fill='white')
        for i,(x,y) in enumerate(locations):
            d.rectangle((x-15,y+18,x+19,y+33),fill=(21,27,34))
            d.text((x-13,y+19),f'CH{i+1}',font=self.small,fill='white')
        d.text((12,430),row.get('hud_status','LIVE'),font=self.small,fill=(242,196,113))
        d.text((12,450),'Channel     dBx       dBy       dBz',font=self.font,fill='white')
        if values is not None:
            for i,v in enumerate(np.asarray(values).reshape(5,3)):
                d.text((12,473+20*i),f'CH{i+1}   {v[0]:+8.2f}  {v[1]:+8.2f}  {v[2]:+8.2f}',font=self.font,fill='white')
        else:
            d.text((12,478),'Waiting for valid baseline; no raw-field glyphs',font=self.small,fill='white')
        d.text((12,582),'Firmware order: center / left / right / up / down',font=self.small,fill='white')
        d.text((12,604),getattr(self,'signal_note','Real B - baseline | not force'),font=self.small,fill='white')
        d.text((12,628),row.get('hud_timing',''),font=self.small,fill='white')
        return np.asarray(image)


class SimSensorHUD(RealSensorHUD):
    heading = 'SIM prediction | T: source'
    reset_key = 'Z'
    signal_note = 'Reference axes -> display axes; NOT calibrated'

    def __init__(self, sim):
        super().__init__(sim)
        self.offset = np.zeros(15)
        self.pending_samples = []
        self.baseline_id = 0
        self.model_view = None
        self.live_overlay = False
        self.live_row = None

    def __enter__(self):
        from .sensor_model_view import SensorModelView
        self.model_view = SensorModelView(self.sim)
        return self

    def __exit__(self, *exc):
        if self.model_view is not None:
            self.model_view.close()
            self.model_view = None

    def background(self, values):
        if self.model_view is not None:
            return self.model_view.render(values, self.full_scale_ut/29), self.model_view.locations
        # Headless unit tests can render labels without a GL context. Never
        # substitute the real PCB photograph for a simulation background.
        return np.full((422,400,3),27,dtype=np.uint8), upstream()['chip_locations']

    def begin_reset(self):
        self.pending_samples = []

    def observe(self, row):
        values = official_values(self.sim, row['magnetic_ut'])
        event = None
        if self.pending_samples is not None:
            self.pending_samples.append(values.copy())
            if len(self.pending_samples) == 5:
                self.offset = np.mean(self.pending_samples, axis=0)
                self.pending_samples = None
                self.baseline_id += 1
                event = dict(count=5, baseline_id=self.baseline_id,
                             display_offset_ut=self.offset.tolist())
        return event

    def render(self, row, probability=None):
        if self.live_overlay:
            self.heading = 'REAL on model | T: source'
            self.reset_key = 'B'
            self.signal_note = 'Measured data overlay; NOT simulation prediction'
            adapted = dict(self.live_row or dict(magnetic_ut=None,hud_status='Waiting for real stream',hud_timing=''))
            return super().render(adapted)
        self.heading = 'SIM prediction | T: source'
        self.reset_key = 'Z'
        self.signal_note = 'Reference axes -> display axes; NOT calibrated'
        values = official_values(self.sim, row['magnetic_ut']) - self.offset
        ready = self.pending_samples is None
        adapted = dict(row, magnetic_ut=values if ready else None,
            hud_status=f'SIM baseline {self.baseline_id}' if ready else 'BASELINE: keep unloaded',
            hud_timing=f"t={row['time_s']:.2f}s | physical correspondence unverified")
        return super().render(adapted)


def comparison_panel(sim_hud, real_hud, sim_row, session):
    real_hud.full_scale_ut = sim_hud.full_scale_ut
    frame, pending, error, age = session.snapshot()
    if frame is None:
        sim_hud.live_row = None
        pixels = Image.new('RGB',(400,real_hud.height),(21,27,34))
        draw = ImageDraw.Draw(pixels)
        draw.text((16,16),'REAL: waiting for serial data',font=real_hud.title,fill='white')
        if error: draw.text((16,65),error[:48],font=real_hud.small,fill='orange')
    else:
        delta = frame['delta_ut']
        real_hud.subtitle = 'Delta B [uT] | baseline '+str(frame['baseline_id']) if delta is not None else 'RAW B [uT] | B: zero when unloaded'
        label = ('ERROR: '+error if error else 'STALE' if age>.5 else 'BASELINE: keep unloaded' if pending or delta is None else 'LIVE | baseline '+str(frame['baseline_id']))
        row = dict(magnetic_ut=delta,
                   time_s=sim_row['time_s'], hud_status=label[:43],
                   hud_timing=f"seq={frame['seq']} | age={age:.3f}s | host time")
        sim_hud.live_row = row
        pixels = Image.fromarray(real_hud.render(row))
    image = Image.new('RGB',(800,max(sim_hud.height,real_hud.height)),(21,27,34))
    image.paste(Image.fromarray(sim_hud.render(sim_row)),(0,0));image.paste(pixels,(400,0))
    return np.asarray(image)


def show_comparison(handle, pixels, keyboard=None, left_reset='Z'):
    if not handle.is_running(): return
    viewport = handle.viewport
    if viewport is None: return
    width = max(1, int(min(800,viewport.width*.60,viewport.height*.82*pixels.shape[1]/pixels.shape[0])))
    height = round(width*pixels.shape[0]/pixels.shape[1])
    resized = np.asarray(Image.fromarray(pixels).resize((width,height),Image.Resampling.LANCZOS))
    rectangle = mujoco.MjrRect(viewport.left+viewport.width-width-10,
                               viewport.bottom+viewport.height-height-10,width,height)
    if keyboard is not None:
        # MjrRect uses framebuffer pixels; AppKit mouse coordinates use points.
        # Normalize both to the content area to support Retina and resizing.
        sx, sy = width/pixels.shape[1], height/pixels.shape[0]
        keyboard.click_regions = tuple(
            ((rectangle.left+(offset+10)*sx-viewport.left)/viewport.width,
             (rectangle.bottom+height-50*sy-viewport.bottom)/viewport.height,
             (rectangle.left+(offset+100)*sx-viewport.left)/viewport.width,
             (rectangle.bottom+height-10*sy-viewport.bottom)/viewport.height, ord(key))
            for offset,key in ([(0,left_reset),(400,'B')] if pixels.shape[1]==800 else [(0,left_reset)]))
    if handle.is_running(): handle.set_images((rectangle,resized))
