import json
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from wowskin_lab.ui.manufacturer import DEFAULT_SCALING, draw, pygame, upstream
from wowskin_lab.data.sensor_compare import SensorComparison


class ManufacturerDisplayChecks(unittest.TestCase):
    def test_exact_default_and_screen_axes(self):
        self.assertEqual(DEFAULT_SCALING,7.)
        np.testing.assert_array_equal(upstream()['chip_locations'],[[201,238],[126,238],[275,238],[201,163],[201,312]])
        surface=pygame.Surface((400,422));surface.fill((0,0,0))
        values=np.zeros((5,3));values[0,0]=70
        draw(surface,values)
        self.assertEqual(tuple(surface.get_at((201,229)))[:3],(0,255,0))
        self.assertEqual(tuple(surface.get_at((201,249)))[:3],(0,0,0))

    def test_circle_sign_and_no_added_cap(self):
        for z,interior in [(70,(255,0,0)),(-70,(0,0,0)),(700,(255,0,0))]:
            surface=pygame.Surface((400,422));surface.fill((0,0,0))
            values=np.zeros((5,3));values[0,2]=z
            draw(surface,values)
            self.assertEqual(tuple(surface.get_at((206,238)))[:3],interior)
            if z==700:self.assertEqual(tuple(surface.get_at((250,248)))[:3],(255,0,0))

    def test_startup_and_reset_use_five_frames_not_long_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=SensorComparison('not-opened',tmp,{},official_baseline=True)
            start=s.started_ns+1_200_000_000
            for i in range(5):s.ingest(np.full((5,4),i,dtype=float),start+i*1_000_000,0)
            np.testing.assert_array_equal(s.baseline,np.full(15,2.))
            s.latest['host_monotonic_ns']=__import__('time').monotonic_ns()
            s.begin_baseline(num_samples=5,duration_s=0.)
            start=s.pending['start']
            for i in range(5):s.ingest(np.full((5,4),10+i,dtype=float),start+i*1_000_000,0)
            np.testing.assert_array_equal(s.baseline,np.full(15,12.))
            s.__exit__(None,None,None)
            events=[json.loads(line) for line in (s.path/'events.jsonl').read_text().splitlines()]
            self.assertEqual([e['count'] for e in events if e['kind']=='baseline'],[5,5])

if __name__=='__main__':unittest.main()

class AlignmentChecks(unittest.TestCase):
    def test_all_channels_preserve_board_direction_through_official_draw(self):
        from wowskin_lab.ui.channel_alignment import official_values, OFFICIAL_TO_REFERENCE
        angles=np.deg2rad([0,180,360,90,270])
        rotations=np.array([[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]] for a in angles])
        sim=SimpleNamespace(rotations=rotations)
        world=np.array([[10+i,20+i,30+i] for i in range(5)])
        local=np.einsum('nji,nj->ni',rotations,world)
        out=official_values(sim,local).reshape(5,3)
        for i,a in enumerate(upstream()['chip_xy_rotations']):
            r=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
            np.testing.assert_allclose(r@out[i,:2],world[OFFICIAL_TO_REFERENCE[i],:2]*[1,-1])
        np.testing.assert_array_equal(out[:,2],world[OFFICIAL_TO_REFERENCE,2])

    def test_sim_reset_is_five_frames_and_does_not_mutate_input(self):
        from wowskin_lab.ui.sensor_compare import SimSensorHUD
        sim=SimpleNamespace(rotations=np.tile(np.eye(3),(5,1,1)))
        hud=SimSensorHUD(sim)
        row=dict(magnetic_ut=np.zeros(15),time_s=0.)
        for i in range(5):
            row['magnetic_ut']=np.full(15,float(i))
            event=hud.observe(row)
            self.assertEqual(event is not None,i==4)
        self.assertEqual(hud.baseline_id,1)
        hud.begin_reset()
        for _ in range(5):event=hud.observe(row)
        np.testing.assert_array_equal(row['magnetic_ut'],np.full(15,4.))
        self.assertEqual(event['count'],5)
        self.assertEqual(hud.render(row).shape,(650,400,3))

    def test_reset_button_hit_regions_and_no_native_click_outside(self):
        from wowskin_lab.ui.keyboard import ViewerKeyboard
        from wowskin_lab.ui.sensor_compare import show_comparison
        k=ViewerKeyboard({ord('B'),ord('Z')},'test')
        handle=SimpleNamespace(is_running=lambda:True, viewport=SimpleNamespace(left=0,bottom=0,width=1800,height=1200),set_images=lambda *_:None)
        show_comparison(handle,np.zeros((650,800,3),dtype=np.uint8),k)
        for box,key in zip(k.click_regions,'ZB'):
            x0,y0,x1,y1,_=box
            self.assertTrue(k.click((x0+x1)/2,(y0+y1)/2))
            self.assertEqual(k.sample().commands,(ord(key),))
        self.assertFalse(k.click(0,0))

class OverlaySourceChecks(unittest.TestCase):
    def test_real_overlay_uses_real_delta_and_switch_preserves_simulation(self):
        from wowskin_lab.ui.sensor_compare import SimSensorHUD
        sim=SimpleNamespace(rotations=np.tile(np.eye(3),(5,1,1)))
        hud=SimSensorHUD(sim)
        seen=[]
        hud.background=lambda values: (seen.append(None if values is None else np.array(values).copy()) or np.zeros((422,400,3),dtype=np.uint8),upstream()['chip_locations'])
        row=dict(magnetic_ut=np.zeros(15),time_s=1.)
        for _ in range(5):hud.observe(row)
        hud.live_row=dict(magnetic_ut=np.arange(15.),hud_status='LIVE',hud_timing='')
        hud.live_overlay=True;hud.render(row)
        np.testing.assert_array_equal(seen[-1],np.arange(15.))
        self.assertEqual(hud.reset_key,'B')
        hud.live_overlay=False;hud.render(row)
        np.testing.assert_allclose(seen[-1],0)
        self.assertEqual(hud.reset_key,'Z')
        np.testing.assert_array_equal(row['magnetic_ut'],np.zeros(15))
