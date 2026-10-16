"""Native FOTS MuJoCo source parity and moving-sensor regressions."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import mujoco
import numpy as np
import torch
from wowskin_lab.core.fots_mujoco import SOURCE, NativeFOTSRenderer, NativeFOTSOptical
from wowskin_lab.envs.optical import OpticalBench


class FOTSChecks(unittest.TestCase):
    def test_original_rgb_shadow_markers_and_repeatability(self):
        seed_function = torch.seed
        renderer = NativeFOTSRenderer()
        self.assertIs(torch.seed, seed_function)
        rows, cols = np.mgrid[:320,:240]
        raw = renderer.baseline-np.maximum(0, .002-((rows-160)**2+(cols-120)**2)*.0000005)
        trajectory = [[0.,0.,0.],[.002,-.001,.2]]
        rgb = renderer.render(raw)
        flow, displacement = renderer.markers(rgb, renderer.baseline-raw, trajectory)
        # Independent source entrypoint, source cwd and its original imports.
        script = '''import os, functools, numpy as np, torch
np.mat = np.asmatrix
torch.load = functools.partial(torch.load, map_location='cpu', weights_only=True)
from tactile_render import get_simapproach
from marker_motion import MarkerMotion
# Compare the adapter against the author's algorithm on the same CPU device.
# Otherwise CUDA availability silently changes the reference arithmetic.
import tactile_render, utils.prepost_mlp
tactile_render.device = torch.device('cpu')
utils.prepost_mlp.device = torch.device('cpu')
p=os.environ['FOTS_PARITY_DIR']
d=np.load(p+'/raw.npy')
with torch.inference_mode():
    image=get_simapproach().generate(d.copy(), shadow=True)
    no_shadow=get_simapproach().generate(d.copy(), shadow=False)
difference=np.load('utils/utils_data/ini_depth_extent.npy')-d
m=MarkerMotion(image,difference,difference>0,[[0.,0.,0.],[.002,-.001,.2]],[.00125,.00021,.00038])
flow=m._marker_motion()
np.savez(p+'/expected.npz',rgb=image[...,::-1],no_shadow=no_shadow[...,::-1],flow=flow[...,::-1])
'''
        with tempfile.TemporaryDirectory() as directory:
            np.save(Path(directory)/'raw.npy', raw)
            subprocess.run([sys.executable,'-B','-c',script], cwd=SOURCE,
                           env=dict(os.environ,FOTS_PARITY_DIR=directory),check=True)
            with np.load(Path(directory)/'expected.npz') as expected:
                np.testing.assert_array_equal(rgb, expected['rgb'])
                np.testing.assert_array_equal(renderer.render(raw,shadow=False), expected['no_shadow'])
                np.testing.assert_array_equal(flow,expected['flow'])
        self.assertEqual(displacement.shape,(9,11,2))
        self.assertGreater(abs(displacement).max(),1)
        for _ in range(3):
            np.testing.assert_array_equal(renderer.render(raw),rgb)
        np.testing.assert_array_equal(renderer.baseline,np.load(SOURCE/'utils/utils_data/ini_depth_extent.npy'))

    def test_native_depth_press_slide_release_and_scene_invariance(self):
        env=OpticalBench(sensor='digit_fots',probe_shape='box')
        optics=NativeFOTSOptical(env.model,env.sensors)
        def capture(target):
            env.data.ctrl[:]=target
            for _ in range(100):
                env.step(np.zeros(6))
            return optics.render(env.data)['DIGIT']
        try:
            unloaded=optics.render(env.data)['DIGIT']
            pressed=capture([0,0,-.012,0,0,0])
            moved=capture([.002,0,-.012,0,0,.35])
            self.assertGreater(pressed.normal_force_n,.1)
            self.assertGreater(pressed.indentation_m.max(),.0001)
            self.assertGreater(abs(pressed.marker_displacement_px-moved.marker_displacement_px).max(),1)
            stationary=optics.render(env.data)['DIGIT']
            np.testing.assert_array_equal(stationary.marker_displacement_px,moved.marker_displacement_px)
            # Room size must not change calibrated depth-buffer input.
            env.model.stat.extent*=10
            env.model.vis.map.znear=.02
            after=optics.render(env.data)['DIGIT']
            np.testing.assert_allclose(after.render_input_depth,moved.render_input_depth,atol=2e-7)
            released=capture([0,0,0,0,0,0])
            self.assertEqual(released.normal_force_n,0)
            self.assertEqual(optics.history,{})
            np.testing.assert_allclose(released.indentation_m,unloaded.indentation_m,atol=2e-7)
            # Do not invent unloaded zeros: retain the author's saved baseline.
            np.testing.assert_array_equal(released.baseline_depth_m,unloaded.baseline_depth_m)
        finally:
            optics.close()

    def test_raw_camera_matches_author_depth_readback(self):
        # Compare against the original MuJoCo scene, not a local proxy scene.
        model=mujoco.MjModel.from_xml_path(str(SOURCE/'assets/gel/fots_mujoco.xml'))
        data=mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        optics=NativeFOTSOptical(model,{'DIGIT':('digit:camera','digit:friction')})
        reference=mujoco.Renderer(model,height=320,width=240)
        try:
            reference._mjr_context.readDepthMap=mujoco.mjtDepthMap.mjDEPTH_ZERONEAR
            reference.update_scene(data,camera='digit:camera')
            reference.render()
            raw=np.empty((320,240),np.float32)
            mujoco.mjr_readPixels(None,raw,reference._rect,reference._mjr_context)
            np.testing.assert_allclose(optics.depth(data,'digit:camera'),np.fliplr(raw),atol=1e-7)
        finally:
            optics.close()
            reference.close()
