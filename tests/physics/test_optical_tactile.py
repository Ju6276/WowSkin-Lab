"""Optional native OpenGL checks: requires requirements/tacto.txt and pinned assets."""
import tempfile
import os
from pathlib import Path
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from wowskin_lab.core.optical_tactile import OpticalFrame, OpticalTactile
from wowskin_lab.envs.optical import OpticalBench, OpticalPandaEnv, digit_geometry
from wowskin_lab.envs.franka import FRANKA_REPO, build_scene
from wowskin_lab.tools.optical_sensor import demo, panel
from wowskin_lab.paths import ROOT


def contact_scene(euler='0 0 0'):
    # Two contacts on left gel; right gel unloaded. Additional sphere/sphere
    # contact elsewhere must not count as a sensor contact. Gravity disabled
    # so a global rigid rotation must preserve mechanics and optical images.
    xml = f'''<mujoco><compiler angle="radian"/><option gravity="0 0 0"/>
      <worldbody><body euler="{euler}">
        <geom name="left" type="box" pos="-.03 0 -.002" size=".01 .015 .002"/>
        <site name="left_cam" pos="-.03 0 .022"/>
        <geom name="right" type="box" pos=".03 0 -.002" size=".01 .015 .002"/>
        <site name="right_cam" pos=".03 0 .022"/>
        <body pos="-.035 -.005 .0035"><joint type="slide" axis="0 0 1"/>
          <geom name="ball1" type="sphere" size=".004" mass=".02"/></body>
        <body pos="-.025 .005 .0035"><joint type="slide" axis="0 0 1"/>
          <geom name="ball2" type="sphere" size=".004" mass=".02"/></body>
        <body pos=".2 0 0"><joint type="slide" axis="0 0 1"/>
          <geom type="sphere" size=".005" mass=".02"/></body>
        <body pos=".2 0 .008"><joint type="slide" axis="0 0 1"/>
          <geom type="sphere" size=".005" mass=".02"/></body>
      </body></worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class OpticalChecks(unittest.TestCase):
    def test_original_renderer_background_noise_and_resolution(self):
        # The original Renderer initialization executes in another process.
        # Only its offscreen GL context is substituted on Mac. This checks
        # camera/lights/force/background/noise, not just that pixels exist.
        import cv2
        model, data = contact_scene()
        optics = OpticalTactile(model, {'left': ('left_cam', 'left')}, ideal=True)
        try:
            actual = optics.render(data)['left']
            self.assertEqual(actual.rgb.shape, (320, 240, 3))
            source = '''import os, sys, numpy as np, cv2, trimesh, pyrender
if sys.platform == 'darwin':
    from wowskin_lab.core.optical_cgl import CGLOffscreenRenderer
    pyrender.OffscreenRenderer = CGLOffscreenRenderer
from tacto import Renderer, get_digit_config_path
path = os.environ['OPTICAL_TEST_DIR']
bg = cv2.imread(os.environ['TACTO_BG'])
r = Renderer(240, 320, bg, get_digit_config_path())
poses = {'a': ([-.005,-.005,-.0185],[0.,0.,0.]),
         'b': ([ .005, .005,-.0185],[0.,0.,0.])}
forces = dict(zip(poses, np.load(path+'/forces.npy')))
for name in poses:
    r.add_object(trimesh.creation.icosphere(subdivisions=3, radius=.004), name)
c,d = r.render(poses, forces, noise=False)
np.savez(path+'/expected.npz', rgb=c[0], depth=r.depth0[0]-d[0])
r.r.delete()
'''
            with tempfile.TemporaryDirectory() as directory:
                np.save(Path(directory)/'forces.npy', [optics.contacts(data, model.geom('left').id)[model.geom(name).id]
                                                      for name in ('ball1', 'ball2')])
                subprocess.run([sys.executable, '-c', source], cwd=ROOT,
                               env=dict(os.environ, OPTICAL_TEST_DIR=directory,
                                        TACTO_BG=optics.background_path), check=True)
                with np.load(Path(directory)/'expected.npz') as expected:
                    np.testing.assert_allclose(actual.rgb, expected['rgb'], atol=1)
                    np.testing.assert_allclose(actual.indentation_m, expected['depth'], atol=1e-7)
            # Unload: the ideal image must equal the source reference photo.
            data.qpos[:2] = .02
            mujoco.mj_forward(model, data)
            clean = optics.render(data)['left']
            background = cv2.imread(optics.background_path)[..., ::-1]
            np.testing.assert_array_equal(clean.rgb, background)
            np.testing.assert_array_equal(clean.contact_mask, False)
            optics.ideal = False
            noise_a, noise_b = optics.render(data)['left'], optics.render(data)['left']
            self.assertFalse(np.array_equal(noise_a.rgb, noise_b.rgb))
            std = (noise_a.rgb.astype(float)-clean.rgb).std()
            self.assertGreater(std, 6.)
            self.assertLess(std, 8.)
            np.testing.assert_array_equal(noise_a.indentation_m, clean.indentation_m)
        finally:
            optics.close()

    def test_depth_preview_contrast_and_mask_preserve_raw_data(self):
        depth = np.full((320, 240), 7.14e-6)
        rgb = np.zeros((320, 240, 3), dtype=np.uint8)
        first = OpticalFrame(rgb.copy(), depth.copy(), .5, (), 0., contact_mask=depth > 0)
        second = OpticalFrame(rgb.copy(), depth.copy()/2, .25, (), 0., contact_mask=depth > 0)
        display = panel({'left': first, 'right': second})
        self.assertEqual(display[100, 300, 0], 255)
        self.assertEqual(display[display.shape[0]//2+100, 300, 0], 127)
        fixed = panel({'left': first}, depth_max_mm=5.)
        self.assertEqual(fixed[100, 300, 0], 0)
        mask = panel({'left': first}, show_mask=True)
        self.assertEqual(mask[100, 300, 0], 255)
        np.testing.assert_array_equal(first.indentation_m, depth)
        np.testing.assert_array_equal(second.indentation_m, depth/2)
        np.testing.assert_array_equal(first.rgb, rgb)

    def test_modules_preserve_stock_fingers_and_opening(self):
        original = ET.parse(FRANKA_REPO/'franka_sim/envs/xmls/panda.xml').getroot()
        self.assertLess(np.ptp(digit_geometry()['meshes'][0].vertices[:, 2]), .035)
        for backend, prefix in (('tacto', 'digit'), ('fots', 'gelsight')):
            env = OpticalPandaEnv(sensor=backend)
            root = ET.fromstring(build_scene(env.cfg, []))
            for side in ('left', 'right'):
                old = original.find(f".//body[@name='{side}_finger']")
                new = root.find(f".//body[@name='{side}_finger']")
                self.assertEqual([g.attrib for g in old.findall('geom')],
                                 [g.attrib for g in new.findall('geom')])
                self.assertEqual(old.find('inertial').attrib, new.find('inertial').attrib)
                self.assertEqual(old.find('joint').attrib, new.find('joint').attrib)
                self.assertIsNotNone(new.find(f"body[@name='{prefix}_{side}_mount']"))
            self.assertEqual(root.find(".//default[@class='finger']/joint").get('range'), '-0.04 0')
            # At zero slide displacement, the original 80 mm opening is retained.
            env.data.qpos[env.finger_q] = 0
            mujoco.mj_forward(env.model, env.data)
            def front(side):
                site = env.model.site(f'{prefix}_{side}_camera').id
                position = env.data.site_xpos[site].copy()
                if backend == 'tacto':
                    position -= .022*env.data.site_xmat[site].reshape(3, 3)[:, 2]
                return position
            self.assertAlmostEqual(np.linalg.norm(front('left')-front('right')), .08)

    def test_multi_contact_and_global_rotation(self):
        frames = []
        for euler in ('0 0 0', '.4 -.7 1.1'):
            model, data = contact_scene(euler)
            optics = OpticalTactile(model, {'left': ('left_cam', 'left'), 'right': ('right_cam', 'right')}, ideal=True)
            try:
                frame = optics.render(data)
                self.assertEqual(set(frame['left'].contacting_geoms), {model.geom('ball1').id, model.geom('ball2').id})
                self.assertEqual(frame['right'].normal_force_n, 0.)
                self.assertGreater(frame['left'].normal_force_n, 0.)
                self.assertGreater(frame['left'].indentation_m.max(), 0.)
                self.assertEqual(float(frame['right'].indentation_m.max()), 0.)
                frames.append(frame['left'])
            finally:
                optics.close()
        self.assertAlmostEqual(frames[0].normal_force_n, frames[1].normal_force_n, places=7)
        np.testing.assert_allclose(frames[0].indentation_m, frames[1].indentation_m, atol=1e-7)
        np.testing.assert_allclose(frames[0].rgb, frames[1].rgb, atol=1)

    def test_press_and_unload(self):
        env = OpticalBench()
        optics = OpticalTactile(env.model, env.sensors, ideal=True)
        try:
            baseline = optics.render(env.data)['DIGIT']
            env.data.ctrl[2] = -.012
            for _ in range(150):
                env.step(np.zeros(3))
            pressed = optics.render(env.data)['DIGIT']
            self.assertGreater(pressed.normal_force_n, 1.)
            self.assertGreater(pressed.indentation_m.max(), .0001)
            # Upstream background calibration halves and blurs the optical
            # difference. Inspect the actual contact patch, not an arbitrary
            # mean diluted by the large unchanged background.
            difference = np.abs(pressed.rgb.astype(float)-baseline.rgb)
            self.assertGreater(difference[pressed.contact_mask].mean(), 1.)
            env.data.ctrl[2] = 0.
            for _ in range(150):
                env.step(np.zeros(3))
            released = optics.render(env.data)['DIGIT']
            self.assertEqual(released.normal_force_n, 0.)
            np.testing.assert_array_equal(released.rgb, baseline.rgb)
            np.testing.assert_array_equal(released.indentation_m, baseline.indentation_m)
        finally:
            optics.close()

    def test_panda_physical_grasp_lift_release(self):
        env = OpticalPandaEnv()
        self.assertEqual(env.model.nflex, 0)
        optics = OpticalTactile(env.model, env.sensors, ideal=True)
        try:
            with tempfile.TemporaryDirectory() as temp:
                report = demo(env, optics, Path(temp))
            self.assertGreater(report['lift_m'], .09)
            self.assertLess(report['released_cube_z_m'], .121)
            for side in ('left', 'right'):
                self.assertGreater(report['lifted'][side]['normal_force_n'], .1)
                self.assertGreater(report['lifted'][side]['max_indentation_mm'], .01)
                self.assertEqual(report['released'][side]['normal_force_n'], 0.)
                self.assertEqual(report['released'][side]['max_indentation_mm'], 0.)
            # Reproduce a broad object behind the curved gel's border. Rays
            # missing the baseline gel must not become -camera_distance depth.
            env.target_xyz = np.array([.40, 0., .125])
            for _ in range(150):
                env.step(env.hold_action())
            env.command_gripper(True)
            for _ in range(100):
                env.step(env.hold_action())
            for frame in optics.render(env.data).values():
                self.assertTrue((~frame.valid_mask).any())
                self.assertGreater(frame.normal_force_n, .1)
                invalid_background = (frame.baseline_depth_m == 0) & (frame.camera_depth_m > 0)
                self.assertTrue(invalid_background.any())
                np.testing.assert_array_equal(frame.indentation_m[invalid_background], 0.)
                self.assertFalse(frame.contact_mask[invalid_background].any())
                np.testing.assert_allclose(frame.indentation_m[frame.valid_mask],
                    (frame.baseline_depth_m-frame.camera_depth_m)[frame.valid_mask])
                self.assertGreater(frame.indentation_m.min(), -.001)
        finally:
            optics.close()


if __name__ == '__main__':
    unittest.main()
