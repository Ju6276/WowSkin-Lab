"""Pose mapping, clutch/reconnect behaviour and the actual local HTTP transport."""
import copy
import json
import time
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import mujoco

from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, response_configuration, rotation_matrix, rotation_vector
from wowskin_lab.teleop.pico import PicoBridge, PicoController, PicoMailbox, validate_frame


def frame(seq=0, grip=0., trigger=0., p=(0., 1., 0.), q=(0., 0., 0., 1.)):
    return dict(session='test', seq=seq, client_ms=seq*20., active=True,
                head=dict(p=[0., 1.6, 0.], q=[0., 0., 0., 1.]),
                right=dict(p=list(p), q=list(q), buttons=[trigger, grip, 0., 0., 0., 0.], mapping='xr-standard', aim=[0.,0.,-1.]),
                left=dict(p=[-.3, 1., 0.], q=[0., 0., 0., 1.], buttons=[0.]*6, mapping='xr-standard'))


class PicoChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = PandaPickCubeTactileEnv(response_configuration(SimpleNamespace(
            task='pick_cube', skin_dof='trilinear' if mujoco.__version__ == '3.15.0' else 'full')), pollen=False)

    def setUp(self):
        self.env.reset()

    def armed(self):
        control = PicoController(scale=.5, camera_azimuth=0.)
        self.assertTrue(control.calibrate(frame()))
        control.action(self.env, frame(), 1., 1, now=1.)
        control.action(self.env, frame(1, grip=1.), 1.02, 1, now=1.02)
        return control

    def test_forward_right_up_and_local_rotation(self):
        for position, axis, sign in (((0., 1., -.02), 0, 1), ((.02, 1., 0.), 1, -1), ((0., 1.02, 0.), 2, 1)):
            with self.subTest(position=position):
                control = self.armed()
                action = control.action(self.env, frame(2, 1., p=position), 1.04, 1, now=1.04)
                self.assertGreater(action[axis]*sign, .0007)
                self.assertLess(np.linalg.norm(np.delete(action[:3], axis)), 1e-8)
        control = self.armed()
        theta = .1
        action = control.action(self.env, frame(2, 1., q=(0., np.sin(theta/2), 0., np.cos(theta/2))), 1.04, 1, now=1.04)
        expected = rotation_vector(self.env.target_rotation.T @ rotation_matrix([0., 0., theta]) @ self.env.target_rotation)
        np.testing.assert_allclose(action[3:6]/np.linalg.norm(action[3:6]), expected/np.linalg.norm(expected), atol=1e-8)

    def test_translated_scene_uses_its_own_workspace(self):
        # Feeding lives outside the cube's world-coordinate box. Even an
        # unchanged held controller must not drag it towards that cube box.
        workspace = np.array([[-2.01, -.36, -np.inf], [-1.26, .52, 1.68]])
        with patch.object(self.env, 'workspace', workspace, create=True):
            self.env.data.site_xpos[self.env.tcp_site] = [-1.7, .1, 1.4]
            control = self.armed()
            action = control.action(self.env, frame(2, 1.), 1.04, 1, now=1.04)
            np.testing.assert_allclose(action[:3], 0., atol=1e-10)

    def test_explicit_unlimited_mode_does_not_slew_or_clamp_cartesian_target(self):
        self.env.cfg['disable_cartesian_limits'] = True
        try:
            control = self.armed()
            before = self.env.target_xyz.copy()
            # A deliberately narrow task workspace would clip this ordinary
            # 10 cm tracked controller motion in the default profile.
            with patch.object(self.env, 'workspace', np.array([before-.001, before+.001]), create=True):
                command = control.action(self.env, frame(2, 1., p=(0., 1., -.10)), 1.04, 1, now=1.04)
            np.testing.assert_allclose(command[:3], [.05, 0, 0], atol=1e-10)
        finally:
            self.env.cfg.pop('disable_cartesian_limits', None)

    def test_stale_loss_session_and_missed_release_require_rearm(self):
        control = self.armed()
        action = control.action(self.env, frame(2, 1., trigger=.8, p=(0., 1.03, 0.)), 1.04, 1, now=1.04)
        self.env.step(action)
        self.assertTrue(control.engaged)
        held_closure = self.env.gripper_fraction(target=True)
        for f, timestamp, epoch, now in ((frame(3, 1.), 1.05, 1, 2.), (frame(4, 1.), 2.01, 2, 2.01)):
            action = control.action(self.env, f, timestamp, epoch, now=now)
            np.testing.assert_array_equal(action[:6], 0)
            self.assertAlmostEqual(action[6], held_closure)
            self.assertFalse(control.engaged)
        control.action(self.env, frame(5), 2.02, 3, now=2.02)
        control.action(self.env, frame(6, 1.), 2.04, 3, now=2.04)
        self.assertTrue(control.engaged)
        control.action(self.env, frame(8, 1.), 2.08, 4, now=2.08)  # Release was received between control ticks.
        self.assertFalse(control.engaged)
        changed = frame(9, 1.); changed['session'] = 'recentered'
        np.testing.assert_array_equal(control.action(self.env, changed, 2.1, 5, now=2.1)[:6], 0)

    def test_physics_move_release_and_trigger(self):
        control = self.armed()
        before = self.env.data.site_xpos[self.env.tcp_site].copy()
        for i in range(80):
            f = frame(i+2, 1., .35, p=(0., 1.+min(i*.0008, .06), 0.))
            self.env.step(control.action(self.env, f, 2+i*.01, 1, now=2+i*.01))
        after = self.env.data.site_xpos[self.env.tcp_site].copy()
        self.assertGreater(after[2]-before[2], .02)
        self.assertAlmostEqual(self.env.gripper_fraction(target=True), .35)
        for i in range(30):
            self.env.step(control.action(self.env, frame(100+i), 3+i*.01, 2+i, now=3+i*.01))
        self.assertLess(np.linalg.norm(self.env.data.site_xpos[self.env.tcp_site]-after), .002)
        np.testing.assert_array_equal(self.env.data.warning.number, 0)

    def test_screen_alignment_with_headset_facing_user(self):
        # Two opposite XR reference headings must produce the same screen-right action.
        actions = []
        for direction in (-1., 1.):
            control = PicoController()
            f = frame()
            f['right']['aim'] = [0., 0., direction]
            f['head']['q'] = [0., 1., 0., 0.]  # HMD facing the operator is irrelevant.
            self.assertTrue(control.calibrate(f))
            control.action(self.env, f, 1., 1, now=1.)
            f['right']['buttons'][1] = 1.
            control.action(self.env, f, 1., 1, now=1.)
            f['right']['p'][0] = -direction*.02
            action = control.action(self.env, f, 1.02, 1, now=1.02)
            # The actual 135-degree camera's horizontal screen-right is +X,+Y.
            self.assertGreater(action[0], 0.)
            self.assertGreater(action[1], 0.)
            self.assertAlmostEqual(action[0], action[1])
            actions.append(action)
        np.testing.assert_allclose(actions[0], actions[1], atol=1e-10)

    def test_two_modes_auto_align_without_a_and_do_not_move_on_switch(self):
        actions=[]
        for mode, head_q in (('worn', [0.,0.,0.,1.]), ('desk', [0.,1.,0.,0.])):
            control=PicoController()
            f=frame(); f['mode']=mode; f['head']['q']=head_q
            control.action(self.env, f, 1., 1, now=1.)
            self.assertTrue(control.armed)
            f['right']['buttons'][1]=1.
            control.action(self.env, f, 1.02, 1, now=1.02)
            f['right']['p'][0]=.02
            action=control.action(self.env, f, 1.04, 1, now=1.04)
            self.assertGreater(action[0], 0.)
            self.assertGreater(action[1], 0.)
            actions.append(action)
            f['mode']='desk' if mode=='worn' else 'worn'
            np.testing.assert_array_equal(control.action(self.env, f, 1.06, 1, now=1.06)[:6], 0)
            self.assertFalse(control.engaged)
        np.testing.assert_allclose(actions[0],actions[1],atol=1e-10)

    def test_responsive_profile_tracks_ramp_and_stops_on_release(self):
        cfg = response_configuration(SimpleNamespace(task='pick_cube',
            skin_dof='trilinear' if mujoco.__version__ == '3.15.0' else 'full'))
        cfg.update(tcp_speed_m_s=.25, tcp_angular_speed_rad_s=1.5)
        env = PandaPickCubeTactileEnv(cfg, pollen=False)
        control = PicoController(scale=1., camera_azimuth=0.)
        control.action(env, frame(), 1., 1, now=1.)
        control.action(env, frame(1, 1.), 1.01, 1, now=1.01)
        origin = env.data.site_xpos[env.tcp_site].copy()
        for i in range(1, 51):
            sample = frame(i+1, 1., p=(0., 1.+i*.0015, 0.))
            env.step(control.action(env, sample, 1.01+i*.01, 1, now=1.01+i*.01))
        actual = env.data.site_xpos[env.tcp_site].copy()
        self.assertLess(abs(.075-(actual[2]-origin[2])), .015)
        for i in range(30):
            env.step(control.action(env, frame(100+i), 2+i*.01, 2+i, now=2+i*.01))
        self.assertLess(np.linalg.norm(env.data.site_xpos[env.tcp_site]-actual), .001)
        # Rotation uses the same profile, still integrating actual robot dynamics.
        reference = env.data.site_xmat[env.tcp_site].reshape(3,3).copy()
        control.action(env, frame(199), 2.99, 40, now=2.99)
        control.action(env, frame(200,1.), 3., 40, now=3.)
        for i in range(1, 31):
            sample = frame(200+i,1.,q=(0.,np.sin(i*.005),0.,np.cos(i*.005)))
            env.step(control.action(env,sample,3+i*.01,40,now=3+i*.01))
        angle = np.linalg.norm(rotation_vector(reference.T @ env.data.site_xmat[env.tcp_site].reshape(3,3)))
        self.assertGreater(angle,.20)
        self.assertLess(angle,.36)
        np.testing.assert_array_equal(env.data.warning.number,0)

    def test_http_rejects_nonfinite_unauthorized_and_out_of_order(self):
        with PicoBridge(port=0) as bridge:
            url=f'http://127.0.0.1:{bridge.port}'
            def post(payload, token=bridge.token):
                return urlopen(Request(url+'/input', data=json.dumps(payload).encode(),
                                       headers={'X-Pico-Token': token, 'Content-Type':'application/json'}))
            with post(frame()) as reply:
                self.assertEqual(json.load(reply)['hands'], ['left', 'right'])
            with self.assertRaises(HTTPError) as ctx:
                post(frame(1), token='wrong')
            self.assertEqual(ctx.exception.code, 403)
            invalid = frame(1); invalid['right']['p'][0] = float('nan')
            with self.assertRaises(HTTPError):
                post(invalid)
            self.assertIsNone(bridge.mailbox.frame)
            post(frame(2)).close()
            with self.assertRaises(HTTPError):
                post(frame(1))
            self.assertIsNone(bridge.mailbox.frame)

    def test_record_edges_only_once_and_no_reconnection_replay(self):
        mailbox = PicoMailbox()
        base = time.monotonic()
        mailbox.publish(frame(), base)
        down=frame(1);down['left']['buttons'][4]=1.
        mailbox.publish(copy.deepcopy(down), base+.02)
        self.assertEqual(mailbox.sample()[3], (ord('B'),))
        down['seq']=2;down['client_ms']=40.
        mailbox.publish(copy.deepcopy(down), base+.04)
        self.assertEqual(mailbox.sample()[3], ())
        down['seq']=3;down['client_ms']=60.;down['session']='new'
        mailbox.publish(down, base+.06)
        self.assertEqual(mailbox.sample()[3], ())
        bad=frame();bad['right']['q']=[0.,0.,0.,0.]
        with self.assertRaises(ValueError):
            validate_frame(bad)

    def test_restarted_server_rejects_old_token_and_allows_same_origin_reconnect(self):
        with PicoBridge(port=0) as old:
            old_token = old.token
        with PicoBridge(port=0) as new:
            url = f'http://127.0.0.1:{new.port}'
            payload = json.dumps(frame()).encode()
            with self.assertRaises(HTTPError) as failure:
                urlopen(Request(url+'/input', data=payload, headers={'X-Pico-Token': old_token}))
            self.assertEqual(failure.exception.code, 403)
            with self.assertRaises(HTTPError):
                urlopen(Request(url+'/connection', headers={'Sec-Fetch-Site': 'cross-site'}))
            with urlopen(Request(url+'/connection', headers={'Sec-Fetch-Site': 'same-origin'})) as reply:
                token = json.load(reply)['token']
            self.assertNotEqual(token, old_token)
            with urlopen(Request(url+'/input', data=payload, headers={'X-Pico-Token': token})) as reply:
                self.assertEqual(json.load(reply)['frames'], 1)

    def test_pico_commands_feed_single_shared_dataset(self):
        import h5py
        from wowskin_lab.data.lerobot_recording import LeRobotSimRecorder, validate_dataset
        from wowskin_lab.data.media import video_path
        from wowskin_lab.paths import ROOT
        self.env = PandaPickCubeTactileEnv(response_configuration(SimpleNamespace(
            task='pick_cube', skin_dof='trilinear', ideal=True,
            magnetic_fit=ROOT/'configs/magnetic_fit_pollen_nonslip.json')), pollen=False)
        with tempfile.TemporaryDirectory() as temporary:
            with LeRobotSimRecorder(self.env, Path(temporary), source='pico') as recorder:
                recorder.start()
                control = self.armed()
                for i in range(20):
                    sample = frame(i+2, 1., .3, p=(0., 1.+i*.0005, 0.))
                    action = control.action(self.env, sample, 2+i*.01, 1, now=2+i*.01)
                    recorder.step(self.env.step(action)[-1])
                result = recorder.finish('success')  # Format test, not a task success claim.
            vi = json.loads((result/'metadata.json').read_text())
            self.assertEqual(vi['source'], 'pico')
            self.assertIn('pico', vi['collection_method'])
            self.assertEqual(vi['pose_frame'], 'robot_base')
            self.assertEqual(vi['camera_ids'], [1, 2, 52])
            self.assertTrue(all(video_path(result, i).suffix == '.mp4' for i in vi['camera_ids']))
            self.assertTrue(validate_dataset(Path(temporary)/'pick_cube/lerobot')['official_loader_passed'])
            with h5py.File(result/'measurements.h5') as data:
                self.assertEqual(data['sensor_states'].shape[1], 15)
                self.assertGreater(data['timestamp'].shape[0], 15)
                self.assertEqual(data['arm_qpos'].shape[1], 7)

    def test_desk_skips_preview_render_and_worn_resumes(self):
        with PicoBridge(port=0) as bridge, patch('wowskin_lab.teleop.pico.mujoco.Renderer') as factory:
            renderer = factory.return_value
            renderer.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            bridge.mailbox.publish(frame())
            bridge.update_preview(self.env, None)
            self.assertEqual(renderer.render.call_count, 1)
            self.assertIsNotNone(bridge.mailbox.jpeg)
            desk = frame(1); desk['mode'] = 'desk'
            bridge.mailbox.publish(desk)
            for _ in range(5):
                bridge.update_preview(self.env, None, recording=True)
            self.assertEqual(renderer.render.call_count, 1)
            self.assertIsNone(bridge.mailbox.jpeg)
            self.assertFalse(bridge.mailbox.report()['preview_enabled'])
            self.assertTrue(bridge.mailbox.report()['recording'])
            self.assertEqual(bridge.mailbox.report()['hands'], ['left', 'right'])
            bridge.mailbox.publish(frame(2))
            bridge.update_preview(self.env, None)
            self.assertEqual(renderer.render.call_count, 2)
            self.assertIsNotNone(bridge.mailbox.jpeg)
            self.assertTrue(bridge.mailbox.report()['preview_enabled'])


if __name__ == '__main__':
    unittest.main()
