"""USB/WebXR input for the simulated Panda. No real-robot transport.

WebXR poses are metres and xyzw quaternions. Only the input adapter changes;
the environment's seven-dimensional command and dataset writers stay intact.
"""
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import shutil
import subprocess
import threading
import time
from urllib.parse import urlsplit

import mujoco
import numpy as np

from wowskin_lab.paths import ROOT
from wowskin_lab.data.cameras import capture_options


def quaternion_matrix(xyzw):
    q = np.asarray(xyzw, dtype=float)[[3, 0, 1, 2]]
    q /= np.linalg.norm(q)
    out = np.empty(9)
    mujoco.mju_quat2Mat(out, q)
    return out.reshape(3, 3)


def validate_frame(frame):
    if not isinstance(frame, dict):
        raise ValueError('Expected an XR frame')
    if not isinstance(frame.get('session'), str) or not 1 <= len(frame['session']) <= 100:
        raise ValueError('Missing session')
    if type(frame.get('seq')) is not int or frame['seq'] < 0:
        raise ValueError('Invalid sequence')
    if type(frame.get('active')) is not bool:
        raise ValueError('Invalid active flag')
    if frame.get('mode', 'worn') not in ('worn', 'desk'):
        raise ValueError('Unknown headset placement mode')
    if not np.isfinite(float(frame['client_ms'])):
        raise ValueError('Invalid timestamp')
    for key in ('head', 'left', 'right'):
        pose = frame.get(key)
        if pose is None:
            continue
        p, q = np.asarray(pose['p'], dtype=float), np.asarray(pose['q'], dtype=float)
        if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all():
            raise ValueError('Invalid pose')
        if not .9 < np.linalg.norm(q) < 1.1 or np.max(np.abs(p)) > 100:
            raise ValueError('Invalid pose range')
        if key != 'head':
            b = np.asarray(pose['buttons'], dtype=float)
            if b.ndim != 1 or not 2 <= len(b) <= 12 or not np.isfinite(b).all() or np.any((b < 0) | (b > 1)):
                raise ValueError('Invalid buttons')
            if pose.get('mapping') != 'xr-standard':
                raise ValueError('Only xr-standard controllers are supported')
            if 'aim' in pose:
                aim = np.asarray(pose['aim'], dtype=float)
                if aim.shape != (3,) or not np.isfinite(aim).all() or not .9 < np.linalg.norm(aim) < 1.1:
                    raise ValueError('Invalid aiming direction')
    return frame


class PicoMailbox:
    """Latest pose plus edge events; network threads never touch MuJoCo."""
    def __init__(self):
        self.lock = threading.Lock()
        self.frame, self.received = None, 0.
        self.epoch, self.frames = 0, 0
        self.commands = deque(maxlen=16)
        self.status = {'control': 'Waiting for headset', 'recording': False}
        self.jpeg = None
        self.error = ''

    def publish(self, frame, now=None):
        frame = validate_frame(frame)
        now = time.monotonic() if now is None else now
        with self.lock:
            old = self.frame
            same = old is not None and old['session'] == frame['session']
            if same and (frame['seq'] <= old['seq'] or frame['client_ms'] < old['client_ms']):
                raise ValueError('Out-of-order frame')
            continuous = same and now-self.received < .25 and old['active'] and frame['active']
            right = frame.get('right')
            if not continuous or not right or right['buttons'][1] < .5:
                self.epoch += 1
            # Recording buttons only work with the motion clutch released.
            # First frames / reconnections cannot replay a held button.
            if continuous and right and right['buttons'][1] < .5:
                previous_right = old.get('right')
                if (previous_right and len(right['buttons']) > 4 and len(previous_right['buttons']) > 4
                        and right['buttons'][4] > .5 >= previous_right['buttons'][4]):
                    self.commands.append((now, ord('G')))  # Right A: calibrate facing direction.
                left, previous = frame.get('left'), old.get('left')
                if left and previous:
                    for index, code in ((4, ord('B')), (5, 257), (3, ord('N'))):
                        if (len(left['buttons']) > index and len(previous['buttons']) > index
                                and left['buttons'][index] > .5 >= previous['buttons'][index]):
                            self.commands.append((now, code))
            self.frame, self.received = frame, now
            self.frames += 1
            self.error = ''

    def invalidate(self, message):
        with self.lock:
            self.epoch += 1
            self.frame = None
            self.commands.clear()
            self.error = message

    def sample(self, drain=True):
        with self.lock:
            commands = tuple(code for stamp, code in self.commands if time.monotonic()-stamp < .5)
            if drain:
                self.commands.clear()
            return self.frame, self.received, self.epoch, commands

    def report(self):
        with self.lock:
            return dict(self.status, frames=self.frames, error=self.error,
                        age_s=round(time.monotonic()-self.received, 3) if self.frame else None,
                        hands=[k for k in ('left', 'right') if self.frame and self.frame.get(k)])


class PicoController:
    """Relative 6-DoF clutch with release-to-rearm after stale tracking."""
    def __init__(self, scale=1., camera_azimuth=135., speed=.25, angular_speed=1.5):
        if not np.isfinite(scale) or not 0 < scale <= 2:
            raise ValueError('Pico position scale must be in (0, 2]')
        self.scale = scale
        if not np.isfinite([speed, angular_speed]).all() or not (0 < speed <= .5 and 0 < angular_speed <= 3.):
            raise ValueError('Invalid Pico speed limits')
        self.speed, self.angular_speed = speed, angular_speed
        self.camera_azimuth = camera_azimuth
        self.basis = None
        self.mode = None
        self.reset()

    def reset(self):
        self.engaged, self.armed = False, False
        self.session, self.epoch = getattr(self, 'session', None), None
        self.previous = None
        self.status = 'Release right grip to arm'

    def calibrate(self, frame):
        """Set operator forward from the controller ray, independently of HMD pose."""
        right = frame.get('right') if frame else None
        if not frame or not frame['active'] or not right or right['buttons'][1] >= .5 or 'aim' not in right:
            self.status = 'Release right grip; point at screen and press A'
            return False
        forward = np.asarray(right['aim'], dtype=float).copy()
        forward[1] = 0.
        if np.linalg.norm(forward) < .2:
            self.status = 'Point horizontally at screen; press A'
            return False
        forward /= np.linalg.norm(forward)
        up = np.array([0., 1., 0.])
        yaw = np.deg2rad(self.camera_azimuth)
        world_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0.],
                              [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
        self.basis = world_yaw @ np.stack([forward, np.cross(up, forward), up])
        self.session = frame['session']
        self.mode = frame.get('mode', 'worn')
        self.engaged, self.armed = False, False
        self.status = 'Direction calibrated; release right grip'
        return True

    def action(self, env, frame, received, epoch, now=None):
        from wowskin_lab.envs.pick_cube import hold_current_pose, limit_vector, rotation_vector, WORKSPACE
        now = time.monotonic() if now is None else now
        right = frame.get('right') if frame else None
        valid = frame and frame['active'] and right and frame.get('head') and now-received <= .25
        changed = frame and self.session != frame['session']
        mode_changed = frame and self.mode != frame.get('mode', 'worn')
        if not valid or changed or mode_changed:
            if self.engaged:
                hold_current_pose(env)
            self.engaged, self.armed = False, False
            self.previous = None
            if mode_changed or (changed and self.session is not None):
                self.basis = None  # Reference-space reset requires a new forward direction.
            self.session = frame['session'] if frame else None
            self.status = 'Tracking paused; release right grip'
        if not valid:
            return env.hold_action()
        if self.basis is None:
            if right['buttons'][1] >= .5:
                self.status = 'Release right grip to set direction'
                return env.hold_action()
            # Facing the user reverses the horizontal forward/right reference.
            # Freeze this basis until a mode/session change or explicit A calibration.
            forward = -quaternion_matrix(frame['head']['q'])[:, 2]
            if frame.get('mode', 'worn') == 'desk':
                forward = -forward
            reference = dict(frame, right=dict(right, aim=forward.tolist()))
            if not self.calibrate(reference):
                return env.hold_action()
        grip = right['buttons'][1] >= .5
        if grip and self.epoch is not None and self.epoch != epoch:
            if self.engaged:
                hold_current_pose(env)
            self.engaged, self.armed = False, False
        self.epoch = epoch
        if not grip:
            if self.engaged:
                hold_current_pose(env)
            self.engaged, self.armed, self.previous = False, True, None
            self.status = 'Ready: hold right grip to move'
            return env.hold_action()
        if not self.armed:
            return env.hold_action()
        p, r = np.asarray(right['p']), quaternion_matrix(right['q'])
        if not self.engaged:
            hold_current_pose(env)
            self.origin_p, self.origin_r = p.copy(), r.copy()
            self.robot_p, self.robot_r = env.target_xyz.copy(), env.target_rotation.copy()
            self.engaged = True
        if self.previous is not None:
            old_p, old_r = self.previous
            if np.linalg.norm(p-old_p) > .15 or np.linalg.norm(rotation_vector(old_r.T @ r)) > .8:
                hold_current_pose(env)
                self.engaged, self.armed = False, False
                self.status = 'Pose jump: release right grip'
                return env.hold_action()
        self.previous = (p.copy(), r.copy())
        desired_p = self.robot_p + self.scale*self.basis @ (p-self.origin_p)
        limited = not env.cfg.get('disable_cartesian_limits', False)
        if limited:
            desired_p = np.clip(desired_p, *getattr(env, 'workspace', WORKSPACE))
        desired_r = self.basis @ r @ self.origin_r.T @ self.basis.T @ self.robot_r
        # Bound motion per simulated second, and bound target lead at contact.
        actual_p = env.data.site_xpos[env.tcp_site]
        actual_r = env.data.site_xmat[env.tcp_site].reshape(3, 3)
        from wowskin_lab.envs.pick_cube import rotation_matrix
        if limited:
            env.target_xyz = np.clip(env.target_xyz, actual_p-.03, actual_p+.03)
            env.target_rotation = actual_r @ rotation_matrix(limit_vector(rotation_vector(actual_r.T @ env.target_rotation), .10))
        hz = env.cfg['sample_hz']
        delta_p = desired_p-env.target_xyz
        delta_r = rotation_vector(env.target_rotation.T @ desired_r)
        if limited:
            delta_p = limit_vector(delta_p, self.speed/hz)
            delta_r = limit_vector(delta_r, self.angular_speed/hz)
        closure = float(right['buttons'][0])
        self.status = 'Moving: right trigger closes gripper'
        return np.r_[delta_p, delta_r, closure]


class PicoBridge:
    """Loopback-only HTTP transport, reached on the headset by adb reverse."""
    def __init__(self, port=8088, scale=1., speed=.25, angular_speed=1.5):
        self.mailbox = PicoMailbox()
        self.controller = PicoController(scale, speed=speed, angular_speed=angular_speed)
        self.token = secrets.token_urlsafe(24)
        self.renderer = None
        self.preview_time = 0.
        self.port = port
        self.device = None
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, code, data, mime='application/json'):
                self.send_response(code)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                path = urlsplit(self.path).path
                if path == '/':
                    html = Path(__file__).with_name('pico.html').read_text().replace('__TOKEN__', bridge.token)
                    self.reply(200, html.encode(), 'text/html; charset=utf-8')
                elif path == '/status':
                    self.reply(200, json.dumps(bridge.mailbox.report()).encode())
                elif path == '/connection':
                    # Same-origin bootstrap for a page surviving a server restart.
                    # No CORS headers: other origins cannot read this token.
                    if self.headers.get('Sec-Fetch-Site') == 'cross-site':
                        self.reply(403, b'{}')
                    else:
                        self.reply(200, json.dumps({'token': bridge.token}).encode())
                elif path == '/preview.jpg':
                    with bridge.mailbox.lock:
                        jpeg = bridge.mailbox.jpeg
                    self.reply(200 if jpeg else 503, jpeg or b'', 'image/jpeg')
                else:
                    self.reply(404, b'{}')

            def do_POST(self):
                if self.path != '/input' or self.headers.get('X-Pico-Token') != bridge.token:
                    self.reply(403, b'{}')
                    return
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size < 16384:
                        raise ValueError('Invalid payload length')
                    self.connection.settimeout(2)
                    frame = json.loads(self.rfile.read(size))
                    bridge.mailbox.publish(frame)
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    bridge.mailbox.invalidate(str(exc))
                    self.reply(400, json.dumps({'error': str(exc)}).encode())
                    return
                self.reply(200, json.dumps(bridge.mailbox.report()).encode())

        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def open_usb(self):
        adb = ROOT/'.tools/platform-tools/adb'
        if not adb.exists():
            installed = shutil.which('adb')
            if not installed:
                raise RuntimeError('Install Android Platform-Tools in .tools/platform-tools or add adb to PATH')
            adb = Path(installed)
        result = subprocess.run([str(adb), 'devices'], capture_output=True, text=True, check=True, timeout=10)
        devices = [line.split()[0] for line in result.stdout.splitlines()[1:] if line.endswith('\tdevice')]
        if len(devices) != 1:
            raise RuntimeError('Connect exactly one USB-debugging-authorized Pico; check `adb devices`')
        self.device = devices[0]
        self.adb = [str(adb), '-s', self.device]
        subprocess.run([*self.adb, 'reverse', f'tcp:{self.port}', f'tcp:{self.port}'], check=True, timeout=10)
        subprocess.run([*self.adb, 'shell', 'am', 'start', '-a', 'android.intent.action.VIEW',
                        '-d', f'http://localhost:{self.port}/?launch={secrets.token_hex(4)}', '-p', 'com.pico.browser'], check=True, timeout=10)
        print(f'Pico USB: open http://localhost:{self.port}/ in headset; click Enter VR. Right GRIP = motion clutch.', flush=True)

    def update_preview(self, env, camera, recording=False):
        from PIL import Image
        import io
        with self.mailbox.lock:
            frame = self.mailbox.frame
            mode = frame.get('mode', 'worn') if frame else self.controller.mode
            preview_enabled = mode != 'desk'
            self.mailbox.status = {'control': self.controller.status, 'recording': recording,
                                   'mode': mode, 'preview_enabled': preview_enabled,
                                   'sim_time_s': round(env.data.time, 3)}
            if not preview_enabled:
                self.mailbox.jpeg = None
        if not preview_enabled:
            # The Mac viewer and learning cameras are independent. Desk mode
            # skips only the extra HMD render and JPEG encoding, not physics/input.
            self.preview_time = 0.
            return
        if time.monotonic()-self.preview_time < .15:
            return
        self.preview_time = time.monotonic()
        if self.renderer is None:
            self.renderer = mujoco.Renderer(env.model, height=480, width=640)
        from types import SimpleNamespace
        options = capture_options()
        if hasattr(env, 'update_viewer_options') and isinstance(camera, mujoco.MjvCamera):
            env.update_viewer_options(SimpleNamespace(cam=camera, opt=options))
        self.renderer.update_scene(env.data, camera=camera, scene_option=options)
        buf = io.BytesIO()
        Image.fromarray(self.renderer.render()).save(buf, format='JPEG', quality=75)
        with self.mailbox.lock:
            self.mailbox.jpeg = buf.getvalue()

    def __exit__(self, *_):
        if self.renderer:
            self.renderer.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        if self.device:
            subprocess.run([*self.adb, 'reverse', '--remove', f'tcp:{self.port}'],
                           capture_output=True, timeout=5)
