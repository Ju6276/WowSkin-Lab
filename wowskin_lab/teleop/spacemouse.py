"""USB SpaceMouse input for simulation; never sends commands to a real robot.

HID protocol: 3Dconnexion report 1 = six signed little-endian axes (or XYZ),
report 2 = rotation on older devices; report 3 = button bits. Status/battery
reports do not refresh motion. Axis convention follows robosuite's device
decoder, with our camera-plane transform applied separately.
"""
from collections import deque
from dataclasses import dataclass
import threading
import time

import numpy as np


@dataclass(frozen=True)
class MouseSnapshot:
    axes: np.ndarray
    received: np.ndarray
    buttons: int
    presses: tuple
    error: str = ''


class MouseState:
    def __init__(self):
        self.lock = threading.Lock()
        self.axes = np.zeros(6)
        self.received = np.full(2, -np.inf)
        self.buttons = 0
        self.presses = deque(maxlen=16)
        self.error = ''

    def feed(self, report, now=None):
        now = time.monotonic() if now is None else now
        if not report:
            return
        with self.lock:
            if report[0] == 1 and len(report) in (7, 13):
                values = np.frombuffer(bytes(report[1:]), dtype='<i2').astype(float)
                self.axes[:len(values)] = np.clip(values / 350., -1., 1.)
                self.received[0] = now
                if len(values) == 6:
                    self.received[1] = now
            elif report[0] == 2 and len(report) == 7:
                self.axes[3:] = np.clip(np.frombuffer(bytes(report[1:]), dtype='<i2') / 350., -1., 1.)
                self.received[1] = now
            elif report[0] == 3 and len(report) >= 2:
                buttons = int(report[1]) & 3
                rising = buttons & ~self.buttons
                # Both buttons together hold rather than choose a direction.
                if buttons == 3:
                    self.presses.append('hold')
                elif rising:
                    self.presses.append('close' if rising & 1 else 'open')
                self.buttons = buttons

    def sample(self):
        with self.lock:
            result = MouseSnapshot(self.axes.copy(), self.received.copy(), self.buttons,
                                   tuple(self.presses), self.error)
            self.presses.clear()
        return result


class SpaceMouse:
    """Single HID reader; retains only the most recent axes, not a motion queue."""
    def __init__(self):
        self.state = MouseState()
        self.stop = threading.Event()
        self.thread = None
        self.device = None
        self.info = None

    def __enter__(self):
        import hid
        # Multi-axis controller usage excludes duplicate vendor interfaces.
        devices = {d['path']: d for d in hid.enumerate()
                   if d['vendor_id'] in (0x256f, 0x046d)
                   and 'spacemouse' in (d.get('product_string') or '').lower()
                   and d.get('usage_page') == 1 and d.get('usage') == 8}
        if len(devices) != 1:
            raise RuntimeError(f'Expected one USB SpaceMouse, found {len(devices)}. Connect its USB data cable.')
        path, description = next(iter(devices.items()))
        self.info = {k: description.get(k) for k in ('product_string', 'vendor_id', 'product_id')}
        # macOS otherwise may route the same relative-axis reports to the
        # desktop pointer. Explicitly seize only this SpaceMouse while open.
        import sys
        if sys.platform == 'darwin':
            import ctypes
            backend = ctypes.CDLL(hid.__file__)
            exclusive = backend.hid_darwin_set_open_exclusive
            exclusive.argtypes = [ctypes.c_int]
            exclusive.restype = None
            exclusive(1)
            self.info['exclusive_hid'] = True
        self.device = hid.device()
        self.device.open_path(path)
        self.thread = threading.Thread(target=self._read, name='SpaceMouse-HID', daemon=True)
        self.thread.start()
        return self

    def _read(self):
        try:
            while not self.stop.is_set():
                report = self.device.read(64, 50)
                self.state.feed(report)
        except Exception as exc:
            with self.state.lock:
                self.state.error = f'SpaceMouse disconnected: {exc}'

    def __exit__(self, *exc):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1.)
        if self.device:
            self.device.close()


class SpaceMouseController:
    def __init__(self, speed=.12, angular_speed=.5, deadzone=.08):
        if not np.isfinite([speed, angular_speed, deadzone]).all() or not (
                0 < speed <= .25 and 0 < angular_speed <= 1.5 and 0 <= deadzone < .5):
            raise ValueError('SpaceMouse: speed (0,.25] m/s, rotation (0,1.5] rad/s, deadzone [0,.5)')
        self.speed, self.angular_speed, self.deadzone = speed, angular_speed, deadzone
        self.reset()

    def reset(self):
        self.armed = False
        self.moving = False
        self.status = 'Center knob and release buttons'

    def settings(self):
        return dict(speed_m_s=self.speed, angular_speed_rad_s=self.angular_speed,
                    deadzone=self.deadzone, timeout_s=.25, full_scale=350,
                    hid_axis_signs=[1,-1,-1,-1,-1,1],
                    translation_frame='camera ground plane + world Z',
                    rotation_frame='camera ground plane + world Z, converted to TCP-local rotvec',
                    buttons='left close; right open; both hold')

    def action(self, env, snapshot, azimuth, focused=True, now=None):
        from wowskin_lab.envs.pick_cube import (
            hold_current_pose, view_translation, limit_vector, rotation_matrix, rotation_vector, key_command,
        )
        now = time.monotonic() if now is None else now
        raw = snapshot.axes
        centered = np.max(np.abs(raw)) <= self.deadzone and snapshot.buttons == 0
        fresh = np.repeat(now-snapshot.received <= .25, 3)
        stale_motion = np.any((np.abs(raw) > self.deadzone) & ~fresh)
        if not focused or snapshot.error or stale_motion:
            if self.moving:
                hold_current_pose(env)
            self.reset()
            self.status = snapshot.error or ('Click MuJoCo window, then center knob' if not focused
                                             else 'No fresh motion; center knob to resume')
            return env.hold_action()
        if not self.armed:
            if centered:
                self.armed = True
                self.status = 'Ready'
            return env.hold_action()
        for press in snapshot.presses:
            if press == 'hold':
                key_command(env, ord('C'))  # Same coordinate conversion as keyboard hold.
            else:
                env.command_gripper(press == 'close')
        axes = np.sign(raw)*np.maximum(np.abs(raw)-self.deadzone, 0)/(1-self.deadzone)
        axes *= np.array([1,-1,-1,-1,-1,1])*fresh
        translation = view_translation(limit_vector(axes[:3], 1.), azimuth)*self.speed
        world_rotation = view_translation(limit_vector(axes[3:], 1.), azimuth)*self.angular_speed
        rotation = env.data.site_xmat[env.tcp_site].reshape(3,3)
        local_rotation = rotation.T @ world_rotation
        moving = bool(np.any(translation) or np.any(local_rotation))
        if moving != self.moving:
            hold_current_pose(env)
        if moving:
            actual = env.data.site_xpos[env.tcp_site]
            env.target_xyz = np.clip(env.target_xyz, actual-.006, actual+.006)
            lead = rotation_vector(rotation.T @ env.target_rotation)
            env.target_rotation = rotation @ rotation_matrix(limit_vector(lead, .03))
        self.moving = moving
        self.status = 'Moving' if moving else 'Ready / centered'
        return np.r_[translation/env.cfg['sample_hz'], local_rotation/env.cfg['sample_hz'],
                     env.gripper_fraction(target=True)]


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Inspect SpaceMouse without moving any robot')
    parser.add_argument('--seconds', type=float, default=30.)
    args = parser.parse_args()
    with SpaceMouse() as mouse:
        print(json.dumps(mouse.info), flush=True)
        end = time.monotonic()+args.seconds
        while time.monotonic() < end:
            state = mouse.state.sample()
            print(json.dumps(dict(axes=state.axes.round(3).tolist(), buttons=state.buttons,
                                  presses=state.presses, error=state.error)), flush=True)
            time.sleep(.1)


if __name__ == '__main__':
    main()
