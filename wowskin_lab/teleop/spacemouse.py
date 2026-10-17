"""USB SpaceMouse input for simulation; never sends commands to a real robot.

HID protocol: 3Dconnexion report 1 = six signed little-endian axes (or XYZ),
report 2 = rotation on older devices; report 3 = button bits. Status/battery
reports do not refresh motion. Axis convention follows robosuite's device
decoder, with our camera-plane transform applied separately.

Linux may read the same raw axes from spacenavd (libspnav protocol 1) when the
daemon has grabbed hidraw. Those events are turned back into HID reports so
scaling, signs and buttons stay on one path. Full scale remains 350.
"""
from collections import deque
from dataclasses import dataclass
import ctypes
import threading
import time

import numpy as np

# Raw HID -> operator frame (screen right, tabletop away, world up).
# Positive raw Rx is forward cap tilt on the connected Wireless BT. A
# downward-facing tool must pitch away from the operator, so use +Rx here.
# This is the user's control convention, not a robot or sensor calibration.
HID_AXIS_SIGNS = np.array([1., -1., -1., 1., -1., 1.])
_SPACEMOUSE_VENDORS = (0x256f, 0x046d)
# libspnav v1: RAWAXIS=5, RAWBUTTON=6, DEV=3, DEV_RM=1.
# Mask 0x34 is DEV | RAWAXIS | RAWBUTTON, skipping the daemon's axis map.
_SPNAV_RAW_AXIS = 5
_SPNAV_RAW_BUTTON = 6
_SPNAV_DEVICE = 3
_SPNAV_DEVICE_REMOVE = 1
_SPNAV_RAW_MASK = 0x34


def apply_spacenav_event(kind, a, b, raw, buttons):
    """Apply one libspnav event. ``raw`` is six little-endian int16 device units.

    The first three ints of ``spnav_event`` are (type, a, b): for a raw axis,
    index and value; for a raw button, press and button number; for a device
    event, operation and id. Non-raw motion is ignored so a transformed daemon
    event cannot fight the raw value. Returns (buttons, removed, button_changed).
    """
    kind, a, b = int(kind), int(a), int(b)
    buttons = int(buttons)
    if kind == _SPNAV_RAW_AXIS and 0 <= a < 6:
        raw[a] = max(-32768, min(32767, b))
        return buttons, False, False
    if kind == _SPNAV_RAW_BUTTON and b in (0, 1):
        bit = 1 << b
        buttons = (buttons | bit) if a else (buttons & ~bit)
        return buttons & 3, False, True
    if kind == _SPNAV_DEVICE and a == _SPNAV_DEVICE_REMOVE:
        return buttons, True, False
    return buttons, False, False


def spacenav_reports(events, raw, buttons):
    """Fold raw libspnav events into the HID reports MouseState already decodes.

    Absolute axes are reported only when they change, so the caller also invokes
    this with an empty event list to keep a held deflection inside the 0.25 s
    freshness window. A device-remove event raises ConnectionError.
    """
    reports = []
    for kind, a, b in events:
        buttons, removed, changed = apply_spacenav_event(kind, a, b, raw, buttons)
        if removed:
            raise ConnectionError('SpaceMouse removed')
        if changed:
            reports.append(bytes((3, buttons & 0xFF)))
    reports.append(bytes((1,)) + np.asarray(raw, dtype='<i2').tobytes())
    return buttons, reports


def _load_spnav():
    try:
        lib = ctypes.CDLL('libspnav.so.0')
    except OSError:
        return None
    signatures = {
        'spnav_open': [],
        'spnav_close': [],
        'spnav_fd': [],
        'spnav_protocol': [],
        'spnav_evmask': [ctypes.c_uint],
        'spnav_dev_usbid': [ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)],
        'spnav_dev_name': [ctypes.c_char_p, ctypes.c_int],
        'spnav_poll_event': [ctypes.c_void_p],
    }
    for name, args in signatures.items():
        fn = getattr(lib, name, None)
        if fn is None:
            return None
        fn.argtypes = args
        fn.restype = ctypes.c_int
    return lib


def _spnav_peer_closed(fd):
    """True when the spacenavd socket is readable because the peer hung up.

    MSG_PEEK | MSG_DONTWAIT does not consume bytes libspnav still has to read.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    recv = libc.recv
    recv.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    recv.restype = ctypes.c_ssize_t
    buf = ctypes.create_string_buffer(1)
    n = recv(fd, buf, 1, 0x02 | 0x40)  # MSG_PEEK | MSG_DONTWAIT on Linux
    if n == 0:
        return True
    if n > 0:
        return False
    return ctypes.get_errno() not in (4, 11, 35)  # EINTR / EAGAIN / EWOULDBLOCK


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
    """Single reader; retains only the most recent axes, not a motion queue.

    macOS opens the HID interface and seizes it so the knob does not move the
    desktop pointer. Linux prefers spacenavd when that daemon is running,
    because it usually owns hidraw; otherwise it uses the same HID reports.
    """
    def __init__(self):
        self.state = MouseState()
        self.stop = threading.Event()
        self.thread = None
        self.device = None
        self.info = None
        self._spnav_fd = -1

    def __enter__(self):
        import sys
        linux = sys.platform.startswith('linux')
        if linux and self._open_spacenavd():
            return self
        import hid
        # Multi-axis controller usage excludes duplicate vendor interfaces.
        devices = {d['path']: d for d in hid.enumerate()
                   if d['vendor_id'] in _SPACEMOUSE_VENDORS
                   and 'spacemouse' in (d.get('product_string') or '').lower()
                   and d.get('usage_page') == 1 and d.get('usage') == 8}
        if len(devices) == 1:
            path, description = next(iter(devices.items()))
            try:
                self._open_hid(hid, path, description)
            except Exception:
                self._release_open_device()
                raise
            return self
        extra = ''
        if linux:
            extra = (' On Linux, start spacenavd when it owns the device (libspnav.so.0, '
                     '/var/run/spnav.sock), or allow this user to open hidraw.')
        raise RuntimeError(f'Expected one USB SpaceMouse, found {len(devices)}. Connect its USB data cable.{extra}')

    def _open_hid(self, hid, path, description):
        import sys
        self.info = {k: description.get(k) for k in ('product_string', 'vendor_id', 'product_id')}
        self.info['backend'] = 'hid'
        # macOS otherwise may route the same relative-axis reports to the
        # desktop pointer. Explicitly seize only this SpaceMouse while open.
        if sys.platform == 'darwin':
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

    def _read(self):
        try:
            while not self.stop.is_set():
                report = self.device.read(64, 50)
                self.state.feed(report)
        except Exception as exc:
            with self.state.lock:
                self.state.error = f'SpaceMouse disconnected: {exc}'

    def _open_spacenavd(self):
        """Use spacenavd raw events when the daemon socket is actually open."""
        lib = _load_spnav()
        if lib is None or lib.spnav_open() == -1:
            return False
        opened = False
        try:
            if lib.spnav_protocol() < 1 or lib.spnav_evmask(_SPNAV_RAW_MASK) == -1:
                return False
            vendor, product = ctypes.c_uint(), ctypes.c_uint()
            if lib.spnav_dev_usbid(ctypes.byref(vendor), ctypes.byref(product)) == -1:
                return False
            if vendor.value not in _SPACEMOUSE_VENDORS:
                raise RuntimeError(f'spacenavd device vendor 0x{vendor.value:04x} is not a SpaceMouse')
            name = ctypes.create_string_buffer(256)
            label = 'SpaceMouse'
            if lib.spnav_dev_name(name, len(name)) != -1 and name.value:
                label = name.value.decode(errors='replace')
            fd = lib.spnav_fd()
            if fd < 0:
                return False
            self.device = lib
            self._spnav_fd = fd
            self.info = dict(product_string=label, vendor_id=int(vendor.value), product_id=int(product.value),
                             backend='spacenavd')
            self.thread = threading.Thread(target=self._read_spacenavd, name='SpaceMouse-spnav', daemon=True)
            self.thread.start()
            opened = True
            return True
        finally:
            if not opened:
                self.device = None
                self.thread = None
                self.info = None
                self._spnav_fd = -1
                lib.spnav_close()

    def _read_spacenavd(self):
        import select
        lib = self.device
        event = (ctypes.c_int * 64)()
        raw, buttons = np.zeros(6, dtype='<i2'), 0
        fd = self._spnav_fd
        try:
            while not self.stop.is_set():
                try:
                    ready, _, _ = select.select([fd], [], [], 0.05)
                except InterruptedError:
                    continue
                if ready and _spnav_peer_closed(fd):
                    raise ConnectionError('spacenavd closed the connection')
                events = []
                for _ in range(64):
                    kind = lib.spnav_poll_event(ctypes.addressof(event))
                    if not kind:
                        break
                    events.append((int(kind), int(event[1]), int(event[2])))
                buttons, reports = spacenav_reports(events, raw, buttons)
                for report in reports:
                    self.state.feed(report)
        except Exception as exc:
            with self.state.lock:
                self.state.error = f'SpaceMouse disconnected: {exc}'

    def _release_open_device(self):
        device, info = self.device, self.info
        self.device = None
        self.info = None
        self._spnav_fd = -1
        if device is None:
            return
        if (info or {}).get('backend') == 'spacenavd':
            device.spnav_close()
        else:
            device.close()

    def __exit__(self, *exc):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1.)
        self._release_open_device()


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

    def settings(self, frame='view'):
        return dict(speed_m_s=self.speed, angular_speed_rad_s=self.angular_speed,
                    deadzone=self.deadzone, timeout_s=.25, full_scale=350,
                    hid_axis_signs=HID_AXIS_SIGNS.tolist(),
                    translation_frame='robot base XYZ' if frame == 'base' else 'camera ground plane + world Z',
                    rotation_frame=('robot base XYZ' if frame == 'base' else 'camera ground plane + world Z')+', converted to TCP-local rotvec',
                    pitch_convention=('Forward cap tilt pitches a downward-facing tool toward base +X; independent of the preview'
                                      if frame == 'base' else 'Forward cap tilt pitches a downward-facing tool away from the viewer about screen right'),
                    buttons='left close; right open; both hold')

    def action(self, env, snapshot, azimuth, focused=True, now=None):
        from wowskin_lab.envs.pick_cube import (
            hold_current_pose, keyboard_translation, limit_vector, rotation_matrix, rotation_vector, key_command, WORKSPACE,
        )
        from wowskin_lab.teleop.target_limits import straight_step, workspace_bounds
        now = time.monotonic() if now is None else now
        raw = snapshot.axes
        centered = np.max(np.abs(raw)) <= self.deadzone and snapshot.buttons == 0
        fresh = np.repeat(now-snapshot.received <= .25, 3)
        stale_motion = np.any((np.abs(raw) > self.deadzone) & ~fresh)
        if not focused or snapshot.error or stale_motion:
            if self.moving or env.cfg.get('disable_cartesian_limits', False):
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
        axes *= HID_AXIS_SIGNS*fresh
        translation = keyboard_translation(env, limit_vector(axes[:3], 1.), azimuth)*self.speed
        world_rotation = keyboard_translation(env, limit_vector(axes[3:], 1.), azimuth, rotation=True)*self.angular_speed
        rotation = env.data.site_xmat[env.tcp_site].reshape(3,3)
        local_rotation = rotation.T @ world_rotation
        moving = bool(np.any(translation) or np.any(local_rotation))
        limited = not env.cfg.get('disable_cartesian_limits', False)
        if limited and moving != self.moving:
            hold_current_pose(env)
        if limited and moving:
            actual = env.data.site_xpos[env.tcp_site]
            translation = straight_step(env.target_xyz, translation/env.cfg['sample_hz'], actual,
                                        workspace_bounds(env, WORKSPACE))*env.cfg['sample_hz']
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
