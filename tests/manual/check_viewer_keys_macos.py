"""In-process macOS event injection for our own viewer (no global input hooks)."""
import ctypes as C
import faulthandler
import os
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
FRANKA = '--franka' in sys.argv
PICK = '--pick' in sys.argv
objc = C.CDLL('/usr/lib/libobjc.A.dylib')
objc.objc_getClass.restype = C.c_void_p
objc.objc_getClass.argtypes = [C.c_char_p]
objc.sel_registerName.restype = C.c_void_p
objc.sel_registerName.argtypes = [C.c_char_p]
class Point(C.Structure):
    _fields_ = [('x', C.c_double), ('y', C.c_double)]

def send(obj, selector, restype=C.c_void_p, types=(), args=()):
    call = C.CFUNCTYPE(restype, C.c_void_p, C.c_void_p, *types)(C.cast(objc.objc_msgSend, C.c_void_p).value)
    return call(obj, objc.sel_registerName(selector.encode()), *args)

def key(char, code):
    app = send(objc.objc_getClass(b'NSApplication'), 'sharedApplication')
    windows = send(app, 'windows')
    assert send(windows, 'count', restype=C.c_ulong) > 0, 'Viewer window is not ready'
    window = send(windows, 'objectAtIndex:', types=(C.c_ulong,), args=(0,))
    number = send(window, 'windowNumber', restype=C.c_long)
    string = send(objc.objc_getClass(b'NSString'), 'stringWithUTF8String:', types=(C.c_char_p,), args=(char.encode(),))
    for event_type in (10, 11):
        event = send(objc.objc_getClass(b'NSEvent'),
            'keyEventWithType:location:modifierFlags:timestamp:windowNumber:context:characters:charactersIgnoringModifiers:isARepeat:keyCode:',
            types=(C.c_ulong, Point, C.c_ulong, C.c_double, C.c_long, C.c_void_p, C.c_void_p, C.c_void_p, C.c_bool, C.c_ushort),
            args=(event_type, Point(0,0), 0, 0., number, None, string, string, False, code))
        send(app, 'postEvent:atStart:', restype=None, types=(C.c_void_p,C.c_bool), args=(event,False))

ready = threading.Event()

def sequence():
    if not ready.wait(30):
        return
    time.sleep(.5)
    keys = [('p',35),('s',1),('v',9),('a',0),('d',2),('l',37),('b',11),('p',35),('l',37),('b',11),('v',9)] if FRANKA else [('p',35),('v',9),('l',37),('2',19),('p',35),('l',37),('3',20),('4',21),('5',23),('1',18),('z',6),('v',9)]
    if PICK:
        keys = [('[',33),('v',9),(']',30),('\x1b',53),('d',2),('a',0),('e',14),('q',12),('x',7),('z',6),('.',47),(',',43),('r',15),('\x7f',51),('f',3)]
    for char, code in keys:
        print('INJECT',char,flush=True)
        key(char,code)
        time.sleep(1.2)


print('PID',os.getpid(),flush=True)
threading.Thread(target=sequence,daemon=True).start()
import mujoco.viewer
from wowskin_lab.ui.keyboard import KeyState
original = mujoco.viewer.launch_passive
seen = []
dispatched = []
original_event = KeyState.event
def tracked_event(self, code, down, repeat=False):
    consumed = original_event(self, code, down, repeat)
    if consumed and down and not repeat:
        dispatched.append(code)
        print('CONSUMED', code, flush=True)
    return consumed
KeyState.event = tracked_event
def launch(*args, **kwargs):
    callback = kwargs['key_callback']
    def tracked(code):
        seen.append(code)
        dispatched.append(code)
        print('CALLBACK',code,flush=True)
        callback(code)
    kwargs['key_callback'] = tracked
    handle = original(*args, **kwargs)
    faulthandler.dump_traceback_later(40 if PICK else (35 if FRANKA else 22))
    ready.set()
    return handle
mujoco.viewer.launch_passive = launch
if PICK:
    sys.argv = ['pick_cube.py','viewer','--seconds','28','--output','outputs/diagnostics/pick_gui']
    from wowskin_lab.envs import pick_cube
    from wowskin_lab.ui.tactile import TactileHUD
    panels = []
    original_show = TactileHUD.show
    def tracked_show(self, handle, row, probability=None):
        original_show(self, handle, row, probability)
        panels.append((self.full_scale_ut, row['time_s'], int(handle.cam.type), dispatched[-1] if dispatched else None,
                       handle.opt.flags.copy(), handle.user_scn.flags.copy()))
    TactileHUD.show = tracked_show
    result = pick_cube.main()
    assert not result['faults'], result
    assert len(panels) > 20
    assert any(p[0] == 5. for p in panels) and any(p[0] == 10. for p in panels)
    for code in (ord('['), ord(']')):
        assert any(p[2] == int(mujoco.mjtCamera.mjCAMERA_FIXED) and p[3] == code for p in panels), panels
    for code in (ord('V'), 256):
        assert any(p[2] == int(mujoco.mjtCamera.mjCAMERA_FREE) and p[3] == code for p in panels), panels
    import numpy as np
    assert all(np.array_equal(p[4], panels[0][4]) and np.array_equal(p[5], panels[0][5]) for p in panels)
    assert not (set(seen) & set(map(ord, 'VDAEQXZRF,.'))), seen
    print('NATIVE_SHORTCUTS_CONSUMED_PASS', flush=True)
    print('CAMERA_RECOVERY_PASS', flush=True)
    print('TACTILE_PANEL_PASS',len(panels),flush=True)
elif FRANKA:
    sys.argv = ['franka_sensor.py','viewer','--pollen','--seconds','22']
    from wowskin_lab.envs import franka as franka_sensor
    readings = []
    passive_readings = []
    original_reading = franka_sensor.FrankaTactileSim.reading
    def tracked_reading(self):
        row = original_reading(self)
        readings.append((self.mode, row['normal_force_n'], row['slip']))
        passive_readings.append((self.probe_side, self.mode, row['passive_normal_force_n']))
        return row
    franka_sensor.FrankaTactileSim.reading = tracked_reading
    franka_sensor.main()
    assert max(force for mode, force, slip in readings if mode == 'press') > .01
    assert any(slip for mode, force, slip in readings if mode == 'slide')
    assert readings[-1][0] == 'off' and readings[-1][1] < 1e-4
    assert max(force for side, mode, force in passive_readings if side == 'right' and mode == 'press') > .01
    assert passive_readings[-1][0] == 'left' and passive_readings[-1][2] < 1e-4
else:
    sys.argv = ['run.py','viewer','--seconds','19']
    from wowskin_lab.tools import sensor as run
    run.main()
faulthandler.cancel_dump_traceback_later()
assert all(ord(c) in dispatched for c in ('VDAEQXZRF[],.' if PICK else ('PSVLADB' if FRANKA else 'PVL12345Z'))),dispatched
print('KEYBOARD_SMOKE_PASS',dispatched,flush=True)
