"""X11 focus and held keys for the Linux MuJoCo window.

GLFW owns one Display on the viewer thread. This module calls XInitThreads
before that display is opened, then uses a second Display from the control
thread. Wayland-only sessions do not take this path.
"""
import ctypes
import os
import sys
import time

_READY = False
_THREADS = False
_LIB = None

_XERROR = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)


def _ignore_x_error(display, event):
    return 0


_IGNORE = _XERROR(_ignore_x_error)

# Control, Alt, Super and Meta. Shift stays out: it does not change the keycode.
_MODIFIER_KEYSYMS = (0xffe3, 0xffe4, 0xffe9, 0xffea, 0xffeb, 0xffec, 0xffe7, 0xffe8)


def keysyms_for(code):
    """X keysyms that should report as one reserved ViewerKeyboard code."""
    from wowskin_lab.ui.keyboard import (
        BACKSPACE, DOWN, ENTER, ESC, LEFT, PAGE_DOWN, PAGE_UP, RIGHT, UP,
    )
    special = {
        ESC: (0xff1b,),
        ENTER: (0xff0d, 0xff8d),
        BACKSPACE: (0xff08,),
        RIGHT: (0xff53,),
        LEFT: (0xff51,),
        DOWN: (0xff54,),
        UP: (0xff52,),
        PAGE_UP: (0xff55,),
        PAGE_DOWN: (0xff56,),
    }
    if code in special:
        return special[code]
    if code == 32:
        return (0x20,)
    if 33 <= code <= 126:
        if 65 <= code <= 90:
            return (code, code + 32)
        return (code,)
    return ()


def keymap_bits(keymap):
    """Keycodes whose bit is set in a 32-byte XQueryKeymap buffer."""
    data = bytes(keymap[:32])
    if len(data) < 32:
        data = data.ljust(32, b'\x00')
    return {code for code in range(256) if data[code // 8] & (1 << (code % 8))}


def held_keys(down, code_to_key, modifier_codes):
    """Reserved key codes that are down, or nothing while a shortcut modifier is down."""
    down = set(down)
    if set(modifier_codes) & down:
        return set()
    return {code_to_key[code] for code in down if code in code_to_key}


def title_matches(names, needle):
    return any(bool(needle) and needle in name for name in names)


def x11_is_ready():
    return _READY


def _library():
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL('libX11.so.6')
        lib.XInitThreads.argtypes = []
        lib.XInitThreads.restype = ctypes.c_int
        lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
        lib.XOpenDisplay.restype = ctypes.c_void_p
        lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
        lib.XCloseDisplay.restype = ctypes.c_int
        lib.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.XSync.restype = ctypes.c_int
        lib.XSetErrorHandler.argtypes = [ctypes.c_void_p]
        lib.XSetErrorHandler.restype = ctypes.c_void_p
        lib.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.XQueryKeymap.restype = ctypes.c_int
        lib.XGetInputFocus.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_int)]
        lib.XGetInputFocus.restype = ctypes.c_int
        lib.XQueryTree.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)), ctypes.POINTER(ctypes.c_uint)]
        lib.XQueryTree.restype = ctypes.c_int
        lib.XFree.argtypes = [ctypes.c_void_p]
        lib.XFree.restype = ctypes.c_int
        lib.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        lib.XKeysymToKeycode.restype = ctypes.c_ubyte
        lib.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        lib.XInternAtom.restype = ctypes.c_ulong
        lib.XFetchName.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_char_p)]
        lib.XFetchName.restype = ctypes.c_int
        lib.XGetWindowProperty.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_long, ctypes.c_long, ctypes.c_int,
            ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte))]
        lib.XGetWindowProperty.restype = ctypes.c_int
        _LIB = lib
    return _LIB


def prepare_x11_threads():
    """Make later Xlib calls thread-safe and ask GLFW for an X11 window.

    XInitThreads has to be the first Xlib call in the process, so this runs
    before launch_passive. A missing display leaves GLFW on its default platform.
    """
    global _READY, _THREADS
    if not sys.platform.startswith('linux'):
        return False
    if _READY:
        return True
    if not os.environ.get('DISPLAY'):
        return False
    try:
        lib = _library()
    except OSError:
        return False
    if not _THREADS:
        if not lib.XInitThreads():
            return False
        _THREADS = True
    display = lib.XOpenDisplay(None)
    if not display:
        return False
    lib.XCloseDisplay(display)
    import glfw
    glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
    _READY = True
    return True


class X11WindowInput:
    """Poll whether the named MuJoCo window is focused, and which keys are down."""

    def __init__(self, title_contains, reserved):
        self.title_contains = title_contains
        self.lib = _library()
        self.display = self.lib.XOpenDisplay(None)
        self._code_to_key = {}
        self._modifiers = set()
        self._cached_window = None
        self._cached_focused = False
        if not self.display:
            raise RuntimeError('XOpenDisplay failed')
        try:
            self._net_wm_name = self.lib.XInternAtom(self.display, b'_NET_WM_NAME', False)
            self._utf8 = self.lib.XInternAtom(self.display, b'UTF8_STRING', False)
            self._build_keymap(reserved)
            self._wait_for_window()
        except Exception:
            self.close()
            raise

    def close(self):
        display, self.display = self.display, None
        if display:
            self.lib.XCloseDisplay(display)

    def state(self):
        if not self.display:
            return False, set()
        try:
            focused, pressed = self._guard(self._read_state)
        except Exception:
            return False, set()
        return focused, pressed

    def _build_keymap(self, reserved):
        for code in reserved:
            for sym in keysyms_for(code):
                keycode = int(self.lib.XKeysymToKeycode(self.display, sym))
                if keycode:
                    self._code_to_key[keycode] = code
        for sym in _MODIFIER_KEYSYMS:
            keycode = int(self.lib.XKeysymToKeycode(self.display, sym))
            if keycode:
                self._modifiers.add(keycode)

    def _wait_for_window(self):
        deadline = time.monotonic() + 0.6
        while True:
            if self._guard(self._contains_title):
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f'No X11 window title contains {self.title_contains!r}')
            time.sleep(0.05)

    def _read_state(self):
        window = self._focus_window()
        if window != self._cached_window:
            self._cached_window = window
            self._cached_focused = title_matches(self._names_around_focus(window), self.title_contains) if window else False
        keymap = ctypes.create_string_buffer(32)
        self.lib.XQueryKeymap(self.display, keymap)
        pressed = held_keys(keymap_bits(keymap.raw), self._code_to_key, self._modifiers) if self._cached_focused else set()
        return self._cached_focused, pressed

    def _focus_window(self):
        window = ctypes.c_ulong()
        revert = ctypes.c_int()
        if not self.lib.XGetInputFocus(self.display, ctypes.byref(window), ctypes.byref(revert)):
            return 0
        current = int(window.value)
        return 0 if current <= 1 else current

    def _contains_title(self):
        # C callers use a macro. libX11 also exports the function for everyone else.
        root_fn = getattr(self.lib, 'XDefaultRootWindow', None)
        if root_fn is None:
            raise RuntimeError('libX11 is missing XDefaultRootWindow, so the MuJoCo window cannot be found')
        root_fn.argtypes = [ctypes.c_void_p]
        root_fn.restype = ctypes.c_ulong
        root = int(root_fn(self.display))
        pending, seen = [root], set()
        while pending and len(seen) < 400:
            window = pending.pop()
            if window <= 1 or window in seen:
                continue
            seen.add(window)
            if title_matches(self._names(window), self.title_contains):
                return True
            pending.extend(self._children(window))
        return False

    def _names_around_focus(self, window):
        names, seen, current = [], set(), window
        for _ in range(16):
            if current <= 1 or current in seen:
                break
            seen.add(current)
            names.extend(self._names(current))
            parent, _ = self._tree(current)
            current = parent
        self._collect(window, names, 0, [32])
        return names

    def _collect(self, window, names, depth, budget):
        if depth > 3 or window <= 1 or budget[0] <= 0:
            return
        budget[0] -= 1
        names.extend(self._names(window))
        for child in self._children(window):
            self._collect(child, names, depth + 1, budget)

    def _children(self, window):
        _, children = self._tree(window)
        return children

    def _tree(self, window):
        root = ctypes.c_ulong()
        parent = ctypes.c_ulong()
        children = ctypes.POINTER(ctypes.c_ulong)()
        count = ctypes.c_uint()
        if not self.lib.XQueryTree(self.display, ctypes.c_ulong(window), ctypes.byref(root), ctypes.byref(parent),
                                   ctypes.byref(children), ctypes.byref(count)):
            return 0, []
        try:
            kids = [int(children[i]) for i in range(int(count.value))] if children else []
        finally:
            if children:
                self.lib.XFree(ctypes.cast(children, ctypes.c_void_p))
        return int(parent.value), kids

    def _names(self, window):
        names = []
        fetched = ctypes.c_char_p()
        if self.lib.XFetchName(self.display, ctypes.c_ulong(window), ctypes.byref(fetched)) and fetched:
            if fetched.value:
                names.append(fetched.value.decode('utf-8', 'replace'))
            self.lib.XFree(ctypes.cast(fetched, ctypes.c_void_p))
        net = self._utf8_name(window)
        if net and net not in names:
            names.append(net)
        return names

    def _utf8_name(self, window):
        actual_type = ctypes.c_ulong()
        actual_format = ctypes.c_int()
        nitems = ctypes.c_ulong()
        bytes_after = ctypes.c_ulong()
        prop = ctypes.POINTER(ctypes.c_ubyte)()
        status = self.lib.XGetWindowProperty(
            self.display, ctypes.c_ulong(window), self._net_wm_name, 0, 1024, False, self._utf8,
            ctypes.byref(actual_type), ctypes.byref(actual_format), ctypes.byref(nitems),
            ctypes.byref(bytes_after), ctypes.byref(prop))
        if status != 0 or not prop:
            if prop:
                self.lib.XFree(ctypes.cast(prop, ctypes.c_void_p))
            return ''
        try:
            if actual_format.value != 8 or not nitems.value:
                return ''
            raw = ctypes.string_at(prop, int(nitems.value))
            return raw.split(b'\x00', 1)[0].decode('utf-8', 'replace')
        finally:
            self.lib.XFree(ctypes.cast(prop, ctypes.c_void_p))

    def _guard(self, fn):
        previous = self.lib.XSetErrorHandler(ctypes.cast(_IGNORE, ctypes.c_void_p))
        try:
            return fn()
        finally:
            try:
                if self.display:
                    self.lib.XSync(self.display, False)
            finally:
                self.lib.XSetErrorHandler(previous)
