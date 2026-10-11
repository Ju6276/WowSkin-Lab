"""Window-local key down/up input for MuJoCo's macOS passive viewer.

AppKit consumes application shortcuts before GLFW/MuJoCo sees them. No global
keyboard listener, event injection, or accessibility permission is involved.
"""
from collections import deque
from dataclasses import dataclass
import sys
import threading

import numpy as np

ESC, ENTER, BACKSPACE = 256, 257, 259
RIGHT, LEFT, DOWN, UP, PAGE_UP, PAGE_DOWN = 262, 263, 264, 265, 266, 267
MOVE_AXES = {
    ord('W'): (0, 1, 0), UP: (0, 1, 0),
    ord('S'): (0, -1, 0), DOWN: (0, -1, 0),
    ord('D'): (1, 0, 0), RIGHT: (1, 0, 0),
    ord('A'): (-1, 0, 0), LEFT: (-1, 0, 0),
    ord('E'): (0, 0, 1), PAGE_UP: (0, 0, 1),
    ord('Q'): (0, 0, -1), PAGE_DOWN: (0, 0, -1),
}


def direction(held, axes=MOVE_AXES):
    # Aliases do not double the speed; diagonals keep the same total speed.
    vectors = {axes[k] for k in held if k in axes}
    value = np.sum(list(vectors), axis=0) if vectors else np.zeros(3)
    return value/max(1., np.linalg.norm(value))


@dataclass(frozen=True)
class KeySnapshot:
    commands: tuple
    held: frozenset


class KeyState:
    def __init__(self, reserved):
        self.reserved = set(reserved)
        self._lock = threading.Lock()
        self._held = set()
        self._commands = deque()

    def event(self, key, down, repeat=False):
        if key not in self.reserved:
            return False
        with self._lock:
            if down and not repeat and key not in self._held:
                self._held.add(key)
                self._commands.append(key)
            elif not down:
                self._held.discard(key)
        return True

    def clear(self):
        with self._lock:
            self._held.clear()
            self._commands.clear()

    def discard_busy_input(self):
        """No queued B/Enter/motion after save; preserve a requested exit."""
        with self._lock:
            quit_requested = ord('P') in self._commands
            self._held.clear()
            self._commands.clear()
            if quit_requested:
                self._commands.append(ord('P'))

    def peek(self):
        """Observe pending commands without consuming key-down events."""
        with self._lock:
            return KeySnapshot(tuple(self._commands), frozenset(self._held))

    def sample(self):
        with self._lock:
            result = KeySnapshot(tuple(self._commands), frozenset(self._held))
            self._commands.clear()
        return result


class ViewerKeyboard(KeyState):
    """Use as a context inside launch_passive, while the UI loop is alive.

    On non-macOS the existing native callback remains a single-step fallback.
    This fallback does not claim to provide held keys or consume native events.
    """
    def __init__(self, reserved, title_contains):
        super().__init__(reserved)
        self.title_contains = title_contains
        self.continuous = sys.platform == 'darwin'
        self._active = False
        self.focused = False
        self._monitor = None
        self._observers = []
        self._captured = {}
        self.click_regions = ()
        self._mouse_captured = False

    def click(self, x, y):
        """Normalized content coordinates, origin bottom left; queue only."""
        for left, bottom, right, top, key in self.click_regions:
            if left <= x <= right and bottom <= y <= top:
                self.event(key, True)
                self.event(key, False)
                return True
        return False

    def native_callback(self, code):
        if not self.continuous:
            self.event(code, True)
            self.event(code, False)

    def _ui_call(self, function):
        from Foundation import NSThread
        from PyObjCTools import AppHelper
        if NSThread.isMainThread():
            return function()
        done, failures = threading.Event(), []
        def invoke():
            try:
                function()
            except Exception as exc:
                failures.append(exc)
            finally:
                done.set()
        AppHelper.callAfter(invoke)
        if not done.wait(5):
            raise RuntimeError('MuJoCo UI thread did not respond to keyboard setup/cleanup')
        if failures:
            raise failures[0]

    def __enter__(self):
        if self.continuous:
            self._ui_call(self._install)
        return self

    def _install(self):
        import AppKit as A
        from Foundation import NSNotificationCenter
        windows = [w for w in A.NSApplication.sharedApplication().windows()
                   if self.title_contains in str(w.title())]
        if len(windows) != 1:
            raise RuntimeError(f'Expected one {self.title_contains!r} window, found {len(windows)}')
        self._window = windows[0]
        self._active = True
        specials = {53: ESC, 36: ENTER, 76: ENTER, 51: BACKSPACE, 123: LEFT, 124: RIGHT,
                    125: DOWN, 126: UP, 116: PAGE_UP, 121: PAGE_DOWN}
        def monitor(event):
            if not self._active or event.window() != self._window:
                return event
            kind = event.type()
            if kind in (A.NSEventTypeLeftMouseDown, A.NSEventTypeLeftMouseUp, A.NSEventTypeLeftMouseDragged):
                if kind == A.NSEventTypeLeftMouseDown:
                    view = self._window.contentView()
                    p = view.convertPoint_fromView_(event.locationInWindow(), None)
                    size = view.bounds().size
                    self._mouse_captured = self.click(p.x/size.width, p.y/size.height)
                captured = self._mouse_captured
                if kind == A.NSEventTypeLeftMouseUp: self._mouse_captured = False
                return None if captured else event
            scan = int(event.keyCode())
            modifiers = event.modifierFlags()
            system_modifiers = A.NSEventModifierFlagCommand | A.NSEventModifierFlagControl | A.NSEventModifierFlagOption
            if kind == A.NSEventTypeFlagsChanged:
                if modifiers & system_modifiers:
                    self.clear()
                return event  # Preserve native Ctrl+mouse perturbations.
            if kind == A.NSEventTypeKeyUp and scan in self._captured:
                self.event(self._captured.pop(scan), False)
                return None
            if modifiers & system_modifiers:
                return event  # Preserve e.g. Cmd+Q and Ctrl+Q.
            chars = str(event.charactersIgnoringModifiers() or '').upper()
            key = specials.get(scan, ord(chars) if len(chars) == 1 else -1)
            if key not in self.reserved:
                return event
            if kind == A.NSEventTypeKeyDown:
                self._captured[scan] = key
                self.event(key, True, bool(event.isARepeat()))
                return None  # Consume BEFORE GLFW and MuJoCo's display shortcuts.
            return event
        self._handler = monitor  # Keep Python blocks alive until removal.
        self._monitor = A.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
            A.NSEventMaskKeyDown | A.NSEventMaskKeyUp | A.NSEventMaskFlagsChanged |
            A.NSEventMaskLeftMouseDown | A.NSEventMaskLeftMouseUp | A.NSEventMaskLeftMouseDragged, monitor)
        if self._monitor is None:
            raise RuntimeError('Could not install the window-local keyboard handler')
        center = NSNotificationCenter.defaultCenter()
        def focus_changed(notification=None):
            self.focused = bool(self._window.isKeyWindow() and A.NSApplication.sharedApplication().isActive())
            if not self.focused:
                self.clear()
        self._focus_handler = focus_changed
        focus_changed()
        for name, obj in ((A.NSWindowDidResignKeyNotification, self._window),
                          (A.NSApplicationDidResignActiveNotification, None),
                          (A.NSWindowDidBecomeKeyNotification, self._window),
                          (A.NSApplicationDidBecomeActiveNotification, None)):
            self._observers.append(center.addObserverForName_object_queue_usingBlock_(
                name, obj, None, self._focus_handler))
        def closing(notification):
            # On manual window close the mjpython UI loop can stop before the
            # worker exits its context. Remove on the UI thread while alive.
            self._active = False
            self.focused = False
            self.clear()
            self._remove()
        self._close_handler = closing
        self._observers.append(center.addObserverForName_object_queue_usingBlock_(
            A.NSWindowWillCloseNotification, self._window, None, closing))

    def __exit__(self, *exc):
        self._active = False
        self.focused = False
        self.clear()
        if self.continuous and self._monitor is not None:
            self._ui_call(self._remove)

    def _remove(self):
        import AppKit as A
        from Foundation import NSNotificationCenter
        if self._monitor is not None:
            A.NSEvent.removeMonitor_(self._monitor)
        for observer in self._observers:
            NSNotificationCenter.defaultCenter().removeObserver_(observer)
        self._observers.clear()
        self._monitor = None
        self._captured.clear()
