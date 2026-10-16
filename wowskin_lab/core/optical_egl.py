"""Keep the shared EGL display alive while individual TACTO contexts close.

Pyrender terminates the device display when deleting one renderer. Both fingers
and MuJoCo can share that display, so defer termination until process exit.
Only the owned platform instance is adapted; upstream rendering is unchanged.
"""
import atexit
import ctypes
from types import MethodType

_displays = set()


def share_display(offscreen):
    from OpenGL import EGL
    from pyrender.platforms.egl import EGLPlatform
    platform = offscreen._platform
    if not isinstance(platform, EGLPlatform):
        return
    display = platform._egl_display
    key = ctypes.cast(display, ctypes.c_void_p).value
    if key not in _displays:
        _displays.add(key)
        atexit.register(EGL.eglTerminate, display)

    def delete_context(self):
        if self._egl_context is not None:
            EGL.eglMakeCurrent(self._egl_display, EGL.EGL_NO_SURFACE,
                               EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
            EGL.eglDestroyContext(self._egl_display, self._egl_context)
            self._egl_context = None
        self._egl_display = None

    platform.delete_context = MethodType(delete_context, platform)
