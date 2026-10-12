"""Native macOS offscreen context for TACTO alongside MuJoCo's viewer.

Pyglet creates an NSWindow, which cannot run on mjpython's simulation thread.
CGL creates no window. Only context/scene initialization is specialized here;
TACTO lighting, rasterization and force-dependent pose adjustment are inherited.
The scene initialization below is adapted from TACTO (MIT; see
assets/reference/tacto/LICENSE). No global monkeypatch or source rewrite.
"""
import ctypes

import pyrender
from pyrender.platforms.base import Platform
from mujoco.cgl import cgl
from tacto import Renderer


class CGLPlatform(Platform):
    def init_context(self):
        self.context, self.pixel = cgl.CGLContextObj(), cgl.CGLPixelFormatObj()
        attribute = cgl.CGLPixelFormatAttribute
        values = [attribute.CGLPFAOpenGLProfile, cgl.CGLOpenGLProfile.CGLOGLPVersion_GL4_Core,
                  attribute.CGLPFAColorSize, 24, attribute.CGLPFAAlphaSize, 8,
                  attribute.CGLPFADepthSize, 24, attribute.CGLPFAAccelerated, 0]
        attributes = (ctypes.c_int*len(values))(*values)
        count = cgl.GLint()
        try:
            cgl.CGLChoosePixelFormat(attributes, ctypes.byref(self.pixel), ctypes.byref(count))
            cgl.CGLCreateContext(self.pixel, 0, ctypes.byref(self.context))
        except Exception:
            self.delete_context()
            raise

    def make_current(self):
        cgl.CGLSetCurrentContext(self.context)

    def make_uncurrent(self):
        cgl.CGLSetCurrentContext(None)

    def delete_context(self):
        if getattr(self, 'context', None):
            self.make_uncurrent()
            cgl.CGLReleaseContext(self.context)
            self.context = None
        if getattr(self, 'pixel', None):
            cgl.CGLReleasePixelFormat(self.pixel)
            self.pixel = None

    def supports_framebuffers(self):
        return True


class CGLOffscreenRenderer(pyrender.OffscreenRenderer):
    def _create(self):
        self._platform = CGLPlatform(self.viewport_width, self.viewport_height)
        self._platform.init_context()
        self._platform.make_current()
        self._renderer = pyrender.Renderer(self.viewport_width, self.viewport_height)


class CGLTactoRenderer(Renderer):
    def _init_pyrender(self):
        self.scene, self.scene_depth = pyrender.Scene(), pyrender.Scene()
        self.object_nodes, self.current_object_nodes = {}, {}
        self.current_light_nodes, self.cam_light_ids = [], []
        self._init_gel()
        self._init_camera()
        self._init_light()
        self.r = CGLOffscreenRenderer(self.width, self.height)
        colors, depths = self.render(object_poses=None, noise=False, calibration=False)
        self.depth0, self._background_sim = depths, colors
