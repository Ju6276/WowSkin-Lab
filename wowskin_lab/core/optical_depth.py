"""Orthographic depth readback for the pinned pyrender version.

That version always uses the perspective depth equation. For an orthographic
camera z = near + buffer*(far-near). Framebuffer plumbing adapted from pyrender
(MIT, copyright 2019 Matthew Matl; assets/reference/pyrender/LICENSE).
"""
import sys

import numpy as np
import pyrender
from OpenGL.GL import (glBindFramebuffer, glBlitFramebuffer, glReadPixels,
    GL_READ_FRAMEBUFFER, GL_DRAW_FRAMEBUFFER, GL_DEPTH_BUFFER_BIT, GL_NEAREST,
    GL_DEPTH_COMPONENT, GL_FLOAT)


class OrthographicDepthRenderer(pyrender.Renderer):
    def _read_main_framebuffer(self, scene, flags):
        camera = scene.main_camera_node.camera
        if not isinstance(camera, pyrender.OrthographicCamera) or not flags & pyrender.RenderFlags.DEPTH_ONLY:
            raise ValueError('This renderer supports orthographic depth only')
        width, height = self._main_fb_dims
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._main_fb_ms)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self._main_fb)
        glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._main_fb)
        buffer = glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT)
        raw = np.frombuffer(buffer, dtype=np.float32).reshape(height, width)[::-1]
        depth = np.where(raw < 1., camera.znear+raw*(camera.zfar-camera.znear), 0.)
        return self._resize_image(depth) if sys.platform == 'darwin' else depth


def depth_renderer(width, height):
    if sys.platform == 'darwin':
        from .optical_cgl import CGLOffscreenRenderer as Base
    else:
        Base = pyrender.OffscreenRenderer

    class Offscreen(Base):
        def _create(self):
            super()._create()
            self._renderer.delete()
            self._renderer = OrthographicDepthRenderer(self.viewport_width, self.viewport_height)

    return Offscreen(width, height)
