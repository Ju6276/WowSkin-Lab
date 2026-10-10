"""Top view rendered from the current MuJoCo state, with projected chip XY."""
import time
import mujoco
import numpy as np
from .channel_alignment import OFFICIAL_TO_REFERENCE
from .manufacturer import draw, pygame


class SensorModelView:
    def __init__(self, sim):
        self.sim = sim
        self.renderer = mujoco.Renderer(sim.model, height=422, width=400)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0, 0, sim.cfg['total_height_m']]
        self.camera.distance = .080
        self.camera.azimuth, self.camera.elevation = 90, -90
        self.last_render = -float('inf')
        self.frame = None
        self.locations = None

    def render(self, values, scaling):
        if time.monotonic()-self.last_render >= .1:
            self.renderer.update_scene(self.sim.data, camera=self.camera)
            self.frame = self.renderer.render().copy()
            cams = self.renderer.scene.camera
            pos = (np.array(cams[0].pos)+np.array(cams[1].pos))/2
            forward, up = np.array(cams[0].forward), np.array(cams[0].up)
            right = np.cross(forward, up)
            points = self.sim.sensor_xyz[OFFICIAL_TO_REFERENCE].copy()
            # Mark each chip's XY on the skin surface, not its hidden depth.
            points[:, 2] = self.sim.cfg['total_height_m']
            relative = points-pos
            depth = relative@forward
            focal = 422*cams[0].frustum_near/(cams[0].frustum_top-cams[0].frustum_bottom)
            self.locations = np.c_[200+focal*(relative@right)/depth,
                                    211-focal*(relative@up)/depth]
            self.last_render = time.monotonic()
        surface = pygame.image.fromstring(self.frame.tobytes(), (400,422), 'RGB')
        if values is not None:
            draw(surface, values, scaling=scaling, locations=self.locations)
        return np.transpose(pygame.surfarray.array3d(surface),(1,0,2)).copy()

    def close(self):
        self.renderer.close()
