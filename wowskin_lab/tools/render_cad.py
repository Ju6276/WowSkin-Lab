"""Render the unmodified OnRobot CAD in MuJoCo, separately from elastic simulation."""
import argparse
import time

import mujoco
import numpy as np

from wowskin_lab.paths import ROOT


def load():
    path = ROOT/'assets/cad/anyskin/onrobot/OnRobot.stl'
    # Preserve geometry and dimensions; only rigid pose changes for inspection.
    from wowskin_lab.tools.inspect_cad import read_parts
    vertices = read_parts(path)[0].reshape(-1, 3)/1000
    center = (vertices.min(0)+vertices.max(0))/2
    rotation = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
    translation = -rotation@center+np.array([0., 0., .011])
    xml = f'''<mujoco model="Original AnySkin OnRobot CAD reference">
      <visual><global offwidth="1000" offheight="750"/><map znear=".01"/>
        <headlight ambient=".4 .4 .4" diffuse=".7 .7 .7" specular=".15 .15 .15"/></visual>
      <asset><mesh name="onrobot" file="{path}"/>
        <material name="plastic" rgba=".2 .23 .26 1" specular=".2" shininess=".3"/></asset>
      <worldbody><light pos=".04 -.06 .1" dir="-.3 .3 -1" diffuse=".8 .8 .8"/>
        <light pos="-.04 .04 .06" dir=".2 -.2 -1" diffuse=".5 .5 .5"/>
        <geom type="plane" size=".1 .1 .001" rgba=".88 .89 .90 1" contype="0" conaffinity="0"/>
        <geom type="mesh" mesh="onrobot" material="plastic" contype="0" conaffinity="0"
          pos="{' '.join(map(str,translation))}" quat="{' '.join(map(str,quaternion))}"/>
      </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--seconds', type=float, default=0)
    args = parser.parse_args()
    model, data = load()
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, .011]
    camera.distance, camera.azimuth, camera.elevation = .10, 115, -30
    if args.viewer:
        from mujoco import viewer
        deadline = time.monotonic()+args.seconds if args.seconds else float('inf')
        with viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as handle:
            handle.cam.lookat[:] = camera.lookat
            handle.cam.distance, handle.cam.azimuth, handle.cam.elevation = camera.distance, camera.azimuth, camera.elevation
            handle.set_texts((None, None, 'Original AnySkin OnRobot.stl\nGeometry reference only; no magnetic skin simulation', ''))
            while handle.is_running() and time.monotonic() < deadline:
                handle.sync()
                time.sleep(.03)
        return
    from PIL import Image, ImageDraw
    images = []
    with mujoco.Renderer(model, height=600, width=800) as renderer:
        for label, azimuth, elevation in [('OnRobot CAD - oblique',115,-30), ('OnRobot CAD - opposite side',-65,-30), ('OnRobot CAD - top',90,-89)]:
            camera.azimuth, camera.elevation = azimuth, elevation
            renderer.update_scene(data, camera=camera)
            im = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(im)
            draw.rectangle((0,0,800,54),fill=(245,246,248))
            draw.text((14,10),label+' | original dimensions preserved',fill=(20,30,40))
            draw.text((14,30),'43.588 x 19.000 x 42.282 mm | reference mounting part, not verified WowSkin CAD',fill=(70,80,90))
            images.append(im)
    canvas = Image.new('RGB',(2400,600),'white')
    for i,im in enumerate(images):canvas.paste(im,(800*i,0))
    path = ROOT/'assets/cad/onrobot_mujoco.png'
    canvas.save(path)
    print(path)


if __name__ == '__main__':
    main()
