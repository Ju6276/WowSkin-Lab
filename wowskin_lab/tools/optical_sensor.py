"""Preview DIGIT/TACTO or GelSight/FOTS; separate from magnetic recording."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from wowskin_lab.core.optical_tactile import OpticalTactile
from wowskin_lab.envs.optical import OpticalBench, OpticalPandaEnv


def panel(frames):
    """Fixed depth display scale, 0..5 mm; numerical frames are not clipped."""
    width, height = 240, 190
    out = Image.new('RGB', (width, height*len(frames)), (22, 28, 34))
    draw = ImageDraw.Draw(out)
    for i, (name, frame) in enumerate(frames.items()):
        y = i*height
        draw.text((4, y+3), f'{name}: RGB | indentation 0..5 mm', fill='white')
        out.paste(Image.fromarray(frame.rgb).resize((120, 160)), (0, y+17))
        depth = np.uint8(255*np.clip(frame.indentation_m/.005, 0, 1))
        out.paste(Image.fromarray(depth).convert('RGB').resize((120, 160)), (120, y+17))
        draw.text((4, y+177), f'Contact normal: {frame.normal_force_n:.3f} N', fill='white')
    return np.asarray(out)


def demo(env, optics, destination):
    """Physical press/unload or grasp/lift/release; no object attachment."""
    destination.mkdir(parents=True, exist_ok=True)
    report = {}
    def capture(name):
        frames = optics.render(env.data)
        Image.fromarray(panel(frames)).save(destination/f'{name}.png')
        arrays = {}
        for side, frame in frames.items():
            arrays[f'{side}_rgb'] = frame.rgb
            arrays[f'{side}_indentation_m'] = frame.indentation_m
        np.savez_compressed(destination/f'{name}.npz', **arrays)
        report[name] = {side: dict(normal_force_n=f.normal_force_n,
                                   max_indentation_mm=float(f.indentation_m.max()*1000),
                                   contacting_geoms=list(f.contacting_geoms)) for side, f in frames.items()}
    capture('unloaded')
    if isinstance(env, OpticalBench):
        env.data.ctrl[2] = -.012
        for _ in range(150):
            env.step(np.zeros(3))
        capture('pressed')
        env.data.ctrl[2] = 0.
        for _ in range(150):
            env.step(np.zeros(3))
        capture('released')
    else:
        initial_z = float(env.data.xpos[env.probe_body, 2])
        def move(xyz, seconds):
            env.target_xyz = np.asarray(xyz, dtype=float)
            for _ in range(round(seconds*env.cfg['sample_hz'])):
                env.step(env.hold_action())
        move([.40, 0., .17], 1.5)
        move([.40, 0., .125], 1.5)
        env.command_gripper(True)
        move([.40, 0., .125], 1.)
        capture('grasped')
        move([.40, 0., .23], 2.)
        capture('lifted')
        report['lift_m'] = float(env.data.xpos[env.probe_body, 2]-initial_z)
        env.command_gripper(False)
        move([.40, 0., .23], 1.)
        capture('released')
        report['released_cube_z_m'] = float(env.data.xpos[env.probe_body, 2])
    (destination/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return report


def viewer(env, optics, input_kind):
    import mujoco.viewer
    from wowskin_lab.ui.keyboard import ViewerKeyboard, MOVE_AXES, direction, ESC, BACKSPACE
    from wowskin_lab.envs.pick_cube import MOTION_KEYS, key_command, manual_action, task_camera
    from wowskin_lab.envs.franka import apply_free_camera, ViewerStyleGuard
    from wowskin_lab.teleop.spacemouse import SpaceMouse, SpaceMouseController
    panda = isinstance(env, OpticalPandaEnv)
    keys = ViewerKeyboard(set(MOTION_KEYS) | {ord(x) for x in 'XZCRP'} | {ESC, BACKSPACE}, 'MuJoCo')
    mouse_context = SpaceMouse() if input_kind == 'spacemouse' else nullcontext(None)
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=keys.native_callback,
                                     show_left_ui=False, show_right_ui=False) as handle:
        cam = task_camera(env) if panda else mujoco.MjvCamera()
        if not panda:
            cam.lookat[:] = [0, .08, .05]
            cam.distance, cam.azimuth, cam.elevation = .28, 120, -35
        apply_free_camera(handle, cam)
        with keys, mouse_context as mouse:
            control = SpaceMouseController()
            style = ViewerStyleGuard(handle, 'wasdeqikljuoxzcrp')
            moving, next_image = False, 0.
            while handle.is_running():
                start = time.monotonic()
                state = keys.sample()
                if ord('P') in state.commands or ESC in state.commands:
                    break
                for key in state.commands:
                    if panda:
                        key_command(env, key)
                    elif key in (ord('R'), BACKSPACE):
                        env.reset()
                held = state.held if keys.continuous else frozenset(state.commands)
                if panda:
                    if mouse:
                        action = control.action(env, mouse.state.sample(), handle.cam.azimuth, focused=keys.focused)
                    else:
                        action, moving = manual_action(env, held, moving, handle.cam.azimuth)
                    env.step(action)
                else:
                    # Bench axes only: W/S +/-Y; A/D +/-X; E/Q +/-Z.
                    env.step(direction(held, MOVE_AXES)*.01)
                if not np.isfinite(env.data.qpos).all():
                    raise RuntimeError('Nonfinite physics state; optical output is invalid')
                if start >= next_image:
                    pixels = panel(optics.render(env.data))
                    if not handle.is_running():
                        break
                    viewport = handle.viewport
                    if viewport is None:
                        break
                    scale = min(2., (viewport.height-30)/pixels.shape[0], viewport.width*.3/pixels.shape[1])
                    w, h = max(1, int(pixels.shape[1]*scale)), max(1, int(pixels.shape[0]*scale))
                    pixels = np.asarray(Image.fromarray(pixels).resize((w, h)))
                    message = ('W/S +/-base X | A/D -/+base Y | E/Q up/down\nX/Z close/open | I/K L/J U/O rotate' if panda
                               else 'W/S probe +/-Y | A/D -/+X\nE/Q up/down (Q presses)')
                    handle.set_texts([(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        optics.label+' optical preview\n'+message+'\nR reset | P exit | No dataset recording', '')])
                    if handle.is_running():
                        handle.set_images([(mujoco.MjrRect(viewport.left+viewport.width-w-10,
                                           viewport.bottom+10, w, h), pixels)])
                    next_image = time.monotonic()+.05
                style.sync(handle)
                time.sleep(max(0., .01-(time.monotonic()-start)))


def main(argv=None, *, mounted=False, sensor='tacto'):
    parser = argparse.ArgumentParser(description=__doc__)
    if mounted:
        parser.add_argument('--task', choices=['pick_cube'], default='pick_cube')
        parser.add_argument('--no-cartesian-limits', action='store_true')
    parser.add_argument('--input', choices=['keyboard', 'spacemouse'] if mounted else ['keyboard'], default='keyboard')
    parser.add_argument('--ideal', action='store_true', help='Disable TACTO RGB noise; FOTS adds no stochastic noise')
    parser.add_argument('--demo', action='store_true', help='Run physical contact/unload without a GUI')
    parser.add_argument('--output', type=Path, default=Path('outputs/diagnostics')/sensor)
    args = parser.parse_args(argv)
    env = OpticalPandaEnv(args, sensor=sensor) if mounted else OpticalBench(sensor=sensor)
    if sensor == 'fots':
        from wowskin_lab.core.fots import FOTSOptical as Backend
    else:
        Backend = OpticalTactile
    optics = Backend(env.model, env.sensors, ideal=args.ideal)
    try:
        if args.demo:
            return demo(env, optics, args.output)
        viewer(env, optics, args.input)
    finally:
        optics.close()
