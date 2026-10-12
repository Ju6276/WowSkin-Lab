"""Preview DIGIT/TACTO or GelSight/FOTS; separate from magnetic recording."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wowskin_lab.core.optical_tactile import OpticalTactile
from wowskin_lab.envs.optical import OpticalBench, OpticalPandaEnv


def panel(frames, *, depth_max_mm=None, show_mask=False):
    """Shared, labelled depth contrast for display only; never alter raw frames."""
    has_flow = any(frame.marker_flow_rgb is not None for frame in frames.values())
    column_width, image_height, height = 240, 320, 428
    width = column_width*(3 if has_flow else 2)
    minimum = min(0., *(float(frame.indentation_m.min()) for frame in frames.values()))
    maximum = max(0., *(float(frame.indentation_m.max()) for frame in frames.values()))
    lo, hi = (minimum, maximum) if depth_max_mm is None else (0., depth_max_mm/1000)
    if not np.isfinite([lo, hi]).all():
        raise ValueError('Nonfinite optical depth cannot be displayed as a valid frame')
    out = Image.new('RGB', (width, height*len(frames)), (22, 28, 34))
    draw = ImageDraw.Draw(out)
    draw.font = ImageFont.load_default(size=18)
    for i, (name, frame) in enumerate(frames.items()):
        y = i*height
        draw.text((4, y+3), f'{name}: RGB', fill='white')
        draw.text((column_width+4, y+3), 'Contact mask' if show_mask else 'Depth (um)', fill='white')
        if has_flow:
            draw.text((2*column_width+4, y+3), 'Marker flow', fill='white')
        out.paste(Image.fromarray(frame.rgb).resize((column_width, image_height)), (0, y+28))
        depth = (np.uint8(255*np.clip((frame.indentation_m-lo)/(hi-lo), 0, 1)) if hi > lo
                 else np.zeros(frame.indentation_m.shape, dtype=np.uint8))
        if show_mask:
            if frame.contact_mask is None:
                raise ValueError('This frame has no contact mask')
            depth = frame.contact_mask.astype(np.uint8)*255
        depth_rgb = np.repeat(depth[..., None], 3, axis=-1)
        if frame.valid_mask is not None:
            depth_rgb[~frame.valid_mask] = [60, 70, 85]
        out.paste(Image.fromarray(depth_rgb).resize((column_width, image_height)), (column_width, y+28))
        if frame.marker_flow_rgb is not None:
            out.paste(Image.fromarray(frame.marker_flow_rgb).resize((column_width, image_height)),
                      (2*column_width, y+28))
        mode = 'AUTO shared' if depth_max_mm is None else 'FIXED'
        draw.text((4, y+351), f'{mode}: {lo*1e6:.2f}..{hi*1e6:.2f} um', fill='#8bd0e7')
        draw.text((4, y+376), f'Max: {frame.indentation_m.max()*1e6:.2f} um | Normal: {frame.normal_force_n:.3f} N', fill='white')
        draw.text((4, y+401), 'M: depth / mask', fill='#8bd0e7')
        if frame.valid_mask is not None and not frame.valid_mask.all():
            draw.text((column_width+4, y+401), 'Blue-gray: outside gel', fill='#8bd0e7')
        if frame.marker_displacement_px is not None:
            displacement = np.linalg.norm(frame.marker_displacement_px, axis=-1).max()
            draw.text((column_width+4, y+401), f'Marker max: {displacement:.3f} px (raw)', fill='white')
        if hi == lo:
            draw.text((column_width+4, y+351), 'No optical indentation', fill='white')
    return np.asarray(out)


def demo(env, optics, destination, *, depth_max_mm=None):
    """Physical press/unload or grasp/lift/release; no object attachment."""
    destination.mkdir(parents=True, exist_ok=True)
    report = {}
    def capture(name):
        frames = optics.render(env.data)
        Image.fromarray(panel(frames, depth_max_mm=depth_max_mm)).save(destination/f'{name}.png')
        arrays = {}
        for side, frame in frames.items():
            arrays[f'{side}_rgb'] = frame.rgb
            arrays[f'{side}_indentation_m'] = frame.indentation_m
            arrays[f'{side}_contact_mask'] = frame.contact_mask
            arrays[f'{side}_valid_mask'] = frame.valid_mask
            if frame.camera_depth_m is not None:
                arrays[f'{side}_camera_depth_m'] = frame.camera_depth_m
                arrays[f'{side}_baseline_depth_m'] = frame.baseline_depth_m
            arrays[f'{side}_time_s'] = frame.time_s
            arrays[f'{side}_normal_force_n'] = frame.normal_force_n
            if frame.marker_flow_rgb is not None:
                arrays[f'{side}_marker_flow_rgb'] = frame.marker_flow_rgb
                arrays[f'{side}_marker_displacement_px'] = frame.marker_displacement_px
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


def viewer(env, optics, input_kind, *, depth_max_mm=None):
    import mujoco.viewer
    from wowskin_lab.ui.keyboard import ViewerKeyboard, MOVE_AXES, direction, ESC, BACKSPACE
    from wowskin_lab.envs.pick_cube import MOTION_KEYS, key_command, manual_action, task_camera
    from wowskin_lab.envs.franka import apply_free_camera, ViewerStyleGuard
    from wowskin_lab.teleop.spacemouse import SpaceMouse, SpaceMouseController
    panda = isinstance(env, OpticalPandaEnv)
    keys = ViewerKeyboard(set(MOTION_KEYS) | {ord(x) for x in 'XZCRPM'} | {ESC, BACKSPACE}, 'MuJoCo')
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
            style = ViewerStyleGuard(handle, 'wasdeqikljuoxzcrpm')
            moving, next_image, show_mask = False, 0., False
            while handle.is_running():
                start = time.monotonic()
                state = keys.sample()
                if ord('P') in state.commands or ESC in state.commands:
                    break
                for key in state.commands:
                    if key == ord('M'):
                        show_mask = not show_mask
                    if panda:
                        key_command(env, key)
                    elif key in (ord('R'), BACKSPACE):
                        env.reset()
                    if key in (ord('R'), BACKSPACE) and hasattr(optics, 'reset'):
                        optics.reset()
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
                    pixels = panel(optics.render(env.data), depth_max_mm=depth_max_mm, show_mask=show_mask)
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
    parser.add_argument('--depth-display-max-mm', type=float, default=None,
                        help='Fixed grayscale maximum in mm; default auto contrast shared by both fingers. Display only.')
    if sensor == 'tacto':
        parser.add_argument('--tacto-config', type=Path, help='TACTO YAML for camera/lights/noise/force settings')
        backgrounds = parser.add_mutually_exclusive_group()
        backgrounds.add_argument('--tacto-background', type=Path, help='Real empty-contact image; default upstream DIGIT background')
        backgrounds.add_argument('--no-tacto-background', action='store_true', help='Pure optical rendering without background calibration')
    if sensor == 'fots':
        parser.add_argument('--no-fots-shadow', action='store_true', help='Disable the original FOTS planar shadow for comparison')
    parser.add_argument('--demo', action='store_true', help='Run physical contact/unload without a GUI')
    parser.add_argument('--output', type=Path, default=Path('outputs/diagnostics')/sensor)
    args = parser.parse_args(argv)
    if args.depth_display_max_mm is not None and (not np.isfinite(args.depth_display_max_mm) or args.depth_display_max_mm <= 0):
        parser.error('--depth-display-max-mm must be finite and positive')
    env = OpticalPandaEnv(args, sensor=sensor) if mounted else OpticalBench(sensor=sensor)
    if sensor == 'fots':
        from wowskin_lab.core.fots import FOTSOptical as Backend
    else:
        Backend = OpticalTactile
    options = (dict(shadow=not args.no_fots_shadow) if sensor == 'fots' else
               dict(background=None if args.no_tacto_background else (args.tacto_background or 'upstream'),
                    config_path=args.tacto_config))
    optics = Backend(env.model, env.sensors, ideal=args.ideal, **options)
    try:
        if args.demo:
            return demo(env, optics, args.output, depth_max_mm=args.depth_display_max_mm)
        viewer(env, optics, args.input, depth_max_mm=args.depth_display_max_mm)
    finally:
        optics.close()
