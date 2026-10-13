"""Sensor-only bench: mechanical deformation -> five three-axis magnetic readings."""
from contextlib import nullcontext
import csv
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from wowskin_lab.core.physics import CHANNELS, TactileSim, configuration
from wowskin_lab.ui.keyboard import ViewerKeyboard
from wowskin_lab.ui.sensor_compare import SimSensorHUD, show_comparison
from wowskin_lab.ui.channel_alignment import OFFICIAL_TO_REFERENCE, POSITIONS, official_values


def bench_configuration(args):
    cfg = configuration(args.seed)
    if tuple(map(int, mujoco.__version__.split('.')[:2])) >= (3, 15):
        cfg.update(skin_dof='trilinear', integrator='discrete',
                   skin_model_status='Trilinear interpolation with bounding-box bottom attachment; standalone bench, not the old full-node calibration model',
                   engine_version=mujoco.__version__)
    cfg.update(duration_s=3.5, offset_x_m=args.x_mm/1000,
               offset_y_m=args.y_mm/1000, normal_load_n=args.load_n)
    # This initial bench stays in a small, numerically checked region.
    if not (0 < args.load_n <= .075):
        raise ValueError("Initial bench load range: 0 < --load-n <= 0.075 N")
    if abs(args.x_mm) > 7 or abs(args.y_mm) > 7:
        raise ValueError("Initial bench location range: +/-7 mm in X and Y")
    if getattr(args, "ideal", False):
        cfg.update(noise_ut=0., drift_ut_per_s=0.)
    return cfg


def normal_press(sim):
    """One fixed-location press, hold and release; no lateral trajectory."""
    t = sim.data.time
    load = None
    if .5 <= t < 2.1:
        a = np.clip((t-.5)/.3, 0, 1)
        load = sim.cfg["normal_load_n"]*a*a*(3-2*a)
    sim.apply_probe_load([sim.cfg["offset_x_m"], sim.cfg["offset_y_m"]], load)


def sensor_reading(sim, controller):
    row = sim.sample(controller)
    # Expose sensor output and mechanical diagnostics, without task labels.
    keys = ("time_s", "magnetic_ut", "ideal_delta_ut", "raw_field_ut",
            "normal_force_n", "probe_xyz_m", "min_volume_ratio")
    result = {key: row[key] for key in keys}
    result["skin_vertices_m"] = sim.data.flexvert_xpos.copy()
    result["max_displacement_m"] = float(np.linalg.norm(
        sim.data.flexvert_xpos-sim.rest, axis=1).max())
    return result


def camera():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, .007]
    cam.distance, cam.azimuth, cam.elevation = .105, 100, -38
    return cam


def structure_view(sim, transparent):
    """Render transparency only; the mechanical geometry is unchanged."""
    sim.model.geom_rgba[sim.model.geom("standard_shell").id, 3] = .12 if transparent else 1.
    sim.model.flex_rgba[:, 3] = .25 if transparent else 1.


def plot_response(values, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True, layout="constrained")
    t = values["time_s"]
    b = values["magnetic_ut"].reshape(-1, 5, 3)
    for axis, component in enumerate("XYZ"):
        for chip in range(5):
            axes[axis].plot(t, b[:, chip, axis], lw=1, label=f"U{chip+1}")
        axes[axis].set_ylabel(f"Delta B{component} (uT)")
        axes[axis].grid(alpha=.2)
    axes[0].legend(ncol=5)
    axes[3].plot(t, values["normal_force_n"], color="#216a8a")
    axes[3].set_ylabel("Simulated normal load (N)")
    twin = axes[3].twinx()
    twin.plot(t, values["max_displacement_m"]*1e6, color="#b55c35")
    twin.set_ylabel("Max skin displacement (um)", color="#b55c35")
    axes[3].set_xlabel("Time (s) | fixed-point press, hold, release")
    fig.suptitle("WowSkin sensor prototype | magnetic response is UNCALIBRATED")
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def demo(args):
    cfg = bench_configuration(args)
    sim = TactileSim(cfg)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    rows, frames = [], []
    renderer = mujoco.Renderer(sim.model, height=600, width=800) if args.render else None
    try:
        for index in range(round(cfg["duration_s"]*cfg["sample_hz"])):
            row = sensor_reading(sim, normal_press)
            rows.append(row)
            if renderer and index % 5 == 0:
                from PIL import Image, ImageDraw
                renderer.update_scene(sim.data, camera=camera())
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 800, 55), fill=(18, 22, 28))
                draw.text((12, 8), "WowSkin standard | approximate geometry, UNCALIBRATED response", fill="white")
                draw.text((12, 29), f"Fixed-point press / release | t={row['time_s']:.2f}s | "
                          f"load={row['normal_force_n']:.3f} N | "
                          f"max deformation={row['max_displacement_m']*1e6:.1f} um", fill=(255, 190, 120))
                frames.append(np.array(frame))
                if index == 150:
                    import imageio.v3 as iio
                    iio.imwrite(destination/"scene.png", frames[-1])
                    structure_view(sim, True)
                    renderer.update_scene(sim.data, camera=camera())
                    internal = Image.fromarray(renderer.render())
                    ImageDraw.Draw(internal).text((12, 12),
                        "REFERENCE LAYOUT | ReSkin chip positions; WowSkin internals unconfirmed", fill="white")
                    iio.imwrite(destination/"structure.png", np.array(internal))
                    structure_view(sim, False)
    finally:
        if renderer:
            renderer.close()
    values = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    np.savez_compressed(destination/"sensor_response.npz", **values)
    with (destination/"sensor_response.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sim_time_s", *CHANNELS, "sim_normal_force_n", "max_displacement_m"])
        for row in rows:
            writer.writerow([row["time_s"], *row["magnetic_ut"], row["normal_force_n"], row["max_displacement_m"]])
    metadata = dict(status="UNCALIBRATED SENSOR PROTOTYPE", experiment="fixed-point normal press / release",
                    config=cfg, channels=CHANNELS, units="microtesla",
                    baseline_field_ut=sim.baseline.tolist(), sensor_layout=sim.layout,
                    output_description="magnetic_ut: filtered, delayed, noisy delta B; raw_field_ut: model field",
                    diagnostics_description="force/displacement are simulation references, not physical sensor outputs",
                    warnings=sim.data.warning.number.tolist())
    (destination/"sensor_response.json").write_text(json.dumps(metadata, indent=2)+"\n")
    (destination/"scene.xml").write_text(sim.xml)
    plot_response(values, destination/"response.png")
    if frames:
        import imageio.v3 as iio
        iio.imwrite(destination/"press_release.gif", np.stack(frames), duration=50, loop=0)
    print(json.dumps(dict(output=str(destination.resolve()), samples=len(rows),
                          max_displacement_um=float(values["max_displacement_m"].max()*1e6),
                          min_volume_ratio=float(values["min_volume_ratio"].min())), indent=2))


def viewer(args):
    import mujoco.viewer
    # 3.3.7/3.4.0 set_texts can wait with the GIL held while OnKey waits to
    # acquire it, deadlocking both UI and Python. 3.5.0 fixes that native wait.
    if tuple(map(int, mujoco.__version__.split('.')[:3])) < (3, 5, 0):
        raise RuntimeError("Interactive viewer requires MuJoCo >= 3.5.0 to avoid the keyboard/overlay deadlock; install requirements.txt with Python >= 3.10.")
    cfg = bench_configuration(args)
    sim = TactileSim(cfg)
    keyboard = ViewerKeyboard(set(map(ord, "PLVZ12345,.QBMT")),
                              title_contains="Uncalibrated magnetic tactile prototype")
    hud = SimSensorHUD(sim)
    comparison = None
    if getattr(args, 'real_port', None):
        from wowskin_lab.data.sensor_compare import SensorComparison
        from wowskin_lab.ui.sensor_compare import RealSensorHUD, comparison_panel
        comparison = SensorComparison(args.real_port, args.capture_dir, cfg, official_baseline=True)
        comparison.event("sim_initial_baseline", baseline_ut=sim.baseline.reshape(15).tolist())
        real_hud = RealSensorHUD(sim)
        hud.live_overlay = True
        comparison.event('display_source', left='real_on_mujoco_model', simulation_is_independent=True)
        print(f'Real/sim recording: {comparison.path.resolve()}', flush=True)
        print('B: reset real baseline from 5 frames (manufacturer default); M: save panel + mark; Z: sim baseline only', flush=True)
    next_hud = 0.
    panel = None
    # Keep the initial view focused on the sensor; show the bench probe on P.
    sim.model.geom_rgba[sim.probe_geom, 3] = 0.
    sim.model.site_rgba[sim.model.site("probe_marker").id, 3] = 0.
    xy = np.array([cfg["offset_x_m"], cfg["offset_y_m"]])
    pressed, transparent = False, False
    load, message = 0., ("Real data overlay; T switches to independent SIM prediction." if comparison else "Unloaded. 1-5 selects position; P applies load.")
    deadline = time.monotonic()+args.seconds if args.seconds else float("inf")
    print("Sensor bench: P press | L lift | 1-5 position while lifted | Z zero | V structure | ,/. display scale | Q exit.", flush=True)

    def control(current):
        current.apply_probe_load(xy, load if pressed else None)

    with (comparison or nullcontext()), hud, mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=keyboard.native_callback,
                                    show_left_ui=False, show_right_ui=False) as handle, keyboard:
        if comparison and getattr(args, "real_baseline", None):
            comparison.load_baseline(args.real_baseline)
        cam = camera()
        handle.cam.lookat[:] = cam.lookat
        handle.cam.distance, handle.cam.azimuth, handle.cam.elevation = cam.distance, cam.azimuth, cam.elevation
        index = 0
        while handle.is_running() and time.monotonic() < deadline:
            start = time.monotonic()
            commands = keyboard.sample().commands
            if ord("Q") in commands:
                break
            for key in commands:
                hud.key(key)
                if comparison:
                    comparison.event('key', key=chr(key), sim_time_s=float(sim.data.time),
                                     selected_xy_m=xy.tolist(), sim_pressed=pressed)
                if key == ord('T') and comparison:
                    hud.live_overlay = not hud.live_overlay
                    comparison.event('display_source', left='real_on_mujoco_model' if hud.live_overlay else 'simulation_prediction')
                    message = 'Real data on MuJoCo model; geometry is not driven by real data.' if hud.live_overlay else 'SIM prediction: 1-5 select, P press, L lift.'
                if key == ord('B') and comparison:
                    try:
                        comparison.begin_baseline(num_samples=5, duration_s=0.)
                        message = 'Real baseline: 5-frame mean; keep unloaded.'
                    except ValueError as error:
                        message = str(error)
                if key == ord('M') and comparison and panel is not None:
                    from PIL import Image
                    filename = f'panel_{time.monotonic_ns()}.png'
                    Image.fromarray(panel).save(comparison.path/filename)
                    comparison.event('panel_snapshot', file=filename, sim_time_s=float(sim.data.time))
                    message = 'Saved panel snapshot and time marker.'
                if key == ord("P"):
                    pressed = True
                    with handle.lock():
                        sim.model.geom_rgba[sim.probe_geom, 3] = 1.
                        sim.model.site_rgba[sim.model.site("probe_marker").id, 3] = 1.
                    message = "Applying normal load; XY target fixed."
                elif key == ord("L"):
                    pressed, load = False, 0.
                    with handle.lock():
                        sim.model.geom_rgba[sim.probe_geom, 3] = 0.
                        sim.model.site_rgba[sim.model.site("probe_marker").id, 3] = 0.
                    message = "Lifting probe. Wait before selecting a position or zeroing."
                elif ord("1") <= key <= ord("5"):
                    clearance = sim.data.xpos[sim.probe_body, 2]-cfg["probe_radius_m"]-cfg["total_height_m"]
                    if pressed or clearance < .0005:
                        message = "Press L and wait for the probe to clear the skin."
                    else:
                        channel = key-ord("1")
                        xy[:] = sim.sensor_xyz[OFFICIAL_TO_REFERENCE[channel], :2]
                        message = f"CH{channel+1}: {POSITIONS[channel]} (manufacturer display)."
                elif key == ord("Z"):
                    try:
                        if pressed:
                            raise ValueError("Press L before zeroing")
                        hud.begin_reset()
                        message = "Sim display baseline: collecting 5 frames."
                        if comparison:
                            comparison.event('sim_display_baseline_requested', sim_time_s=float(sim.data.time), count=5)
                    except ValueError as error:
                        message = str(error)
                elif key == ord("V"):
                    transparent = not transparent
                    with handle.lock():
                        structure_view(sim, transparent)
                    message = "Structure view on." if transparent else "Structure view off."
                if key in map(ord, "PLVZ12345"):
                    print(f"Key {chr(key)}: {message}", flush=True)
            if pressed:
                load = min(cfg["normal_load_n"], load+cfg["normal_load_n"]/(.3*cfg["sample_hz"]))
            row = sensor_reading(sim, control)
            baseline_event = hud.observe(row)
            if comparison:
                if baseline_event:
                    comparison.event('sim_display_baseline', sim_time_s=row['time_s'], **baseline_event)
                row['manufacturer_display_ut'] = official_values(sim, row['magnetic_ut'])-hud.offset
                row['manufacturer_display_baseline_id'] = hud.baseline_id
                comparison.record_sim(row, selected_xy_m=xy.tolist(), pressed=pressed, commanded_load_n=load)
            if time.monotonic() >= next_hud:
                if comparison:
                    panel = comparison_panel(hud, real_hud, row, comparison)
                    show_comparison(handle, panel, keyboard, left_reset='B' if hud.live_overlay else 'Z')
                else:
                    show_comparison(handle, hud.render(row), keyboard)
                labels = "WowSkin sensor prototype\nBase / fitted (mm)\nState\nXY target (mm)\nSim. load (N)\nSkin motion (um)"
                readings = (f"UNCALIBRATED\n{cfg['base_height_m']*1000:g} / {cfg['total_height_m']*1000:g}\n"
                            f"{'PRESS' if pressed else 'UNLOADED'}\n{xy[0]*1000:+.1f}, {xy[1]*1000:+.1f}\n"
                            f"{row['normal_force_n']:.4f}\n{row['max_displacement_m']*1e6:.1f}")
                handle.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT, labels, readings),
                                  (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                                   "P press | L lift | 1-5 position | Z sim zero | V structure\n,/. display scale | Q exit\n"+("T: real overlay / SIM prediction | B real zero | M snapshot\n" if comparison else "")+message, "")])
                next_hud = time.monotonic()+1/60
            handle.sync()
            if index % 100 == 0:
                print(f"t={row['time_s']:.2f}s Fz={row['normal_force_n']:.4f}N "
                      f"max displacement={row['max_displacement_m']*1e6:.2f}um", flush=True)
            index += 1
            time.sleep(max(0, 1/cfg["sample_hz"]-(time.monotonic()-start)))
