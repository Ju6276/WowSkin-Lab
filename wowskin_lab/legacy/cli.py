"""WowSkin sensor bench. Earlier slip experiments remain explicit optional commands."""
import argparse
import csv
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from wowskin_lab.core.physics import CHANNELS, TactileSim, configuration


def run_episode(cfg, frame_callback=None, predictor=None):
    sim = TactileSim(cfg)
    rows = []
    for index in range(round(cfg["duration_s"]*cfg["sample_hz"])):
        sample = sim.sample()
        sample["probability"] = predictor.update(sample["magnetic_ut"]) if predictor else float("nan")
        rows.append(sample)
        if frame_callback and index % 5 == 0:
            frame_callback(sim, sample)
    return sim, {key: np.array([r[key] for r in rows]) for key in rows[0]}


def save_episode(destination, cfg, sim, values):
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination.with_suffix(".npz"), **values)
    metadata = dict(config=cfg, status="UNCALIBRATED SYNTHETIC DATA", channels=CHANNELS,
                    field_units="microtesla from a point-dipole approximation",
                    baseline_field_ut=sim.baseline.tolist(),
                    sensor_layout=sim.layout,
                    label="contact-force gated, force-weighted tangential relative speed",
                    warnings=sim.data.warning.number.tolist())
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    # The mag column names match Pollen, but simulation time is explicitly named.
    with destination.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sim_time_s", *CHANNELS, "slip", "contact", "normal_force_n", "slip_speed_m_per_s"])
        for i, t in enumerate(values["time_s"]):
            writer.writerow([t, *values["magnetic_ut"][i], values["slip"][i],
                             values["contact"][i], values["normal_force_n"][i],
                             values["slip_speed_m_per_s"][i]])


def plot_episode(values, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 1, figsize=(11, 10), sharex=True, layout="constrained")
    t = values["time_s"]
    b = values["magnetic_ut"].reshape(-1, 5, 3)
    for axis, component in enumerate("XYZ"):
        for sensor in range(5):
            axes[axis].plot(t, b[:, sensor, axis], lw=1, label=f"U{sensor+1}")
        axes[axis].set_ylabel(f"Delta B{component} (uT)")
        axes[axis].grid(alpha=.2)
    axes[0].legend(ncol=5, loc="upper right")
    axes[3].plot(t, values["normal_force_n"], color="#216a8a")
    axes[3].set_ylabel("Normal load (N)")
    twin = axes[3].twinx()
    twin.plot(t, values["slip_speed_m_per_s"]*1000, color="#c25835", alpha=.8)
    twin.set_ylabel("Relative slip (mm/s)", color="#c25835")
    axes[4].step(t, values["slip"], where="post", color="#cb6038", label="Simulation ground truth")
    if np.any(np.isfinite(values["probability"])):
        axes[4].plot(t, values["probability"], color="#236d96", label="LSTM prediction")
        axes[4].axhline(.5, color="gray", ls=":", lw=1)
    axes[4].legend(loc="upper right")
    axes[4].set_ylabel("Slip / probability")
    axes[4].set_xlabel("Simulation time (s)")
    fig.suptitle("MuJoCo magnetic tactile prototype — UNCALIBRATED", fontsize=15)
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def demo(args):
    cfg = configuration(args.seed, args.randomize)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    predictor = None
    if args.checkpoint:
        from wowskin_lab.legacy.slip import Predictor
        predictor = Predictor(args.checkpoint)
        if predictor.sample_hz != cfg["sample_hz"]:
            raise ValueError("Sample rate differs from training")
    renderer = None
    frames = []
    def capture(sim, row):
        nonlocal renderer
        from PIL import Image, ImageDraw
        if renderer is None:
            renderer = mujoco.Renderer(sim.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, .007]
        camera.distance, camera.azimuth, camera.elevation = .10, 100, -38
        renderer.update_scene(sim.data, camera=camera)
        frame = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, 640, 48), fill=(18, 22, 28))
        draw.text((12, 6), "UNCALIBRATED SIMULATION | elastic skin + magnetic dipoles", fill="white")
        label = "SLIDING" if row["slip"] else ("CONTACT" if row["contact"] else "NO CONTACT")
        probability = row["probability"]
        suffix = f" | model p={probability:.2f}" if np.isfinite(probability) else ""
        draw.text((12, 26), f"t={row['time_s']:.2f}s | ground truth: {label}"+suffix,
                  fill=(255, 180, 110))
        frames.append(np.array(frame))
    try:
        sim, values = run_episode(cfg, capture if args.render else None, predictor)
    finally:
        if renderer is not None:
            renderer.close()
    save_episode(destination/"episode", cfg, sim, values)
    (destination/"scene.xml").write_text(sim.xml)
    plot_episode(values, destination/"signals.png")
    if frames:
        import imageio.v3 as iio
        iio.imwrite(destination/"demo.gif", np.stack(frames), duration=50, loop=0)
        iio.imwrite(destination/"scene.png", frames[min(35, len(frames)-1)])
    print(json.dumps(dict(output=str(destination.resolve()), samples=len(values["time_s"]),
                          slip_samples=int(values["slip"].sum()),
                          min_volume_ratio=float(values["min_volume_ratio"].min())), indent=2))


def generate(args):
    if args.episodes < 6:
        raise ValueError("Use at least 6 episodes for train/val/test")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if (root/"manifest.json").exists():
        raise FileExistsError("Dataset already exists; choose another --output")
    ntrain = max(2, int(args.episodes*.65))
    nval = max(1, int(args.episodes*.15))
    manifest = dict(status="UNCALIBRATED SIMULATION", sample_hz=configuration()["sample_hz"],
                    splits={"train": [], "val": [], "test": []}, episodes=[])
    for index in range(args.episodes):
        seed = args.seed+index
        cfg = configuration(seed, randomize=True)
        sim, values = run_episode(cfg)
        name = f"episode_{seed:05d}"
        if (root/f"{name}.npz").exists():
            raise FileExistsError(f"Refusing to overwrite episode {name}")
        save_episode(root/name, cfg, sim, values)
        split = "train" if index < ntrain else ("val" if index < ntrain+nval else "test")
        manifest["splits"][split].append(name+".npz")
        manifest["episodes"].append(dict(seed=seed, split=split, config=cfg,
                                         slip_fraction=float(values["slip"].mean())))
        print(f"episode {index+1}/{args.episodes}: seed={seed}, split={split}, "
              f"slip={values['slip'].mean():.1%}", flush=True)
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


def viewer(args):
    import mujoco.viewer
    cfg = configuration(args.seed, args.randomize)
    sim = TactileSim(cfg)
    predictor = None
    if args.checkpoint:
        from wowskin_lab.legacy.slip import Predictor
        predictor = Predictor(args.checkpoint)
    print("Uncalibrated simulation; automatic press / hold / slide / lift. Close window to exit.")
    deadline = time.monotonic()+args.seconds if args.seconds else float("inf")
    with mujoco.viewer.launch_passive(sim.model, sim.data) as handle:
        handle.cam.lookat[:] = [0, 0, .007]
        handle.cam.distance, handle.cam.azimuth, handle.cam.elevation = .10, 100, -38
        index = 0
        while handle.is_running() and time.monotonic() < deadline:
            start = time.monotonic()
            row = sim.sample()
            probability = predictor.update(row["magnetic_ut"]) if predictor else float("nan")
            if index % 25 == 0:
                print(f"t={row['time_s']:.2f} contact={row['contact']} "
                      f"true_slip={row['slip']} model_p={probability:.3f}", flush=True)
            handle.sync()
            index += 1
            if sim.data.time >= cfg["duration_s"]:
                mujoco.mj_resetData(sim.model, sim.data)
                mujoco.mj_forward(sim.model, sim.data)
                sim.filtered[:] = 0
                sim.delay.clear()
                sim.delay.extend(np.zeros((5, 3)) for _ in range(cfg["delay_samples"]))
                if predictor:
                    predictor.history.clear()
            time.sleep(max(0, 1/cfg["sample_hz"]-(time.monotonic()-start)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "viewer"):
        p = commands.add_parser(name, help="Sensor-only press/release inspection")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--x-mm", type=float, default=0.)
        p.add_argument("--y-mm", type=float, default=0.)
        p.add_argument("--load-n", type=float, default=.05)
        p.add_argument("--ideal", action="store_true", help="Disable additive noise and drift")
        if name == "demo":
            p.add_argument("--output", default="outputs/sensor")
            p.add_argument("--render", action="store_true")
            p.set_defaults(func=lambda a: __import__("wowskin_lab.envs.bench", fromlist=["*"]).demo(a))
        else:
            p.add_argument("--seconds", type=float, default=0)
            p.add_argument("--real-port", help="Real WowSkin serial port; enables paired recording")
            p.add_argument("--real-baseline", help="Load a saved noise_summary.json baseline for the same device")
            p.add_argument("--capture-dir", default="outputs/calibration", help="Parent directory for a new real/sim recording")
            p.set_defaults(func=lambda a: __import__("wowskin_lab.envs.bench", fromlist=["*"]).viewer(a))
    for name in ("slip-demo", "slip-viewer"):
        p = commands.add_parser(name)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--randomize", action="store_true")
        p.add_argument("--checkpoint")
        if name == "slip-demo":
            p.add_argument("--output", default="outputs/demo")
            p.add_argument("--render", action="store_true")
        else:
            p.add_argument("--seconds", type=float, default=0, help="Close after wall-clock seconds; 0 means unlimited")
        p.set_defaults(func=demo if name == "slip-demo" else viewer)
    p = commands.add_parser("generate")
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--output", default="outputs/dataset")
    p.set_defaults(func=generate)
    p = commands.add_parser("train")
    p.add_argument("--dataset", default="outputs/dataset")
    p.add_argument("--output", default="outputs/model")
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--epochs", type=int, default=25)
    p.set_defaults(func=lambda a: __import__("wowskin_lab.legacy.slip", fromlist=["*"]).train(a.dataset, a.output, a.window, a.epochs))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
