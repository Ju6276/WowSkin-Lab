"""Franka-mounted deformable WowSkin bench using Ju6276/franka_xbox assets.

Single instrumented finger, matching passive silicone pad and external probe; not a reproduced
grasp/handoff experiment or a validated real-sensor digital twin.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from wowskin_lab.core.geometry import standard_xml
from wowskin_lab.core.physics import CHANNELS, ROOT, TactileSim, configuration, flex_tetrahedra, flex_vertex_slice
from wowskin_lab.ui.keyboard import ViewerKeyboard

FRANKA_REPO = ROOT/"third_party/franka_xbox"
FRANKA_REVISION = "a43327ad8446642acd8d64b6c20829fd135a9f28"
HOME = np.array([0., -.405, 0., -2.86, 0., 2.43, np.pi/4])


def mount_reference(cfg):
    """Register the module's proximal edge to the removed stock pad's edge.

    OBJ coordinates are finger-local metres. The exposed central metal seat
    defines the backing plane; this is a geometric fit, not a fabricated bracket.
    """
    directory = FRANKA_REPO/"franka_sim/envs/xmls/franka_emika_panda/assets"
    def vertices(filename):
        return np.array([[float(x) for x in line.split()[1:4]]
                         for line in (directory/filename).read_text().splitlines()
                         if line.startswith("v ")])
    pad, metal = vertices("finger_1.obj"), vertices("finger_0.obj")
    proximal = float(pad[:, 2].min())
    seat = metal[(metal[:, 2] >= proximal) & (metal[:, 2] <= pad[:, 2].max())
                 & (np.abs(metal[:, 0]) < .8*np.max(np.abs(pad[:, 0])))]
    plane_y = float(seat[:, 1].min())
    return dict(position_m=[0., plane_y, proximal+cfg["length_m"]/2],
                removed_pad_proximal_z_m=proximal, exposed_seat_y_m=plane_y,
                alignment="housing proximal edge at old black pad proximal edge; backing on metal seat")


def build_scene(cfg, layout):
    directory = FRANKA_REPO/"franka_sim/envs/xmls"
    if not (directory/"panda.xml").is_file():
        raise FileNotFoundError("Clone https://github.com/Ju6276/franka_xbox into third_party/franka_xbox")
    root = ET.parse(directory/"panda.xml").getroot()
    root.set("model", "Franka Panda with one uncalibrated WowSkin")
    asset = root.find("asset")
    for element in list(asset):
        if element.tag == "mesh" and element.get("file"):
            if element.get("file").startswith("gear/"):
                asset.remove(element)
            else:
                element.set("file", str((directory/element.get("file")).resolve()))
    ET.SubElement(root, "option", timestep=str(cfg["timestep_s"]), integrator="implicitfast",
                  gravity="0 0 -9.81", cone="elliptic", iterations="50", tolerance="1e-10")
    ET.SubElement(root, "statistic", center=".3 0 .35", extent=".8")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1000", offheight="750")
    ET.SubElement(visual, "headlight", ambient=".4 .4 .4", diffuse=".65 .65 .65")
    ET.SubElement(visual, "map", znear=".005")
    # Override the inherited panda joint defaults for elastic nodes.
    tactile_default = ET.SubElement(root.find("default"), "default", {"class": "tactile"})
    ET.SubElement(tactile_default, "joint", armature="0",
                  damping="0" if cfg.get('skin_dof') == 'trilinear' else ".002", limited="false")
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1", diffuse=".7 .7 .7")
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 .01",
                  rgba=".32 .36 .4 1")
    if cfg.get('sensor') in ('tacto', 'fots'):
        from .optical import mount_digits, mount_gelsights
        mount = mount_gelsights if cfg['sensor'] == 'fots' else mount_digits
        mount(root, friction=cfg['friction'])
        return ET.tostring(root, encoding="unicode")
    finger = root.find(".//body[@name='left_finger']")
    for name in ("left_finger", "right_finger"):
        body = root.find(f".//body[@name='{name}']")
        for geom in list(body.findall("geom")):
            if geom.get("mesh") == "finger_1" or geom.get("class", "").startswith("fingertip_pad_collision_"):
                body.remove(geom)
    reference = mount_reference(cfg)
    # +Z of sensor points into the jaw gap (-Y of the left finger).
    mount = ET.SubElement(finger, "body", name="wowskin_mount",
                          pos=" ".join(map(str, reference["position_m"])),
                          quat=".7071067811865476 .7071067811865476 0 0", childclass="tactile")
    extra_asset, housing, flex = standard_xml(cfg)
    asset.append(ET.fromstring(extra_asset))
    shell = ET.fromstring(housing)
    shell.set("mass", ".025")
    mount.append(shell)
    skin = ET.fromstring(flex)
    skin.find("contact").set("contype", "2")
    # Internal vertex/face contacts are skipped by MuJoCo unless the flex's
    # own contype intersects conaffinity. selfcollide remains none.
    skin.find("contact").set("conaffinity", "6")
    mount.append(skin)
    for sensor in layout:
        ET.SubElement(mount, "site", name=sensor["name"],
                      pos=" ".join(map(str, sensor["position_m"])), size=".0007", rgba=".2 .9 .8 1")
    # The upstream right finger already has a 180-degree Z rotation. Using
    # the same local mount pose makes the pads face one another at equal height.
    right = root.find(".//body[@name='right_finger']")
    passive_mount = ET.SubElement(right, "body", name="silicone_mount",
                                  pos=mount.get("pos"), quat=mount.get("quat"), childclass="tactile")
    backing = ET.fromstring(housing)
    backing.set("name", "silicone_backing")
    backing.set("mass", ".025")
    passive_mount.append(backing)
    passive = ET.fromstring(flex)
    passive.set("name", "silicone")
    passive.set("rgba", ".86 .83 .74 1")
    passive.find("contact").set("contype", "2")
    passive.find("contact").set("conaffinity", "6")
    passive_mount.append(passive)
    # This second flex is ordinary silicone: no chip sites or magnetic dipoles.
    # This probe moves in world XYZ; its controller acts in the PCB frame.
    probe = ET.SubElement(world, "body", name="probe")
    for axis, vector in zip("xyz", ["1 0 0", "0 1 0", "0 0 1"]):
        ET.SubElement(probe, "joint", name=f"probe_{axis}", type="slide", axis=vector, damping=".02")
    ET.SubElement(probe, "geom", name="probe_geom", type="sphere", size=str(cfg["probe_radius_m"]),
                  mass=str(cfg["probe_mass_kg"]), rgba=".2 .57 .92 1", contype="4", conaffinity="2",
                  friction=f"{cfg['friction']} .005 .0001", solref=".002 1")
    return ET.tostring(root, encoding="unicode")


class FrankaTactileSim(TactileSim):
    def __init__(self, cfg=None):
        cfg = configuration() if cfg is None else cfg
        super().__init__(cfg, xml_factory=build_scene, mount_body="wowskin_mount")
        self.initialize_mounts_and_arm()
        self.set_probe_side("left")
        self.baseline = self.magnetic_field()
        self.mode = "auto"
        self.phase = "unloaded"
        # Establish an unloaded baseline after gravitational settling.
        for _ in range(round(.25/cfg["timestep_s"])):
            self.control(mode="off")
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.baseline = self.magnetic_field().copy()
        self.data.time = 0.

    def initialize_mounts_and_arm(self):
        self.passive_mount = self.model.body("silicone_mount").id
        self.passive_flex = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_FLEX, "silicone")
        self.passive_vertices = flex_vertex_slice(self.model, self.passive_flex)
        self.passive_tets = flex_tetrahedra(self.model, self.passive_flex)
        self.passive_rest = self.passive_vertices_local()
        vertices = self.passive_rest[self.passive_tets]
        self.passive_dm_inverse = np.linalg.inv(np.stack(
            [vertices[:, k]-vertices[:, 0] for k in (1, 2, 3)], axis=-1))
        self.probe_side = "left"
        self.probe_mount = self.mount_body
        self.arm_joints = [self.model.joint(f"joint{i}").id for i in range(1, 8)]
        self.arm_q = self.model.jnt_qposadr[self.arm_joints]
        self.arm_v = self.model.jnt_dofadr[self.arm_joints]
        self.arm_controls = [self.model.actuator(f"actuator{i}").id for i in range(1, 8)]
        self.joint_target = HOME.copy()
        self.data.qpos[self.arm_q] = HOME
        self.data.ctrl[self.model.actuator("fingers_actuator").id] = 0.
        mujoco.mj_forward(self.model, self.data)

    def to_world(self, local, mount=None):
        mount = self.mount_body if mount is None else mount
        return self.data.xpos[mount]+self.data.xmat[mount].reshape(3, 3) @ local

    def passive_vertices_local(self):
        return ((self.data.flexvert_xpos[self.passive_vertices]-self.data.xpos[self.passive_mount])
                @ self.data.xmat[self.passive_mount].reshape(3, 3))

    def set_probe_side(self, side):
        """Reposition the test probe unloaded; this is a fixture reset, not motion planning."""
        if side not in ("left", "right"):
            raise ValueError("Probe side must be left or right")
        self.probe_side = side
        self.probe_mount = self.mount_body if side == "left" else self.passive_mount
        self.mode, self.phase = "off", "unloaded"
        height = self.cfg["total_height_m"]+self.cfg["probe_radius_m"]+.001
        q = self.model.jnt_qposadr[self.model.joint("probe_x").id]
        self.data.qpos[q:q+3] = self.to_world(np.array([0., 0., height]), self.probe_mount)
        self.data.qvel[self.probe_dof:self.probe_dof+3] = 0.
        self.data.xfrc_applied[self.probe_body] = 0.
        mujoco.mj_forward(self.model, self.data)

    def control(self, mode=None):
        d, m, cfg = self.data, self.model, self.cfg
        error = self.joint_target-d.qpos[self.arm_q]
        torque = 600*error-45*d.qvel[self.arm_v]+d.qfrc_bias[self.arm_v]
        limits = m.actuator_ctrlrange[self.arm_controls]
        d.ctrl[self.arm_controls] = np.clip(torque, limits[:, 0], limits[:, 1])
        rotation = d.xmat[self.probe_mount].reshape(3, 3)
        position = (d.xpos[self.probe_body]-d.xpos[self.probe_mount]) @ rotation
        jacobian = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacobian, None, d.xpos[self.probe_body], self.probe_mount)
        relative_velocity = (d.qvel[self.probe_dof:self.probe_dof+3]-jacobian @ d.qvel) @ rotation
        t, mode = d.time, mode or self.mode
        target_x, load = 0., None
        if mode == "auto":
            if .5 <= t < 3.:
                a = np.clip((t-.5)/.3, 0, 1)
                load = cfg["normal_load_n"]*a*a*(3-2*a)
                target_x = .004*np.clip((t-1.4)/1.2, 0, 1)
            self.phase = "unloaded" if t < .5 else ("press/hold" if t < 1.4 else ("slide" if t < 2.6 else ("hold" if t < 3 else "release")))
        elif mode in ("press", "slide"):
            load = cfg["normal_load_n"]
            target_x = .003*np.sin(t*2) if mode == "slide" else 0.
            self.phase = mode
        else:
            self.phase = "unloaded"
        force = np.zeros(3)
        force[:2] = 80*(np.array([target_x, 0.])-position[:2])-.12*relative_velocity[:2]
        if load is None:
            height = cfg["total_height_m"]+cfg["probe_radius_m"]+.001
            force[2] = 80*(height-position[2])-.2*relative_velocity[2]
        else:
            force[2] = -load-.08*relative_velocity[2]
        d.xfrc_applied[self.probe_body, :3] = rotation @ force-cfg["probe_mass_kg"]*m.opt.gravity

    def tactile_quality(self):
        quality = super().tactile_quality()
        passive = self.passive_vertices_local()
        vertices = passive[self.passive_tets]
        ds = np.stack([vertices[:, k]-vertices[:, 0] for k in (1, 2, 3)], axis=-1)
        quality['passive_min_volume_ratio'] = float(np.linalg.det(ds @ self.passive_dm_inverse).min())
        return quality

    def reading(self):
        row = self.sample(lambda _: self.control())
        passive = self.passive_vertices_local()
        row["passive_max_displacement_m"] = float(np.linalg.norm(passive-self.passive_rest, axis=1).max())
        right_contact = self.contact_state(self.passive_flex)
        row["passive_normal_force_n"] = right_contact["normal_force_n"]
        row["passive_slip"] = right_contact["slip"]
        row["max_displacement_m"] = float(np.linalg.norm(self.skin_vertices_local()-self.rest, axis=1).max())
        row["joint_positions"] = self.data.qpos[self.arm_q].copy()
        row["mount_position_m"] = self.data.xpos[self.mount_body].copy()
        return row


def camera(sim, close=False, side=None):
    cam = mujoco.MjvCamera()
    if close:
        side = sim.probe_side if side is None else side
        mount = sim.mount_body if side == "left" else sim.passive_mount
        cam.lookat[:] = sim.to_world(np.array([0., 0., .009]), mount)
        cam.distance, cam.azimuth, cam.elevation = .20, (225 if side == "left" else 45), -10
    else:
        cam.lookat[:] = [.28, 0., .32]
        cam.distance, cam.azimuth, cam.elevation = 1.3, 225, -25
    return cam


def apply_free_camera(handle, cam):
    """Reset the camera mode too: changing lookat does not exit a fixed camera."""
    with handle.lock():
        handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        handle.cam.fixedcamid = -1
        handle.cam.trackbodyid = -1
        handle.cam.lookat[:] = cam.lookat
        handle.cam.distance = cam.distance
        handle.cam.azimuth, handle.cam.elevation = cam.azimuth, cam.elevation


class ViewerStyleGuard:
    """Reserve application keys without triggering MuJoCo display shortcuts.

    The passive callback does not consume native key events. Restore only the
    visualization flags bound to our keys; other viewer options remain usable.
    """
    def __init__(self, handle, keys):
        self.vis = [i for i, entry in enumerate(mujoco.mjVISSTRING) if entry[2] and entry[2] in keys]
        self.rnd = [i for i, entry in enumerate(mujoco.mjRNDSTRING) if entry[2] and entry[2] in keys]
        with handle.lock():
            self.vis_values = handle.opt.flags[self.vis].copy()
            self.rnd_values = handle.user_scn.flags[self.rnd].copy()

    def sync(self, handle):
        # sync copies native render changes back into user_scn. A second sync
        # is needed only on a conflict, so restoring flags reaches native GL.
        handle.sync()
        with handle.lock():
            changed = (not np.array_equal(handle.opt.flags[self.vis], self.vis_values)
                       or not np.array_equal(handle.user_scn.flags[self.rnd], self.rnd_values))
            handle.opt.flags[self.vis] = self.vis_values
            handle.user_scn.flags[self.rnd] = self.rnd_values
        if changed:
            handle.sync()


def demo(args):
    sim = FrankaTactileSim()
    predictor = None
    if args.pollen:
        from wowskin_lab.baselines.pollen import PollenPredictor
        predictor = PollenPredictor()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, frames = [], []
    renderer = mujoco.Renderer(sim.model, height=600, width=800) if args.render else None
    try:
        for i in range(350):
            row = sim.reading()
            row["pollen_probability"] = predictor.update(row["magnetic_ut"]) if predictor else float("nan")
            rows.append(row)
            if renderer and i % 10 == 0:
                from PIL import Image, ImageDraw
                renderer.update_scene(sim.data, camera=camera(sim))
                picture = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(picture)
                draw.rectangle((0, 0, 800, 52), fill=(20, 25, 32))
                draw.text((12, 7), "Franka Panda | left: WowSkin | right: plain silicone | UNCALIBRATED", fill="white")
                draw.text((12, 29), f"{sim.phase} | contact={row['normal_force_n']:.4f} N | Pollen p={row['pollen_probability']:.3f}", fill="white")
                frames.append(np.asarray(picture))
                if i == 120:
                    import imageio.v3 as iio
                    iio.imwrite(output/"franka_sensor.png", np.asarray(picture))
                    renderer.update_scene(sim.data, camera=camera(sim, True))
                    iio.imwrite(output/"finger_closeup.png", renderer.render())
                    renderer.update_scene(sim.data, camera=camera(sim, True, "right"))
                    iio.imwrite(output/"finger_opposite.png", renderer.render())
    finally:
        if renderer:
            renderer.close()
    values = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    from wowskin_lab.tools.sensor_diagnostics import save_episode
    save_episode(output/"episode", sim.cfg, sim, values)
    import csv
    with (output/"inference.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sim_time_s", *CHANNELS, "sim_slip", "pollen_probability", "sim_normal_force_n"])
        writer.writerows((r["time_s"], *r["magnetic_ut"], r["slip"], r["pollen_probability"], r["normal_force_n"]) for r in rows)
    plot_response(values, output/"response.png")
    if frames:
        import imageio.v3 as iio
        iio.imwrite(output/"franka_sensor.gif", np.stack(frames), duration=100, loop=0)
    actual_revision = subprocess.check_output(["git", "-C", str(FRANKA_REPO), "rev-parse", "HEAD"], text=True).strip()
    report = dict(status="SIMULATION INTEGRATION ONLY; magnetic model and channel mapping unvalidated",
                  robot="Franka Panda assets from Ju6276/franka_xbox", robot_revision=actual_revision,
                  fixture="left WowSkin + same-size right passive silicone; external sphere, not a full grasp",
                  mount_reference=mount_reference(sim.cfg),
                  channels=CHANNELS, samples=len(rows), max_force_n=float(values["normal_force_n"].max()),
                  pad_dimensions_mm=[sim.cfg["width_m"]*1000, sim.cfg["length_m"]*1000, sim.cfg["skin_thickness_m"]*1000],
                  passive_min_volume_ratio=float(values["passive_min_volume_ratio"].min()),
                  passive_max_force_n=float(values["passive_normal_force_n"].max()),
                  max_displacement_um=float(values["max_displacement_m"].max()*1e6),
                  min_volume_ratio=float(values["min_volume_ratio"].min()),
                  pollen_checkpoint="published checkpoint, unchanged" if predictor else None,
                  pollen_probability_range=[float(np.nanmin(values["pollen_probability"])), float(np.nanmax(values["pollen_probability"]))] if predictor else None,
                  warnings=sim.data.warning.number.tolist())
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    (output/"scene.xml").write_text(sim.xml)
    print(json.dumps(report, indent=2))


def plot_response(values, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True, layout="constrained")
    t = values["time_s"]
    axes[0].plot(t, values["normal_force_n"])
    axes[0].set_ylabel("Contact load (N)")
    axes[1].plot(t, values["slip_speed_m_per_s"]*1000)
    axes[1].set_ylabel("Relative slip (mm/s)")
    axes[2].plot(t, values["magnetic_ut"], lw=.65)
    axes[2].set_ylabel("15 delta-B channels (uT)")
    axes[3].step(t, values["slip"], where="post", label="Simulation contact/velocity label")
    axes[3].plot(t, values["pollen_probability"], label="Published Pollen checkpoint")
    axes[3].set(xlabel="Simulation time (s)", ylabel="Slip / model score", ylim=(-.05, 1.05))
    axes[3].legend(loc="upper right")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("Franka-mounted WowSkin prototype\nUncalibrated magnetic response; Pollen transfer is NOT validated")
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def viewer(args):
    import mujoco.viewer
    sim = FrankaTactileSim()
    from wowskin_lab.ui.tactile import TactileHUD
    hud = TactileHUD(sim)
    sim.mode = "off"
    predictor = None
    if args.pollen:
        from wowskin_lab.baselines.pollen import PollenPredictor
        predictor = PollenPredictor()
    keyboard = ViewerKeyboard(set(map(ord, "PSLBVADQ,.")) | {256},
                              title_contains="Franka Panda with one uncalibrated WowSkin")
    deadline = time.monotonic()+args.seconds if args.seconds else float("inf")
    close = False
    print("Franka sensor: P press | S slide | L release | B change probe side (unloaded reset) | V close-up | A/D arm | Q exit", flush=True)
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=keyboard.native_callback,
                                    show_left_ui=False, show_right_ui=False) as handle, keyboard:
        cam = camera(sim)
        apply_free_camera(handle, cam)
        style = None if keyboard.continuous else ViewerStyleGuard(handle, "PSLBVADQ,.")
        was_moving = False
        index = 0
        while handle.is_running() and time.monotonic() < deadline:
            start = time.monotonic()
            snapshot = keyboard.sample()
            for key in snapshot.commands:
                hud.key(key)
                if key == ord("Q"):
                    return
                if key in (ord("P"), ord("S"), ord("L")):
                    sim.mode = {ord("P"): "press", ord("S"): "slide", ord("L"): "off"}[key]
                elif key == ord("B"):
                    sim.set_probe_side("right" if sim.probe_side == "left" else "left")
                    if close:
                        cam = camera(sim, close)
                        apply_free_camera(handle, cam)
                elif key in (ord("A"), ord("D")) and not keyboard.continuous:
                    sim.joint_target[0] = np.clip(sim.joint_target[0]+(.05 if key == ord("A") else -.05), -.5, .5)
                elif key == ord("V"):
                    close = not close
                    cam = camera(sim, close)
                    apply_free_camera(handle, cam)
                elif key == 256:
                    close = False
                    apply_free_camera(handle, camera(sim))
            if keyboard.continuous:
                velocity = .3*((ord("A") in snapshot.held)-(ord("D") in snapshot.held))
                if was_moving and not velocity:
                    sim.joint_target[0] = sim.data.qpos[sim.arm_q[0]]
                sim.joint_target[0] = np.clip(sim.joint_target[0]+velocity/sim.cfg['sample_hz'], -.5, .5)
                was_moving = bool(velocity)
            row = sim.reading()
            probability = predictor.update(row["magnetic_ut"]) if predictor else float("nan")
            if index % 10 == 0:
                labels = ["Left / right", "Contact controls", "Probe side", "View / arm", "Mode",
                          "Left / right load [N]", "Pollen (left only)", *[f"U{i} dB XYZ [uT]" for i in range(1, 6)]]
                readings = ["WowSkin / plain silicone", "P press | S slide | L release",
                            f"{sim.probe_side} | B switch + unload", "V view | A/D arm | Q exit",
                            sim.phase, f"{row['normal_force_n']:.4f} / {row['passive_normal_force_n']:.4f}", f"{probability:.3f} (uncalibrated)",
                            *["  ".join(f"{v:+.2f}" for v in chip) for chip in row["magnetic_ut"].reshape(5, 3)]]
                handle.set_texts([(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                   "\n".join(labels), "\n".join(readings))])
            hud.show(handle, row, probability)
            if index % 100 == 0:
                print(f"t={row['time_s']:.2f} load={row['normal_force_n']:.4f} pollen={probability:.3f}", flush=True)
            if style:
                style.sync(handle)
            else:
                handle.sync()
            index += 1
            time.sleep(max(0., .01-(time.monotonic()-start)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["demo", "viewer"])
    parser.add_argument("--pollen", action="store_true", help="Use unchanged published Pollen checkpoint; transfer is unvalidated")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", default="outputs/franka_sensor")
    parser.add_argument("--seconds", type=float, default=0.)
    args = parser.parse_args()
    (demo if args.command == "demo" else viewer)(args)


if __name__ == "__main__":
    main()
