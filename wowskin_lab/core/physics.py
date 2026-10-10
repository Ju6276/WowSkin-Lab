"""Deformable contact + advected magnetic dipoles; deliberately uncalibrated.

Mechanical state drives the observation. Slip labels and contact force NEVER
enter the magnetic forward model. SI internally, magnetic output in microtesla.
"""
from collections import deque
from pathlib import Path
import json

import mujoco
import numpy as np

from wowskin_lab.core.geometry import standard_xml

from wowskin_lab.paths import ROOT
CHANNELS = [f"mag{i}_{axis}" for i in range(1, 6) for axis in "xyz"]


def configuration(seed=0, randomize=False):
    cfg = json.loads((ROOT / "configs/sensor.json").read_text())
    # One physical profile for the standalone bench and mounted Franka skins.
    cfg.update(json.loads((ROOT / "configs/sensor_hardware.json").read_text()))
    cfg["seed"] = int(seed)
    cfg["randomized"] = bool(randomize)
    if randomize:
        rng = np.random.default_rng(seed)
        for key, lo, hi in [
            ("young_pa", 24000, 38000), ("friction", .35, .85),
            ("normal_load_n", .025, .075), ("travel_m", .0025, .005),
            ("direction_rad", -np.pi, np.pi),
            ("offset_x_m", -.001, .001), ("offset_y_m", -.001, .001),
            ("magnetization_a_per_m", 3200, 4800), ("noise_ut", .2, 1.0),
        ]:
            cfg[key] = float(rng.uniform(lo, hi))
    return cfg


def dipole_field(sensor_xyz, source_xyz, moments):
    """Magnetostatic point dipoles, mu0/(4*pi)=1e-7; return microtesla."""
    r = np.asarray(sensor_xyz)[:, None, :] - np.asarray(source_xyz)[None, :, :]
    distance = np.linalg.norm(r, axis=-1, keepdims=True)
    if np.any(distance < 1e-5):
        raise ValueError("Magnetic source is too close to a sensor.")
    unit = r / distance
    dot = np.sum(unit * moments[None, :, :], axis=-1, keepdims=True)
    return .1 * np.sum((3 * unit * dot - moments[None, :, :]) / distance**3, axis=1)


def flex_vertex_slice(model, flex_id):
    start = int(model.flex_vertadr[flex_id])
    return slice(start, start+int(model.flex_vertnum[flex_id]))


def flex_tetrahedra(model, flex_id):
    """Element indices are local to each flex, not the combined vertex array."""
    if model.flex_dim[flex_id] != 3:
        raise ValueError("The magnetic skin requires a volumetric tetrahedral flex")
    start = int(model.flex_elemdataadr[flex_id])
    count = int(model.flex_elemnum[flex_id])*4
    return model.flex_elem[start:start+count].reshape(-1, 4).copy()


def scene_xml(cfg, layout):
    extra_asset, housing, skin = standard_xml(cfg)
    probe_height = cfg["total_height_m"]+cfg["probe_radius_m"]+.001
    sites = "\n".join(
        f'<site name="{s["name"]}" pos="{" ".join(map(str, s["position_m"]))}" '
        'size=".0007" rgba=".2 .9 .8 1"/>' for s in layout
    )
    return f'''<mujoco model="Uncalibrated magnetic tactile prototype">
  <option timestep="{cfg['timestep_s']}" integrator="implicitfast"
          gravity="0 0 0" cone="elliptic" iterations="50" tolerance="1e-10"/>
  <statistic center="0 0 .006" extent=".06"/>
  <visual><global offwidth="960" offheight="720"/><map znear=".01"/>
    <headlight ambient=".4 .4 .4" diffuse=".7 .7 .7" specular=".05 .05 .05"/></visual>
  <default><geom friction="{cfg['friction']} .005 .0001" solref=".002 1"/></default>
  <asset>{extra_asset}<material name="board" rgba=".04 .18 .18 1"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".17 .19 .23"
             rgb2=".22 .24 .28" width="256" height="256"/>
    <material name="ground" rgba=".83 .85 .87 1" specular="0"/>
  </asset>
  <worldbody>
    <light pos=".03 -.03 .08" dir="-.3 .3 -1"/>
    <light pos="-.04 .02 .05" dir=".3 -.2 -1" diffuse=".5 .5 .5"/>
    <geom type="plane" pos="0 0 {cfg.get('ground_z_m', -.003)}" size=".1 .1 .001" material="ground"
          contype="0" conaffinity="0"/>
    {housing}
    {sites}
    {skin}
    <body name="probe" pos="{cfg['offset_x_m']} {cfg['offset_y_m']} {probe_height}">
      <joint name="probe_x" type="slide" axis="1 0 0"/>
      <joint name="probe_y" type="slide" axis="0 1 0"/>
      <joint name="probe_z" type="slide" axis="0 0 1"/>
      <geom name="probe_geom" type="sphere" size="{cfg['probe_radius_m']}"
            mass="{cfg['probe_mass_kg']}" rgba=".2 .57 .92 1"/>
      <site name="probe_marker" pos="0 0 .0038" size=".00045" rgba="1 1 1 1"/>
    </body>
  </worldbody>
</mujoco>'''


class TactileSim:
    def __init__(self, cfg, xml_factory=None, mount_body=None, contact_body="probe", contact_geom="probe_geom"):
        self.cfg = dict(cfg)
        self.magnetic_gains = np.ones(5)
        self.magnetic_fit = None
        if cfg.get('magnetic_fit'):
            if cfg.get('reskin_model'):
                raise ValueError('Choose fitted dipole OR ReSkin network output, not both')
            profile = json.loads(Path(cfg['magnetic_fit']).read_text())
            gains = np.asarray(profile.get('gains'), dtype=float)
            if (profile.get('schema') != 'wowskin_sim_magnetic_gain_v1' or gains.shape != (5,)
                    or not np.isfinite(gains).all() or np.any(gains <= 0)):
                raise ValueError('Expected a versioned profile with five finite positive magnetic gains')
            if not np.isclose(profile['fit_magnetization_a_per_m'], cfg['magnetization_a_per_m']):
                raise ValueError('Magnetization differs from the fitted reference; refit instead of double scaling')
            self.magnetic_gains, self.magnetic_fit = gains, profile
            self.cfg['magnetic_fit_profile'] = profile
        self.rng = np.random.default_rng(cfg["seed"])
        ref = json.loads((ROOT / "assets/reference/sensor_layout.json").read_text())
        self.layout = ref["sensors"]
        for sensor in self.layout:
            sensor["position_m"][2] = cfg["total_height_m"]-cfg["skin_thickness_m"]-cfg["sensor_gap_m"]
        self.sensor_xyz = np.array([s["position_m"] for s in self.layout])
        angles = np.deg2rad([s["rotation_deg"] for s in self.layout])
        self.rotations = np.array([
            [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
            for a in angles])
        self.xml = (xml_factory or scene_xml)(cfg, self.layout)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.mount_body = self.model.body(mount_body).id if mount_body else None
        self.skin_flex = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_FLEX, "skin")
        if self.skin_flex < 0:
            raise ValueError("Scene must contain the magnetic flex named skin")
        self.skin_vertices = flex_vertex_slice(self.model, self.skin_flex)
        self.probe_body = self.model.body(contact_body).id
        self.probe_geom = self.model.geom(contact_geom).id
        self.contact_geoms = {self.probe_geom}
        self.probe_dof = self.model.body_dofadr[self.probe_body]
        # Numerical damping on movable FEM nodes, not an identified material value.
        if self.mount_body is None:
            self.model.dof_damping[:] = .002
        if self.model.body_dofnum[self.probe_body] == 3:
            self.model.dof_damping[self.probe_dof:self.probe_dof+3] = .02
        mujoco.mj_forward(self.model, self.data)
        self.rest = self.skin_vertices_local()
        self.tets = flex_tetrahedra(self.model, self.skin_flex)
        vertices = self.rest[self.tets]
        dm = np.stack([vertices[:, k] - vertices[:, 0] for k in (1, 2, 3)], axis=-1)
        self.dm_inverse = np.linalg.inv(dm)
        self.volume = np.abs(np.linalg.det(dm)) / 6
        self.reference_moments = np.zeros((len(self.tets), 3))
        self.reference_moments[:, 2] = cfg["magnetization_a_per_m"] * self.volume
        self.baseline = self.magnetic_field()
        self.zero_time = 0.
        self.filtered = np.zeros((5, 3))
        self.delay = deque([np.zeros((5, 3)) for _ in range(cfg["delay_samples"])])
        self.drift_rate = self.rng.normal(0, cfg["drift_ut_per_s"], (5, 3))
        self.steps_per_sample = round(1 / cfg["sample_hz"] / cfg["timestep_s"])
        if not np.isclose(self.steps_per_sample * cfg["timestep_s"], 1 / cfg["sample_hz"]):
            raise ValueError("sample period must be an integer multiple of timestep")

    def skin_vertices_local(self):
        """Use the PCB frame so a rigid arm rotation cannot mimic deformation."""
        vertices = self.data.flexvert_xpos[self.skin_vertices]
        if self.mount_body is None:
            return vertices.copy()
        rotation = self.data.xmat[self.mount_body].reshape(3, 3)
        return (vertices-self.data.xpos[self.mount_body]) @ rotation

    def magnetic_field(self):
        v = self.skin_vertices_local()[self.tets]
        ds = np.stack([v[:, k] - v[:, 0] for k in (1, 2, 3)], axis=-1)
        deformation = ds @ self.dm_inverse
        self.min_volume_ratio = float(np.linalg.det(deformation).min())
        if self.min_volume_ratio <= .15:
            raise RuntimeError(
                f"Magnetic silicone FEM invalid: min V/V0={self.min_volume_ratio:.3f} "
                f"at t={self.data.time:.3f}s (limit 0.15). Physics stopped, not a gripper stop")
        u, _, vh = np.linalg.svd(deformation)
        rotation = u @ vh
        moments = np.einsum("nij,nj->ni", rotation, self.reference_moments)
        world = dipole_field(self.sensor_xyz, v.mean(axis=1), moments)
        return np.einsum("nji,nj->ni", self.rotations, world)*self.magnetic_gains[:, None]

    def controls(self):
        t = self.data.time
        cfg = self.cfg
        origin = np.array([cfg["offset_x_m"], cfg["offset_y_m"]])
        direction = np.array([np.cos(cfg["direction_rad"]), np.sin(cfg["direction_rad"])])
        def ease(a):
            a = np.clip(a, 0, 1)
            return a*a*(3-2*a)
        travel = cfg["travel_m"]
        if t < 1.2:
            x = 0.
        elif t < 2.0:
            x = travel * ease((t-1.2)/.8)
        elif t < 2.5:
            x = travel
        elif t < 3.3:
            x = travel * (1-2*ease((t-2.5)/.8))
        else:
            x = -travel * (1-ease((t-3.9)/.5))
        desired = origin + direction*x
        load = cfg["normal_load_n"] * ease((t-.3)/.3) if .3 <= t < 3.7 else None
        self.apply_probe_load(desired, load)

    def apply_probe_load(self, xy, load_n=None):
        """External bench control: fixed XY; None lifts, otherwise applies -Z load."""
        cfg = self.cfg
        xyz = self.data.xpos[self.probe_body]
        velocity = self.data.qvel[self.probe_dof:self.probe_dof+3]
        force = np.zeros(3)
        force[:2] = 80*(np.asarray(xy)-xyz[:2]) - .12*velocity[:2]
        if load_n is not None:
            if not np.isfinite(load_n) or load_n < 0:
                raise ValueError("Normal load must be finite and nonnegative")
            force[2] = -load_n - .08*velocity[2]
        else:
            clearance_height = cfg["total_height_m"]+cfg["probe_radius_m"]+.001
            force[2] = 80*(clearance_height-xyz[2]) - .2*velocity[2]
        self.data.xfrc_applied[self.probe_body, :3] = force

    def zero(self):
        """Reset simulated electronics only when the skin is unloaded and settled."""
        if self.contact_state()["normal_force_n"] > 1e-6:
            raise ValueError("Release the probe before zeroing")
        skin_dofs = self.data.qvel[:self.probe_dof]
        displacement = np.linalg.norm(self.skin_vertices_local()-self.rest, axis=1)
        if np.max(displacement) > 2e-6 or np.max(np.abs(skin_dofs)) > 1e-4:
            raise ValueError("Wait for the skin to settle before zeroing")
        self.baseline = self.magnetic_field().copy()
        self.zero_time = float(self.data.time)
        self.filtered[:] = 0
        self.delay.clear()
        self.delay.extend(np.zeros((5, 3)) for _ in range(self.cfg["delay_samples"]))

    def contact_state(self, flex_id=None):
        """Force-weighted tangential relative speed at actual flex contacts.

        Both object rotation and arm motion are included through point
        Jacobians. Flex velocities are interpolated from tetrahedral vertices,
        so elastic sticking is not labelled as slip just because the arm moves.
        """
        d, m = self.data, self.model
        flex_id = self.skin_flex if flex_id is None else flex_id
        selected = flex_vertex_slice(m, flex_id)
        positions = d.flexvert_xpos[selected]
        tets = self.tets if flex_id == self.skin_flex else flex_tetrahedra(m, flex_id)
        body_ids = m.flex_vertbodyid[selected]
        velocities = np.zeros_like(positions)
        for i, bid in enumerate(body_ids):
            if self.mount_body is not None:
                # World point Jacobians include motion inherited from the arm,
                # including the pinned vertices on a moving finger.
                jacobian = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jacobian, None, positions[i], int(bid))
                velocities[i] = jacobian @ d.qvel
            elif m.body_dofnum[bid]:
                adr = m.body_dofadr[bid]
                velocities[i] = d.qvel[adr:adr+3]
        normal_force = 0.
        weighted_speed = 0.
        force_on_probe = np.zeros(3)
        count = 0
        for i in range(d.ncon):
            contact = d.contact[i]
            object_geom = next((int(g) for g in contact.geom if int(g) in self.contact_geoms), None)
            if object_geom is None or flex_id not in contact.flex:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, wrench)
            fn = max(float(wrench[0]), 0.)
            if fn < 1e-8:
                continue
            side = int(np.flatnonzero(contact.flex == flex_id)[0])
            if contact.vert[side] >= 0:
                skin_v = velocities[contact.vert[side]]
            else:
                ids = tets[contact.elem[side]]
                v = positions[ids]
                bary = np.linalg.solve((v[1:]-v[0]).T, contact.pos-v[0])
                weights = np.clip(np.r_[1-bary.sum(), bary], 0, 1)
                weights /= weights.sum()
                skin_v = weights @ velocities[ids]
            normal = contact.frame.reshape(3, 3)[0]
            jacobian = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jacobian, None, contact.pos, int(m.geom_bodyid[object_geom]))
            probe_v = jacobian @ d.qvel
            relative = probe_v-skin_v
            tangent = relative-normal*np.dot(relative, normal)
            normal_force += fn
            weighted_speed += fn*np.linalg.norm(tangent)
            world_force = contact.frame.reshape(3, 3).T @ wrench[:3]
            force_on_probe += world_force * (1 if contact.geom[1] == object_geom else -1)
            count += 1
        speed = weighted_speed/normal_force if normal_force else 0.
        in_contact = normal_force >= self.cfg["contact_threshold_n"]
        slip = int(in_contact and speed >= self.cfg["slip_speed_m_per_s"])
        return dict(contact=int(in_contact), slip=slip, normal_force_n=normal_force,
                    slip_speed_m_per_s=speed, force_on_probe_n=force_on_probe,
                    contacts=count)

    def sample(self, controller=None):
        start = self.data.time
        after_step = getattr(self, 'after_physics_step', None)
        for _ in range(self.steps_per_sample):
            if controller is None:
                self.controls()
            else:
                controller(self)
            mujoco.mj_step(self.model, self.data)
            if after_step is not None:
                after_step()
        expected = start + self.steps_per_sample*self.cfg["timestep_s"]
        if not np.isclose(self.data.time, expected, atol=1e-8) or np.any(self.data.warning.number):
            raise RuntimeError("MuJoCo instability/warning; refusing to export invalid data")
        # Make positions, velocities and contact forces refer to the same time.
        mujoco.mj_forward(self.model, self.data)
        field = self.magnetic_field()
        ideal = field-self.baseline
        alpha = 1-np.exp(-1/self.cfg["sample_hz"]/max(self.cfg["filter_tau_s"], 1e-9))
        self.filtered += alpha*(ideal-self.filtered)
        measured = self.filtered + self.drift_rate*(self.data.time-self.zero_time)
        measured += self.rng.normal(0, self.cfg["noise_ut"], measured.shape)
        self.delay.append(measured.copy())
        output = self.delay.popleft()
        result = self.contact_state()
        result.update(time_s=float(self.data.time), magnetic_ut=output.reshape(15),
                      ideal_delta_ut=ideal.reshape(15), raw_field_ut=field.reshape(15),
                      probe_xyz_m=self.data.xpos[self.probe_body].copy(),
                      min_volume_ratio=self.min_volume_ratio)
        if self.magnetic_fit:
            result.update(magnetic_source='dipole_with_simulation_fitted_readout_gains',
                          magnetic_fit_validated_real=False,
                          unfitted_ideal_delta_ut=(ideal/self.magnetic_gains[:, None]).reshape(15),
                          magnetic_fit_gains=self.magnetic_gains.copy())
        return result
