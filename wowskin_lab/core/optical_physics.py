"""Rigid optical sensor mechanics used by the shared task/controller runner.

No magnetic fields or FEM placeholders. RGB/depth are rendered separately at
the acquisition camera clock from this exact physical state.
"""
import xml.etree.ElementTree as ET
import mujoco
import numpy as np


def enabled(cfg):
    return cfg.get('sensor') in ('tacto', 'digit_fots', 'gelsight')


def initialize(env, cfg, factory, contact_body, contact_geom):
    from wowskin_lab.envs.optical import configure_optical_solver, sensor_registration, sensor_prefix
    env.cfg = dict(cfg)
    env.rng = np.random.default_rng(cfg.get('seed', 0))
    root = ET.fromstring(factory(cfg, []))
    # A task may set its own solver for AnySkin. Source optical settings win
    # explicitly at the final assembly boundary, including reset/rollout.
    configure_optical_solver(root.find('option'), cfg['sensor'])
    if cfg.get('optical_task_integrator'):
        root.find('option').set('integrator', cfg['optical_task_integrator'])
    size = root.find('size')
    if size is None:
        size = ET.SubElement(root, 'size')
    # Dense source Newton needs more arena for the original multi-part peg.
    # Allocate memory; do not drop contacts or change the solver.
    for name in ('njmax', 'nconmax', 'nstack'):
        size.attrib.pop(name, None)
    size.set('memory', '128M')
    env.xml = ET.tostring(root, encoding='unicode')
    env.model = mujoco.MjModel.from_xml_string(env.xml)
    env.data = mujoco.MjData(env.model)
    env.probe_body = env.model.body(contact_body).id
    env.probe_geom = env.model.geom(contact_geom).id
    env.contact_geoms = {env.probe_geom}
    env.steps_per_sample = round(1/cfg['sample_hz']/cfg['timestep_s'])
    if not np.isclose(env.steps_per_sample*cfg['timestep_s'], 1/cfg['sample_hz']):
        raise ValueError('Optical physics step must divide the state sample period')
    prefix = sensor_prefix(cfg['sensor'])
    env.sensors = sensor_registration(cfg['sensor'], {s: f'{prefix}_{s}' for s in ('left', 'right')})
    env.optical_gel_geoms = {}
    for side, (_, names) in env.sensors.items():
        env.optical_gel_geoms[side] = {env.model.geom(n).id for n in
                                     ((names,) if isinstance(names, str) else names)}
    env.mount_body = env.model.body(f'{prefix}_left_module').id
    env.passive_mount = env.model.body(f'{prefix}_right_module').id
    env.magnetic_fit = None
    mujoco.mj_forward(env.model, env.data)


def surfaces(env):
    from wowskin_lab.envs.optical import sensor_geometry
    local = sensor_geometry(env.cfg['sensor'])['source_surface']
    return np.array([env.data.xpos[b]+env.data.xmat[b].reshape(3,3)@local
                     for b in (env.mount_body, env.passive_mount)])


def contact_state(env, side, objects=None):
    m, d = env.model, env.data
    geoms = env.optical_gel_geoms[side]
    total, weighted, count = 0., 0., 0
    wrench = np.zeros(6)
    for i, contact in enumerate(d.contact):
        which = next((s for s in (0, 1) if contact.geom[s] in geoms), None)
        if which is None:
            continue
        other = int(contact.geom[1-which])
        if other < 0 or (objects is not None and other not in objects):
            continue
        mujoco.mj_contactForce(m, d, i, wrench)
        force = max(0., float(wrench[0]))
        if force == 0:
            continue
        velocity = []
        for geom in contact.geom:
            jac = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jac, None, contact.pos, int(m.geom_bodyid[geom]))
            velocity.append(jac@d.qvel)
        relative = velocity[1]-velocity[0]
        normal = contact.frame[:3]
        weighted += force*np.linalg.norm(relative-normal*np.dot(relative, normal))
        total += force
        count += 1
    return dict(normal_force_n=total, contacts=count, contact=bool(count),
                slip_speed_m_per_s=weighted/total if total else 0.)


def sample(env, controller):
    start = float(env.data.time)
    callback = getattr(env, 'after_physics_step', None)
    for _ in range(env.steps_per_sample):
        controller(env)
        mujoco.mj_step(env.model, env.data)
        if callback:
            callback()
    if (not np.isclose(env.data.time, start+1/env.cfg['sample_hz'], atol=1e-8)
            or np.any(env.data.warning.number) or not np.isfinite(env.data.qpos).all()):
        raise RuntimeError('MuJoCo instability/warning; refusing to export invalid optical state')
    mujoco.mj_forward(env.model, env.data)
    left, right = (contact_state(env, side) for side in ('left', 'right'))
    target = [contact_state(env, side, env.contact_geoms)['normal_force_n'] for side in ('left', 'right')]
    return dict(**left, passive_normal_force_n=right['normal_force_n'],
                passive_slip_speed_m_per_s=right['slip_speed_m_per_s'],
                target_normal_forces_n=np.array(target), time_s=float(env.data.time),
                tactile_modality='optical', tactile_status='rigid contact; source optical renderer',
                joint_positions=env.data.qpos[env.arm_q].copy(),
                probe_xyz_m=env.data.xpos[env.probe_body].copy())
