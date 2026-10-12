"""Replay actual edge-contact failures, then verify an ordinary cable grasp.

These checks cover numerical validity, not measured silicone/field accuracy.
"""
from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from wowskin_lab.core.physics import configuration
from wowskin_lab.envs.cable import PandaCableEnv
from wowskin_lab.envs.pick_cube import (
    hold_current_pose, limit_vector, response_configuration,
    rotation_matrix, rotation_vector,
)


def keyboard_replay_step(env, action, moving, was_moving):
    # Replay world-frame commands, preserving the original keyboard clutch
    # and lead limit; there is no need to reproduce the user's camera orbit.
    if moving != was_moving:
        hold_current_pose(env)
    if moving:
        actual = env.data.site_xpos[env.tcp_site]
        env.target_xyz = np.clip(env.target_xyz, actual-.006, actual+.006)
        rotation = env.data.site_xmat[env.tcp_site].reshape(3, 3)
        lead = rotation_vector(rotation.T @ env.target_rotation)
        env.target_rotation = rotation @ rotation_matrix(limit_vector(lead, .03))
    return env.step(action)[-1]


@unittest.skipUnless(mujoco.__version__.startswith('3.5.'),
                     'Legacy full-node MuJoCo 3.5 recordings; current reduced contacts covered by test_contact_integrity')
class CableStabilityChecks(unittest.TestCase):
    def test_single_clip_spacemouse_rim_contact_and_release(self):
        cfg = response_configuration(SimpleNamespace(task='cable', cable_layout='single_clip', skin_dof='full', ideal=True))
        cfg.update(tcp_speed_m_s=.12, tcp_angular_speed_rad_s=.5, gripper_control_mode='panda_menagerie_position')
        env = PandaCableEnv(cfg, pollen=False)
        with np.load(Path(__file__).resolve().parents[1]/'fixtures/cable_single_clip_contact.npz') as source:
            # Resume the old-profile state immediately before the recorded
            # one-sided shear. The old contact settings fail at t=26.31 s.
            # The recorded scene hid the adapter; restoring its static body
            # adds six external-wrench entries but no dynamic coordinates.
            original_xml = ET.fromstring(env.xml)
            original_xml.find('worldbody').remove(original_xml.find(".//body[@name='adapter_0']"))
            original_model = mujoco.MjModel.from_xml_string(ET.tostring(original_xml, encoding='unicode'))
            original_data = mujoco.MjData(original_model)
            state_type = int(mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_setState(original_model, original_data, source['state'], state_type)
            for bit in range(16):
                field = 1 << bit
                if not state_type & field or field == int(mujoco.mjtState.mjSTATE_XFRC_APPLIED):
                    continue
                state = np.empty(mujoco.mj_stateSize(original_model, field))
                mujoco.mj_getState(original_model, original_data, state, field)
                mujoco.mj_setState(env.model, env.data, state, field)
            for i in range(original_model.nbody):
                name = mujoco.mj_id2name(original_model, mujoco.mjtObj.mjOBJ_BODY, i)
                dest = env.model.body(name).id if name else 0
                env.data.xfrc_applied[dest] = original_data.xfrc_applied[i]
            for key in ('joint_target', 'target_xyz', 'servo_xyz', 'target_rotation',
                        'servo_rotation', 'grip_target', 'baseline', 'filtered'):
                setattr(env, key, source[key].copy())
            env.delay.clear()
            env.delay.extend(source['delay'])
            mujoco.mj_forward(env.model, env.data)
            rows, was_moving = [], bool(source['was'])
            for action in source['actions']:
                moving = bool(np.any(action[:6]))
                rows.append(keyboard_replay_step(env, action, moving, was_moving))
                was_moving = moving
        hold_current_pose(env)
        env.command_gripper(False)
        for i in range(120):
            action = env.hold_action()
            if i < 80:
                action[2] = .0005
            rows.append(env.step(action)[-1])
        self.check_rows(env, rows)
        self.assertGreater(max(r['normal_force_n'] for r in rows), .1)
        self.assertLess(rows[-1]['normal_force_n'], .001)
        self.assertLess(rows[-1]['passive_normal_force_n'], .001)
        self.assertGreater(rows[-1]['min_volume_ratio'], .95)
        print(f"Single-clip rim: min V/V0={min(r['min_volume_ratio'] for r in rows):.3f}, "
              f"released V/V0={rows[-1]['min_volume_ratio']:.3f}", flush=True)

    def test_mesh_quality_profile_preserves_shape_volume_and_conforming_faces(self):
        from collections import Counter
        from wowskin_lab.core.geometry import cap_mesh
        cfg=configuration();cfg['mesh_outline_samples']=12
        outline,points,_,_,volume=cap_mesh(cfg)
        cfg['cap_top_inner_fraction']=.85
        new_outline,new_points,tets,_,new_volume=cap_mesh(cfg)
        np.testing.assert_allclose(outline,new_outline)
        np.testing.assert_allclose(points.min(0),new_points.min(0))
        np.testing.assert_allclose(points.max(0),new_points.max(0))
        self.assertAlmostEqual(new_volume,volume,places=14)
        faces=Counter(tuple(sorted(np.delete(tet,i))) for tet in tets for i in range(4))
        self.assertTrue(all(count in (1,2) for count in faces.values()))

    def make_env(self):
        cfg = response_configuration(SimpleNamespace(task='cable', skin_dof='full', ideal=True))
        cfg['scene_design'] = 'ebim_task1_single_arm'
        cfg.update(keyboard_speed_m_s=.12, tcp_speed_m_s=.12, gripper_control_mode='panda_menagerie_position')
        env = PandaCableEnv(cfg, pollen=False)
        self.assertTrue(np.all(env.model.flex_internal))
        self.assertTrue(np.all(env.model.flex_contype & env.model.flex_conaffinity))
        self.assertIsNone(env.reskin_response)
        return env

    def check_rows(self, env, rows, network_preview=False):
        self.assertGreater(min(r['min_volume_ratio'] for r in rows), .15)
        self.assertGreater(min(r['passive_min_volume_ratio'] for r in rows), .15)
        if network_preview:
            from tests.physics.test_reduced_skin import ReducedTaskChecks
            ReducedTaskChecks().check_network_rows(rows)
        else:
            self.assertTrue(np.isfinite([r['magnetic_ut'] for r in rows]).all())
        np.testing.assert_allclose(np.diff([r['time_s'] for r in rows]), .01, atol=1e-8)
        np.testing.assert_array_equal(env.data.warning.number, 0)

    def test_recorded_edge_contacts_continue_and_release(self):
        with np.load(Path(__file__).resolve().parents[1]/'fixtures/cable_edge_commands.npz') as source:
            for episode in (1, 3):
                with self.subTest(episode=episode):
                    env = self.make_env()
                    actions = source[f'episode_{episode}_action']
                    moving = source[f'episode_{episode}_moving']
                    rows, was_moving = [], False
                    for action, held in zip(actions, moving):
                        rows.append(keyboard_replay_step(env, action, held, was_moving))
                        was_moving = held
                    # Include the failing command and another 90 ms beyond it.
                    for _ in range(10):
                        rows.append(keyboard_replay_step(env, actions[-1], True, was_moving))
                        was_moving = True
                    # Opening and retracting must recover, not leave a frozen
                    # gripper or a collapsed skin that merely escaped detection.
                    hold_current_pose(env)
                    env.command_gripper(False)
                    for i in range(140):
                        action = env.hold_action()
                        if i < 80:
                            action[2] = .0005
                        rows.append(env.step(action)[-1])
                    self.check_rows(env, rows)
                    self.assertGreater(max(r['normal_force_n'] for r in rows), .1)
                    self.assertLess(rows[-1]['normal_force_n'], .001)
                    self.assertLess(rows[-1]['passive_normal_force_n'], .001)
                    self.assertGreater(rows[-1]['min_volume_ratio'], .95)
                    print(f"Edge replay {episode}: {len(rows)} samples, "
                          f"min V/V0={min(r['min_volume_ratio'] for r in rows):.3f}, "
                          f"released V/V0={rows[-1]['min_volume_ratio']:.3f}", flush=True)

    def test_mid_cable_grasp_lift_and_release(self):
        env = self.make_env()
        # Match the stored six-second grasp diagnostic's approach speed.
        env.cfg['tcp_speed_m_s'] = .08
        rows = []
        initial = env.data.xpos[env.cable_bodies, 2].copy()
        for _ in range(600):
            t = env.data.time
            if t < 2.4:
                target = [.45, -.12, .185]
            elif t < 3.3:
                target = [.45, -.12, .1275]
            elif t < 4.2:
                target = [.45, -.12, .1275]
                env.command_gripper(True)
            elif t < 4.9:
                target = [.45, -.12, .16]
            else:
                target = [.45, -.10, .16]
                if t >= 5.4:
                    env.command_gripper(False)
            env.target_xyz[:] = target
            rows.append(env.step(env.hold_action())[-1])
        self.check_rows(env, rows)
        contact_frames = sum(r['grasped'] for r in rows)
        lift = max(float(np.max(r['cable_centerline_m'][:, 2]-initial)) for r in rows)
        self.assertGreater(contact_frames, 100)
        self.assertGreater(lift, .04)
        self.assertLess(rows[-1]['normal_force_n'], .001)
        self.assertLess(rows[-1]['passive_normal_force_n'], .001)
        print(f"Mid-cable: {contact_frames} two-pad contact frames, {lift*1000:.1f} mm lift", flush=True)


if __name__ == '__main__':
    unittest.main()
