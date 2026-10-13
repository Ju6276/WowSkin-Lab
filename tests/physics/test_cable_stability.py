"""Replay recorded edge-contact commands on the reduced skin, then release.

These checks cover numerical validity, not measured silicone/field accuracy.
"""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from wowskin_lab.core.geometry import cap_mesh
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


@unittest.skipUnless(mujoco.__version__ == '3.15.0', 'Requires reduced environment')
class CableStabilityChecks(unittest.TestCase):
    def test_mesh_quality_profile_preserves_shape_volume_and_conforming_faces(self):
        cfg = response_configuration(SimpleNamespace(task='cable', skin_dof='trilinear', ideal=True))
        cfg['mesh_outline_samples'] = 48
        outline, points, _, _, volume = cap_mesh(dict(cfg, cap_top_inner_fraction=1.))
        new_outline, new_points, tets, _, new_volume = cap_mesh(dict(cfg, cap_top_inner_fraction=.85))
        np.testing.assert_allclose(outline, new_outline)
        np.testing.assert_allclose(points.min(0), new_points.min(0))
        np.testing.assert_allclose(points.max(0), new_points.max(0))
        self.assertAlmostEqual(new_volume, volume, places=14)
        faces = Counter(tuple(sorted(np.delete(tet, i))) for tet in tets for i in range(4))
        self.assertTrue(all(count in (1, 2) for count in faces.values()))

    def make_env(self):
        cfg = response_configuration(SimpleNamespace(task='cable', skin_dof='trilinear', ideal=True))
        cfg['scene_design'] = 'ebim_task1_single_arm'
        cfg.update(keyboard_speed_m_s=.12, tcp_speed_m_s=.12, gripper_control_mode='panda_menagerie_position')
        env = PandaCableEnv(cfg, pollen=False)
        self.assertEqual(env.cfg['cap_top_inner_fraction'], .85)
        self.assertIsNone(env.reskin_response)
        return env

    def check_rows(self, env, rows):
        self.assertGreater(min(r['min_volume_ratio'] for r in rows), .15)
        self.assertGreater(min(r['passive_min_volume_ratio'] for r in rows), .15)
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
                    # Include the last recorded command and another 90 ms beyond it.
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


if __name__ == '__main__':
    unittest.main()
