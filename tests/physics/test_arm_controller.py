"""Optional torque controller must not change the default or initial grasp."""
from types import SimpleNamespace
import unittest
import numpy as np
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv, response_configuration, manual_action, keyboard_translation
from wowskin_lab.envs.cable import PandaCableEnv
from wowskin_lab.envs.feeding import PandaFeedingEnv
from wowskin_lab.envs.peg_insert import PandaPegInsertEnv
from wowskin_lab.core.arm_control import upstream_controller
from wowskin_lab.data.robot_state import robot_metadata, world_from_base
from wowskin_lab.ui.keyboard import direction


class ArmControllerTests(unittest.TestCase):
    def test_all_tasks_share_front_facing_keys_and_actual_lateral_motion(self):
        directions = {'W': [1,0,0], 'S': [-1,0,0], 'A': [0,-1,0],
                      'D': [0,1,0], 'E': [0,0,1], 'Q': [0,0,-1]}
        for task, factory in [('pick_cube',PandaPickCubeTactileEnv),('cable',PandaCableEnv),
                              ('feeding',PandaFeedingEnv),('peg_insert',PandaPegInsertEnv)]:
            for controller in ('ik_pd', 'cartesian_impedance'):
                cfg = response_configuration(SimpleNamespace(task=task, skin_dof='trilinear', ideal=True,
                    arm_controller=controller, cable_layout='single_clip'))
                cfg.update(teleop_frame='base', teleop_base_lateral_sign=1., disable_workspace_limits=True,
                           keyboard_speed_m_s=.04, tcp_speed_m_s=.08)
                env = factory(cfg)
                base = world_from_base(env)[:3,:3]
                for yaw in (0.,135.,270.):
                    for key, expected in directions.items():
                        with self.subTest(task=task, controller=controller, key=key, yaw=yaw):
                            actual = base.T @ keyboard_translation(env,direction({ord(key)}),yaw)
                            np.testing.assert_allclose(actual,expected,atol=1e-12)
                for key, sign in (('A',-1.), ('D',1.)):
                    with self.subTest(task=task,controller=controller,movement=key):
                        env.reset()
                        origin = env.data.site_xpos[env.tcp_site].copy()
                        moving = False
                        for i in range(40):
                            action,moving=manual_action(env,{ord(key)},moving,135 if i<20 else 270)
                            env.step(action)
                            target = base.T @ (env.target_xyz-origin)
                            np.testing.assert_allclose(target[[0,2]],0.,atol=1e-9)
                            self.assertGreater(sign*target[1],0.)
                        actual = base.T @ (env.data.site_xpos[env.tcp_site]-origin)
                        self.assertGreater(sign*actual[1],.003)
                        self.assertFalse(np.any(env.data.warning.number))

    def test_all_task_resets_and_native_torque(self):
        for task, factory in [('pick_cube',PandaPickCubeTactileEnv),('cable',PandaCableEnv),
                              ('feeding',PandaFeedingEnv),('peg_insert',PandaPegInsertEnv)]:
            with self.subTest(task=task):
                cfg = response_configuration(SimpleNamespace(task=task, skin_dof='trilinear', ideal=True))
                self.assertNotIn('arm_controller',cfg)
                pd = factory(cfg,pollen=False)
                cart = factory(dict(cfg,arm_controller='cartesian_impedance'),pollen=False)
                np.testing.assert_allclose(pd.data.qpos,cart.data.qpos,atol=1e-12)
                cart.servo_xyz[0] += .001
                expected = upstream_controller()(cart.model,cart.data,cart.tcp_site,cart.arm_v,
                    pos=cart.servo_xyz,ori=cart.servo_rotation,joint=cart.arm_nullspace_posture,
                    **cart.cfg['cartesian_impedance'])
                limits=cart.model.actuator_ctrlrange[cart.arm_controls]
                cart.control()
                np.testing.assert_allclose(cart.data.ctrl[cart.arm_controls],np.clip(expected,*limits.T))
                for _ in range(10):cart.step(cart.hold_action())
                self.assertFalse(np.any(cart.data.warning.number))
                cart.reset()
                np.testing.assert_allclose(cart.arm_nullspace_posture,cart.data.qpos[cart.arm_q])

    def test_w_s_free_motion_has_straight_targets(self):
        for controller in ('ik_pd','cartesian_impedance'):
            for key in ('W','S'):
                with self.subTest(controller=controller,key=key):
                    cfg=response_configuration(SimpleNamespace(task='cable',skin_dof='trilinear',ideal=True,
                                               arm_controller=controller,cable_layout='single_clip'))
                    cfg.update(teleop_frame='base',disable_workspace_limits=True,
                               keyboard_speed_m_s=.12,tcp_speed_m_s=.12)
                    env=PandaCableEnv(cfg)
                    contract = robot_metadata(env)['controller_contract']
                    self.assertFalse(contract['workspace_limits_enabled'])
                    self.assertIsNone(contract['workspace_world_m'])
                    self.assertTrue(contract['cartesian_limits_enabled'])
                    self.assertEqual(contract['arm_controller'], controller)
                    self.assertEqual(contract['teleop_frame'], 'base')
                    origin=env.data.site_xpos[env.tcp_site].copy()
                    moving=False
                    # Short free-space stroke, before the folded arm approaches
                    # joint limits when moving back toward its base.
                    for i in range(50):
                        action,moving=manual_action(env,{ord(key)},moving,135 if i<25 else 270)
                        env.step(action)
                        np.testing.assert_allclose(env.target_xyz[1:],origin[1:],atol=1e-10)
                        self.assertLess(np.linalg.norm(env.data.site_xpos[env.tcp_site,1:]-origin[1:]),.001)
                    self.assertGreater((1 if key=='W' else -1)*(env.data.site_xpos[env.tcp_site,0]-origin[0]),.01)


if __name__=='__main__':unittest.main()
