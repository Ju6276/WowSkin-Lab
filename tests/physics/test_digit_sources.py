"""Source mechanical fidelity, sustained grasp and empty closure regressions."""
import unittest
import mujoco
import numpy as np
from wowskin_lab.envs.digit_sources import source, geometry, CONTACT_FIELDS
from wowskin_lab.envs.optical import OpticalPandaEnv
from wowskin_lab.core.optical_backends import make_optics


class DigitSourceChecks(unittest.TestCase):
    def test_rounded_ends_point_distally_and_sensing_faces_oppose(self):
        # The two source CADs have different axes. A facing-only check
        # misses a sensor mounted sideways about its sensing normal.
        for sensor, distal, normal in (
            ('tacto', [0, 0, -1], [0, 1, 0]),
            ('digit_fots', [-1, 0, 0], [0, 0, 1]),
        ):
            with self.subTest(sensor=sensor):
                env = OpticalPandaEnv(sensor=sensor)
                model, data = env.model, env.data
                centers, normals = [], []
                for side in ('left', 'right'):
                    module = model.body(f'digit_{side}_module').id
                    finger = model.body(f'{side}_finger').id
                    rotation = data.xmat[module].reshape(3, 3)
                    finger_rotation = data.xmat[finger].reshape(3, 3)
                    np.testing.assert_allclose(rotation @ distal,
                                               finger_rotation[:, 2], atol=1e-10)
                    centers.append(data.xpos[module] + rotation @ geometry(sensor)['source_surface'])
                    normals.append(rotation @ normal)
                gap = centers[1] - centers[0]
                direction = gap / np.linalg.norm(gap)
                self.assertGreater(np.dot(normals[0], direction), .999999)
                self.assertGreater(np.dot(normals[1], -direction), .999999)

    def test_sensor_parameters_mass_meshes_and_camera_preserved(self):
        for sensor in ('tacto','digit_fots'):
            with self.subTest(sensor=sensor):
                _, original, _=source(sensor)
                env=OpticalPandaEnv(sensor=sensor)
                model=env.model
                for side in ('left','right'):
                    for g in range(original.ngeom):
                        new=model.geom(f'digit_{side}_g{g}').id
                        for attr in (*CONTACT_FIELDS,'size','rgba','type'):
                            np.testing.assert_allclose(getattr(model,'geom_'+attr)[new],
                                getattr(original,'geom_'+attr)[g],atol=1e-8,err_msg=f'{sensor} {g} {attr}')
                        if original.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                            oldmesh,newmesh=original.geom_dataid[g],model.geom_dataid[new]
                            self.assertEqual(original.mesh_vertnum[oldmesh],model.mesh_vertnum[newmesh])
                            self.assertEqual(original.mesh_facenum[oldmesh],model.mesh_facenum[newmesh])
                    for b in range(1,original.nbody):
                        oldname=original.body(b).name
                        newname=f'digit_{side}_module' if oldname=='source_digit' else f'digit_{side}_{oldname}'
                        new=model.body(newname).id
                        for attr in ('mass','ipos','inertia'):
                            np.testing.assert_allclose(getattr(model,'body_'+attr)[new],
                                getattr(original,'body_'+attr)[b],atol=1e-10)
                    if sensor=='digit_fots':
                        cam=model.camera('optical_digit_'+side+'_camera').id
                        src=original.camera('digit:camera').id
                        for attr in ('pos','quat','fovy'):
                            np.testing.assert_array_equal(getattr(model,'cam_'+attr)[cam],getattr(original,'cam_'+attr)[src])
                np.testing.assert_array_equal(model.opt.gravity,[0,0,-9.81])
                self.assertEqual(model.nflex,0)

    def test_sixty_second_hold_and_reopen(self):
        for sensor in ('tacto','digit_fots','gelsight'):
            with self.subTest(sensor=sensor):
                env=OpticalPandaEnv(sensor=sensor)
                optics=make_optics(env,sensor,ideal=True)
                def move(xyz,seconds):
                    env.target_xyz=np.asarray(xyz,float)
                    for _ in range(round(seconds*100)):
                        env.step(env.hold_action())
                try:
                    move([.4,0,.17],1.5)
                    move([.4,0,.125],1.5)
                    env.command_gripper(True)
                    move([.4,0,.125],1)
                    move([.4,0,.23],2)
                    initial=env.data.xpos[env.probe_body].copy()
                    self.assertGreater(initial[2],.22)
                    first=optics.render(env.data)
                    for _ in range(60):
                        move([.4,0,.23],1)
                        frames=optics.render(env.data)
                        for side,frame in frames.items():
                            self.assertGreater(frame.normal_force_n,.1)
                            if frame.marker_displacement_px is not None:
                                # Original FOTS has integer-pixel shear: 0.1 mm can
                                # legitimately move 1.2 px, plus <1 px quantization.
                                # This is a test tolerance, never an output clamp.
                                self.assertLess(abs(frame.marker_displacement_px-first[side].marker_displacement_px).max(),.0001*12000+1)
                    self.assertLess(np.linalg.norm(env.data.xpos[env.probe_body]-initial),.0001)
                    env.command_gripper(False)
                    move([.4,0,.23],1)
                    self.assertLess(env.data.xpos[env.probe_body,2],.121)
                    # Closing without an object must not require a local gap
                    # or finger-contact-stop and must reopen after contact.
                    env.command_gripper(True)
                    move([.4,0,.23],2)
                    self.assertTrue(np.isfinite(env.data.qpos).all())
                    env.command_gripper(False)
                    move([.4,0,.23],1)
                    self.assertLess(env.gripper_fraction(),.01)
                finally:
                    optics.close()
