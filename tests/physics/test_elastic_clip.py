import unittest
from scripts.diagnostics.check_elastic_clip import run
from wowskin_lab.envs.elastic_clip import configuration


class ElasticClipTests(unittest.TestCase):
    def test_panda_inserts_a_grasped_cable_and_releases(self):
        from types import SimpleNamespace
        import numpy as np
        from wowskin_lab.envs.cable import PandaCableEnv
        from wowskin_lab.envs.pick_cube import response_configuration
        c=response_configuration(SimpleNamespace(task='cable',cable_layout='single_clip',
                                                 clip_model='elastic_beam',ideal=True))
        e=PandaCableEnv(c)
        opening, lower_bend, held=0.,0.,0
        for i in range(1200):
            t=i*.01
            if t<2.4:
                p=[.53,-.12,.185]
            elif t<4.2:
                p=[.53,-.12,.123]
            elif t<8.:
                p=[.53,-.12+.025*(t-4.2),.123]
            else:
                p=[.53,-.025,.123]
            if t>=3.3:
                e.command_gripper(t<10.)
            e.target_xyz[:]=p
            r=e.step(e.hold_action())[-1]
            opening=max(opening,r['clip_tip_distance_m'])
            lower_bend=max(lower_bend,r['clip_lower_max_bend_rad'])
            held+=int(r['grasped'])
            self.assertGreater(min(r['min_volume_ratio'],r['passive_min_volume_ratio']),0.)
        self.assertTrue(e.success)
        self.assertGreater(held,100)
        self.assertGreater(opening,.007)
        self.assertGreater(lower_bend,.001)
        self.assertLess(r['normal_force_n'],.001)
        self.assertFalse(np.any(e.data.warning.number))
        print(f'Elastic cable insertion: success={e.success}, after-release inside={r["clip_inside"]}, opening={opening*1000:.2f} mm')

    def test_lower_lip_bends_and_returns_without_moving_mount(self):
        _, _, report = run(configuration(), probe_height=.007)
        self.assertFalse(any(report['warnings']))
        self.assertGreater(report['lower_max_tip_displacement_mm'], 1.)
        self.assertLess(report['lower_final_tip_displacement_mm'], .05)
        self.assertEqual(report['mounting_base_displacement_m'], 0.)

    def test_passive_clip_opens_and_recovers_after_withdrawal(self):
        _, values, report = run(configuration())
        self.assertFalse(any(report['warnings']))
        self.assertGreater(report['maximum_tip_distance_mm'],7.)
        self.assertLess(abs(report['final_tip_distance_mm']-report['initial_tip_distance_mm']),.05)
        self.assertLess(report['final_max_bend_rad'],.001)
        self.assertGreater(values[:,3].max(),1e-4)
        self.assertLess(report['final_elastic_energy_j'],1e-7)


if __name__=='__main__':unittest.main()
