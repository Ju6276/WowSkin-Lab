import unittest
import numpy as np
from wowskin_lab.envs.peg_insert import PegInsertEvaluator
from wowskin_lab.envs.task_evaluation import task_config


class PegSuccessTests(unittest.TestCase):
    def test_requires_insertion_alignment_and_continuous_dwell(self):
        c = task_config('peg_insert')
        score = PegInsertEvaluator(c)
        origin = np.array([0., 0., c['socket_entry_local_z_m']+.025-c['tip_local_m'][2]])
        rotation = np.eye(3)
        def update(t, offset):
            return score.update(t, origin+offset, rotation, np.zeros(3), rotation)
        self.assertFalse(update(0., [0,0,-.04])['peg_inserted'])
        self.assertFalse(update(.1, [.01,0,0])['peg_inserted'])
        self.assertFalse(update(.2, [0,0,0])['peg_success'])
        self.assertFalse(update(.4, [.01,0,0])['peg_success'])
        self.assertFalse(update(.5, [0,0,0])['peg_success'])
        self.assertTrue(update(.8, [0,0,0])['peg_success'])
        score.reset()
        self.assertFalse(score.success)


if __name__ == '__main__':
    unittest.main()
