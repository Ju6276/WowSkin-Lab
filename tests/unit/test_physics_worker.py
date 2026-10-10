"""Viewer remains responsive without sharing live MuJoCo state."""
import threading
import time
import unittest

import mujoco
import numpy as np

from wowskin_lab.ui.keyboard import KeyState
from wowskin_lab.ui.physics_worker import PhysicsWorker, ViewerInterrupted


class Env:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body pos="0 0 1"><freejoint/><geom type="sphere" size=".01"/></body>
        </worldbody></mujoco>''')
        self.data = mujoco.MjData(self.model)

    def advance(self, n, delay=0.):
        for _ in range(n):
            if getattr(self, '_viewer_progress', None):
                self._viewer_progress()
            mujoco.mj_step(self.model, self.data)
            if delay:
                time.sleep(delay)
        return self.data.qpos.copy()


class Handle:
    def __init__(self):
        self.count = 0
        self.on_sync = lambda: None
        self.threads = set()

    def is_running(self):
        return True

    def sync(self):
        self.count += 1
        self.threads.add(threading.get_ident())
        self.on_sync()


class PhysicsWorkerTests(unittest.TestCase):
    def test_detached_view_and_identical_dynamics(self):
        expected, actual = Env(), Env()
        reference = expected.advance(40)
        handle, keyboard = Handle(), KeyState({ord('P')})
        model, data = PhysicsWorker.display(actual)
        def ui_edit():
            model.opt.gravity[:] = 100  # Native UI cannot alter live physics.
        handle.on_sync = ui_edit
        with PhysicsWorker(actual, handle, keyboard, model, data) as worker:
            result = worker.run(actual.advance, 40, .002)
        np.testing.assert_array_equal(result, reference)
        np.testing.assert_array_equal(actual.model.opt.gravity, [0, 0, -9.81])
        self.assertGreater(handle.count, 3)
        self.assertEqual(handle.threads, {threading.get_ident()})
        self.assertIsNone(actual._viewer_progress)

    def test_exit_interrupts_long_reset_and_preserves_command(self):
        env, handle, keys = Env(), Handle(), KeyState({ord('P')})
        model, data = PhysicsWorker.display(env)
        def quit_during_sync():
            if handle.count == 3:
                keys.event(ord('P'), True)
                keys.event(ord('P'), False)
        handle.on_sync = quit_during_sync
        aborts = []
        with PhysicsWorker(env, handle, keys, model, data, aborts.append) as worker:
            with self.assertRaises(ViewerInterrupted):
                worker.run(env.advance, 1000, .002)
        self.assertLess(env.data.time, .2)
        self.assertEqual(keys.sample().commands, (ord('P'),))
        self.assertEqual(len(aborts), 1)

    def test_worker_fault_propagates_and_next_reset_can_run(self):
        env, handle, keys = Env(), Handle(), KeyState(set())
        model, data = PhysicsWorker.display(env)
        with PhysicsWorker(env, handle, keys, model, data) as worker:
            def failure():
                raise RuntimeError('invalid FEM')
            with self.assertRaisesRegex(RuntimeError, 'invalid FEM'):
                worker.run(failure)
            worker.run(env.advance, 2)
        self.assertGreater(env.data.time, 0.)
