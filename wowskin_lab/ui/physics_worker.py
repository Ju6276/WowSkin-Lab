"""Run physics off the viewer loop; render coherent, detached snapshots.

The worker exclusively owns the live environment during an operation. The
viewer owns a separate model/data, so sync() cannot reset or perturb live
physics halfway through a solver step. Simulation clocks are not changed.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import mujoco
import numpy as np


class ViewerInterrupted(Exception):
    """A user stopped an operation between physics steps."""


class PhysicsWorker:
    def __init__(self, env, handle, keyboard, display_model, display_data, on_abort=None):
        self.env, self.handle, self.keyboard = env, handle, keyboard
        self.model, self.data = display_model, display_data
        self.on_abort = on_abort
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mujoco-physics')
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._state = np.empty(mujoco.mj_stateSize(env.model, self._spec))
        self._version = self._shown = 0
        self._solver = int(env.model.opt.solver)
        self._next = 0.
        self.last_operation_wall_s = 0.
        self.on_present = None
        self._previous_hook = getattr(env, '_viewer_progress', None)
        self._previous_run = getattr(env, '_viewer_run', None)

    @staticmethod
    def display(env):
        model = copy.copy(env.model)
        data = mujoco.MjData(model)
        mujoco.mj_copyData(data, model, env.data)
        return model, data

    def __enter__(self):
        self.env._viewer_progress = self.checkpoint
        self.env._viewer_run = self.run
        return self

    def __exit__(self, kind, value, tb):
        self._cancel.set()
        self._pool.shutdown(wait=True)
        self.env._viewer_progress = self._previous_hook
        self.env._viewer_run = self._previous_run
        return kind is ViewerInterrupted

    def checkpoint(self, force=False):
        if self._cancel.is_set():
            raise ViewerInterrupted()
        now = time.monotonic()
        if force or now >= self._next:
            with self._lock:
                mujoco.mj_getState(self.env.model, self.env.data, self._state, self._spec)
                self._solver = int(self.env.model.opt.solver)
                self._version += 1
            self._next = now + 1/30

    def present(self):
        # Never access live MjData here: the worker may be inside mj_step.
        with self._lock:
            if self._version != self._shown:
                mujoco.mj_setState(self.model, self.data, self._state, self._spec)
                self.model.opt.solver = self._solver
                self._shown = self._version
                changed = True
            else:
                changed = False
        if changed:
            mujoco.mj_forward(self.model, self.data)
        if self.handle.is_running():
            self.handle.sync()
            if self.on_present is not None:
                self.on_present(self.handle)

    def run(self, function, *args):
        self._cancel.clear()
        if function.__name__ == 'reset' or (function.__name__ == 'key_command' and args[-1] in (ord('R'), 259)):
            self.handle.set_texts([(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                   'Resetting simulation', 'Settling physical contacts | P exits')])
        def work():
            started = time.perf_counter()
            try:
                result = function(*args)
                self.checkpoint(force=True)
                return result
            finally:
                self.last_operation_wall_s = time.perf_counter()-started
        future = self._pool.submit(work)
        try:
            while not future.done():
                if not self.handle.is_running() or ord('P') in self.keyboard.peek().commands:
                    self._cancel.set()
                self.present()
                self._cancel.wait(1/60) if not self._cancel.is_set() else time.sleep(.001)
            result = future.result()
            self.present()
            return result
        except BaseException:
            # Stop/join before recorders, the viewer or the environment close.
            self._cancel.set()
            try:
                future.result()
            except BaseException:
                pass
            if self.on_abort:
                self.on_abort('Viewer operation interrupted or failed')
            raise
