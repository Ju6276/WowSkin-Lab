"""Long captures must not relax optical/robot clock synchronization."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import h5py
import mujoco
import numpy as np

from wowskin_lab.core.optical_tactile import OpticalFrame
from wowskin_lab.data.optical_recording import OpticalWriter, validate_optical


class OpticalTimestampTests(unittest.TestCase):
    def writer(self, path, frame_time):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><site name="left"/><site name="right"/></worldbody></mujoco>')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        data.time = 1000.
        env = SimpleNamespace(model=model, data=data, cfg={'sensor': 'tacto'},
                              sensors={side: (side, ()) for side in ('left', 'right')})
        frame = OpticalFrame(np.zeros((320, 240, 3), dtype=np.uint8),
                             np.zeros((320, 240)), 0., (), frame_time)
        optics = SimpleNamespace(render=lambda data: {'left': frame, 'right': frame})
        return OpticalWriter(path, env, optics)

    def test_delayed_frame_rejected_even_after_long_running_simulation(self):
        with TemporaryDirectory() as directory:
            writer = self.writer(Path(directory)/'optical.h5', 1000.-.005)
            try:
                with self.assertRaisesRegex(ValueError, 'Stale optical frame'):
                    writer.capture()
            finally:
                writer.close()

    def test_readback_rejects_stale_frame_inside_synchronized_container(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'optical.h5'
            writer = self.writer(path, 1000.)
            try:
                writer.capture()
            finally:
                writer.close()
            validate_optical(path, np.array([1000.]))
            with h5py.File(path, 'r+') as f:
                f['frames/000000/left'].attrs['time_s'] = 1000.-.005
            with self.assertRaisesRegex(ValueError, 'Stale optical sample'):
                validate_optical(path, np.array([1000.]))
