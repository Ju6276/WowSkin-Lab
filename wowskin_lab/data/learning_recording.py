"""One canonical raw dataset; ViSk and FTF exports happen after collection."""
from pathlib import Path

from .raw_recording import RawRobotRecorder


class LearningSimRecorder(RawRobotRecorder):
    def __init__(self, env, directory, source='keyboard', *, instruction=None, record_depth=True):
        task = getattr(env, 'task_name', 'pick_cube')
        if task == 'cable' and env.cfg.get('cable_layout') == 'single_clip':
            task = 'cable_insert'
        super().__init__(env, Path(directory)/task, source, instruction=instruction, record_depth=record_depth)
        self.phase = 0

    def _idle_roll(self, count):
        run = getattr(self.env, '_viewer_run', None)
        for _ in range(count):
            row = (self.env.step(self.env.hold_action()) if run is None else
                   run(self.env.step, self.env.hold_action()))[-1]
            self.capture(row)

    def start(self):
        self.phase = 0
        super().start()
        try:
            self.metadata.update(phase_codes={'0': 'actual pre-roll', '1': 'demonstration', '2': 'actual post-roll'},
                                 pre_roll_samples=61, post_roll_samples=20,
                                 algorithm='none; select ViSk or FTF at export time',
                                 joint_recording='measurements.h5: arm_qpos (rad), arm_qvel (rad/s), joint order in metadata')
            self._metadata()
            # Genuine observations for FTF's 40-sample tactile history and
            # pause boundaries. Save every sample once; never pad by copying.
            self.set_progress('PREPARING', 'Recording 0.6 s of actual baseline observations')
            self._idle_roll(60)
            self.phase = 1
            self.set_progress('RECORDING', 'Demonstration recording started')
        except BaseException as exc:
            self.abort(str(exc))
            raise

    def capture(self, row):
        super().capture(row)
        self._append('phase', self.phase)
        self.h5.flush()

    def finish(self, status='unreviewed'):
        if not self.active:
            return None
        if self.phase == 1 and status != 'interrupted':
            self.phase = 2
            self.set_progress('SAVING', 'Recording final 0.2 s of actual observations')
            try:
                self._idle_roll(20)
                # A final image must have an actually issued following action.
                # Record one real hold update if the last camera hit the final
                # state tick; never fabricate a terminal action by copying.
                if abs(float(self.h5['camera_timestamp'][-1])-float(self.h5['timestamp'][-1])) < 1e-8:
                    self._idle_roll(1)
            except BaseException as exc:
                self.abort(str(exc))
                raise
        return super().finish(status)

    def abort(self, reason):
        if self.active:
            self.metadata['error'] = reason
        return super().finish('interrupted')
