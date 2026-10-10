"""Record one teleoperation session for both official learning workflows."""
from pathlib import Path
import uuid

from .ftf_recording import FTFSimRecorder
from .visk_recording import ViSkSimRecorder


class LearningSimRecorder:
    def __init__(self, env, directory, source='keyboard'):
        root = Path(directory)
        task = getattr(env, 'task_name', 'pick_cube')
        self.ftf = FTFSimRecorder(env, root/'ftf-robot/extracted_data'/task, source)
        try:
            self.visk = ViSkSimRecorder(env, root/'visk/raw'/task, source)
        except BaseException:
            self.ftf.close()
            raise

    @property
    def active(self):
        return self.visk.active or self.ftf.active

    @property
    def writer(self):
        return self.visk.writer if self.visk.active else self.ftf.writer

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def start(self):
        if self.active:
            raise RuntimeError('Finish the current demonstration first')
        # FTF records its genuine simulation pre-roll first. ViSk then starts;
        # no simulation steps are advanced between the two per-step captures.
        self.ftf.start()
        try:
            self.visk.start()
            pair = uuid.uuid4().hex
            for recorder in (self.ftf, self.visk):
                recorder.metadata['paired_recording_id'] = pair
            self.ftf._write_metadata()
            self.visk._metadata()
        except BaseException:
            self.finish('interrupted')
            raise

    def step(self, row):
        if self.active:
            self.visk.step(row)
            self.ftf.step(row)

    def finish(self, status='unreviewed'):
        # Close ViSk before FTF advances its post-roll simulation steps.
        try:
            visk = self.visk.finish(status)
        except BaseException:
            self.ftf.finish('interrupted')
            raise
        ftf = self.ftf.finish(status)
        return dict(visk=visk, ftf=ftf) if visk or ftf else None

    def close(self):
        try:
            self.visk.close()
        finally:
            self.ftf.close()

    def abort(self, reason):
        try:
            visk = self.visk.abort(reason)
        finally:
            ftf = self.ftf.abort(reason)
        return dict(visk=visk, ftf=ftf)
