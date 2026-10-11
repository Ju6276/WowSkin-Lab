from pathlib import Path
from types import SimpleNamespace
import unittest

from wowskin_lab.ui.teleop_hud import saved_recording_notice


class RecordingNoticeTests(unittest.TestCase):
    def test_manual_success_is_not_automatic_success(self):
        recorder = SimpleNamespace(metadata={'task_evaluation': {'success': False}})
        notice = saved_recording_notice(recorder, Path('capture_000001'), 'success')
        self.assertIn('SAVED capture_000001', notice)
        self.assertIn('manual: SUCCESS', notice)
        self.assertIn('auto: NOT PASSED', notice)
        recorder.metadata['task_evaluation']['success'] = True
        self.assertIn('auto: PASS', saved_recording_notice(recorder, Path('capture_000002'), 'success'))

    def test_idle_enter_does_not_claim_save(self):
        notice = saved_recording_notice(SimpleNamespace(), None, 'success')
        self.assertIn('No active recording', notice)


if __name__ == '__main__':
    unittest.main()
