import json
import struct
import tempfile
import time
import unittest
import numpy as np
from wowskin_lab.data.sensor_compare import BurstDecoder, SensorComparison
from wowskin_lab.ui.sensor_compare import RealSensorHUD, comparison_panel
from wowskin_lab.ui.tactile import TactileHUD
from types import SimpleNamespace


class ComparisonChecks(unittest.TestCase):
    def test_fragmentation_recovery_and_invalid_packet(self):
        values = np.arange(20,dtype=float)
        packet = struct.pack('<20f',*values)+b'\r\n'
        bad = struct.pack('<20f',*([float('nan')]*20))+b'\r\n'
        stream = b'bad'+packet+bad+packet
        decoder = BurstDecoder(); frames=[]
        for i in range(0,len(stream),7): frames.extend(decoder.feed(stream[i:i+7]))
        self.assertEqual(len(frames),2)
        np.testing.assert_array_equal(frames[0].reshape(-1),values)
        self.assertEqual(decoder.discarded,85)
        self.assertEqual(decoder.buffer,b'')

    def test_raw_preserved_baseline_logged_and_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = SensorComparison('not-opened',tmp,{})
            now=time.monotonic_ns();wall=time.time_ns()
            frame=np.arange(20,dtype=float).reshape(5,4)
            session.ingest(frame,now,wall)
            self.assertIsNone(session.snapshot()[0]['delta_ut'])
            session.begin_baseline(); start=session.pending['start']
            for i in range(101): session.ingest(frame,start+i*20_000_000,wall+i*20_000_000)
            self.assertEqual(session.baseline_id,1)
            session.ingest(frame+3,start+3_000_000_000,wall+3_000_000_000)
            snapshot=session.snapshot()[0]
            np.testing.assert_array_equal(snapshot['delta_ut'],np.full(15,3))
            np.testing.assert_array_equal(snapshot['raw_magnetic_ut'],(frame+3)[:,1:].reshape(15))
            session.__exit__(None,None,None)
            events=[json.loads(x) for x in (session.path/'events.jsonl').read_text().splitlines()]
            baseline=next(x for x in events if x['kind']=='baseline')
            np.testing.assert_array_equal(baseline['std_ut'],np.zeros(15))
            self.assertEqual(len((session.path/'real.jsonl').read_text().splitlines()),103)
            self.assertEqual(json.loads((session.path/'manifest.json').read_text())['status'],'closed')

    def test_real_layout_and_stale_panel(self):
        sim=SimpleNamespace(rotations=np.tile(np.eye(3),(5,1,1)),sensor_xyz=np.zeros((5,3)))
        real=RealSensorHUD(sim)
        session=SimpleNamespace(snapshot=lambda: (dict(delta_ut=[0.]*15,raw_magnetic_ut=[2.]*15,
                                                      baseline_id=1,seq=10),False,None,2.))
        panel=comparison_panel(TactileHUD(sim),real,dict(magnetic_ut=np.zeros(15),time_s=1.),session)
        self.assertEqual(panel.shape,(650,800,3))

if __name__=='__main__': unittest.main()
