"""Real WowSkin acquisition alongside simulation; host-time comparison, not force calibration.

Wire layout follows the installed official anyskin.sensor.AnySkinBase burst reader:
5 * (temperature, Bx, By, Bz) float32, followed by CR LF. No dummy fallback.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import threading
import time
import uuid

import numpy as np


def resolve_port(port):
    if port != 'auto':
        return port
    from serial.tools import list_ports
    candidates = [p.device for p in list_ports.comports() if (p.vid, p.pid) == (0x239A, 0x80CB)]
    if len(candidates) != 1:
        raise ValueError(f'Expected one WowSkin QT Py M0; found {candidates}. Supply --real-port explicitly.')
    return candidates[0]


class BurstDecoder:
    size = 82

    def __init__(self):
        self.buffer = bytearray()
        self.discarded = 0

    def feed(self, chunk):
        self.buffer.extend(chunk)
        frames = []
        while len(self.buffer) >= self.size:
            if self.buffer[80:82] != b'\r\n':
                del self.buffer[0]
                self.discarded += 1
                continue
            values = np.array(struct.unpack('<20f', self.buffer[:80])).reshape(5, 4)
            if not np.isfinite(values).all():
                del self.buffer[:82]
                self.discarded += 82
                continue
            frames.append(values)
            del self.buffer[:82]
        return frames


class SensorComparison:
    def __init__(self, port, output, cfg, official_baseline=False):
        port = resolve_port(port)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.path = Path(output)/(stamp+'_'+uuid.uuid4().hex[:6])
        self.path.mkdir(parents=True, exist_ok=False)
        self.port = port
        self.official_baseline = official_baseline
        self.decoder = BurstDecoder()
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.latest = None
        self.baseline = None
        self.baseline_id = 0
        self.pending = None
        self.error = None
        self.count = 0
        self.started_ns = time.monotonic_ns()
        self.thread = None
        self.files = {name: (self.path/(name+'.jsonl')).open('w') for name in ('real', 'sim', 'events')}
        self.wire = (self.path/'serial.bin').open('wb')
        from serial.tools import list_ports
        device = next((p for p in list_ports.comports() if p.device == port), None)
        identity = None if device is None else dict(vid=device.vid, pid=device.pid,
                    serial_number=device.serial_number, description=device.description)
        self.manifest = dict(device=identity, status='recording', port=port, created_utc=stamp, sim_config=cfg,
            wire_format='little-endian 5 x [temperature, Bx, By, Bz] float32 + CR LF; AnySkin burst format',
            magnetic_units='uT as reported by firmware; scale not independently verified',
            temperature_units='firmware values, unverified',
            timing='host reception monotonic_ns and unix_ns; no device timestamps; frames in one read share reception time',
            association='latest received real sequence at completion of each sim sample; NOT synchronized physical loading',
            channel_mapping='real R1..R5 in firmware order; sim U1..U5 in reference PCB order; correspondence unverified',
            calibration='baseline/noise only; no force or mechanical fitting',
            baseline_policy='startup/B: 5-frame mean from manufacturer' if official_baseline else 'explicit 2-second >=100-frame statistics',
            provenance='serial decoder implements official wire format; comparison and simulation are local project code')
        self._manifest()

    def _manifest(self):
        (self.path/'manifest.json').write_text(json.dumps(self.manifest, indent=2)+'\n')

    def _write(self, stream, row):
        self.files[stream].write(json.dumps(row, allow_nan=False)+'\n')

    def event(self, kind, **fields):
        with self.lock:
            self._write('events', dict(kind=kind, host_monotonic_ns=time.monotonic_ns(),
                                       host_unix_ns=time.time_ns(), **fields))

    def begin_baseline(self, num_samples=100, duration_s=2.):
        with self.lock:
            if self.latest is None or time.monotonic_ns()-self.latest['host_monotonic_ns'] > 500_000_000:
                raise ValueError('No fresh real data; baseline not started')
            self.pending = dict(start=time.monotonic_ns(), values=[], seq=[], num_samples=num_samples, duration_ns=int(duration_s*1e9))
            self._write('events', dict(kind='baseline_requested', host_monotonic_ns=self.pending['start'],
                                       num_samples=num_samples, duration_s=duration_s,
                                       condition='user declares fixed, unloaded, settled sensor; not automatically verified'))

    def ingest(self, values, monotonic_ns, unix_ns):
        """All frames are saved even when UI/physics is slower than the sensor."""
        magnetic = values[:, 1:].reshape(15)
        with self.lock:
            self.count += 1
            if (self.official_baseline and self.baseline is None and self.pending is None
                    and monotonic_ns-self.started_ns >= 1_100_000_000):
                self.pending = dict(start=monotonic_ns, values=[], seq=[], num_samples=5, duration_ns=0)
                self._write('events', dict(kind='baseline_requested', host_monotonic_ns=monotonic_ns,
                    source='manufacturer startup default', num_samples=5, duration_s=0.,
                    condition='startup baseline; unloaded condition must be ensured by user'))
            if self.pending is not None:
                p = self.pending
                p['values'].append(magnetic.copy()); p['seq'].append(self.count)
                if monotonic_ns-p['start'] >= p['duration_ns'] and len(p['values']) >= p['num_samples']:
                    data = np.asarray(p['values'])
                    self.baseline = data.mean(axis=0)
                    self.baseline_id += 1
                    self._write('events', dict(kind='baseline', baseline_id=self.baseline_id,
                        host_monotonic_ns=monotonic_ns, host_unix_ns=unix_ns,
                        first_seq=p['seq'][0], last_seq=p['seq'][-1], count=len(data),
                        method='frame mean', requested_samples=p['num_samples'], requested_duration_s=p['duration_ns']/1e9,
                        baseline_ut=self.baseline.tolist(), std_ut=data.std(axis=0, ddof=1).tolist(),
                        peak_to_peak_ut=np.ptp(data, axis=0).tolist(),
                        condition='user-declared unloaded; motion/contact may contaminate statistics'))
                    self.pending = None
            row = dict(seq=self.count, host_monotonic_ns=monotonic_ns, host_unix_ns=unix_ns,
                       raw_magnetic_ut=magnetic.tolist(), temperature=values[:,0].tolist(),
                       baseline_id=self.baseline_id,
                       delta_ut=None if self.baseline is None else (magnetic-self.baseline).tolist())
            self.latest = row
            self._write('real', row)

    def load_baseline(self, path):
        path = Path(path)
        report = json.loads(path.read_text())
        source = json.loads((path.parent/'manifest.json').read_text())
        expected, actual = source.get('device') or {}, self.manifest.get('device') or {}
        if not expected.get('serial_number') or expected.get('serial_number') != actual.get('serial_number'):
            raise ValueError('Baseline belongs to an unknown or different device')
        baseline = np.asarray(report['baseline']['baseline_ut'], dtype=float)
        if baseline.shape != (15,) or not np.isfinite(baseline).all():
            raise ValueError('Invalid baseline')
        with self.lock:
            self.baseline = baseline.copy()
            self.baseline_id += 1
            self._write('events', dict(kind='baseline_loaded', baseline_id=self.baseline_id,
                baseline_ut=baseline.tolist(), source=str(path.resolve()),
                host_monotonic_ns=time.monotonic_ns(), condition='same device; mounting must remain unchanged'))

    def _run(self):
        try:
            import serial
            with serial.Serial(self.port, 115200, timeout=.1, exclusive=True) as stream:
                stream.reset_input_buffer()
                last_flush = time.monotonic()
                while not self.stopping.is_set():
                    chunk = stream.read(min(max(stream.in_waiting, 1), 8192))
                    now, wall = time.monotonic_ns(), time.time_ns()
                    self.wire.write(chunk)
                    for frame in self.decoder.feed(chunk):
                        self.ingest(frame, now, wall)
                    if time.monotonic()-last_flush > .5:
                        with self.lock:
                            for file in self.files.values(): file.flush()
                        self.wire.flush()
                        last_flush = time.monotonic()
        except Exception as error:
            with self.lock:
                self.error = f'{type(error).__name__}: {error}'
                self._write('events', dict(kind='serial_error', message=self.error,
                                           host_monotonic_ns=time.monotonic_ns()))

    def __enter__(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def snapshot(self):
        with self.lock:
            row = None if self.latest is None else dict(self.latest)
            pending = self.pending is not None
            error = self.error
        if row is None and error is None and time.monotonic_ns()-self.started_ns > 3_000_000_000:
            error = "No frames: check USB / sensor output"
        age = None if row is None else (time.monotonic_ns()-row['host_monotonic_ns'])/1e9
        return row, pending, error, age

    def record_sim(self, row, **control):
        if 'manufacturer_display_ut' in row:
            control.update(manufacturer_display_ut=row['manufacturer_display_ut'].tolist(),
                           manufacturer_display_baseline_id=row['manufacturer_display_baseline_id'])
        with self.lock:
            now = time.monotonic_ns()
            real = self.latest
            self._write('sim', dict(host_monotonic_ns=now, host_unix_ns=time.time_ns(),
                sim_time_s=row['time_s'], magnetic_ut=row['magnetic_ut'].tolist(),
                raw_field_ut=row['raw_field_ut'].tolist(), ideal_delta_ut=row['ideal_delta_ut'].tolist(),
                normal_force_n=row['normal_force_n'], max_displacement_m=row['max_displacement_m'],
                probe_xyz_m=row['probe_xyz_m'].tolist(), **control,
                real_seq=None if real is None else real['seq'],
                real_age_s=None if real is None else (now-real['host_monotonic_ns'])/1e9,
                baseline_id=self.baseline_id))

    def __exit__(self, exc_type, exc, tb):
        self.stopping.set()
        if self.thread: self.thread.join(timeout=3)
        if self.thread and self.thread.is_alive():
            raise RuntimeError('Serial reader did not stop; refusing to close files underneath it')
        with self.lock:
            self.manifest.update(status='error' if self.error or exc else 'no_data' if self.count == 0 else 'closed',
                                 error=self.error or (str(exc) if exc else None),
                                 real_frames=self.count, discarded_bytes=self.decoder.discarded,
                                 trailing_bytes=len(self.decoder.buffer), baseline_count=self.baseline_id)
            for file in self.files.values(): file.close()
            self.wire.close()
            self._manifest()
