"""Record a user-confirmed unloaded reference; no force/elastic calibration."""
import json,time
from pathlib import Path
from argparse import Namespace
import numpy as np
from PIL import Image
from wowskin_lab.envs.bench import bench_configuration,sensor_reading
from wowskin_lab.core.physics import TactileSim
from wowskin_lab.data.sensor_compare import SensorComparison
from wowskin_lab.ui.sensor_compare import RealSensorHUD,comparison_panel
from wowskin_lab.ui.tactile import TactileHUD


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',required=True)
    p.add_argument('--unloaded-confirmed',action='store_true',required=True)
    a=p.parse_args()
    cfg=bench_configuration(Namespace(seed=0,x_mm=0.,y_mm=0.,load_n=.05,ideal=True))
    sim=TactileSim(cfg)
    with SensorComparison(a.port,'outputs/calibration/unloaded',cfg) as s:
        timeout=time.monotonic()+5
        while s.snapshot()[0] is None:
            if s.error or time.monotonic()>timeout: raise RuntimeError('No real sensor data: '+str(s.snapshot()[2]))
            time.sleep(.01)
        s.event('unloaded_confirmed',source='user confirmation',purpose='baseline/noise only')
        s.begin_baseline()
        start=time.monotonic()
        while time.monotonic()-start<10:
            row=sensor_reading(sim,lambda sim:sim.apply_probe_load([0,0],None))
            s.record_sim(row,pressed=False,selected_xy_m=[0,0],commanded_load_n=0.)
        frame,pending,error,age=s.snapshot()
        if error or pending or age>.5: raise RuntimeError(f'Invalid capture: {error}, pending={pending}, age={age}')
        hud=TactileHUD(sim,full_scale_ut=10.,title='SIMULATION | U1-U5')
        Image.fromarray(comparison_panel(hud,RealSensorHUD(sim),row,s)).save(s.path/'unloaded_panel.png')
        destination=s.path
    frames=[json.loads(line) for line in (destination/'real.jsonl').read_text().splitlines()]
    events=[json.loads(line) for line in (destination/'events.jsonl').read_text().splitlines()]
    baseline=next(e for e in events if e['kind']=='baseline')
    held=[f for f in frames if f['seq']>baseline['last_seq']]
    x=np.array([f['raw_magnetic_ut'] for f in held])
    t=(np.array([f['host_monotonic_ns'] for f in held])-held[0]['host_monotonic_ns'])/1e9
    centered=x-x.mean(axis=0)
    slope=((t-t.mean())[:,None]*centered).sum(axis=0)/np.square(t-t.mean()).sum()
    report=dict(samples=len(frames),post_baseline_samples=len(held),baseline=baseline,
        post_baseline_std_ut=x.std(axis=0,ddof=1).tolist(),
        post_baseline_peak_to_peak_ut=np.ptp(x,axis=0).tolist(),
        post_baseline_linear_slope_ut_per_s=slope.tolist(),
        interpretation='One user-declared unloaded run. Statistics include drift, quantization and environmental changes; do not treat the short-run slope as a permanent drift model.',
        applies_to='this device and mounting condition only; not applied to sim parameters')
    (destination/'noise_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (Path('outputs/calibration')/'latest_unloaded.json').write_text(json.dumps(dict(path=str(destination.resolve())),indent=2))
    print(json.dumps(dict(path=str(destination.resolve()),samples=len(frames),
         std_range_ut=[float(x.std(axis=0,ddof=1).min()),float(x.std(axis=0,ddof=1).max())],
         discarded_bytes=s.decoder.discarded),indent=2))

if __name__=='__main__': main()
