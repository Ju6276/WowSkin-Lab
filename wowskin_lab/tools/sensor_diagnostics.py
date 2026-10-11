"""Standalone magnetic-physics regression helpers; not policy training data."""
import csv
import json
import numpy as np
from wowskin_lab.core.physics import CHANNELS, TactileSim

def run_episode(cfg, frame_callback=None, predictor=None):
    sim = TactileSim(cfg)
    rows = []
    for index in range(round(cfg["duration_s"]*cfg["sample_hz"])):
        sample = sim.sample()
        sample["probability"] = predictor.update(sample["magnetic_ut"]) if predictor else float("nan")
        rows.append(sample)
        if frame_callback and index % 5 == 0:
            frame_callback(sim, sample)
    return sim, {key: np.array([r[key] for r in rows]) for key in rows[0]}

def save_episode(destination, cfg, sim, values):
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination.with_suffix(".npz"), **values)
    metadata = dict(config=cfg, status="UNCALIBRATED SYNTHETIC DATA", channels=CHANNELS,
                    field_units="microtesla from a point-dipole approximation",
                    baseline_field_ut=sim.baseline.tolist(),
                    sensor_layout=sim.layout,
                    label="contact-force gated, force-weighted tangential relative speed",
                    warnings=sim.data.warning.number.tolist())
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    # The mag column names match Pollen, but simulation time is explicitly named.
    with destination.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sim_time_s", *CHANNELS, "slip", "contact", "normal_force_n", "slip_speed_m_per_s"])
        for i, t in enumerate(values["time_s"]):
            writer.writerow([t, *values["magnetic_ut"][i], values["slip"][i],
                             values["contact"][i], values["normal_force_n"][i],
                             values["slip_speed_m_per_s"][i]])
