"""
One-time precompute for tests/fleet/test_real_ngafid_fleet.py: runs 6 real
NGAFID flights through the Phase-A/B/C-calibrated PhysicsTwin +
IdentifiabilityGate to get each flight's real local fingerprint and
regime-encounter history, and persists the result to
data/ngafid_fleet_fixture.parquet.

Rerun this (`python3 tests/real_data/build_ngafid_fleet_fixture.py` from
the repo root) whenever model/ is recalibrated, since the fingerprints
depend on it. Takes several minutes -- each flight is ~15-18k real
telemetry frames run through the full twin+gate pipeline, including a
12-regime Jacobian finite-difference pass.

The flight_ids below were chosen from a prior scan across the Cooling/
Induction/Mechanical/Lubrication subsystems for having real RPM/IAS
excursions (not ground-only recordings) and were confirmed to span more
than one real REGIME_GRID_V1 core bin over their full length:
  23517: bins [5,8]   27142: bins [5,8]   15379: bins [5,8]
  30799: bins [2,5,8] 28763: bins [5,8]   30252: bin [5] only (the
  genuine real cruise-only aircraft used in the borrow test)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from real_data_loader import RealFlightLoader  # noqa: E402
from replan_to_learn.physics_twin.physics_twin import PhysicsTwin  # noqa: E402
from replan_to_learn.gate.identifiability import IdentifiabilityGate  # noqa: E402

MODEL_DIR = ROOT / "model"
DATA_ROOT = ROOT / "data"
THETA_IDX = 2  # theta_cool

FLIGHT_IDS = ["23517", "27142", "15379", "30799", "28763", "30252"]


def main() -> None:
    loader = RealFlightLoader(DATA_ROOT)
    rows = []

    for fid in FLIGHT_IDS:
        t0 = time.time()
        df = loader.load_flight(fid)
        if df is None:
            continue
        frames = loader.to_telemetry_frames(df, fid)
        if len(frames) < 500:
            continue

        twin = PhysicsTwin(model_dir=MODEL_DIR, dt=0.1)
        gate = IdentifiabilityGate(physics_twin=twin, n_theta=15)
        twin.reset(frames[0])

        regime_seconds = {r: 0 for r in range(12)}
        last_rf = None
        for frame in frames:
            try:
                rf = twin.step(frame)
            except Exception:
                continue
            if rf.status != 0:
                continue
            if rf.regime < 12:
                regime_seconds[rf.regime] += 1
            last_rf = rf

        if last_rf is None or sum(regime_seconds.values()) == 0:
            print(f"  {fid}: no quasi-steady time accumulated, skipping")
            continue

        gate.update(last_rf)  # populates gate._last_fingerprints + regime_encounters
        f_i_regime = gate._regime_summary(gate.fingerprint(THETA_IDX))

        n_seconds = np.array([regime_seconds[r] for r in range(12)], dtype=np.float64)
        elapsed = time.time() - t0
        total_qs = int(n_seconds.sum())
        print(f"  {fid}: {total_qs}s quasi-steady, regimes={np.where(n_seconds > 0)[0].tolist()}, {elapsed:.1f}s")

        rows.append({
            "flight_id": fid,
            "fingerprint_regime": f_i_regime.tolist(),
            "n_seconds": n_seconds.tolist(),
        })

    out = pl.DataFrame(rows)
    out_path = DATA_ROOT / "ngafid_fleet_fixture.parquet"
    out.write_parquet(out_path)
    print(f"\nWrote {out_path}: {len(rows)} real flight contributions")


if __name__ == "__main__":
    main()
