"""
SIH26054 Replan to Learn: Real Data Sanity Check.
Loads real flight data and verifies it flows through PhysicsTwin + Stage3Orchestrator.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from real_data_loader import RealFlightLoader
from replan_to_learn.physics_twin.physics_twin import PhysicsTwin
from replan_to_learn.stage3 import Stage3Orchestrator, Stage3Config
from replan_to_learn.contracts.residuals import ResidualFrame

DATA_ROOT = Path("/Users/vatxn1907__/Desktop/V1/data")
loader = RealFlightLoader(DATA_ROOT)
manifest = loader.load_manifest()

print(f"Manifest: {manifest.shape[0]} flights")
print(f"Subsystems: {manifest['subsystem'].unique().to_list()}")

sample_flights = manifest.head(3).to_dicts()

twin = PhysicsTwin(model_dir=DATA_ROOT.parent / "model", dt=0.1)
config = Stage3Config()
orchestrator = Stage3Orchestrator(physics_twin=twin, config=config)

results = []
for flight_info in sample_flights:
    flight_id = str(flight_info["flight_id"])
    subsystem = str(flight_info["subsystem"])
    flight_df = loader.load_flight(flight_id)
    if flight_df is None or flight_df.shape[0] == 0:
        print(f"Skipping flight {flight_id}: no data")
        continue

    frames = loader.to_telemetry_frames(flight_df, flight_id)
    print(f"\nFlight {flight_id} ({subsystem}): {len(frames)} frames")

    residual_frames = []
    for frame in frames[:3]:
        try:
            rf = twin.step(frame)
            residual_frames.append(rf)
        except Exception as e:
            print(f"  PhysicsTwin step error at t={frame.t}: {e}")
            break

    print(f"  Generated {len(residual_frames)} residual frames")
    if not residual_frames:
        continue

    valid_residuals = [rf for rf in residual_frames if rf.status == 0]
    print(f"  Valid residuals: {len(valid_residuals)}")

    if valid_residuals:
        sample_rf = valid_residuals[len(valid_residuals) // 2]
        try:
            panel, metadata = orchestrator.process_residual_frame(sample_rf)
            verdict = panel.diagnosis.verdict
            health = panel.engine_health.status
            rul_p50 = panel.prognostics.rul_p50
            print(f"  Verdict: {verdict}, Health: {health}, RUL p50: {rul_p50:.1f}h")
            results.append({
                "flight_id": flight_id,
                "subsystem": subsystem,
                "verdict": verdict,
                "health": health,
                "rul_p50": rul_p50,
                "n_valid_residuals": len(valid_residuals),
            })
        except Exception as e:
            print(f"  Stage 3 error: {e}")

print("\n" + "=" * 60)
print("REAL DATA TEST SUMMARY")
print("=" * 60)
for r in results:
    print(f"Flight {r['flight_id']:>6} ({r['subsystem']:>15}): {r['verdict']:>10} | {r['health']:>15} | RUL={r['rul_p50']:>6.1f}h | valid={r['n_valid_residuals']}")
