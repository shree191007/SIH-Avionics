"""
SIH26054 Replan to Learn: Stage 5 Demo 1 -- Early detection
(05_gcs_integration_validation_deployment.md Sec 6, Demo 1).

Replays the real NTSB WPR22LA211 Rotax 915iS accident telemetry fixture
and compares two detection strategies on the SAME real data:

  A) raw-threshold alert  -- current practice: alert when any sensor
     crosses a published Rotax operating limit (same limits as eval/
     arms.py's Arm A).
  B) physics-residual alert -- alert on the physics twin's own whitened
     residual (ResidualFrame.z) crossing a 3-sigma anomaly threshold,
     BEFORE any single sensor has crossed its raw limit.

Only twin.step() is used here (not the full gate/orchestrator), since
this demo's claim is specifically about the physics RESIDUAL catching
the anomaly earlier than a raw threshold would -- not about naming which
subsystem, which is Demo 2's job. twin.step() is cheap (~2ms/frame), so
the full ~4659-frame fixture replays in seconds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "real_data"))

from ntsb_915is_loader import load_ntsb_915is_fixture  # noqa: E402
from replan_to_learn.physics_twin.physics_twin import PhysicsTwin  # noqa: E402

MODEL_DIR = ROOT / "model"
Z_ANOMALY_THRESHOLD = 3.0


def _rotax_limits() -> dict:
    return json.loads((MODEL_DIR / "rotax_915is_reference_data.json").read_text())


# Frames to skip before either detector starts looking: covers the real
# engine-start transient (oil pressure has not yet built up, all sensors
# are still ramping toward steady state) -- same 20-25s quasi-steady
# stabilization convention used throughout this session's gate tests
# (tests/gate/test_gate_naming_accuracy.py). Without this, BOTH detectors
# trivially fire at frame 0 on cold-start oil pressure, which says
# nothing about the actual developing fault this demo is about.
STABILIZATION_FRAMES = 25


def raw_threshold_alert_index(frames) -> "int | None":
    limits = _rotax_limits()
    egt_max_k = limits["egt_limits"]["max_c"] + 273.15
    cht_max_k = limits["coolant_limits"]["temperature_c"]["normal_max"] + 273.15
    oil_min_pa = limits["oil_limits"]["pressure_bar"]["min_below_3500rpm"] * 1e5
    for i in range(STABILIZATION_FRAMES, len(frames)):
        f = frames[i]
        if max(f.egt) > egt_max_k or f.cht > cht_max_k or f.p_oil < oil_min_pa:
            return i
    return None


def physics_residual_alert_index(frames, twin: PhysicsTwin, z_threshold: float = Z_ANOMALY_THRESHOLD) -> "int | None":
    twin.reset(frames[0])
    for i, f in enumerate(frames):
        rf = twin.step(f)
        if i >= STABILIZATION_FRAMES and any(abs(z) > z_threshold for z in rf.z):
            return i
    return None


def main() -> None:
    frames = load_ntsb_915is_fixture()
    print(f"Loaded {len(frames)} real telemetry frames from NTSB WPR22LA211 (N46JH, 915iS SN13785)")

    raw_idx = raw_threshold_alert_index(frames)
    twin = PhysicsTwin(model_dir=MODEL_DIR, dt=0.1)
    physics_idx = physics_residual_alert_index(frames, twin)

    print()
    print("=" * 60)
    print("threshold alert")
    if raw_idx is None:
        print("  never fired: no raw channel crossed its published Rotax limit")
    else:
        t = frames[raw_idx].t
        print(f"  fired at frame {raw_idx} (t={t:.1f}s): {frames[raw_idx].egt}, cht={frames[raw_idx].cht - 273.15:.1f}C, p_oil={frames[raw_idx].p_oil / 1e5:.2f}bar")
    print("vs")
    print("physics residual alert")
    if physics_idx is None:
        print(f"  never fired: no channel's whitened residual exceeded {Z_ANOMALY_THRESHOLD} sigma")
    else:
        t = frames[physics_idx].t
        print(f"  fired at frame {physics_idx} (t={t:.1f}s)")
    print("=" * 60)

    if raw_idx is not None and physics_idx is not None:
        lead_s = frames[raw_idx].t - frames[physics_idx].t
        print(f"\nlead time = {lead_s:.1f}s ({'physics ahead' if lead_s > 0 else 'threshold ahead' if lead_s < 0 else 'simultaneous'})")
    elif physics_idx is not None and raw_idx is None:
        print(f"\nphysics residual alert fired at t={frames[physics_idx].t:.1f}s; raw threshold never fired at all on this flight -- an unbounded lead time (the incident would have gone undetected by threshold monitoring alone).")
    else:
        print("\nInsufficient alerts on this fixture to report a lead time.")


if __name__ == "__main__":
    main()
