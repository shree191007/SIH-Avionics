"""
SIH26054 Replan to Learn: Stage 5 Demo 4 -- Mission save
(05_gcs_integration_validation_deployment.md Sec 6, Demo 4).

Shows the SAME planned mission profile simulated twice with the L7
MissionSimulator, once at the healthy nominal theta and once at a
degraded theta (theta_cool, reduced cooling effectiveness) -- the
current engine state changing the recommendation:

  nominal mission -> risk assessment -> derate / altitude change / return
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replan_to_learn.physics_twin.physics_twin import PhysicsTwin  # noqa: E402
from replan_to_learn.planner.mission_simulator import MissionSimulator, MissionState  # noqa: E402

MODEL_DIR = ROOT / "model"
NOMINAL = np.concatenate([np.ones(9), np.zeros(6)])

# A sustained near-max-continuous climb (5200rpm, close to Rotax's
# 5500rpm MCP), fine-grained (5s cadence over 600s) so PhysicsTwin.predict's
# per-frame state integration (real inter-frame dt, see physics_twin.py
# predict()'s docstring) actually accumulates thermal load across the
# mission instead of resetting to a near-nominal state every call.
MISSION_PROFILE = [
    MissionState(altitude_m=1500.0 + 2.0 * i, ias_kt=110.0, rpm=5200.0, fuel_kg=50.0 - 0.01 * i, elapsed_s=float(i), regime=5)
    for i in range(0, 600, 5)
]


def _report(label: str, theta: np.ndarray, sim: MissionSimulator) -> None:
    result = sim.simulate(theta=theta, mission_profile=MISSION_PROFILE, current_fuel_kg=MISSION_PROFILE[0].fuel_kg)
    print(f"{label}")
    print(f"  risk={result.risk}  risk_probability={result.risk_probability:.2f}")
    print(f"  recommended_action={result.recommended_action}")
    if result.margin_warnings:
        print(f"  margin_warnings={list(result.margin_warnings)}")
    if result.alternatives:
        print(f"  alternatives={result.alternatives}")
    return result


def main() -> None:
    twin = PhysicsTwin(model_dir=MODEL_DIR, dt=0.1)
    sim = MissionSimulator(physics_twin=twin)

    print("=" * 60)
    print("nominal mission")
    print("=" * 60)
    healthy = _report("healthy engine (theta = nominal)", NOMINAL.copy(), sim)

    print()
    print("  |")
    print("  v")
    print("risk assessment")
    print("  |")
    print("  v")
    print()

    print("=" * 60)
    print("degraded engine (theta_cool reduced -- weaker cooling)")
    print("=" * 60)
    degraded_levels = [0.5, 0.3, 0.2]
    degraded = None
    for level in degraded_levels:
        theta = NOMINAL.copy()
        theta[2] = level  # theta_cool
        degraded = _report(f"theta_cool={level}", theta, sim)
        if degraded.risk != "LOW":
            break

    print()
    if healthy.risk == "LOW" and degraded is not None and degraded.risk != "LOW":
        print(f"Mission save: identical planned profile, but the CURRENT engine state changed the recommendation "
              f"from '{healthy.recommended_action}' (healthy) to '{degraded.recommended_action}' (degraded).")
    else:
        print(f"(Healthy risk={healthy.risk}, most-degraded risk tried={degraded.risk if degraded else 'n/a'} -- "
              "did not observe a risk-tier change across the theta_cool levels tried; reporting honestly rather than forcing one.")


if __name__ == "__main__":
    main()
