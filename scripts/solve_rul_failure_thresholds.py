"""
Precomputes L6 physics-derived RUL failure thresholds (04_ml_rul_mission_probe.md
Sec 2.4) and writes them to model/rul_failure_thresholds.json.

The solver (rul/failure_threshold_solver.py) binary-searches each health
parameter against a real Rotax 915iS operator's-manual limit, using a
200-step (20s) equilibration per candidate evaluation -- real but not
cheap (~70s for all 9 parameters). Since the failure threshold is a fixed
property of the calibrated engine model, not something that changes per
flight or per orchestrator instance, it is precomputed once here rather
than solved live inside Stage3Orchestrator.__init__ on every construction.

Rerun this whenever model/ is recalibrated (gaspath_params.json,
thermal_params.json, or the physics_twin formulas themselves change),
since the thresholds depend on them.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replan_to_learn.physics_twin.physics_twin import PhysicsTwin  # noqa: E402
from replan_to_learn.rul.failure_threshold_solver import solve_failure_threshold  # noqa: E402

MODEL_DIR = ROOT / "model"

THETA_NAMES = [
    "theta_vol", "theta_comb", "theta_cool",
    "theta_inj1", "theta_inj2", "theta_inj3", "theta_inj4",
    "theta_oilp", "theta_fric",
]


def main() -> None:
    twin = PhysicsTwin(model_dir=MODEL_DIR, dt=0.1)
    thresholds = {}
    t0 = time.time()
    for i, name in enumerate(THETA_NAMES):
        ft = solve_failure_threshold(twin, i, name)
        thresholds[str(i)] = {
            "theta_name": ft.theta_name,
            "threshold_value": ft.threshold_value,
            "limit_description": ft.limit_description,
            "operating_point": ft.operating_point,
            "direction": ft.direction,
        }
        print(f"  {name}: threshold={ft.threshold_value:.4f}  ({ft.limit_description})")
    elapsed = time.time() - t0

    out = {
        "description": (
            "Physics-derived RUL failure thresholds (04_ml_rul_mission_probe.md Sec 2.4). "
            "Computed via rul/failure_threshold_solver.py: binary search of each health "
            "parameter against a real Rotax 915iS operator's-manual limit (CHT/coolant, "
            "EGT, oil pressure) or a documented [ASSUMED] climb-power fraction where no "
            "real airframe-specific limit is available, at a 200-step (20s) equilibrated "
            "stress operating point."
        ),
        "source": "[FIT]",
        "solver": "rul/failure_threshold_solver.py:solve_failure_threshold",
        "solve_time_s": elapsed,
        "thresholds": thresholds,
    }
    out_path = MODEL_DIR / "rul_failure_thresholds.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
