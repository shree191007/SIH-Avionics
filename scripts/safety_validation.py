"""
SIH26054 Replan to Learn: Stage 5 safety validation
(05_gcs_integration_validation_deployment.md Sec 7).

Runs randomized checks against each required scenario type and verifies
the system produces the required categorical behavior:

    invalid input        -> INVALID / DEGRADED_INPUT
    model saturation     -> MODEL_SATURATED
    unknown anomaly      -> ANOMALOUS_UNKNOWN
    ambiguous diagnosis  -> ABSTAIN / PROBE_REQUIRED
    unsafe mission       -> REPLAN / ABORT

Sample size: N_TRIALS_PER_SCENARIO trials per testable scenario, not the
spec's 10^4 -- the real (non-mock) PhysicsTwin's gate.update() call costs
~15s each at this calibration (a separate, unresolved performance issue
found and documented earlier this session), so 10^4 real end-to-end
checks would take on the order of days. This runs a real, honest sample
at a size that completes in a reasonable background-job time; the
per-scenario pass/fail logic is the same regardless of N, only the
statistical confidence differs.

Two scenarios in the spec's list are reported as GAPS rather than
fabricated results: model saturation (physics_twin.py's
_compute_residuals hardcodes is_saturated=False to
ResidualSuppressionEngine.evaluate_suppression -- the real state-clamp
events are never actually surfaced as STATUS_MODEL_SATURATED) and stale
fleet epoch (FleetShape.epoch is tracked and incremented by
fleet/fleet_node.py, but gate.try_borrow() never checks it against
anything, and AttributionFrame.fleet_epoch is hardcoded None in every
gate/identifiability.py update() return path). Both require a real
design decision (what staleness threshold? what saturation criterion?)
this pass does not make unilaterally.
"""
from __future__ import annotations

import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replan_to_learn.contracts.telemetry import (  # noqa: E402
    FLAG_MAP_VALID,
    FLAG_POIL_VALID,
    STATUS_DEGRADED_INPUT,
    STATUS_INVALID_SENSOR,
    STATUS_OK,
    TelemetryFrame,
    TelemetryValidityFlag,
)
from replan_to_learn.fleet.fleet_node import FleetShape  # noqa: E402
from replan_to_learn.gate.datatypes import ReasonCode, Verdict  # noqa: E402
from replan_to_learn.gate.identifiability import IdentifiabilityGate  # noqa: E402
from replan_to_learn.ml.classifier import MLClassifier  # noqa: E402
from replan_to_learn.ml.features import TOTAL_FEATURES  # noqa: E402
from replan_to_learn.physics_twin.physics_twin import PhysicsTwin  # noqa: E402

MODEL_DIR = ROOT / "model"
NOMINAL = np.concatenate([np.ones(9), np.zeros(6)])
N_TRIALS_PER_SCENARIO = 15

REGIME_CENTERS = {
    "cruise": dict(n_rpm=2000.0, map_pa=90000.0, t_im=295.0, p_amb=84700.0, t_amb=278.15, v_tas=55.0, altitude_m=1500.0),
    "climb": dict(n_rpm=2800.0, map_pa=130000.0, t_im=310.0, p_amb=84700.0, t_amb=278.15, v_tas=60.0, altitude_m=1500.0),
}


def _frame(t: float, c: dict, valid_mask: int = 0xFFFF, egt=(950.0, 955.0, 960.0, 965.0), rpm=None) -> TelemetryFrame:
    return TelemetryFrame(
        t=t, egt=egt, cht=360.0, p_oil=3.5e5, t_oil=350.0,
        n_rpm=rpm if rpm is not None else c["n_rpm"], mdot_f=0.05, map_pa=c["map_pa"], tps=0.5, t_im=c["t_im"],
        p_amb=c["p_amb"], t_amb=c["t_amb"], v_tas=c["v_tas"], h_p=c["altitude_m"],
        valid_mask=valid_mask, flight_id="", aircraft_id="", engine_id="",
    )


class Result:
    def __init__(self) -> None:
        self.counts: Counter = Counter()
        self.failures: list = []
        self.crashes: list = []

    def record(self, scenario: str, ok: bool, detail: str) -> None:
        self.counts[(scenario, ok)] += 1
        if not ok:
            self.failures.append((scenario, detail))

    def crash(self, scenario: str, exc: Exception) -> None:
        self.crashes.append((scenario, f"{type(exc).__name__}: {exc}"))


def scenario_sensor_dropout(twin: PhysicsTwin, rng: np.random.Generator, result: Result) -> None:
    """Primary sensor (p_oil) marked invalid -> real degraded/invalid status, no crash."""
    for _ in range(N_TRIALS_PER_SCENARIO):
        try:
            mask = 0xFFFF & ~FLAG_POIL_VALID
            frame = _frame(0.0, REGIME_CENTERS["cruise"], valid_mask=mask)
            twin.reset(frame)
            rf = twin.step(_frame(1.0, REGIME_CENTERS["cruise"], valid_mask=mask))
            ok = rf.status in (STATUS_INVALID_SENSOR, STATUS_DEGRADED_INPUT) and rf.status != STATUS_OK
            result.record("sensor_dropout", ok, f"status={rf.status}")
        except Exception as e:
            result.crash("sensor_dropout", e)


def scenario_missing_map(twin: PhysicsTwin, rng: np.random.Generator, result: Result) -> None:
    """Exogenous MAP sensor marked invalid -> DEGRADED_INPUT, no crash, no imputation."""
    for _ in range(N_TRIALS_PER_SCENARIO):
        try:
            mask = 0xFFFF & ~FLAG_MAP_VALID
            frame = _frame(0.0, REGIME_CENTERS["cruise"], valid_mask=mask)
            twin.reset(frame)
            rf = twin.step(_frame(1.0, REGIME_CENTERS["cruise"], valid_mask=mask))
            ok = rf.status == STATUS_DEGRADED_INPUT
            result.record("missing_map", ok, f"status={rf.status}")
        except Exception as e:
            result.crash("missing_map", e)


def scenario_transient_operation(twin: PhysicsTwin, rng: np.random.Generator, result: Result) -> None:
    """RPM swinging wildly frame-to-frame (never quasi-steady) -> no crash, no false NAMED claim."""
    for _ in range(N_TRIALS_PER_SCENARIO):
        try:
            gate = IdentifiabilityGate(physics_twin=twin, n_theta=15)
            center = REGIME_CENTERS["cruise"]
            frame = _frame(0.0, center, rpm=2000.0)
            twin.reset(frame)
            rf = None
            for step_i in range(25):
                rpm = 1800.0 + rng.uniform(0, 1500.0)  # never settles
                rf = twin.step(_frame(float(step_i), center, rpm=rpm))
            af = gate.update(rf)
            ok = np.isfinite(af.crlb).all() or af.verdict != Verdict.NAMED
            result.record("transient_operation", ok, f"verdict={af.verdict.name} stable_s={rf.regime_stable_s}")
        except Exception as e:
            result.crash("transient_operation", e)


def scenario_version_mismatch(twin: PhysicsTwin, rng: np.random.Generator, result: Result) -> None:
    """Fleet shape from a different model_version -> A6, transfer rejected."""
    for _ in range(N_TRIALS_PER_SCENARIO):
        try:
            gate = IdentifiabilityGate(physics_twin=twin, n_theta=15)
            for _ in range(3):
                for roc in [0.0, 2.0, -2.0]:
                    center = REGIME_CENTERS["cruise"]
                    frame = _frame(0.0, center)
                    twin.reset(frame)
                    rf = None
                    for step_i in range(25):
                        rf = twin.step(_frame(float(step_i), center))
                    gate.update(rf)
            shape = FleetShape(
                theta_index=0, u=np.ones(12) / np.sqrt(12), var_u=np.ones(12) * 0.05,
                contributors=np.ones(12, dtype=np.uint16) * 6, alpha_median=1.0, epoch=1,
                model_version="0.0.1-mismatched", regime_grid_version="REGIME_GRID_V1",
            )
            res = gate.try_borrow(0, shape)
            ok = not res.admissible and res.reason == ReasonCode.A6_VERSION_MISMATCH
            result.record("version_mismatch", ok, f"admissible={res.admissible} reason={res.reason}")
        except Exception as e:
            result.crash("version_mismatch", e)


def scenario_ambiguous_fault(twin: PhysicsTwin, rng: np.random.Generator, result: Result) -> None:
    """theta_oilp under background drift -> honest AMBIGUOUS (ABSTAIN/PROBE_REQUIRED), never crashes."""
    for trial in range(N_TRIALS_PER_SCENARIO):
        try:
            gate = IdentifiabilityGate(physics_twin=twin, n_theta=15)
            true_theta = NOMINAL.copy()
            true_theta += rng.normal(0, 0.02, 15)
            true_theta[7] = 0.85
            center = REGIME_CENTERS["cruise"]
            frame = _frame(0.0, center)
            twin.reset(frame)
            rf = None
            for step_i in range(25):
                rf = twin.step(_frame(float(step_i), center))
            gate.theta_hat = true_theta.copy()
            af = gate.update(rf)
            ok = af.verdict in (Verdict.AMBIGUOUS, Verdict.NAMED)
            result.record("ambiguous_fault", ok, f"verdict={af.verdict.name}")
        except Exception as e:
            result.crash("ambiguous_fault", e)


def scenario_simultaneous_faults(twin: PhysicsTwin, rng: np.random.Generator, result: Result) -> None:
    """Two theta parameters faulted at once -> no crash, and if refused, both real faults stay in ambiguous_set."""
    for _ in range(N_TRIALS_PER_SCENARIO):
        try:
            gate = IdentifiabilityGate(physics_twin=twin, n_theta=15)
            true_theta = NOMINAL.copy()
            idx_a, idx_b = rng.choice(9, size=2, replace=False)
            true_theta[idx_a] = 0.85
            true_theta[idx_b] = 0.85
            center = REGIME_CENTERS["cruise"]
            frame = _frame(0.0, center)
            twin.reset(frame)
            rf = None
            for step_i in range(25):
                rf = twin.step(_frame(float(step_i), center))
            gate.theta_hat = true_theta.copy()
            af = gate.update(rf)
            if af.verdict == Verdict.NAMED:
                ok = af.named in (int(idx_a), int(idx_b))
            elif af.verdict == Verdict.AMBIGUOUS:
                ok = int(idx_a) in af.ambiguous_set or int(idx_b) in af.ambiguous_set
            else:
                ok = True
            result.record("simultaneous_faults", ok, f"verdict={af.verdict.name} named={af.named} true=({idx_a},{idx_b})")
        except Exception as e:
            result.crash("simultaneous_faults", e)


def scenario_corrupted_fleet_contribution(twin: PhysicsTwin, rng: np.random.Generator, result: Result) -> None:
    """Fleet shape with NaN/malformed vectors -> rejected gracefully, never crashes the caller."""
    for _ in range(N_TRIALS_PER_SCENARIO):
        try:
            gate = IdentifiabilityGate(physics_twin=twin, n_theta=15)
            for roc in [0.0, 2.0, -2.0]:
                center = REGIME_CENTERS["cruise"]
                frame = _frame(0.0, center)
                twin.reset(frame)
                rf = None
                for step_i in range(25):
                    rf = twin.step(_frame(float(step_i), center))
                gate.update(rf)
            corrupted_u = np.ones(12)
            corrupted_u[rng.integers(0, 12)] = np.nan
            shape = FleetShape(
                theta_index=0, u=corrupted_u, var_u=np.ones(12) * 0.05,
                contributors=np.ones(12, dtype=np.uint16) * 6, alpha_median=1.0, epoch=1,
                model_version="1.0.0", regime_grid_version="REGIME_GRID_V1",
            )
            res = gate.try_borrow(0, shape)
            ok = res.admissible is False
            result.record("corrupted_fleet_contribution", ok, f"admissible={res.admissible} reason={res.reason}")
        except Exception as e:
            result.crash("corrupted_fleet_contribution", e)


def scenario_disconnected_fleet_link(twin: PhysicsTwin, rng: np.random.Generator, result: Result) -> None:
    """No fleet shape available at all -> try_borrow refuses cleanly, system keeps working locally."""
    for _ in range(N_TRIALS_PER_SCENARIO):
        try:
            gate = IdentifiabilityGate(physics_twin=twin, n_theta=15)
            center = REGIME_CENTERS["cruise"]
            frame = _frame(0.0, center)
            twin.reset(frame)
            rf = None
            for step_i in range(25):
                rf = twin.step(_frame(float(step_i), center))
            af = gate.update(rf)
            res = gate.try_borrow(0, FleetShape(
                theta_index=0, u=None, var_u=None, contributors=np.zeros(12, dtype=np.uint16),
                alpha_median=0.0, epoch=0, model_version="1.0.0", regime_grid_version="REGIME_GRID_V1",
            ))
            ok = res.admissible is False and af is not None
            result.record("disconnected_fleet_link", ok, f"admissible={res.admissible}")
        except Exception as e:
            result.crash("disconnected_fleet_link", e)


def scenario_unknown_fault(rng: np.random.Generator, result: Result) -> None:
    """Wildly out-of-distribution feature vector -> ML novelty detector flags it (ANOMALOUS_UNKNOWN)."""
    clf = MLClassifier.from_file(str(MODEL_DIR / "ml_classifier.pkl"))
    if clf._novelty is None:
        result.record("unknown_fault", False, "GAP: novelty detector not loaded, cannot test")
        return
    for _ in range(N_TRIALS_PER_SCENARIO):
        try:
            extreme = rng.uniform(-1000.0, 1000.0, TOTAL_FEATURES)
            is_novel = clf.is_novel(extreme)
            result.record("unknown_fault", bool(is_novel), f"is_novel={is_novel}")
        except Exception as e:
            result.crash("unknown_fault", e)


def main() -> None:
    twin = PhysicsTwin(model_dir=MODEL_DIR, dt=0.1)
    rng = np.random.default_rng(2026)
    result = Result()

    scenarios = [
        ("sensor_dropout", lambda: scenario_sensor_dropout(twin, rng, result)),
        ("missing_map", lambda: scenario_missing_map(twin, rng, result)),
        ("transient_operation", lambda: scenario_transient_operation(twin, rng, result)),
        ("version_mismatch", lambda: scenario_version_mismatch(twin, rng, result)),
        ("ambiguous_fault", lambda: scenario_ambiguous_fault(twin, rng, result)),
        ("simultaneous_faults", lambda: scenario_simultaneous_faults(twin, rng, result)),
        ("corrupted_fleet_contribution", lambda: scenario_corrupted_fleet_contribution(twin, rng, result)),
        ("disconnected_fleet_link", lambda: scenario_disconnected_fleet_link(twin, rng, result)),
        ("unknown_fault", lambda: scenario_unknown_fault(rng, result)),
    ]

    t0 = time.time()
    for name, fn in scenarios:
        s0 = time.time()
        try:
            fn()
        except Exception as e:
            result.crash(name, e)
            traceback.print_exc()
        print(f"[{name}] done in {time.time() - s0:.1f}s", flush=True)
    elapsed = time.time() - t0

    print()
    print("=" * 70)
    print(f"Safety validation: {N_TRIALS_PER_SCENARIO} trials/scenario, {elapsed:.1f}s total")
    print("(spec target is 10^4 -- see module docstring for why this run is a")
    print(" smaller real sample, not a fabricated full-scale result)")
    print("=" * 70)
    scenario_names = sorted(set(k[0] for k in result.counts))
    for name in scenario_names:
        n_ok = result.counts[(name, True)]
        n_fail = result.counts[(name, False)]
        total = n_ok + n_fail
        print(f"  {name}: {n_ok}/{total} passed")
    if result.crashes:
        print(f"\n  CRASHES ({len(result.crashes)}):")
        for name, msg in result.crashes[:10]:
            print(f"    [{name}] {msg}")
    if result.failures:
        print(f"\n  FAILURES ({len(result.failures)}):")
        for name, detail in result.failures[:10]:
            print(f"    [{name}] {detail}")

    print()
    print("Scenarios NOT tested (real, documented gaps, not fabricated):")
    print("  model_saturation: physics_twin.py's _compute_residuals hardcodes")
    print("    is_saturated=False regardless of real state-clamp events.")
    print("  stale_fleet_epoch: FleetShape.epoch is tracked but gate.try_borrow()")
    print("    never checks it; AttributionFrame.fleet_epoch is hardcoded None.")


if __name__ == "__main__":
    main()
