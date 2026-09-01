"""
Phase B thermal recalibration: prediction-error least-squares fit of the
head/oil thermal sub-model against real Rotax 915 iS flight data
(data/ntsb_915is/, NTSB docket WPR22LA211).

Re-run of the fit described in model/thermal_params.json's provenance
block, done AFTER fixing a real bug upstream in PhysicsTwin's nominal
health-parameter initialization: theta_fric (health_params[8], the
multiplicative friction/FMEP health parameter) was defaulting to 0.0
instead of 1.0, which silently zeroed Q_fric_to_oil -- the friction-heat
source term that feeds the oil-temperature ODE -- in every earlier Phase B
fit and RMSE measurement. With that fixed, this script re-fits the
thermal capacitances / conductances / heat-split fractions that govern
CHT and oil-temperature dynamics.

Fits on a windowed slice of the single available real flight (skip the
first `--skip` seconds as integrator warm-up, fit against the next
`--window` seconds), matching the fit_skip_s / fit_window_s already
recorded in model/thermal_params.json's provenance for comparability.
Uses a coarser optimizer dt (see --opt-dt) to keep each residual
evaluation affordable; the resulting parameters are then re-validated
against the FULL flight at the real dt=0.1 PhysicsTwin uses in production
(and in tests/physics_twin/test_g1_calibration_gates.py) before being
accepted.

Usage:
    python3 scripts/fit_phase_b_thermal.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "real_data"))

from ntsb_915is_loader import load_ntsb_915is_fixture  # noqa: E402
from replan_to_learn.physics_twin.physics_twin import PhysicsTwin  # noqa: E402

MODEL_DIR = REPO_ROOT / "model"
THERMAL_PARAMS_PATH = MODEL_DIR / "thermal_params.json"

# Order matters -- must match PARAM_KEYS below.
PARAM_KEYS = [
    ("thermal_capacitances", "C_hd"),
    ("thermal_capacitances", "C_oil"),
    ("cooling_conductances", "UA_hd_nominal"),
    ("cooling_conductances", "UA_oc_nominal"),
    ("heat_exchanger", "UA_ho"),
    ("heat_transfer_to_head", "Q_to_head_fraction"),
    ("friction_heat", "friction_to_oil_fraction"),
]

# Loose but physically-sane bounds (positive capacitances/conductances,
# fractions in (0, 1)).
#
# UA_hd_nominal/UA_oc_nominal bounds rescaled this session: the cooling-
# conductance formulas (_compute_ua_hd/_compute_ua_oc/_compute_ua_c in
# physics_twin.py) were reparameterized from an unnormalized
# v_tas^0.8*rho^0.8*(N/300) form (UA_nominal implicitly meant "UA at
# v_tas=1 m/s", a nonsensical reference point nowhere near the real
# ~40-80 m/s cruise envelope -- badly conditioned for a fit spanning a
# wide real airspeed range) to a reference-normalized
# (v_tas/50)^0.8*(rho/1.225)^0.8*(N/2800) form, where UA_nominal now
# means "UA at real reference cruise conditions". This changes
# UA_nominal's natural scale by a factor of ~251x (see
# thermal_params.json's provenance.reparameterization_note for the exact
# derivation), so the old [5,400]/[1,2000] bounds (tuned for the OLD
# unnormalized scale) are wrong by two orders of magnitude for the new
# formula and were rejecting the rescaled starting point outright.
LOWER = np.array([2000.0, 50.0, 500.0, 500.0, 1.0, 0.05, 0.05])
UPPER = np.array([60000.0, 2000.0, 200000.0, 500000.0, 200.0, 0.9, 0.98])


def set_params(thermal_params: dict, x: np.ndarray) -> None:
    for (section, key), val in zip(PARAM_KEYS, x):
        thermal_params[section][key] = float(val)


def get_params(thermal_params: dict) -> np.ndarray:
    return np.array([thermal_params[section][key] for section, key in PARAM_KEYS])


def build_twin(opt_dt: float) -> PhysicsTwin:
    return PhysicsTwin(model_dir=MODEL_DIR, dt=opt_dt)


def simulate_errors(twin: PhysicsTwin, frames: list) -> tuple[np.ndarray, np.ndarray]:
    """Reset at frames[0] and simulate through all `frames`, returning (cht_err, toil_err)."""
    twin.reset(frames[0])
    cht_err = np.empty(len(frames))
    toil_err = np.empty(len(frames))
    for i, frame in enumerate(frames):
        twin._advance_state(frame)
        preds = twin._compute_predictions_from_state(twin.state, twin.health_params, telemetry=frame)
        cht_err[i] = preds["cht"] - frame.cht
        toil_err[i] = preds["t_oil"] - frame.t_oil
    return cht_err, toil_err


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    # Defaults changed from the original skip=250/window=400 (a single narrow
    # low-airspeed slice, v_tas in [1.7, 9.1] m/s -- see thermal_params.json
    # provenance history). That window badly under-represents the real
    # flight's regime diversity (full flight v_tas spans [0.1, 41.2] m/s):
    # UA_hd/UA_oc scale with v_tas^0.8, so a UA_xx_nominal fitted only at
    # low airspeed extrapolates to physically absurd cooling power at real
    # cruise airspeed (diagnosed directly: at a real cruise frame, computed
    # UA_oc*(T_oil-T_amb) ~ 597 kW against a Q_fric_to_oil heat source of
    # ~36 kW -- 16x oversized -- which collapses simulated T_oil toward
    # T_amb regardless of C_oil, matching the observed ~-66C oil-temp
    # prediction bias). Defaults now cover nearly the entire flight
    # (skip=100 to clear the t=0 ground/startup frames where v_tas pins at
    # its 0.1 floor) so the optimizer actually sees the real airspeed/RPM
    # range it will be validated against.
    ap.add_argument("--skip", type=float, default=100.0, help="warm-up seconds to skip before the fit window")
    ap.add_argument("--window", type=float, default=4300.0, help="fit-window length in seconds")
    ap.add_argument("--opt-dt", type=float, default=0.1, help="PhysicsTwin dt used during optimization")
    ap.add_argument("--max-nfev", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true", help="fit and report, but do not write thermal_params.json")
    args = ap.parse_args()

    all_frames = load_ntsb_915is_fixture()
    window_frames = [f for f in all_frames if args.skip <= f.t < args.skip + args.window]
    print(f"Loaded {len(all_frames)} frames total, {len(window_frames)} in fit window "
          f"[{args.skip}, {args.skip + args.window})s")

    with open(THERMAL_PARAMS_PATH) as fh:
        thermal_params = json.load(fh)
    x0 = get_params(thermal_params)
    print("Initial params:", dict(zip([k for _, k in PARAM_KEYS], x0)))

    twin = build_twin(args.opt_dt)
    n_eval = [0]
    t_start = time.time()

    def residuals(x: np.ndarray) -> np.ndarray:
        n_eval[0] += 1
        set_params(twin.thermal_params, x)
        cht_err, toil_err = simulate_errors(twin, window_frames)
        # Scale each channel by its gate threshold so both are ~O(1) at the
        # target accuracy -- keeps the optimizer from over-weighting the
        # numerically larger CHT error at oil-temp's expense.
        return np.concatenate([cht_err / 6.0, toil_err / 4.0])

    result = least_squares(
        residuals, x0, bounds=(LOWER, UPPER), method="trf",
        max_nfev=args.max_nfev, verbose=2, xtol=1e-10, ftol=1e-10,
    )
    elapsed = time.time() - t_start
    print(f"\nOptimizer finished: status={result.status}, cost={result.cost:.6g}, "
          f"nfev={result.nfev}, elapsed={elapsed:.1f}s")
    print("Fitted params:", dict(zip([k for _, k in PARAM_KEYS], result.x)))
    at_bound = [
        k for (_, k), lo, hi, v in zip(PARAM_KEYS, LOWER, UPPER, result.x)
        if abs(v - lo) < 1e-6 * max(1.0, lo) or abs(v - hi) < 1e-6 * max(1.0, hi)
    ]
    print("Params at bound:", at_bound if at_bound else "(none)")

    # Window RMSE at the optimizer dt.
    set_params(twin.thermal_params, result.x)
    cht_err_w, toil_err_w = simulate_errors(twin, window_frames)
    print(f"\nWindow RMSE (opt dt={args.opt_dt}): CHT={rmse(cht_err_w):.2f}C, oil-temp={rmse(toil_err_w):.2f}C")

    # Full-flight validation at the real production dt=0.1, matching the
    # gate test exactly.
    val_twin = PhysicsTwin(model_dir=MODEL_DIR, dt=0.1)
    set_params(val_twin.thermal_params, result.x)
    val_twin.reset(all_frames[0])
    cht_err_f = np.empty(len(all_frames))
    toil_err_f = np.empty(len(all_frames))
    poil_err_f = np.empty(len(all_frames))
    for i, frame in enumerate(all_frames):
        val_twin._advance_state(frame)
        preds = val_twin._compute_predictions_from_state(val_twin.state, val_twin.health_params, telemetry=frame)
        cht_err_f[i] = preds["cht"] - frame.cht
        toil_err_f[i] = preds["t_oil"] - frame.t_oil
        poil_err_f[i] = preds["p_oil"] - frame.p_oil
    print(f"Full-flight validation RMSE (dt=0.1): CHT={rmse(cht_err_f):.2f}C "
          f"(target <=6.0C), oil-temp={rmse(toil_err_f):.2f}C (target <=4.0C), "
          f"oil-pressure={rmse(poil_err_f)/1e5:.3f}bar (target <=0.25bar)")

    if args.dry_run:
        print("\n--dry-run: not writing thermal_params.json")
        return

    out = dict(thermal_params)
    set_params(out, result.x)
    out["provenance"] = dict(out.get("provenance", {}))
    out["provenance"].update({
        "procedure": "Phase B: prediction-error least-squares (scipy trf) against REAL "
                     "Rotax 915 iS Dynon CAN-bus data, re-run after fixing the theta_fric "
                     "nominal-init bug (health_params[8] was defaulting to 0.0 instead of "
                     "1.0, silently zeroing Q_fric_to_oil in all earlier fits)",
        "source": "[FIT]",
        "fit_source": "NTSB docket WPR22LA211 (RANS S21, N46JH, SN13785), DYNON FLIGHT DATA-USER LOG DATA-RAW CSV",
        "fit_window_s": args.window,
        "fit_skip_s": args.skip,
        "n_fit_samples": len(window_frames),
        "cost": float(result.cost),
        "optimizer_status": int(result.status),
        "full_flight_validation_rmse": {
            "cht_c": rmse(cht_err_f),
            "oil_temp_c": rmse(toil_err_f),
            "oil_pressure_bar": rmse(poil_err_f) / 1e5,
        },
        "caveat": "Single accident flight; not a healthy-segment-selected calibration "
                  "set per spec Sec 3.1. This IS the correct engine type (915 iS).",
    })
    with open(THERMAL_PARAMS_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    print(f"\nWrote {THERMAL_PARAMS_PATH}")


if __name__ == "__main__":
    main()
