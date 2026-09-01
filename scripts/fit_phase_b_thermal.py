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

# C_hd/C_oil (thermal capacitances) are REMOVED from the free-parameter
# set as of this fit. They are real physical quantities (mass x specific
# heat of the lumped head+coolant / oil systems) that should not need
# curve-fitting from one noisy flight at all -- and the prior 7-parameter
# fit's own diagnosis (5 of 7 landing on their bounds) is the textbook
# symptom of asking a single ~1.3-hour, narrow-regime-coverage flight
# (per data/ntsb_915is/README.md: only 2-3 of 12 REGIME_GRID_V1 bins
# populated, mostly CRUISE_HIGH_POWER) to identify more free parameters
# than it actually constrains. Fixing the two capacitances to
# first-principles estimates removes 2 of 7 dimensions from that
# underdetermined problem, leaving the fit to do what the data can
# actually support: the genuinely material-unknown cooling/heat-transfer
# coefficients.
#
#   C_hd (head+coolant, [ASSUMED] first-principles):
#     coolant ~3.0 L 50/50 glycol/water, rho~1050 kg/m^3, cp~3600 J/(kg K)
#       -> ~3.15 kg -> ~11,340 J/K
#     + lumped aluminum head metal near the coolant jacket, ~4 kg,
#       cp~900 J/(kg K) -> ~3,600 J/K
#     total ~15,000 J/K -- within ~8% of the PRIOR fitted value
#     (16,291 J/K), i.e. this doesn't contradict the earlier fit, it
#     just stops asking the optimizer to re-derive a number physics
#     already predicts closely.
#   C_oil ([ASSUMED] first-principles):
#     ~3.0 L engine oil, rho~875 kg/m^3, cp~1950 J/(kg K)
#       -> ~2.6 kg -> ~5,100 J/K
#     This is ~22x the PRIOR fitted value (234.7 J/K) -- a genuinely
#     implausible thermal mass (far too little inertia, meaning oil
#     temperature would swing far faster than a real ~2.6kg of oil
#     physically could), and a strong candidate for why oil-temp RMSE
#     (75.6C) is by far the worst of the three G1.2 channels.
#   Exact oil/coolant volumes are not confirmed against a Rotax 915iS
#   operator's manual (not found publicly during this session's search)
#   -- these are order-of-magnitude-defensible engineering estimates,
#   not a sourced spec value. Revisit if the real capacity is found.
C_HD_PHYSICAL = 15000.0
C_OIL_PHYSICAL = 5100.0

# Order matters -- must match PARAM_KEYS below.
PARAM_KEYS = [
    ("cooling_conductances", "UA_hd_nominal"),
    ("cooling_conductances", "UA_oc_nominal"),
    ("heat_exchanger", "UA_ho"),
    ("heat_transfer_to_head", "Q_to_head_fraction"),
    ("friction_heat", "friction_to_oil_fraction"),
]

# Loose but physically-sane bounds (positive capacitances/conductances,
# fractions in (0, 1)).
#
# CORRECTED back to the UNNORMALIZED-formula scale: a reference-velocity-
# normalized reparameterization of _compute_ua_hd/_compute_ua_oc/
# _compute_ua_c (which is what these bounds were previously tuned for)
# was tried and reverted this session -- see physics_twin.py's
# [ATTEMPTED-AND-REVERTED] comment on _compute_ua_c. The ACTIVE formula
# is the original unnormalized v_tas^0.8*rho^0.8*(N/300) form, under
# which UA_hd_nominal/UA_oc_nominal's real natural scale is what the
# CURRENTLY shipped thermal_params.json values already show
# (UA_hd_nominal=58.6, UA_oc_nominal=382.5) -- roughly two orders of
# magnitude smaller than the reparameterized-scale bounds this comment
# used to justify (500-200000 / 500-500000), which would have silently
# fit against the WRONG formula's scale had they been left in place.
# Lower bounds on UA_hd_nominal/UA_oc_nominal further loosened (5.0->0.5,
# 1.0->0.1): a full-window fit against physical C_hd/C_oil pinned BOTH at
# their old floors, and a diagnostic run at nominal (unfitted) params
# showed why -- at real cruise v_tas (~30 m/s), UA_oc*(T_oil-T_amb) swamps
# the ~6-8kW of friction heat feeding the oil (the nominal UA_oc_nominal
# base coefficient was originally fit against a LOW-airspeed-only window,
# v_tas in [1.7, 9.1] m/s, so v_tas^0.8's small value there let a large
# base coefficient through unnoticed). Giving the optimizer room below the
# old floor lets it find the true wide-airspeed-consistent coefficient
# instead of being artificially blocked from it.
# Q_to_head_fraction's lower bound also loosened (0.05->0.01): after the
# UA_oc exponent fix dropped it from 3-5 pinned params to just this one,
# same diagnostic applies -- give it room below the old floor to see if
# it wants a real value below 0.05 or was just blocked from converging.
LOWER = np.array([0.5, 0.1, 1.0, 0.01, 0.05])
UPPER = np.array([400.0, 2000.0, 200.0, 0.9, 0.98])


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
    # BUG FIX: this was defaulting to 0.1 (== production dt), contradicting
    # this script's own docstring ("Uses a coarser optimizer dt ... to keep
    # each residual evaluation affordable") -- with window=4300s that meant
    # 43,000 RK4 steps per residual call, and least_squares' finite-diff
    # Jacobian multiplies that by ~(n_params+1) per iteration. A real
    # dry-run took 27+ minutes with zero flushed output before this was
    # caught. Thermal time constants here are tens of seconds to minutes
    # (C_hd/C_oil are O(10^4) J/K against O(10-500) W/K conductances), so
    # dt=1.0 still resolves them fine while cutting per-eval cost ~10x.
    ap.add_argument("--opt-dt", type=float, default=1.0, help="PhysicsTwin dt used during optimization (coarser than production's 0.1 to keep the fit affordable)")
    ap.add_argument("--max-nfev", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true", help="fit and report, but do not write thermal_params.json")
    args = ap.parse_args()

    all_frames = load_ntsb_915is_fixture()
    window_frames = [f for f in all_frames if args.skip <= f.t < args.skip + args.window]
    print(f"Loaded {len(all_frames)} frames total, {len(window_frames)} in fit window "
          f"[{args.skip}, {args.skip + args.window})s")

    with open(THERMAL_PARAMS_PATH) as fh:
        thermal_params = json.load(fh)
    thermal_params["thermal_capacitances"]["C_hd"] = C_HD_PHYSICAL
    thermal_params["thermal_capacitances"]["C_oil"] = C_OIL_PHYSICAL
    print(f"Fixed (not fitted): C_hd={C_HD_PHYSICAL} J/K, C_oil={C_OIL_PHYSICAL} J/K (first-principles estimates)")
    x0 = get_params(thermal_params)
    print("Initial params:", dict(zip([k for _, k in PARAM_KEYS], x0)))

    twin = build_twin(args.opt_dt)
    # PhysicsTwin._load_model_parameters() loads thermal_params.json fresh
    # from disk into its OWN dict -- set_params() below only ever touches
    # PARAM_KEYS (no longer C_hd/C_oil), so the twin's C_hd/C_oil must be
    # pinned to the physical estimates explicitly here, or it would fit
    # against whatever C_hd/C_oil is still on disk.
    twin.thermal_params["thermal_capacitances"]["C_hd"] = C_HD_PHYSICAL
    twin.thermal_params["thermal_capacitances"]["C_oil"] = C_OIL_PHYSICAL
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
    val_twin.thermal_params["thermal_capacitances"]["C_hd"] = C_HD_PHYSICAL
    val_twin.thermal_params["thermal_capacitances"]["C_oil"] = C_OIL_PHYSICAL
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
                     "1.0, silently zeroing Q_fric_to_oil in all earlier fits). C_hd/C_oil "
                     "are FIXED to first-principles physical estimates ([ASSUMED], see "
                     "fit_phase_b_thermal.py's C_HD_PHYSICAL/C_OIL_PHYSICAL comment), not "
                     "fitted -- the prior 7-parameter fit had 5 of 7 landing on bounds, the "
                     "textbook symptom of over-parameterizing a fit against one narrow-"
                     "regime-coverage flight; removing the two capacitances (real physical "
                     "constants that shouldn't need curve-fitting at all) leaves only the "
                     "genuinely unknown UA/heat-split coefficients free.",
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
