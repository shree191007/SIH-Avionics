"""
Persisted evaluation harness for the Stage 2 calibration-accuracy gates
(02_physics_twin_residuals.md Sec 0):

  G1.1  Gas-path fit vs Rotax published curves: <=3% power error
  G1.2  Thermal fit vs held-out flight data: CHT RMSE <=6C, oil-temp
        RMSE <=4C, oil-pressure RMSE <=0.25 bar

Before this file, every number backing these gates existed only in
one-off, thrown-away scratch scripts. This makes them real, checked-in,
repeatable tests against the actual calibrated model/ artifacts and real
Rotax 915 iS flight data (see data/ntsb_915is/README.md for provenance).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "real_data"))

from ntsb_915is_loader import load_ntsb_915is_fixture  # noqa: E402
from replan_to_learn.physics_twin.physics_twin import PhysicsTwin  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"

# Real anchor points from the Rotax 915 iS operator's manual (Sec 2.1,
# page 2-2/2-4): the only two numeric (RPM, power) pairs actually published
# in the manual text (the full curve is a proprietary external tool, see
# model/rotax_915is_reference_data.json).
ROTAX_POWER_ANCHORS = [
    {"n_rpm": 5500.0, "map_pa": 1450e2, "t_im": 305.15, "power_target_w": 99_000.0},
    {"n_rpm": 5800.0, "map_pa": 1520e2, "t_im": 305.15, "power_target_w": 104_000.0},
]
G1_1_MAX_POWER_ERROR_FRAC = 0.03

G1_2_CHT_RMSE_MAX_C = 6.0
G1_2_OIL_TEMP_RMSE_MAX_C = 4.0
G1_2_OIL_PRESSURE_RMSE_MAX_BAR = 0.25


def _steady_state_power_w(twin: PhysicsTwin, n_rpm: float, map_pa: float, t_im: float) -> float:
    """
    Predicted brake power at a fixed (RPM, MAP) operating point, using the
    same gas-path formulas PhysicsTwin evaluates internally, at nominal
    theta (as-new engine) -- what G1.1 is actually checking.
    """
    eta_vol = twin._compute_eta_vol(n_rpm, map_pa)
    mdot_air = eta_vol * twin.fixed_params["V_d"] * (n_rpm / 120.0) * map_pa / (twin.fixed_params["R_air"] * t_im)
    mdot_air = max(1e-9, mdot_air)
    lambda_cmd = twin._compute_lambda_cmd(n_rpm, map_pa)
    mdot_f = mdot_air / (lambda_cmd * twin.fixed_params["AFR_st"])
    q_fuel = mdot_f * twin.fixed_params["LHV"]
    lam = mdot_air / (mdot_f * twin.fixed_params["AFR_st"])
    eta_ind_0 = twin._compute_eta_ind(lam, n_rpm, map_pa)
    p_ind = eta_ind_0 * q_fuel
    fmep = twin._c_0 + twin._c_1 * n_rpm + twin._c_2 * n_rpm ** 2
    p_brake = p_ind - fmep * twin.fixed_params["V_d"] * n_rpm / 120.0
    return max(0.0, p_brake)


@pytest.fixture(scope="module")
def calibrated_twin() -> PhysicsTwin:
    return PhysicsTwin(model_dir=MODEL_DIR, dt=0.1)


class TestG11GasPathFit:
    """G1.1: steady-state gas-path fit vs Rotax published curves, <=3% power error."""

    @pytest.mark.parametrize("anchor", ROTAX_POWER_ANCHORS, ids=lambda a: f"{a['n_rpm']:.0f}rpm")
    def test_power_within_3_percent_of_rotax_anchor(self, calibrated_twin, anchor):
        predicted_w = _steady_state_power_w(
            calibrated_twin, anchor["n_rpm"], anchor["map_pa"], anchor["t_im"]
        )
        target_w = anchor["power_target_w"]
        rel_error = abs(predicted_w - target_w) / target_w
        assert rel_error <= G1_1_MAX_POWER_ERROR_FRAC, (
            f"G1.1 FAILED at {anchor['n_rpm']:.0f}rpm: predicted {predicted_w/1000:.1f}kW "
            f"vs Rotax {target_w/1000:.1f}kW ({rel_error:.1%} error, limit {G1_1_MAX_POWER_ERROR_FRAC:.0%})"
        )


class TestG12ThermalFit:
    """
    G1.2: thermal sub-model fit vs held-out healthy flight data.

    Uses the real Rotax 915 iS flight fixture (data/ntsb_915is/). As of the
    current calibration this gate is NOT met.

    Current real-data RMSE (full flight, dt=0.1, model/thermal_params.json
    as shipped): CHT=36.7C (target <=6.0C), oil-temp=75.6C (target <=4.0C),
    oil-pressure=4.56bar (target <=0.25bar). These are a large improvement
    over an earlier stale measurement (CHT~82.8C, oil-temp~75.3C,
    oil-pressure~4.4bar) that predated a real upstream bug fix: PhysicsTwin
    was initializing theta_fric (health_params[8], the multiplicative
    friction/FMEP health parameter) to 0.0 instead of its correct nominal
    value of 1.0, which silently zeroed Q_fric_to_oil -- the friction-heat
    source term in the oil-temperature ODE -- in every earlier measurement.
    CHT roughly halved once that heat source was restored; oil-temp and
    oil-pressure did not meaningfully change from the theta_fric fix alone
    (oil-temp is dominated by other terms; oil-pressure's error is mostly
    inherited from the still-wrong simulated T_oil feeding the
    viscosity model).

    A Phase B recalibration was re-run after the theta_fric fix
    (scripts/fit_phase_b_thermal.py), refitting C_hd, C_oil, UA_hd_nominal,
    UA_oc_nominal, UA_ho, Q_to_head_fraction and friction_to_oil_fraction
    via scipy trf least-squares against a 400s window (skip first 250s) of
    the same NTSB WPR22LA211 flight, matching the window already recorded
    in model/thermal_params.json's provenance. The optimizer DID converge
    in the formal sense (ftol/xtol satisfied, cost dropped from 2.02e4 to
    1.03e3) -- a real improvement over the previously-recorded ~1e14
    non-convergent fit -- but 5 of the 7 fitted parameters landed on their
    (generously-set) bounds, the same non-identifiability signature as
    before. Full-flight validation RMSE with the new fit was CHT=19.35C,
    oil-temp=52.78C, oil-pressure=2.34bar: a substantial improvement over
    the shipped calibration, but still 3-13x off every G1.2 threshold, not
    "close to" passing by any reasonable read. Per the params landing on
    bounds, this reads as a single 400s window of one accident flight
    still not carrying enough regime diversity to identify 7 independent
    thermal parameters -- the same open gap the original provenance
    documented, not newly closed. That refit was therefore NOT adopted
    into model/thermal_params.json (adopting a bound-pinned fit could
    regress behavior in regimes outside its narrow fit window without
    actually closing any gate), and none of the three xfail markers below
    were removed.

    A follow-up investigation dug into WHY oil-temp/oil-pressure specifically
    did not move from the theta_fric fix (unlike CHT, which roughly halved).
    Direct diagnostic simulation (state seeded at real telemetry values at a
    representative cruise frame, t=1430s, N=5042rpm, v_tas=27.2m/s) found a
    real, reproducible structural bug: the oil-cooler cooling term
    UA_oc*(T_oil-T_amb) evaluates to ~597kW at that frame, against a
    Q_fric_to_oil heat source of only ~36kW -- a ~16x oversizing that
    collapses simulated T_oil toward T_amb regardless of C_oil, which
    matches the observed full-flight prediction bias directly (mean
    predicted-minus-real T_oil error is a near-constant -65 to -66C once
    C_oil is raised enough to stop the resulting integration instability --
    see below -- confirming this is a bias/scale bug, not just noise). The
    root cause: UA_hd/UA_oc scale with v_tas^0.8 * rho^0.8 with no
    reference-velocity normalization, so "UA_xx_nominal" is implicitly
    "UA at v_tas=1m/s" and blows up hugely by the time v_tas reaches real
    cruise speed (up to 41.2m/s in this flight). The original Phase B fit
    window (skip=250s, window=400s) only ever saw v_tas in [1.7, 9.1]m/s --
    a narrow, low-airspeed slice of the full flight's [0.1, 41.2]m/s range
    -- so the fitted UA_oc_nominal/UA_hd_nominal extrapolate to physically
    absurd cooling power outside that slice. (Separately, C_oil=234.7 J/K
    is also small enough relative to UA_oc's v^0.8-scaled magnitude at
    cruise that the oil-temp ODE's time constant drops below the RK4
    integration substep at production dt=0.1, well past the ~2.785x
    stability ratio for real eigenvalues -- confirmed by direct tau
    computation across the flight, tau falls below that threshold for
    ~33% of frames -- which is why raw T_oil predictions were seen
    oscillating between the state's hard clip bounds of 250-450K before
    any refit was attempted.)

    scripts/fit_phase_b_thermal.py was updated to (a) widen UA_oc_nominal's
    lower bound from 20.0 to 1.0 -- diagnosed as sitting almost exactly at
    the old bound for the fitted value needed to close the cruise-frame
    heat balance above (~29) -- and (b) default to a fit window covering
    nearly the entire flight (skip=100s, window=4300s) instead of the old
    400s low-airspeed-only slice, so the optimizer actually sees the real
    airspeed/RPM range it is validated against. Re-running the fit
    (scipy trf, opt_dt=1.0 for tractability, max_nfev=60) against this
    corrected window DID reduce cost (1.29e6 -> 1.02e6) but did NOT
    converge to a materially better calibration: full-flight validation
    RMSE was CHT=33.39C, oil-temp=69.23C, oil-pressure=3.655bar -- only a
    small improvement over the shipped calibration (36.7C/75.6C/4.56bar),
    nowhere near the 3-13x-off numbers the earlier narrow-window refit
    produced in-sample, and 2 of the 7 parameters (C_hd, UA_ho) landed on
    the (now different) bounds -- the same non-identifiability signature
    as before, just with different parameters pinned. This was NOT adopted
    into model/thermal_params.json (bound-pinned, not a real convergence,
    and not an improvement worth the regression risk).

    Taken together, this is stronger evidence than before that the gap is
    a genuine model-structure limitation, not a fixable data/bounds
    problem: even after directly diagnosing and correcting the fit
    window/bounds bug that explains the oil-temp/pressure divergence
    mechanically, refitting within the existing UA ~ v_tas^0.8 (no
    reference-velocity normalization) functional form still cannot
    represent this flight's oil/head thermal dynamics across its full
    speed range well enough to approach the G1.2 thresholds. Actually
    fixing this would mean redesigning the cooling-conductance functional
    form itself (e.g. normalizing v_tas/rho/N by real reference flight
    conditions instead of the current implicit v=1m/s/rho=1/N=300rpm
    reference) and refitting all thermal parameters against that new
    model -- a materially larger change than a parameter recalibration,
    intentionally left out of scope here rather than forced through
    without a real, checked improvement to show for it.

    This remains a real, currently-open gap, not a test bug, so this is
    marked xfail(strict=True) rather than quietly skipped or loosened: if
    a future recalibration actually closes the gap, this test starts
    failing *because it now unexpectedly passes*, which is exactly the
    signal that the xfail marker should be removed.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def prediction_errors(calibrated_twin):
        frames = load_ntsb_915is_fixture()
        twin = calibrated_twin
        twin.reset(frames[0])
        cht_err, toil_err, poil_err = [], [], []
        for frame in frames:
            twin._advance_state(frame)
            preds = twin._compute_predictions_from_state(twin.state, twin.health_params, telemetry=frame)
            cht_err.append(preds["cht"] - frame.cht)
            toil_err.append(preds["t_oil"] - frame.t_oil)
            poil_err.append(preds["p_oil"] - frame.p_oil)
        return np.array(cht_err), np.array(toil_err), np.array(poil_err)

    @pytest.mark.xfail(reason="Phase B thermal fit does not currently converge against real data (see docstring)", strict=True)
    def test_cht_rmse_within_6c(self, prediction_errors):
        cht_err, _, _ = prediction_errors
        rmse_c = float(np.sqrt(np.mean(cht_err ** 2)))
        assert rmse_c <= G1_2_CHT_RMSE_MAX_C, f"CHT RMSE {rmse_c:.1f}C exceeds {G1_2_CHT_RMSE_MAX_C}C"

    @pytest.mark.xfail(reason="Phase B thermal fit does not currently converge against real data (see docstring)", strict=True)
    def test_oil_temp_rmse_within_4c(self, prediction_errors):
        _, toil_err, _ = prediction_errors
        rmse_c = float(np.sqrt(np.mean(toil_err ** 2)))
        assert rmse_c <= G1_2_OIL_TEMP_RMSE_MAX_C, f"oil-temp RMSE {rmse_c:.1f}C exceeds {G1_2_OIL_TEMP_RMSE_MAX_C}C"

    @pytest.mark.xfail(reason="Oil-pump formula fit does not currently converge to spec-level accuracy against real data", strict=True)
    def test_oil_pressure_rmse_within_0_25_bar(self, prediction_errors):
        _, _, poil_err = prediction_errors
        rmse_bar = float(np.sqrt(np.mean(poil_err ** 2))) / 1e5
        assert rmse_bar <= G1_2_OIL_PRESSURE_RMSE_MAX_BAR, (
            f"oil-pressure RMSE {rmse_bar:.2f}bar exceeds {G1_2_OIL_PRESSURE_RMSE_MAX_BAR}bar"
        )

    def test_thermal_rmse_report(self, prediction_errors, capsys):
        """Not a gate assertion -- always passes, exists to print the current real numbers into test output."""
        cht_err, toil_err, poil_err = prediction_errors
        print(f"\nG1.2 current real-data RMSE: CHT={np.sqrt(np.mean(cht_err**2)):.1f}C "
              f"(target <={G1_2_CHT_RMSE_MAX_C}C), "
              f"oil-temp={np.sqrt(np.mean(toil_err**2)):.1f}C (target <={G1_2_OIL_TEMP_RMSE_MAX_C}C), "
              f"oil-pressure={np.sqrt(np.mean(poil_err**2))/1e5:.2f}bar (target <={G1_2_OIL_PRESSURE_RMSE_MAX_BAR}bar)")
