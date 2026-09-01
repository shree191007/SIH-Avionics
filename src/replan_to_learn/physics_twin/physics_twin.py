"""
SIH26054 Replan to Learn: Physics Twin Implementation.
Implements the mean-value engine model (L1) and residual generation (L2).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from replan_to_learn.contracts.telemetry import TelemetryFrame
from replan_to_learn.contracts.residuals import ResidualFrame
from replan_to_learn.contracts.regimes import RegimeClassifier, BIN_TRANSIENT, BIN_INVALID
from replan_to_learn.contracts.residuals import ResidualSuppressionEngine
from replan_to_learn.contracts.faults import SpatialEGTDecomposer


@dataclass
class PhysicsTwinConfig:
    """Configuration for PhysicsTwin model."""
    dt: float = 0.1
    model_version: str = "1.0.0"
    regime_grid_version: str = "REGIME_GRID_V1"


class PhysicsTwin:
    """
    Physics Twin implementing L1 (mean-value engine model) and L2 (residual generation).

    Implements the contract from Section 6.4 of the specification:
    - __init__(self, model_dir: Path, dt: float = 0.1) -> None
    - reset(self, frame: TelemetryFrame) -> None
    - step(self, frame: TelemetryFrame) -> ResidualFrame
    - predict(self, u: InputSequence, theta: np.ndarray) -> np.ndarray
    - jacobian(self, u: InputFrame, theta: np.ndarray) -> np.ndarray
    """

    def __init__(self, model_dir: Path, dt: float = 0.1) -> None:
        """Initialize PhysicsTwin with calibrated model parameters."""
        self.model_dir = Path(model_dir)
        self.config = PhysicsTwinConfig(dt=dt)
        self._load_model_parameters()
        self._initialize_nominal_state()
        self._regime_classifier = RegimeClassifier()
        self._prev_t: Optional[float] = None

    def _load_model_parameters(self) -> None:
        """Load all calibrated model parameters from JSON files."""
        try:
            with open(self.model_dir / "regime_grid_v1.json", 'r') as f:
                self.regime_grid = json.load(f)
            with open(self.model_dir / "gaspath_params.json", 'r') as f:
                self.gaspath_params = json.load(f)
            with open(self.model_dir / "thermal_params.json", 'r') as f:
                self.thermal_params = json.load(f)
            with open(self.model_dir / "noise_model.json", 'r') as f:
                self.noise_model = json.load(f)
        except FileNotFoundError as e:
            raise RuntimeError(f"Missing model artifact: {e.filename}")

        self._build_interpolators()

    def _build_interpolators(self) -> None:
        """Build scipy interpolators for lookup tables."""
        gp = self.gaspath_params

        eta_vol_lt = gp["volumetric_efficiency"]["lookup_table"]
        self._eta_vol_interp = RegularGridInterpolator(
            (eta_vol_lt["N_breakpoints"], eta_vol_lt["p_im_breakpoints"]),
            np.array(eta_vol_lt["values"]),
            method="linear",
            bounds_error=False,
            fill_value=1.0,
        )

        eta_ind_lt = gp["indicated_efficiency"]["lookup_table"]
        self._eta_ind_interp = RegularGridInterpolator(
            (eta_ind_lt["lambda_breakpoints"], eta_ind_lt["N_breakpoints"], eta_ind_lt["p_im_breakpoints"]),
            np.array(eta_ind_lt["values"]),
            method="linear",
            bounds_error=False,
            fill_value=0.35,
        )

        lambda_lt = gp["lambda_commanded"]["lookup_table"]
        lambda_vals = np.array(lambda_lt["values"])
        if lambda_vals.ndim == 3:
            lambda_vals = np.mean(lambda_vals, axis=0)
        self._lambda_cmd_interp = RegularGridInterpolator(
            (lambda_lt["N_breakpoints"], lambda_lt["p_im_breakpoints"]),
            lambda_vals,
            method="linear",
            bounds_error=False,
            fill_value=1.0,
        )

        self._c_0 = gp["fmep_coefficients"]["c_0"]
        self._c_1 = gp["fmep_coefficients"]["c_1"]
        self._c_2 = gp["fmep_coefficients"]["c_2"]

        self._noise_std = {}
        noise_std_raw = self.noise_model.get("noise_std", {})
        if isinstance(noise_std_raw, dict):
            values_map = noise_std_raw.get("values", noise_std_raw)
            for ch, vals in values_map.items():
                if ch in ("description", "channels", "regime_bins"):
                    continue
                self._noise_std[ch] = np.array(vals, dtype=np.float64)
        self._ar_coeffs = {}
        ar_data = self.noise_model["ar_coefficients"]
        channels = ar_data.get("channels", [])
        values = ar_data.get("values", [])
        for i, ch in enumerate(channels):
            self._ar_coeffs[ch] = float(values[i]) if i < len(values) else 0.0

        self._prev_e_whitened = np.zeros(9, dtype=np.float64)

    def _initialize_nominal_state(self) -> None:
        """Initialize state and health parameters to nominal values."""
        self.state = np.zeros(10)
        self.state[0] = 100000.0
        self.state[1] = 350.0
        self.state[2] = 350.0
        self.state[3] = 600.0
        self.state[4] = 300.0 * 2.0 * math.pi / 60.0
        self.state[5:] = 350.0

        self.health_params = np.zeros(15, dtype=np.float64)
        self.health_params[0:9] = 1.0
        self.health_params[9:] = 0.0

        # Self-calibrated fixed-pitch propeller load-curve constant (see
        # _compute_power_balance_residual): P_prop = k_prop * N_rpm^3 *
        # sigma_density, the standard cube-law propeller load curve
        # (Aircraft Powerplant Handbook). Reset on every reset() so each
        # new flight gets its own fit -- avoids needing an externally
        # published thrust/power constant for an unknown airframe+prop
        # combination. Fit via running least-squares through the origin
        # (k_prop = sum(x*y)/sum(x^2), x=N^3*sigma_density, y=P_brake)
        # across every stable above-idle frame seen so far, not just the
        # first one: a single-sample "calibration" is exactly as noisy as
        # whatever frame happened to be first (if it's mid-transient or
        # otherwise atypical, the whole flight's power-balance residual
        # channel inherits that bias/noise permanently). Accumulated as
        # running sums (not stored raw samples) to stay streaming/
        # fixed-memory, matching this codebase's other streaming
        # estimators. Refit on every call up to _K_PROP_MAX_SAMPLES, then
        # frozen -- keeps early-flight estimates responsive while
        # bounding cost and avoiding drift from a real developing fault
        # (which is exactly the signal this channel exists to detect, so
        # it must NOT keep re-averaging over a genuinely faulted period).
        self._k_prop: Optional[float] = None
        self._k_prop_sum_xy: float = 0.0
        self._k_prop_sum_xx: float = 0.0
        self._k_prop_n_samples: int = 0

        self.fixed_params = {
            "R_air": self.gaspath_params["constants"]["R_air"],
            "V_d": self.gaspath_params["constants"]["V_d"],
            "AFR_st": self.gaspath_params["constants"]["AFR_st"],
            "LHV": self.gaspath_params["constants"]["LHV"],
            "C_hd": self.thermal_params["thermal_capacitances"]["C_hd"],
            "C_oil": self.thermal_params["thermal_capacitances"]["C_oil"],
            "C_c": self.thermal_params["thermal_capacitances"]["C_c"],
            "UA_hd_nom": self.thermal_params["cooling_conductances"]["UA_hd_nominal"],
            "UA_oc_nom": self.thermal_params["cooling_conductances"]["UA_oc_nominal"],
            "UA_c_nom": self.thermal_params["cooling_conductances"]["UA_c_nominal"],
            "a_i": np.array(self.thermal_params["asymmetry_factors"]["a_values"]),
            "Q_to_head_frac": self.thermal_params["heat_transfer_to_head"]["Q_to_head_fraction"],
            "Q_to_head_cyl": np.array(self.thermal_params["heat_transfer_to_head"]["Q_to_head_per_cylinder"]),
            "friction_to_oil": self.thermal_params["friction_heat"]["friction_to_oil_fraction"],
            "UA_ho": self.thermal_params["heat_exchanger"]["UA_ho"],
            "k_probe": self.gaspath_params["probe_recovery"]["k_probe"],
            # Oil-pump pressure model: p_oil = k_oilpump*omega*mu_oil / (k_oilleak + k_oilclear*omega).
            # Defaults match the values this was hardcoded to before it was
            # exposed for calibration, so old model_dir artifacts without an
            # "oil_pump" block keep behaving exactly as before.
            "k_oilpump": self.thermal_params.get("oil_pump", {}).get("k_oilpump", 3000.0),
            "k_oilleak": self.thermal_params.get("oil_pump", {}).get("k_oilleak", 500.0),
            "k_oilclear": self.thermal_params.get("oil_pump", {}).get("k_oilclear", 100.0),
        }

        self._k1 = np.zeros_like(self.state)
        self._k2 = np.zeros_like(self.state)
        self._k3 = np.zeros_like(self.state)
        self._k4 = np.zeros_like(self.state)
        self._state_temp = np.zeros_like(self.state)

    N_THETA = 15  # theta_vol,theta_comb,theta_cool,theta_inj[1..4],theta_oilp,theta_fric,b_egt[1..4],b_cht,b_poil
    RHO_SEA_LEVEL_STD = 1.225  # kg/m^3, ISA sea-level standard density

    # Sanity ceiling for the modeled exhaust gas temperature, T_exh_calc.
    # Was T_im + 800.0 (~1080-1110K for typical cruise T_im), which turned
    # out to sit BELOW the real Rotax 915iS published EGT limit of 950 C
    # = 1223.15 K (model/rotax_915is_reference_data.json, egt_limits.max_c)
    # -- the model's own unclipped T_exh_calc routinely computed ~1145K at
    # ordinary cruise operating points, meaning this "sanity" bound was
    # silently saturating theta_comb's entire EGT response to zero at
    # every regime (dT_exh_calc/dtheta_comb = 0 once clipped), collapsing
    # theta_comb's fingerprint down to the single P_brake channel it
    # shares with theta_fric -- the real cause of their ~0.99 cosine
    # similarity, not a fundamental physical degeneracy. Set generously
    # above the real spec limit (not AT it) so a genuine overheat fault
    # the gate needs to actually see isn't clipped away either; this stays
    # a pure numerical-runaway guard, not a realism constraint.
    T_EXH_CALC_MAX_K = 1400.0

    def _pad_theta(self, theta: np.ndarray) -> np.ndarray:
        """
        The health-parameter vector always has a fixed 15-element physical
        layout (indices 0-8 multiplicative, nominal 1.0; indices 9-14
        additive sensor biases, nominal 0.0). Some callers -- e.g.
        IdentifiabilityGate configured for a reduced fingerprint dimension
        to test a minimal ambiguous-pair scenario -- legitimately carry a
        shorter theta covering only the first n < 15 parameters. Pad the
        rest with their nominal values instead of indexing out of bounds,
        so a caller working with a subset of parameters doesn't have to
        know or care about the other ones.
        """
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape[0] >= self.N_THETA:
            return theta
        nominal = np.zeros(self.N_THETA, dtype=np.float64)
        nominal[0:9] = 1.0
        padded = nominal.copy()
        padded[: theta.shape[0]] = theta
        return padded

    def _get_health_params(self) -> dict:
        """Extract health parameters from vector."""
        return {
            "theta_vol": float(self.health_params[0]),
            "theta_comb": float(self.health_params[1]),
            "theta_cool": float(self.health_params[2]),
            "theta_inj": self.health_params[3:7].copy(),
            "theta_oilp": float(self.health_params[7]),
            "theta_fric": float(self.health_params[8]),
            "b_egt": self.health_params[9:13].copy(),
            "b_cht": float(self.health_params[13]),
            "b_poil": float(self.health_params[14]),
        }

    def _compute_eta_vol(self, N_rpm: float, p_im: float) -> float:
        return float(self._eta_vol_interp([[N_rpm, p_im]])[0])

    def _compute_eta_ind(self, lam: float, N_rpm: float, p_im: float) -> float:
        return float(self._eta_ind_interp([[lam, N_rpm, p_im]])[0])

    def _compute_lambda_cmd(self, N_rpm: float, p_im: float) -> float:
        return float(self._lambda_cmd_interp([[N_rpm, p_im]])[0])

    def _compute_mu_oil(self, T_oil: float) -> float:
        params = self.thermal_params["oil_viscosity"]["parameters"]
        A = params["A"]
        B = params["B"]
        C = params["C"]
        mu = 10.0 ** (A + B / (C + T_oil))
        return float(np.clip(mu, 1e-6, 1.0))

    def _compute_ua_hd(self, v_tas: float, rho_amb: float, N: float) -> float:
        base = self.fixed_params["UA_hd_nom"]
        return base * max(0.1, (v_tas ** 0.8)) * max(0.1, (rho_amb ** 0.8)) * max(0.1, (N / 300.0))

    def _compute_ua_oc(self, v_tas: float, rho_amb: float) -> float:
        base = self.fixed_params["UA_oc_nom"]
        return base * max(0.1, (v_tas ** 0.8)) * max(0.1, (rho_amb ** 0.8))

    def _compute_ua_c(self, v_tas: float, rho_amb: float, N: float, cyl_idx: int) -> float:
        base = self.fixed_params["UA_c_nom"]
        a_i = self.fixed_params["a_i"]
        return base * (1.0 + a_i[cyl_idx]) * max(0.1, (v_tas ** 0.8)) * max(0.1, (rho_amb ** 0.8)) * max(0.1, (N / 300.0))

    # [ATTEMPTED-AND-REVERTED] A reference-velocity-normalized form of the
    # three functions above (UA = UA_nominal * (v_tas/50)^0.8 *
    # (rho/1.225)^0.8 * (N/2800) instead of the raw v_tas^0.8*rho^0.8*
    # (N/300) form kept here) was tried this session to fix G1.2's oil-
    # temp/CHT RMSE gap -- the unnormalized form makes UA_nominal
    # implicitly mean "UA at v_tas=1 m/s", a nonsensical reference point,
    # which genuinely does make a real Phase B least-squares fit badly
    # conditioned (confirmed: a UA_oc_nominal fit against a narrow low-
    # airspeed window extrapolates to ~597kW of cooling power at real
    # cruise conditions against a ~36kW real friction-heat source, a 16x
    # oversizing). However: (1) the max(0.1, ...) floor clamp triggers at
    # a DIFFERENT real airspeed under each parameterization (v_tas<0.056
    # m/s unnormalized vs v_tas<2.8 m/s normalized), so a same-behavior
    # rescale of the existing fitted values is NOT achievable at real
    # low-speed/idle/ground frames the full flight actually contains --
    # confirmed empirically, rescaling made RMSE WORSE (CHT 36.7->69.0C,
    # oil-temp 75.6->87.0C); (2) the refit attempted to fix that was
    # killed mid-run before writing a real result, leaving the (worse)
    # rescaled-only values in place; (3) with those bad intermediate
    # values live, theta_fric's noise-robustness regressed sharply
    # (26/30 -> 11/30, seeds 1002/2002), confirming the corrupted T_hd/
    # T_oil trajectory this caused propagates into the gate's fingerprint
    # channels, not just the G1.2 gates. Reverted cleanly back to the
    # known-good unnormalized form + the original fitted
    # thermal_params.json values (CHT 36.7C, oil-temp 75.6C, oil-pressure
    # 4.56bar, matching the xfail'd G1.2 gates' documented current
    # numbers). The conditioning diagnosis is still real and worth
    # revisiting with a properly-completed refit (bounds already widened
    # in scripts/fit_phase_b_thermal.py for this scale) by whoever picks
    # this up next -- just don't ship a half-migrated intermediate state.

    def reset(self, frame: TelemetryFrame) -> None:
        """Reset the PhysicsTwin state based on initial telemetry frame."""
        self._initialize_nominal_state()
        self._regime_classifier.reset()
        self._prev_t = None
        self._prev_e_whitened = np.zeros(9, dtype=np.float64)

    def step(self, frame: TelemetryFrame) -> ResidualFrame:
        """Process one time step of telemetry and emit residual frame."""
        self._advance_state(frame)
        return self._compute_residuals(frame)

    def _compute_state_derivatives(self, state: np.ndarray, telemetry: TelemetryFrame, theta: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute state derivatives dx/dt for the mean-value engine model."""
        derivatives = np.zeros_like(state)

        if theta is None:
            theta = self.health_params
        theta = self._pad_theta(theta)

        p_im = state[0]
        T_hd = state[1]
        T_oil = state[2]
        T_exh = state[3]
        omega = state[4]
        T_cyl = state[5:9]

        N_rpm = telemetry.n_rpm
        omega_meas = N_rpm * 2.0 * math.pi / 60.0

        T_im = telemetry.t_im
        p_amb = telemetry.p_amb
        T_amb = telemetry.t_amb
        v_tas = telemetry.v_tas
        rho_amb = p_amb / (self.fixed_params["R_air"] * T_amb)

        theta_vol = theta[0]
        theta_comb = theta[1]
        theta_cool = theta[2]
        theta_inj = theta[3:7]
        theta_oilp = theta[7]
        theta_fric = theta[8]
        b_egt = theta[9:13]
        b_cht = theta[13]
        b_poil = theta[14]

        eta_vol = self._compute_eta_vol(N_rpm, p_im)
        mdot_air = theta_vol * eta_vol * self.fixed_params["V_d"] * (N_rpm / (2.0 * 60.0)) * p_im / (self.fixed_params["R_air"] * T_im)
        mdot_air = max(1e-9, mdot_air)

        lambda_cmd = self._compute_lambda_cmd(N_rpm, p_im)
        mdot_f_cyl = theta_inj * mdot_air / (4.0 * lambda_cmd * self.fixed_params["AFR_st"])
        mdot_f = float(np.sum(mdot_f_cyl))
        mdot_f = max(1e-9, mdot_f)

        Q_dot_fuel = mdot_f * self.fixed_params["LHV"]

        lam_i = mdot_air / (4.0 * mdot_f_cyl * self.fixed_params["AFR_st"])
        eta_ind_0 = np.array([self._compute_eta_ind(float(lam), N_rpm, p_im) for lam in lam_i])
        eta_ind = theta_comb * np.mean(eta_ind_0)
        P_ind = theta_comb * np.mean(eta_ind_0) * Q_dot_fuel

        FMEP = theta_fric * (self._c_0 + self._c_1 * N_rpm + self._c_2 * N_rpm ** 2)
        P_brake = P_ind - FMEP * self.fixed_params["V_d"] * N_rpm / (2.0 * 60.0)
        P_brake = max(0.0, P_brake)

        f_ht = 0.15
        T_exh_calc = T_im + (1.0 - eta_ind - f_ht) * Q_dot_fuel / (mdot_air * 1150.0)
        T_exh_calc = max(T_amb, min(self.T_EXH_CALC_MAX_K, T_exh_calc))

        k_probe = self.fixed_params["k_probe"]
        lam_avg = max(1e-9, float(np.mean(lam_i)))
        egt_hat = telemetry.t_amb + k_probe * (T_exh_calc * (lam_i / lam_avg) - telemetry.t_amb) + b_egt

        Q_to_head_total = self.fixed_params["Q_to_head_frac"] * Q_dot_fuel
        Q_to_head_cyl = self.fixed_params["Q_to_head_cyl"] * Q_to_head_total
        Q_fric_total = FMEP * self.fixed_params["V_d"] * N_rpm / (2.0 * 60.0)
        Q_fric_to_oil = self.fixed_params["friction_to_oil"] * Q_fric_total

        UA_hd = self._compute_ua_hd(v_tas, rho_amb, omega_meas)
        UA_oc = self._compute_ua_oc(v_tas, rho_amb)
        UA_ho = self.fixed_params["UA_ho"]

        dT_hd = (Q_to_head_total - theta_cool * UA_hd * (T_hd - T_amb) - UA_ho * (T_hd - T_oil)) / self.fixed_params["C_hd"]
        dT_oil = (Q_fric_to_oil + UA_ho * (T_hd - T_oil) - UA_oc * (T_oil - T_amb)) / self.fixed_params["C_oil"]

        dT_cyl = np.zeros(4)
        for i in range(4):
            UA_c_i = self._compute_ua_c(v_tas, rho_amb, omega_meas, i)
            dT_cyl[i] = (Q_to_head_cyl[i] - UA_c_i * (T_cyl[i] - T_amb) - UA_ho * 0.1 * (T_cyl[i] - T_hd)) / self.fixed_params["C_c"]

        derivatives[0] = 2.0 * (telemetry.map_pa - p_im)
        derivatives[1] = dT_hd
        derivatives[2] = dT_oil
        derivatives[3] = 2.0 * (T_exh_calc - T_exh)
        derivatives[4] = 2.0 * (omega_meas - omega)
        derivatives[5:9] = dT_cyl

        return derivatives

    def _advance_state(self, telemetry: TelemetryFrame) -> None:
        """
        Integrate the physics forward by the actual elapsed time since the
        previous call (telemetry.t - self._prev_t), sub-stepping RK4 at
        config.dt granularity, instead of always advancing by a fixed
        config.dt regardless of how far apart in real time the telemetry
        frames actually are.

        Without this, a twin driven by telemetry sparser than config.dt
        (e.g. 1 Hz real logs vs the default dt=0.1s) silently runs its
        internal clock at a fraction of real time: state under-integrates
        real dynamics, and any derivative computed downstream from
        consecutive states (e.g. the regime classifier's quasi-steady
        |dT_hd/dt| check) compares an artificially small state change
        against a full real-time delta, corrupting exactly the stability
        judgement that the regime axis depends on.
        """
        if self._prev_t is None:
            elapsed = self.config.dt
        else:
            elapsed = telemetry.t - self._prev_t
            if elapsed <= 0.0 or elapsed > 30.0:
                # Non-monotonic timestamp, or a gap large enough that
                # bridging it with dozens of substeps would just extrapolate
                # noise -- fall back to a single nominal step, the same
                # conservative behavior as a fresh reset.
                elapsed = self.config.dt
        self._prev_t = telemetry.t

        n_substeps = max(1, round(elapsed / self.config.dt))
        substep_dt = elapsed / n_substeps
        for _ in range(n_substeps):
            self._rk4_step(telemetry, dt=substep_dt)

    def _rk4_step(self, telemetry: TelemetryFrame, dt: Optional[float] = None) -> None:
        """Perform one RK4 integration step of duration `dt` (defaults to config.dt)."""
        if dt is None:
            dt = self.config.dt
        self._k1 = self._compute_state_derivatives(self.state, telemetry)
        self._state_temp = self.state + 0.5 * dt * self._k1
        self._k2 = self._compute_state_derivatives(self._state_temp, telemetry)
        self._state_temp = self.state + 0.5 * dt * self._k2
        self._k3 = self._compute_state_derivatives(self._state_temp, telemetry)
        self._state_temp = self.state + dt * self._k3
        self._k4 = self._compute_state_derivatives(self._state_temp, telemetry)
        self.state = self.state + (dt / 6.0) * (self._k1 + 2 * self._k2 + 2 * self._k3 + self._k4)

        self.state[0] = np.clip(self.state[0], 50000.0, 250000.0)
        self.state[1] = np.clip(self.state[1], 273.15, 500.0)
        self.state[2] = np.clip(self.state[2], 250.0, 450.0)
        # Upper bound raised from 1200.0 to match T_EXH_CALC_MAX_K (the
        # algebraic T_exh_calc ceiling, raised for the same reason -- see
        # its docstring): with the old 1200K state clip sitting BELOW the
        # new 1400K algebraic ceiling, the T_exh state would relax toward
        # and saturate at 1200 even under ordinary healthy operation
        # (confirmed empirically), silently re-introducing the same
        # signal-clipping problem the algebraic fix was meant to remove,
        # just one layer down.
        self.state[3] = np.clip(self.state[3], 300.0, self.T_EXH_CALC_MAX_K + 50.0)
        self.state[4] = np.clip(self.state[4], 50.0, 600.0)
        self.state[5:9] = np.clip(self.state[5:9], 273.15, 500.0)

    def reset(self, frame: TelemetryFrame) -> None:
        """Reset the PhysicsTwin state based on initial telemetry frame."""
        self._initialize_nominal_state()
        self._regime_classifier.reset()
        self._prev_t = None
        self._prev_e_whitened = np.zeros(9, dtype=np.float64)

    def _compute_predictions(self, telemetry: TelemetryFrame) -> dict:
        """Compute predicted measurements from current state."""
        return self._compute_predictions_from_state(self.state, self.health_params, telemetry)

    def _compute_predictions_from_state(self, state: np.ndarray, theta: np.ndarray, telemetry: Optional[TelemetryFrame] = None) -> dict:
        """Compute predicted measurements from arbitrary state and theta."""
        if telemetry is None:
            raise ValueError("telemetry is required")
        theta = self._pad_theta(theta)

        p_im = state[0]
        T_hd = state[1]
        T_oil = state[2]
        T_exh = state[3]
        omega = state[4]
        T_cyl = state[5:9]

        N_rpm = telemetry.n_rpm
        T_im = telemetry.t_im
        p_amb = telemetry.p_amb
        T_amb = telemetry.t_amb
        v_tas = telemetry.v_tas
        rho_amb = p_amb / (self.fixed_params["R_air"] * T_amb)

        theta_vol = theta[0]
        theta_comb = theta[1]
        theta_cool = theta[2]
        theta_inj = theta[3:7]
        theta_oilp = theta[7]
        theta_fric = theta[8]
        b_egt = theta[9:13]
        b_cht = theta[13]
        b_poil = theta[14]

        eta_vol = self._compute_eta_vol(N_rpm, p_im)
        mdot_air = theta_vol * eta_vol * self.fixed_params["V_d"] * (N_rpm / (2.0 * 60.0)) * p_im / (self.fixed_params["R_air"] * T_im)
        mdot_air = max(1e-9, mdot_air)

        lambda_cmd = self._compute_lambda_cmd(N_rpm, p_im)
        mdot_f_cyl = theta_inj * mdot_air / (4.0 * lambda_cmd * self.fixed_params["AFR_st"])
        mdot_f = float(np.sum(mdot_f_cyl))
        mdot_f = max(1e-9, mdot_f)

        Q_dot_fuel = mdot_f * self.fixed_params["LHV"]

        lam_i = mdot_air / (4.0 * mdot_f_cyl * self.fixed_params["AFR_st"])
        eta_ind_0 = np.array([self._compute_eta_ind(float(lam), N_rpm, p_im) for lam in lam_i])
        eta_ind = theta_comb * np.mean(eta_ind_0)
        P_ind = theta_comb * np.mean(eta_ind_0) * Q_dot_fuel

        FMEP = theta_fric * (self._c_0 + self._c_1 * N_rpm + self._c_2 * N_rpm ** 2)
        P_brake = P_ind - FMEP * self.fixed_params["V_d"] * N_rpm / (2.0 * 60.0)
        P_brake = max(0.0, P_brake)

        f_ht = 0.15
        T_exh_calc = T_im + (1.0 - eta_ind - f_ht) * Q_dot_fuel / (mdot_air * 1150.0)
        T_exh_calc = max(T_amb, min(self.T_EXH_CALC_MAX_K, T_exh_calc))

        k_probe = self.fixed_params["k_probe"]
        lam_avg = max(1e-9, float(np.mean(lam_i)))
        egt_hat = telemetry.t_amb + k_probe * (T_exh_calc * (lam_i / lam_avg) - telemetry.t_amb) + b_egt

        mu_oil = self._compute_mu_oil(T_oil)
        k_oilpump = self.fixed_params["k_oilpump"]
        k_oilleak = self.fixed_params["k_oilleak"]
        k_oilclear = self.fixed_params["k_oilclear"]
        p_oil = theta_oilp * (k_oilpump * omega * mu_oil / (k_oilleak + k_oilclear * omega)) + b_poil
        p_oil = max(0.0, p_oil)

        return {
            "egt": egt_hat,
            "cht": T_hd + b_cht,
            "p_oil": p_oil,
            "t_oil": T_oil,
            "mdot_f": mdot_f,
            "p_brake": P_brake,
        }

    # Number of stable above-idle samples the running least-squares
    # propeller-constant fit accumulates before freezing. Chosen to cover
    # a real flight's early climb/cruise transition (a few minutes at
    # 1Hz-to-10Hz effective sample rates) without running indefinitely
    # into a period where a real fault may have developed.
    _K_PROP_MAX_SAMPLES = 200

    def _compute_power_balance_residual(self, telemetry: TelemetryFrame, p_brake_w: float) -> float:
        """
        Power-balance residual channel: P_brake(model) - P_prop(N_rpm), in kW.

        P_prop uses the standard fixed-pitch propeller load curve (the
        "cube law": HP = K * RPM^3 * sigma_density, from the Aircraft
        Powerplant Handbook / FAA propeller theory) rather than a made-up
        linear RPM proxy. K is not a published constant -- it depends on
        the specific airframe/propeller, which isn't known for arbitrary
        fleet telemetry -- so it is self-calibrated from this twin's own
        stable, above-idle frames each flight (see reset(), which clears
        the accumulator), using the model's own P_brake prediction at
        nominal health as the calibration target. A running least-squares
        fit through the origin (not a single-sample ratio -- see the
        accumulator fields' docstring in _initialize_nominal_state for why
        that was a real, fixed noise source) makes the residual small at
        the samples it was fit against and grows as later brake power
        diverges from what a nominal engine would need to produce to be
        turning the prop at the observed RPM/density -- which is precisely
        the signal theta_fric (mechanical friction loss, the only health
        parameter that does NOT also touch EGT/CHT/mdot_f) needs to become
        separable from theta_vol/theta_comb.
        """
        rho_amb = telemetry.p_amb / (self.fixed_params["R_air"] * telemetry.t_amb)
        sigma_density = rho_amb / self.RHO_SEA_LEVEL_STD
        n_cubed_sigma = (telemetry.n_rpm ** 3) * sigma_density

        if telemetry.n_rpm > 1000.0 and n_cubed_sigma > 1e-9 and self._k_prop_n_samples < self._K_PROP_MAX_SAMPLES:
            self._k_prop_sum_xy += n_cubed_sigma * p_brake_w
            self._k_prop_sum_xx += n_cubed_sigma * n_cubed_sigma
            self._k_prop_n_samples += 1
            if self._k_prop_sum_xx > 1e-9:
                self._k_prop = self._k_prop_sum_xy / self._k_prop_sum_xx

        if self._k_prop is None:
            return 0.0

        p_prop_w = self._k_prop * n_cubed_sigma
        return (p_brake_w - p_prop_w) / 1000.0

    def _compute_residuals(self, telemetry: TelemetryFrame) -> ResidualFrame:
        """Compute normalized residuals from current state and telemetry."""
        preds = self._compute_predictions(telemetry)

        raw_e = np.zeros(9, dtype=np.float64)
        raw_e[0:4] = np.array(telemetry.egt) - preds["egt"]
        raw_e[4] = telemetry.cht - preds["cht"]
        raw_e[5] = telemetry.p_oil - preds["p_oil"]
        raw_e[6] = telemetry.t_oil - preds["t_oil"]
        raw_e[7] = telemetry.mdot_f - preds["mdot_f"]
        raw_e[8] = self._compute_power_balance_residual(telemetry, preds["p_brake"])

        regime, stable_duration = self._regime_classifier.step(
            t=telemetry.t,
            rpm=telemetry.n_rpm,
            map_pa=telemetry.map_pa,
            cht_k=telemetry.cht,
            altitude_m=telemetry.h_p,
            p_brake_kw=preds["p_brake"] / 1000.0,
            is_valid_telemetry=telemetry.are_primary_sensors_valid(),
        )

        regime_idx = max(0, min(11, regime)) if regime < 12 else 11

        sigma = np.ones(9, dtype=np.float64)
        for i, ch in enumerate(["z_EGT_1", "z_EGT_2", "z_EGT_3", "z_EGT_4", "z_CHT", "z_p_oil", "z_t_oil", "z_mdot_f", "z_N"]):
            sigma[i] = self._noise_std.get(ch, np.ones(12))[regime_idx]

        whitened = np.zeros(9, dtype=np.float64)
        for i in range(9):
            ch = ["z_EGT_1", "z_EGT_2", "z_EGT_3", "z_EGT_4", "z_CHT", "z_p_oil", "z_t_oil", "z_mdot_f", "z_N"][i]
            a_c = self._ar_coeffs.get(ch, 0.0)
            whitened[i] = raw_e[i] - a_c * self._prev_e_whitened[i]
        self._prev_e_whitened = whitened.copy()

        z = whitened / sigma

        alpha, beta, s_norm_inf, dominant_cyl = SpatialEGTDecomposer.decompose(z[0:4])

        x_hat = np.zeros(9, dtype=np.float64)
        x_hat[0:4] = preds["egt"]
        x_hat[4] = preds["cht"]
        x_hat[5] = preds["p_oil"]
        x_hat[6] = preds["t_oil"]
        x_hat[7] = preds["mdot_f"]
        x_hat[8] = preds["p_brake"] / 1000.0

        raw_z_list = [float(v) for v in z]
        z_suppressed, status, _ = ResidualSuppressionEngine.evaluate_suppression(
            raw_z_list,
            telemetry.valid_mask,
            is_saturated=False,
        )

        return ResidualFrame(
            t=telemetry.t,
            z=tuple(z_suppressed),
            regime=regime,
            regime_stable_s=stable_duration,
            spatial=(float(alpha), float(beta), float(s_norm_inf), int(dominant_cyl)),
            sigma=tuple(float(v) for v in sigma),
            x_hat=tuple(float(v) for v in x_hat),
            status=status,
            model_version=self.config.model_version,
            regime_grid_version=self.config.regime_grid_version,
            flight_id=getattr(telemetry, 'flight_id', ''),
        )

    def _determine_regime(self, telemetry: TelemetryFrame) -> int:
        """Determine regime index based on telemetry."""
        return 4

    def predict(self, u: Any, theta: np.ndarray) -> np.ndarray:
        """Open-loop forward simulation. Used by L7's mission simulator and by L4's finite-difference sensitivity computation."""
        if isinstance(u, TelemetryFrame):
            frames = [u]
        elif hasattr(u, '__iter__'):
            frames = list(u)
        else:
            raise TypeError("u must be TelemetryFrame or sequence of TelemetryFrame")

        saved_theta = self.health_params.copy()
        saved_state = self.state.copy()
        saved_prev_e = self._prev_e_whitened.copy()
        saved_prev_t = self._prev_t

        # Reset the elapsed-time clock so predict() is deterministic and
        # side-effect free regardless of any live step() history on this
        # twin instance: the first frame of `u` always takes exactly one
        # nominal-dt step (a "fresh start"), and later frames in the same
        # sequence use their own real inter-frame deltas.
        self._prev_t = None

        self.health_params = np.array(theta, dtype=np.float64).copy()
        outputs = []
        for frame in frames:
            self._advance_state(frame)
            preds = self._compute_predictions(frame)
            y_hat = np.array([
                preds["egt"][0], preds["egt"][1], preds["egt"][2], preds["egt"][3],
                preds["cht"],
                preds["p_oil"],
                preds["t_oil"],
                preds["mdot_f"],
                preds["p_brake"] / 1000.0,
            ], dtype=np.float64)
            outputs.append(y_hat)

        self.health_params = saved_theta
        self.state = saved_state
        self._prev_e_whitened = saved_prev_e
        self._prev_t = saved_prev_t

        return np.stack(outputs) if len(outputs) > 1 else outputs[0]

    def jacobian(self, u: Any, theta: np.ndarray) -> np.ndarray:
        """d(y)/d(theta), shape (9, n_theta). Central differences, step 1e-3."""
        n_theta = len(theta)
        y0 = self.predict(u, theta)
        if y0.ndim == 1:
            y0 = y0.reshape(1, -1)
            single = True
        else:
            single = False

        J = np.zeros((y0.shape[0], 9, n_theta), dtype=np.float64)
        eps = 1e-3
        for j in range(n_theta):
            theta_plus = theta.copy()
            theta_plus[j] += eps
            theta_minus = theta.copy()
            theta_minus[j] -= eps
            y_plus = self.predict(u, theta_plus)
            y_minus = self.predict(u, theta_minus)
            if single:
                y_plus = y_plus.reshape(1, -1)
                y_minus = y_minus.reshape(1, -1)
            J[:, :, j] = (y_plus - y_minus) / (2.0 * eps)

        return J.squeeze(axis=0) if single else J
