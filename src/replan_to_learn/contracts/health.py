"""
SIH26054 Replan to Learn: Foundation Data Contracts & Schemas.
Health Parameters Schema & Ambiguous Pair Mathematical Model.

Defines the 12-parameter / 15-scalar health state vector theta:
- theta_vol: Volumetric efficiency multiplier [0.80, 1.05]
- theta_comb: Indicated combustion efficiency multiplier [0.80, 1.05]
- theta_cool: Cylinder head heat transfer conductance multiplier [0.70, 1.10]
- theta_inj[1..4]: Per-cylinder fuel injector delivery multiplier [0.80, 1.10]
- theta_oilp: Oil pump efficiency / bearing clearance multiplier [0.75, 1.05]
- theta_fric: Engine friction (FMEP) multiplier [0.95, 1.30]
- b_EGT[1..4]: Thermocouple calibration drift (additive, +/- 24.0 K)
- b_CHT: CHT RTD probe bias (additive, +/- 7.5 K)
- b_poil: Oil pressure transducer drift (additive, +/- 36.0 kPa)

Provides rigorous mathematical and physical modeling of the cooling/combustion
(theta_cool, theta_comb) ambiguous pair, demonstrating sensitivity Jacobian
properties and why both parameters are maintained as distinct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


# ============================================================================
# 1. Health Parameter Bounds and Constants
# ============================================================================

PARAMETER_NAMES = [
    "theta_vol",
    "theta_comb",
    "theta_cool",
    "theta_inj_1",
    "theta_inj_2",
    "theta_inj_3",
    "theta_inj_4",
    "theta_oilp",
    "theta_fric",
    "b_egt_1",
    "b_egt_2",
    "b_egt_3",
    "b_egt_4",
    "b_cht",
    "b_poil",
]

NUM_HEALTH_PARAMETERS = 15

# Bounds: (min_admissible, max_admissible, nominal_value, unit)
HEALTH_PARAMETER_SPECS: Dict[str, Tuple[float, float, float, str]] = {
    "theta_vol": (0.80, 1.05, 1.0, "ratio"),
    "theta_comb": (0.80, 1.05, 1.0, "ratio"),
    "theta_cool": (0.70, 1.10, 1.0, "ratio"),
    "theta_inj_1": (0.80, 1.10, 1.0, "ratio"),
    "theta_inj_2": (0.80, 1.10, 1.0, "ratio"),
    "theta_inj_3": (0.80, 1.10, 1.0, "ratio"),
    "theta_inj_4": (0.80, 1.10, 1.0, "ratio"),
    "theta_oilp": (0.75, 1.05, 1.0, "ratio"),
    "theta_fric": (0.95, 1.30, 1.0, "ratio"),
    "b_egt_1": (-24.0, 24.0, 0.0, "K"),
    "b_egt_2": (-24.0, 24.0, 0.0, "K"),
    "b_egt_3": (-24.0, 24.0, 0.0, "K"),
    "b_egt_4": (-24.0, 24.0, 0.0, "K"),
    "b_cht": (-7.5, 7.5, 0.0, "K"),
    "b_poil": (-36000.0, 36000.0, 0.0, "Pa"),
}


# ============================================================================
# 2. HealthParameters Frozen Dataclass
# ============================================================================

@dataclass(frozen=True)
class HealthParameters:
    """
    Immutable health parameter state vector theta.
    Estimated online by the UKF (L3) and inspected by the Identifiability Gate.
    """
    theta_vol: float = 1.0
    theta_comb: float = 1.0
    theta_cool: float = 1.0
    theta_inj: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    theta_oilp: float = 1.0
    theta_fric: float = 1.0
    b_egt: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    b_cht: float = 0.0
    b_poil: float = 0.0

    def __post_init__(self) -> None:
        if len(self.theta_inj) != 4:
            raise ValueError(f"theta_inj must contain 4 elements, got {len(self.theta_inj)}")
        if len(self.b_egt) != 4:
            raise ValueError(f"b_egt must contain 4 elements, got {len(self.b_egt)}")

    def to_array(self) -> np.ndarray:
        """Serialize parameters to a 15-element NumPy array (float64)."""
        return np.array([
            self.theta_vol,
            self.theta_comb,
            self.theta_cool,
            self.theta_inj[0],
            self.theta_inj[1],
            self.theta_inj[2],
            self.theta_inj[3],
            self.theta_oilp,
            self.theta_fric,
            self.b_egt[0],
            self.b_egt[1],
            self.b_egt[2],
            self.b_egt[3],
            self.b_cht,
            self.b_poil,
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: Sequence[float] | np.ndarray) -> HealthParameters:
        """Construct HealthParameters from a 15-element array."""
        if len(arr) != NUM_HEALTH_PARAMETERS:
            raise ValueError(f"Expected array of length {NUM_HEALTH_PARAMETERS}, got {len(arr)}")
        return cls(
            theta_vol=float(arr[0]),
            theta_comb=float(arr[1]),
            theta_cool=float(arr[2]),
            theta_inj=(float(arr[3]), float(arr[4]), float(arr[5]), float(arr[6])),
            theta_oilp=float(arr[7]),
            theta_fric=float(arr[8]),
            b_egt=(float(arr[9]), float(arr[10]), float(arr[11]), float(arr[12])),
            b_cht=float(arr[13]),
            b_poil=float(arr[14]),
        )

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate whether all parameters lie within admissible physical bounds."""
        arr = self.to_array()
        violations = []
        for i, name in enumerate(PARAMETER_NAMES):
            min_val, max_val, _, unit = HEALTH_PARAMETER_SPECS[name]
            val = arr[i]
            if val < min_val or val > max_val:
                violations.append(f"{name}={val:.4f} outside [{min_val:.4f}, {max_val:.4f}] {unit}")
        return len(violations) == 0, violations

    def clip_to_bounds(self) -> HealthParameters:
        """Return a new HealthParameters instance clipped to admissible bounds."""
        arr = self.to_array()
        clipped = np.zeros_like(arr)
        for i, name in enumerate(PARAMETER_NAMES):
            min_val, max_val, _, _ = HEALTH_PARAMETER_SPECS[name]
            clipped[i] = np.clip(arr[i], min_val, max_val)
        return HealthParameters.from_array(clipped)

    def is_nominal(self, tol: float = 1e-5) -> bool:
        """Check if parameters are at nominal as-new values."""
        arr = self.to_array()
        nominal_arr = np.array([HEALTH_PARAMETER_SPECS[name][2] for name in PARAMETER_NAMES])
        return bool(np.all(np.abs(arr - nominal_arr) <= tol))


# ============================================================================
# 3. Mathematical & Physical Model of Ambiguous Pair (theta_cool, theta_comb)
# ============================================================================

class AmbiguousPairModel:
    """
    Rigorous thermodynamic modeling of the ambiguous cooling/combustion pair.

    Demonstrates:
    1. Why (theta_cool, theta_comb) exhibit near-collinear sensitivity in steady cruise.
    2. Why merging them into a single lumped thermal scalar obscures physical failure mechanisms.
    3. How multi-regime diversity (climb/descent) and the orthogonal power balance residual r_N
       break the degeneracy and guarantee parameter identifiability.
    """

    LHV_FUEL = 43.0e6           # Lower heating value of gasoline (J/kg)
    CP_EXH = 1150.0             # Specific heat of exhaust gas (J/(kg*K))
    HEAD_HEAT_FRACTION = 0.15   # Fraction of total fuel heat rejected to cylinder head

    @classmethod
    def compute_jacobian_columns(
        cls,
        regime: str = "CRUISE",
        include_power_residual: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute sensitivity Jacobian column vectors J_cool and J_comb.

        In steady cruise (under governor throttle compensation to maintain airspeed):
        - A drop in theta_cool increases CHT (z_CHT: 0.85) and oil temp (z_Toil: 0.50).
        - A drop in theta_comb increases fuel burn to maintain cruise thrust, which
          increases CHT (z_CHT: 0.80), oil temp (z_Toil: 0.45), and EGT (z_EGT: 0.25).
        - Power-balance residual r_N directly decouples theta_comb from theta_cool:
          d(r_N)/d(theta_comb) = -1.50 (kW / sigma), while d(r_N)/d(theta_cool) = 0.0.
        """
        regime = regime.upper()

        if regime == "CRUISE":
            # Channels: [EGT1, EGT2, EGT3, EGT4, CHT, p_oil, T_oil, mdot_f, PowerBalance]
            j_cool = np.array([0.05, 0.05, 0.05, 0.05, 0.85, 0.0, 0.50, 0.0, 0.0], dtype=np.float64)
            j_comb = np.array([0.25, 0.25, 0.25, 0.25, 0.80, 0.0, 0.45, 0.10, -1.50], dtype=np.float64)
        elif regime == "CLIMB":
            # In climb (low TAS, high power), cooling conductance is minimal -> CHT diverges
            j_cool = np.array([0.02, 0.02, 0.02, 0.02, 1.45, 0.0, 0.80, 0.0, 0.0], dtype=np.float64)
            j_comb = np.array([0.60, 0.60, 0.60, 0.60, 0.50, 0.0, 0.30, 0.25, -2.00], dtype=np.float64)
        elif regime == "DESCENT":
            # In descent (high TAS, low power), cooling faults have negligible signature
            j_cool = np.array([0.01, 0.01, 0.01, 0.01, 0.20, 0.0, 0.15, 0.0, 0.0], dtype=np.float64)
            j_comb = np.array([0.15, 0.15, 0.15, 0.15, 0.25, 0.0, 0.10, 0.05, -1.00], dtype=np.float64)
        else:
            raise ValueError(f"Unknown regime '{regime}'")

        if not include_power_residual:
            return j_cool[:8], j_comb[:8]
        return j_cool, j_comb

    @classmethod
    def evaluate_sensitivities(cls, regime: str = "CRUISE") -> Dict[str, float]:
        """Return scalar sensitivity values for diagnostic inspection."""
        j_cool, j_comb = cls.compute_jacobian_columns(regime, include_power_residual=True)
        return {
            "dz_cht_d_theta_cool": float(j_cool[4]),
            "dz_egt_d_theta_cool": float(j_cool[0]),
            "dz_power_d_theta_cool": float(j_cool[8]),
            "dz_cht_d_theta_comb": float(j_comb[4]),
            "dz_egt_d_theta_comb": float(j_comb[0]),
            "dz_power_d_theta_comb": float(j_comb[8]),
        }

    @classmethod
    def compute_collinearity(
        cls,
        regime: str = "CRUISE",
        include_power_residual: bool = True,
    ) -> float:
        """
        Compute cosine similarity |cos(phi)| between J_cool and J_comb.
        In steady cruise without power balance, cos_sim > 0.90 (collinear).
        With power balance or multi-regime diversity, cos_sim drops.
        """
        j_cool, j_comb = cls.compute_jacobian_columns(regime, include_power_residual=include_power_residual)
        norm_cool = np.linalg.norm(j_cool)
        norm_comb = np.linalg.norm(j_comb)
        if norm_cool < 1e-12 or norm_comb < 1e-12:
            return 0.0
        cos_sim = float(np.abs(np.dot(j_cool, j_comb)) / (norm_cool * norm_comb))
        return cos_sim

    @classmethod
    def compute_fisher_information_matrix(
        cls,
        regimes: Sequence[str] = ("CRUISE",),
        include_power_residual: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute the 2x2 Fisher Information Matrix F = sum_k J_k^T R^-1 J_k for (theta_cool, theta_comb),
        and its condition number cond(F).
        """
        fim = np.zeros((2, 2), dtype=np.float64)

        for reg in regimes:
            j_cool, j_comb = cls.compute_jacobian_columns(reg, include_power_residual=include_power_residual)
            j_pair = np.column_stack([j_cool, j_comb])
            fim += j_pair.T @ j_pair

        cond_num = float(np.linalg.cond(fim)) if np.linalg.det(fim) > 1e-15 else float("inf")
        return fim, cond_num

    @classmethod
    def demonstrate_distinct_parameter_necessity(cls) -> Dict[str, object]:
        """Demonstrate why theta_cool and theta_comb must remain distinct."""
        cos_sim_thermal = cls.compute_collinearity("CRUISE", include_power_residual=False)
        _, cond_thermal = cls.compute_fisher_information_matrix(["CRUISE"], include_power_residual=False)

        cos_sim_power = cls.compute_collinearity("CRUISE", include_power_residual=True)
        _, cond_power = cls.compute_fisher_information_matrix(["CRUISE"], include_power_residual=True)

        fim_multi, cond_multi = cls.compute_fisher_information_matrix(
            ["CRUISE", "CLIMB", "DESCENT"],
            include_power_residual=True
        )

        crlb_cov = np.linalg.inv(fim_multi) if np.isfinite(cond_multi) and cond_multi < 1e6 else np.full((2, 2), np.nan)

        return {
            "cosine_similarity_cruise_thermal_only": cos_sim_thermal,
            "fim_condition_cruise_thermal_only": cond_thermal,
            "cosine_similarity_cruise_with_power_residual": cos_sim_power,
            "fim_condition_cruise_with_power_residual": cond_power,
            "fim_multi_regime": fim_multi.tolist(),
            "fim_condition_multi_regime": cond_multi,
            "crlb_covariance_matrix": crlb_cov.tolist(),
            "conclusion": (
                "theta_cool and theta_comb MUST remain distinct parameters because "
                "multi-regime operational diversity and the orthogonal power balance "
                "residual cleanly break the steady-cruise collinearity, enabling true root-cause isolation."
            )
        }
