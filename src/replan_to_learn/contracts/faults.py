"""
SIH26054 Replan to Learn: Foundation Data Contracts & Schemas.
Fault Injection Harness & Spatial EGT Basis Decomposition.

Implements:
1. Injected-fault specification and parameter degradation engine supporting:
   - Non-uniform cooling degradation baseline (w >= 0.35)
   - Indicated combustion efficiency degradation
   - Per-cylinder fuel injector clogging
   - Sensor bias drift (EGT, CHT, p_oil)
   - Oil pump degradation & friction increase
2. Spatial EGT orthogonal basis projection:
   z_EGT = alpha_t * 1 + beta_t * g + s_t
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import math
import numpy as np

from replan_to_learn.contracts.health import HealthParameters


# ============================================================================
# 1. Spatial Basis Definition for 4-Cylinder Engine
# ============================================================================

# Unit-norm common-mode basis vector (1): len 4, ||1||_2 = 1.0
BASIS_COMMON_MODE = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)

# Fore-to-aft cylinder cooling gradient vector (g):
# Physical cylinder fore-aft positions: Cyl 1 & 2 are front, Cyl 3 & 4 are rear.
# Raw gradient: [1.0, 0.5, -0.5, -1.0]. Norm = sqrt(1^2 + 0.5^2 + (-0.5)^2 + (-1)^2) = sqrt(2.5) = sqrt(5/2)
_RAW_G = np.array([1.0, 0.5, -0.5, -1.0], dtype=np.float64)
BASIS_GRADIENT = _RAW_G / np.linalg.norm(_RAW_G)  # Unit norm, orthogonal to BASIS_COMMON_MODE

# Baseline aerodynamic cylinder asymmetry factors [FIT on healthy flight data]
NOMINAL_CYL_AERODYNAMIC_ASYMMETRY = np.array([1.04, 1.02, 0.97, 0.97], dtype=np.float64)


# ============================================================================
# 2. FaultSpec Frozen Dataclass
# ============================================================================

@dataclass(frozen=True)
class FaultSpec:
    """
    Fault injection specification for benchmarks and synthetic degradation validation.
    """
    fault_type: str                  # 'cooling_degradation', 'combustion_degradation', 'injector_clogging',
                                     # 'sensor_bias_egt', 'sensor_bias_cht', 'sensor_bias_poil',
                                     # 'oil_pump_wear', 'friction_increase'
    target_cylinder: int = 0         # 0 for global/engine-wide, 1..4 for specific cylinder
    magnitude: float = 0.15          # Severity (e.g. 0.15 = 15% degradation or absolute shift)
    onset_time_s: float = 0.0        # Flight time at fault inception (s)
    ramp_duration_s: float = 60.0    # Ramp time from 0 to full magnitude (s)
    spatial_asymmetry_weight: float = 0.40  # Non-uniform spatial weighting factor (w >= 0.35)

    def __post_init__(self) -> None:
        if self.target_cylinder not in (0, 1, 2, 3, 4):
            raise ValueError(f"target_cylinder must be in 0..4, got {self.target_cylinder}")
        if self.fault_type == "cooling_degradation" and self.spatial_asymmetry_weight < 0.35:
            # Enforce non-uniform cooling degradation baseline
            raise ValueError(
                f"Cooling degradation requires spatial_asymmetry_weight w >= 0.35 to preserve "
                f"physical non-uniformity, got {self.spatial_asymmetry_weight}"
            )

    def evaluate_severity_factor(self, t: float) -> float:
        """
        Evaluate time-dependent ramp multiplier a(t) in [0.0, magnitude].
        """
        if t < self.onset_time_s:
            return 0.0
        if self.ramp_duration_s <= 1e-6:
            return float(self.magnitude)
        progress = min(1.0, max(0.0, (t - self.onset_time_s) / self.ramp_duration_s))
        return float(self.magnitude * progress)


# ============================================================================
# 3. Fault Injection Engine
# ============================================================================

class FaultInjectionHarness:
    """
    Programmatic Fault Injection Engine.

    Modifies nominal health parameters theta_nom(t) -> theta_degraded(t) according
    to physical degradation models.
    """

    @classmethod
    def inject(
        cls,
        theta_nom: HealthParameters,
        fault: FaultSpec,
        t: float,
    ) -> HealthParameters:
        """
        Apply fault specification to HealthParameters at flight time t.
        """
        severity = fault.evaluate_severity_factor(t)
        if severity == 0.0:
            return theta_nom

        f_type = fault.fault_type.lower()

        if f_type == "cooling_degradation":
            # Non-uniform cooling degradation:
            # theta_cool drops, while spatial asymmetry w creates cylinder-to-cylinder spread
            new_theta_cool = max(0.50, theta_nom.theta_cool - severity)
            # Injector / combustion remains nominal
            return HealthParameters(
                theta_vol=theta_nom.theta_vol,
                theta_comb=theta_nom.theta_comb,
                theta_cool=new_theta_cool,
                theta_inj=theta_nom.theta_inj,
                theta_oilp=theta_nom.theta_oilp,
                theta_fric=theta_nom.theta_fric,
                b_egt=theta_nom.b_egt,
                b_cht=theta_nom.b_cht,
                b_poil=theta_nom.b_poil,
            ).clip_to_bounds()

        elif f_type == "combustion_degradation":
            new_theta_comb = max(0.50, theta_nom.theta_comb - severity)
            return HealthParameters(
                theta_vol=theta_nom.theta_vol,
                theta_comb=new_theta_comb,
                theta_cool=theta_nom.theta_cool,
                theta_inj=theta_nom.theta_inj,
                theta_oilp=theta_nom.theta_oilp,
                theta_fric=theta_nom.theta_fric,
                b_egt=theta_nom.b_egt,
                b_cht=theta_nom.b_cht,
                b_poil=theta_nom.b_poil,
            ).clip_to_bounds()

        elif f_type == "injector_clogging":
            inj_list = list(theta_nom.theta_inj)
            if fault.target_cylinder == 0:
                # All cylinders
                for i in range(4):
                    inj_list[i] = max(0.50, inj_list[i] - severity)
            else:
                idx = fault.target_cylinder - 1
                inj_list[idx] = max(0.50, inj_list[idx] - severity)
            return HealthParameters(
                theta_vol=theta_nom.theta_vol,
                theta_comb=theta_nom.theta_comb,
                theta_cool=theta_nom.theta_cool,
                theta_inj=(inj_list[0], inj_list[1], inj_list[2], inj_list[3]),
                theta_oilp=theta_nom.theta_oilp,
                theta_fric=theta_nom.theta_fric,
                b_egt=theta_nom.b_egt,
                b_cht=theta_nom.b_cht,
                b_poil=theta_nom.b_poil,
            ).clip_to_bounds()

        elif f_type in ("sensor_bias_egt", "sensor_bias"):
            b_egt_list = list(theta_nom.b_egt)
            if fault.target_cylinder == 0:
                for i in range(4):
                    b_egt_list[i] += severity
            else:
                idx = fault.target_cylinder - 1
                b_egt_list[idx] += severity
            return HealthParameters(
                theta_vol=theta_nom.theta_vol,
                theta_comb=theta_nom.theta_comb,
                theta_cool=theta_nom.theta_cool,
                theta_inj=theta_nom.theta_inj,
                theta_oilp=theta_nom.theta_oilp,
                theta_fric=theta_nom.theta_fric,
                b_egt=(b_egt_list[0], b_egt_list[1], b_egt_list[2], b_egt_list[3]),
                b_cht=theta_nom.b_cht,
                b_poil=theta_nom.b_poil,
            ).clip_to_bounds()

        elif f_type == "sensor_bias_cht":
            return HealthParameters(
                theta_vol=theta_nom.theta_vol,
                theta_comb=theta_nom.theta_comb,
                theta_cool=theta_nom.theta_cool,
                theta_inj=theta_nom.theta_inj,
                theta_oilp=theta_nom.theta_oilp,
                theta_fric=theta_nom.theta_fric,
                b_egt=theta_nom.b_egt,
                b_cht=theta_nom.b_cht + severity,
                b_poil=theta_nom.b_poil,
            ).clip_to_bounds()

        elif f_type == "sensor_bias_poil":
            return HealthParameters(
                theta_vol=theta_nom.theta_vol,
                theta_comb=theta_nom.theta_comb,
                theta_cool=theta_nom.theta_cool,
                theta_inj=theta_nom.theta_inj,
                theta_oilp=theta_nom.theta_oilp,
                theta_fric=theta_nom.theta_fric,
                b_egt=theta_nom.b_egt,
                b_cht=theta_nom.b_cht,
                b_poil=theta_nom.b_poil + severity,
            ).clip_to_bounds()

        elif f_type == "oil_pump_wear":
            new_oilp = max(0.50, theta_nom.theta_oilp - severity)
            return HealthParameters(
                theta_vol=theta_nom.theta_vol,
                theta_comb=theta_nom.theta_comb,
                theta_cool=theta_nom.theta_cool,
                theta_inj=theta_nom.theta_inj,
                theta_oilp=new_oilp,
                theta_fric=theta_nom.theta_fric,
                b_egt=theta_nom.b_egt,
                b_cht=theta_nom.b_cht,
                b_poil=theta_nom.b_poil,
            ).clip_to_bounds()

        elif f_type == "friction_increase":
            new_fric = min(1.80, theta_nom.theta_fric + severity)
            return HealthParameters(
                theta_vol=theta_nom.theta_vol,
                theta_comb=theta_nom.theta_comb,
                theta_cool=theta_nom.theta_cool,
                theta_inj=theta_nom.theta_inj,
                theta_oilp=theta_nom.theta_oilp,
                theta_fric=new_fric,
                b_egt=theta_nom.b_egt,
                b_cht=theta_nom.b_cht,
                b_poil=theta_nom.b_poil,
            ).clip_to_bounds()

        else:
            raise ValueError(f"Unsupported fault type '{fault.fault_type}'")

    @classmethod
    def compute_non_uniform_cooling_factors(
        cls,
        fault: FaultSpec,
        t: float,
    ) -> np.ndarray:
        """
        Compute per-cylinder cooling conductance multipliers UA_c,i(t)
        demonstrating non-uniform spatial degradation.

        Returns array of shape (4,) containing per-cylinder conductance factors.
        """
        severity = fault.evaluate_severity_factor(t)
        w = fault.spatial_asymmetry_weight  # w >= 0.35
        g_fore_aft = _RAW_G  # [1.0, 0.5, -0.5, -1.0]

        # Per-cylinder cooling delta: Delta_{cool, i} = a(t) * [1.0 + w * (g_i - g_bar)]
        delta_cool = severity * (1.0 + w * g_fore_aft)

        # Multiplied against baseline aerodynamic asymmetry factors
        ua_factors = NOMINAL_CYL_AERODYNAMIC_ASYMMETRY * (1.0 - delta_cool)
        return np.maximum(0.10, ua_factors)


# ============================================================================
# 4. Spatial EGT Basis Decomposition Engine
# ============================================================================

class SpatialEGTDecomposer:
    """
    Orthogonal Spatial Decomposition of 4-Cylinder EGT Residuals.

    Decomposes z_EGT into:
    z_EGT = alpha_t * 1 + beta_t * g + s_t

    where:
    - 1: Unit common-mode basis vector ([0.5, 0.5, 0.5, 0.5])
    - g: Unit fore-aft cooling gradient vector
    - s_t: Residual sparse component (s_t orthogonal to span{1, g})

    Emits: (alpha_t, beta_t, ||s_t||_inf, dominant_cylinder)
    """

    @classmethod
    def decompose(
        cls,
        z_egt: Sequence[float] | np.ndarray,
    ) -> Tuple[float, float, float, int]:
        """
        Project 4-cylinder EGT residual vector onto orthogonal basis.

        Args:
            z_egt: Array/sequence of 4 EGT residuals.

        Returns:
            Tuple of:
                - alpha_t: Common-mode magnitude
                - beta_t: Spatial gradient projection
                - norm_inf_s: Max absolute localized anomaly ||s_t||_inf
                - dominant_cylinder: 1-indexed cylinder with max localized deviation (1..4)
        """
        if len(z_egt) != 4:
            raise ValueError(f"Expected 4 EGT residuals, got {len(z_egt)}")

        z_arr = np.array(z_egt, dtype=np.float64)

        # Check for missing/NaN values
        if np.any(np.isnan(z_arr)):
            valid_mask = ~np.isnan(z_arr)
            if np.sum(valid_mask) == 0:
                return (0.0, 0.0, 0.0, 0)
            # Partial graceful projection
            mean_val = float(np.mean(z_arr[valid_mask]))
            return (mean_val, 0.0, 0.0, 0)

        # 1. Project onto common-mode 1
        alpha_t = float(np.dot(z_arr, BASIS_COMMON_MODE))

        # 2. Project onto gradient g
        beta_t = float(np.dot(z_arr, BASIS_GRADIENT))

        # 3. Compute sparse residual s_t
        s_t = z_arr - (alpha_t * BASIS_COMMON_MODE + beta_t * BASIS_GRADIENT)

        # 4. Localized infinity norm and dominant cylinder
        abs_s = np.abs(s_t)
        norm_inf_s = float(np.max(abs_s))
        dominant_cylinder = int(np.argmax(abs_s) + 1)  # 1-indexed (1..4)

        return (alpha_t, beta_t, norm_inf_s, dominant_cylinder)

    @classmethod
    def verify_orthogonality(cls) -> Dict[str, float]:
        """
        Verify mathematical properties of the spatial basis:
        ||1|| = 1, ||g|| = 1, 1^T g = 0.
        """
        norm_1 = float(np.linalg.norm(BASIS_COMMON_MODE))
        norm_g = float(np.linalg.norm(BASIS_GRADIENT))
        dot_1_g = float(np.dot(BASIS_COMMON_MODE, BASIS_GRADIENT))
        return {
            "norm_common_mode": norm_1,
            "norm_gradient": norm_g,
            "dot_common_gradient": dot_1_g,
            "is_orthonormal": (abs(norm_1 - 1.0) < 1e-12 and abs(norm_g - 1.0) < 1e-12 and abs(dot_1_g) < 1e-12)
        }
