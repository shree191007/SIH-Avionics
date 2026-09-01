"""
Unit tests for replan_to_learn.contracts.health
Verifies health parameter state vector theta, bounds validation/clipping,
and the rigorous mathematical modeling of the ambiguous cooling/combustion pair.
"""

import math
import pytest
import numpy as np

from replan_to_learn.contracts.health import (
    HEALTH_PARAMETER_SPECS,
    NUM_HEALTH_PARAMETERS,
    PARAMETER_NAMES,
    AmbiguousPairModel,
    HealthParameters,
)


class TestHealthParametersContract:
    """Test HealthParameters frozen dataclass and state vector conversions."""

    def test_nominal_health_parameters(self, nominal_health_parameters: HealthParameters) -> None:
        hp = nominal_health_parameters
        assert hp.theta_vol == 1.0
        assert hp.theta_comb == 1.0
        assert hp.theta_cool == 1.0
        assert hp.theta_inj == (1.0, 1.0, 1.0, 1.0)
        assert hp.theta_oilp == 1.0
        assert hp.theta_fric == 1.0
        assert hp.b_egt == (0.0, 0.0, 0.0, 0.0)
        assert hp.b_cht == 0.0
        assert hp.b_poil == 0.0
        assert hp.is_nominal() is True

    def test_to_and_from_array_roundtrip(self) -> None:
        hp = HealthParameters(
            theta_vol=0.95,
            theta_comb=0.92,
            theta_cool=0.85,
            theta_inj=(0.98, 1.02, 1.00, 0.99),
            theta_oilp=0.90,
            theta_fric=1.05,
            b_egt=(5.0, -3.0, 2.0, -1.0),
            b_cht=2.5,
            b_poil=-5000.0,
        )
        arr = hp.to_array()
        assert len(arr) == NUM_HEALTH_PARAMETERS
        recovered = HealthParameters.from_array(arr)
        assert recovered.theta_vol == pytest.approx(0.95)
        assert recovered.theta_comb == pytest.approx(0.92)
        assert recovered.theta_cool == pytest.approx(0.85)
        assert recovered.theta_inj == (pytest.approx(0.98), pytest.approx(1.02), pytest.approx(1.00), pytest.approx(0.99))
        assert recovered.b_cht == pytest.approx(2.5)
        assert recovered.b_poil == pytest.approx(-5000.0)

    def test_health_bounds_validation(self) -> None:
        # Valid parameters
        hp_valid = HealthParameters(theta_vol=0.95, theta_comb=0.90, theta_cool=0.80)
        is_valid, violations = hp_valid.validate()
        assert is_valid is True
        assert len(violations) == 0

        # Out of bounds parameters (e.g. theta_vol = 0.50 below min 0.80)
        hp_invalid = HealthParameters(theta_vol=0.50, theta_fric=2.0)
        is_valid, violations = hp_invalid.validate()
        assert is_valid is False
        assert len(violations) >= 2

    def test_clip_to_bounds(self) -> None:
        hp_extreme = HealthParameters(theta_vol=0.40, theta_comb=1.50, theta_cool=0.20)
        hp_clipped = hp_extreme.clip_to_bounds()
        assert hp_clipped.theta_vol == pytest.approx(0.80)   # Clipped to min
        assert hp_clipped.theta_comb == pytest.approx(1.05)  # Clipped to max
        assert hp_clipped.theta_cool == pytest.approx(0.70)  # Clipped to min
        is_valid, _ = hp_clipped.validate()
        assert is_valid is True


class TestAmbiguousPairModel:
    """Test thermodynamic sensitivity modeling of (theta_cool, theta_comb)."""

    def test_steady_cruise_collinearity_without_power_residual(self) -> None:
        """In cruise without power balance, thermal sensitivities exhibit high collinearity."""
        cos_sim = AmbiguousPairModel.compute_collinearity("CRUISE", include_power_residual=False)
        assert cos_sim > 0.70, f"Expected high collinearity in cruise thermal channels, got {cos_sim}"

    def test_power_balance_residual_decouples_combustion_from_cooling(self) -> None:
        """Power balance residual r_N has zero sensitivity to cooling, breaking collinearity."""
        sens = AmbiguousPairModel.evaluate_sensitivities("CRUISE")
        # Cooling does not affect shaft power
        assert sens["dz_power_d_theta_cool"] == 0.0
        # Combustion directly affects shaft power (non-zero sensitivity)
        assert abs(sens["dz_power_d_theta_comb"]) > 0.0

        # Including power balance drops the collinearity from 0.92 down to ~0.52
        cos_sim_with_rn = AmbiguousPairModel.compute_collinearity("CRUISE", include_power_residual=True)
        assert cos_sim_with_rn < 0.60

    def test_multi_regime_fisher_information_matrix(self) -> None:
        """Multi-regime diversity (Climb + Cruise + Descent) ensures well-conditioned FIM."""
        # Single regime cruise (thermal only) has ill-conditioned FIM
        fim_single, cond_single = AmbiguousPairModel.compute_fisher_information_matrix(["CRUISE"], include_power_residual=False)

        # Multi regime with power residual has well-conditioned FIM
        fim_multi, cond_multi = AmbiguousPairModel.compute_fisher_information_matrix(
            ["CRUISE", "CLIMB", "DESCENT"],
            include_power_residual=True,
        )

        assert cond_multi < cond_single
        assert cond_multi < 50.0  # Well-conditioned FIM enables unique identification of theta_cool and theta_comb

    def test_demonstrate_distinct_parameter_necessity(self) -> None:
        demo = AmbiguousPairModel.demonstrate_distinct_parameter_necessity()
        assert "fim_condition_multi_regime" in demo
        assert "conclusion" in demo
        assert demo["fim_condition_multi_regime"] < 100.0
