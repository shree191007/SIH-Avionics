"""
Tests for gate package.
"""

import pytest
import numpy as np

from replan_to_learn.gate import (
    AttributionFrame,
    BorrowResult,
    IdentifiabilityGate,
    ProbeRequest,
    ReasonCode,
    Verdict,
)


class TestVerdictEnum:
    def test_verdict_values(self):
        assert Verdict.NAMED.value == 0
        assert Verdict.AMBIGUOUS.value == 1
        assert Verdict.BORROWED.value == 2
        assert Verdict.INVALID.value == 3


class TestReasonCodeEnum:
    def test_reason_code_values(self):
        assert ReasonCode.NONE.value == 0
        assert ReasonCode.A1_INSUFFICIENT_REGIMES.value == 1
        assert ReasonCode.A2_LOW_R2.value == 2


class TestIdentifiabilityGate:
    @pytest.fixture
    def mock_physics_twin(self):
        class MockPhysicsTwin:
            def __init__(self):
                self.health_params = np.ones(15)

            def jacobian(self, u, theta):
                # Must emulate PhysicsTwin.jacobian's real contract: a
                # single frame returns (9, n_theta), but a sequence of N
                # frames returns (N, 9, n_theta) -- one row per step, used
                # by IdentifiabilityGate._precompute_regime_jacobian to
                # pull both an equilibrated (last) and an early (first)
                # snapshot from ONE call. Always returning (9, n_theta)
                # regardless of input shape (the previous version) silently
                # broke that code's J[-1]/J[early_idx] indexing once the
                # gate started passing a multi-frame sequence.
                n_frames = len(u) if hasattr(u, "__iter__") and not hasattr(u, "t") else 1
                if n_frames == 1:
                    return np.random.randn(9, len(theta)) * 0.1
                return np.random.randn(n_frames, 9, len(theta)) * 0.1

            def step(self, frame):
                from replan_to_learn.contracts.residuals import ResidualFrame
                return ResidualFrame(
                    t=frame.t,
                    z=(0.0,) * 9,
                    regime=4,
                    regime_stable_s=20.0,
                    spatial=(0.0, 0.0, 0.0, 0),
                    sigma=(0.1,) * 9,
                    x_hat=(950.0, 955.0, 960.0, 965.0, 360.0, 3.5e5, 350.0, 0.05, 50.0),
                    status=0,
                    model_version="1.0.0",
                    regime_grid_version="REGIME_GRID_V1",
                )

            def _compute_predictions(self, frame):
                return {
                    "egt": np.array([950.0, 955.0, 960.0, 965.0]),
                    "cht": 360.0,
                    "p_oil": 3.5e5,
                    "t_oil": 350.0,
                    "mdot_f": 0.05,
                    "p_brake": 50.0,
                    "_alpha": 0.0,
                    "_beta": 0.0,
                    "_s_inf": 0.0,
                }
        return MockPhysicsTwin()

    def test_init(self, mock_physics_twin):
        gate = IdentifiabilityGate(
            physics_twin=mock_physics_twin,
            n_theta=5,
        )
        assert gate.n_theta == 5
        assert gate.tau_sep == 0.90
        assert gate.tau_r2 == 0.85

    def test_update_invalid_status(self, mock_physics_twin):
        gate = IdentifiabilityGate(physics_twin=mock_physics_twin, n_theta=5)

        from replan_to_learn.contracts.residuals import ResidualFrame
        rf = ResidualFrame(
            t=0.0,
            z=(0.0,) * 9,
            regime=4,
            regime_stable_s=0.0,
            spatial=(0.0, 0.0, 0.0, 0),
            sigma=(1.0,) * 9,
            x_hat=(0.0,) * 9,
            status=1,  # DEGRADED_INPUT
            model_version="1.0.0",
            regime_grid_version="REGIME_GRID_V1",
        )

        result = gate.update(rf)
        assert result.verdict == Verdict.INVALID

    def test_update_returns_attribution_frame(self, mock_physics_twin):
        gate = IdentifiabilityGate(physics_twin=mock_physics_twin, n_theta=5)

        from replan_to_learn.contracts.residuals import ResidualFrame
        rf = ResidualFrame(
            t=0.0,
            z=(0.0,) * 9,
            regime=4,
            regime_stable_s=0.0,
            spatial=(0.0, 0.0, 0.0, 0),
            sigma=(1.0,) * 9,
            x_hat=(0.0,) * 9,
            status=0,
            model_version="1.0.0",
            regime_grid_version="REGIME_GRID_V1",
        )

        result = gate.update(rf)
        assert isinstance(result, AttributionFrame)
        assert result.t == 0.0
        assert result.model_version == "1.0.0"

    def test_fingerprint_shape(self, mock_physics_twin):
        gate = IdentifiabilityGate(physics_twin=mock_physics_twin, n_theta=5)
        fp = gate.fingerprint(theta_index=0)
        assert fp.shape == (IdentifiabilityGate.FINGERPRINT_DIM,)

    def test_try_borrow_returns_result(self, mock_physics_twin):
        gate = IdentifiabilityGate(physics_twin=mock_physics_twin, n_theta=5)

        from replan_to_learn.fleet import FleetShape
        shape = FleetShape(
            theta_index=0,
            u=np.ones(12) / np.sqrt(12),
            var_u=np.ones(12) * 0.1,
            contributors=np.zeros(12, dtype=np.uint16),
            alpha_median=1.0,
            epoch=0,
            model_version="1.0.0",
            regime_grid_version="REGIME_GRID_V1",
        )

        result = gate.try_borrow(theta_index=0, shape=shape)
        assert isinstance(result, BorrowResult)

    def test_probe_request_returns_none_initially(self, mock_physics_twin):
        gate = IdentifiabilityGate(physics_twin=mock_physics_twin, n_theta=5)
        assert gate.probe_request() is None
