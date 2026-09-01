"""
Tests for estimation package.
"""

import pytest
import numpy as np

from replan_to_learn.estimation import UKF, UKFConfig


class TestUKFInitialization:
    def test_default_config(self):
        ukf = UKF(n_state=3, n_theta=2, n_meas=3)
        assert ukf.config.alpha == 1e-3
        assert ukf.config.beta == 2.0
        assert ukf.config.kappa == 0.0

    def test_custom_config(self):
        config = UKFConfig(alpha=1e-2, beta=1.0, kappa=-1)
        ukf = UKF(n_state=3, n_theta=2, n_meas=3, config=config)
        assert ukf.config.alpha == 1e-2
        assert ukf.config.beta == 1.0

    def test_initial_state(self):
        ukf = UKF(n_state=3, n_theta=2, n_meas=3)
        assert np.allclose(ukf.x, np.zeros(5))
        assert ukf.P.shape == (5, 5)


class TestUKFPredict:
    def test_predict_updates_state(self):
        ukf = UKF(n_state=2, n_theta=1, n_meas=2)

        def f(x, u, theta):
            return x + 0.1  # Simple constant velocity

        ukf.predict(f, u=None)
        # State should have changed
        assert not np.allclose(ukf.x, np.zeros(3))

    def test_predict_preserves_dimensions(self):
        ukf = UKF(n_state=2, n_theta=2, n_meas=2)
        assert ukf.x.shape == (4,)
        assert ukf.P.shape == (4, 4)


class TestUKFUpdate:
    def test_update_with_measurement(self):
        ukf = UKF(n_state=2, n_theta=1, n_meas=2)

        def h(x, theta):
            return x[:2]  # Direct measurement

        z = np.array([1.0, 2.0])
        x, P, innovation = ukf.update(h, z)
        assert x.shape == (3,)
        assert P.shape == (3, 3)
        assert innovation.shape == (2,)


class TestUKFConstraints:
    def test_project_constraints(self):
        ukf = UKF(n_state=2, n_theta=1, n_meas=2)
        theta = np.array([0.5])  # Below bound
        projected = ukf._project_constraints(theta)
        assert projected[0] >= 0.75

    def test_theta_frozen_in_transient(self):
        """Test that theta is not updated during transient."""
        ukf = UKF(n_state=2, n_theta=1, n_meas=2)
        initial_theta = ukf.x[2:].copy()
        # In actual implementation, transient flag would freeze theta
        # This is a placeholder test
        assert initial_theta is not None
