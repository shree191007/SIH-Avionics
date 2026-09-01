"""
Tests for L6 RUL Particle Filter.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from replan_to_learn.rul.datatypes import FailureThreshold, RULConfig, RULReport
from replan_to_learn.rul.particle_filter import RULParticleFilter


class TestRULParticleFilter:
    def test_initialize(self) -> None:
        pf = RULParticleFilter(theta_index=1, theta_name="theta_comb")
        pf.initialize(theta_init=0.90, P_theta_jj=0.01, seed=42)
        assert len(pf.particles) == 500
        assert math.isclose(pf.theta_hat, 0.90, abs_tol=0.05)

    def test_freeze_policy(self) -> None:
        pf = RULParticleFilter(theta_index=1, theta_name="theta_comb")
        pf.initialize(theta_init=0.90, P_theta_jj=0.01, seed=42)
        pf.step(theta_obs=0.89, P_obs=0.01, dt_fh=1.0, verdict="NAMED")
        pf.freeze()
        before = pf.theta_hat
        pf.step(theta_obs=0.50, P_obs=0.01, dt_fh=10.0, verdict="AMBIGUOUS")
        assert math.isclose(pf.theta_hat, before, abs_tol=1e-3)

    def test_rul_estimate(self) -> None:
        pf = RULParticleFilter(
            theta_index=1,
            theta_name="theta_comb",
            threshold=FailureThreshold(theta_index=1, theta_name="theta_comb", threshold_value=0.80, limit_description="test", operating_point={}),
        )
        pf.initialize(theta_init=0.90, P_theta_jj=0.01, seed=42)
        for _ in range(10):
            pf.step(theta_obs=0.89, P_obs=0.01, dt_fh=1.0, verdict="NAMED")
        pf.set_mission_end(100.0)
        report = pf.estimate_rul()
        assert report.particles_converged
        assert report.theta_index == 1
        assert report.threshold == 0.80
        assert report.current_theta > 0.0

    def test_rul_no_threshold_raises(self) -> None:
        pf = RULParticleFilter(theta_index=1, theta_name="theta_comb")
        pf.initialize(theta_init=0.90, P_theta_jj=0.01, seed=42)
        with pytest.raises(RuntimeError):
            pf.estimate_rul()
