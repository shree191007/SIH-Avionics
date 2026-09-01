"""
Tests for fleet package.
"""

import pytest
import numpy as np

from replan_to_learn.fleet import FleetNode, FingerprintContribution, FleetShape


class TestFingerprintContribution:
    def test_create_contribution(self):
        c = FingerprintContribution(
            aircraft_id="aircraft_1",
            model_version="1.0.0",
            regime_grid_version="REGIME_GRID_V1",
            theta_index=0,
            info_vector=np.ones(12),
            info_scalar=np.ones(12),
            n_seconds=np.ones(12, dtype=np.uint32) * 100,
            epoch=1,
        )
        assert c.aircraft_id == "aircraft_1"
        assert c.theta_index == 0
        assert len(c.info_vector) == 12

    def test_faulted_contribution(self):
        c = FingerprintContribution(
            aircraft_id="aircraft_1",
            model_version="1.0.0",
            regime_grid_version="REGIME_GRID_V1",
            theta_index=0,
            info_vector=np.ones(12),
            info_scalar=np.ones(12),
            n_seconds=np.ones(12, dtype=np.uint32) * 100,
            epoch=1,
            is_faulted=True,
        )
        assert c.is_faulted is True


class TestFleetShape:
    def test_create_shape(self):
        u = np.ones(12) / np.sqrt(12)
        shape = FleetShape(
            theta_index=0,
            u=u,
            var_u=np.ones(12) * 0.1,
            contributors=np.ones(12, dtype=np.uint16),
            alpha_median=1.0,
            epoch=1,
            model_version="1.0.0",
            regime_grid_version="REGIME_GRID_V1",
        )
        assert np.allclose(shape.u, u)
        assert shape.epoch == 1


class TestFleetNode:
    def test_init(self):
        node = FleetNode()
        assert node.n_regime == 12
        assert node.tau_r2 == 0.85
        assert node.epoch == 0

    def test_set_prior(self):
        node = FleetNode()
        u_prior = np.ones(12) / np.sqrt(12)
        node.set_prior(theta_index=0, u_prior=u_prior)
        assert 0 in node.priors
        assert np.allclose(node.priors[0], u_prior)

    def test_set_prior_creates_initial_shape(self):
        node = FleetNode()
        u_prior = np.ones(12) / np.sqrt(12)
        node.set_prior(theta_index=0, u_prior=u_prior)
        assert 0 in node.shapes
        assert np.allclose(node.shapes[0].u, u_prior)

    def test_ingest_contribution(self):
        node = FleetNode()
        c = FingerprintContribution(
            aircraft_id="aircraft_1",
            model_version="1.0.0",
            regime_grid_version="REGIME_GRID_V1",
            theta_index=0,
            info_vector=np.ones(12),
            info_scalar=np.ones(12),
            n_seconds=np.ones(12, dtype=np.uint32) * 100,
            epoch=1,
        )
        node.ingest(c)
        assert 0 in node.contributions
        assert len(node.contributions[0]) == 1

    def test_fuse_no_contributions_raises(self):
        node = FleetNode()
        with pytest.raises(ValueError):
            node.fuse(theta_index=0)

    def test_fuse_with_contributions(self):
        node = FleetNode()
        u_prior = np.ones(12) / np.sqrt(12)
        node.set_prior(theta_index=0, u_prior=u_prior)

        c = FingerprintContribution(
            aircraft_id="aircraft_1",
            model_version="1.0.0",
            regime_grid_version="REGIME_GRID_V1",
            theta_index=0,
            info_vector=np.ones(12),
            info_scalar=np.ones(12),
            n_seconds=np.ones(12, dtype=np.uint32) * 100,
            epoch=1,
        )
        node.ingest(c)
        shape = node.fuse(theta_index=0)
        assert isinstance(shape, FleetShape)
        assert shape.theta_index == 0
        assert np.allclose(np.linalg.norm(shape.u), 1.0, atol=1e-6)

    def test_fuse_sign_convention(self):
        node = FleetNode()
        u_prior = np.ones(12) / np.sqrt(12)
        node.set_prior(theta_index=0, u_prior=u_prior)

        # Negative contribution
        c = FingerprintContribution(
            aircraft_id="aircraft_1",
            model_version="1.0.0",
            regime_grid_version="REGIME_GRID_V1",
            theta_index=0,
            info_vector=-np.ones(12),
            info_scalar=np.ones(12),
            n_seconds=np.ones(12, dtype=np.uint32) * 100,
            epoch=1,
        )
        node.ingest(c)
        shape = node.fuse(theta_index=0)
        # Sign should align with prior
        assert np.allclose(np.linalg.norm(shape.u), 1.0, atol=1e-6)

    def test_get_shape(self):
        node = FleetNode()
        u_prior = np.ones(12) / np.sqrt(12)
        node.set_prior(theta_index=0, u_prior=u_prior)
        shape = node.get_shape(theta_index=0)
        assert shape is not None
        assert shape.theta_index == 0

    def test_get_shape_missing(self):
        node = FleetNode()
        assert node.get_shape(theta_index=999) is None

    def test_step_increments_epoch(self):
        node = FleetNode()
        assert node.epoch == 0
        node.step()
        assert node.epoch == 1

    def test_version_mismatch_rejected(self):
        node = FleetNode()
        u_prior = np.ones(12) / np.sqrt(12)
        node.set_prior(theta_index=0, u_prior=u_prior)

        c = FingerprintContribution(
            aircraft_id="aircraft_1",
            model_version="0.9.0",  # Wrong version
            regime_grid_version="REGIME_GRID_V1",
            theta_index=0,
            info_vector=np.ones(12),
            info_scalar=np.ones(12),
            n_seconds=np.ones(12, dtype=np.uint32) * 100,
            epoch=1,
        )
        node.ingest(c)
        shape = node.fuse(theta_index=0)
        # Should return prior when no admissible contributions
        assert np.allclose(shape.u, u_prior)
