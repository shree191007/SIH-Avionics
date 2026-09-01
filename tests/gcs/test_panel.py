"""
Tests for GCS Operator Panel.
"""

from __future__ import annotations

import numpy as np
import pytest

from replan_to_learn.gcs.datatypes import (
    DiagnosisPanel,
    EngineHealthSummary,
    FleetPanel,
    GCSPanel,
    MissionPanel,
    PrognosticsPanel,
)
from replan_to_learn.gcs.panel import GCSPanelBuilder


class TestGCSPanelBuilder:
    def test_build_engine_health_low(self) -> None:
        builder = GCSPanelBuilder()
        z = np.array([4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        health = builder.build_engine_health(
            z=z,
            egt=(1000.0, 960.0, 960.0, 960.0),
            cht=400.0,
            p_oil=3.5e5,
            t_oil=350.0,
            n_rpm=2800.0,
            mdot_f=0.05,
        )
        assert health.status in ("DEGRADING", "ACTION REQUIRED")

    def test_build_diagnosis_ambiguous(self) -> None:
        builder = GCSPanelBuilder()
        diagnosis = builder.build_diagnosis(
            verdict="AMBIGUOUS",
            named=None,
            ambiguous_set=(1, 2),
            confidence=0.0,
            separable_set=(),
            cos_matrix=np.array([[1.0, 0.97], [0.97, 1.0]]),
            contributing_channels=(),
            attribution_basis="LOCAL",
        )
        assert "AMBIGUOUS" in diagnosis.verdict
        assert diagnosis.ambiguous_set == (1, 2)

    def test_build_diagnosis_named(self) -> None:
        builder = GCSPanelBuilder()
        diagnosis = builder.build_diagnosis(
            verdict="NAMED",
            named=1,
            ambiguous_set=(),
            confidence=0.95,
            separable_set=("theta_comb",),
            cos_matrix=None,
            contributing_channels=(4,),
            attribution_basis="LOCAL",
        )
        assert diagnosis.verdict == "NAMED"

    def test_build_prognostics(self) -> None:
        from replan_to_learn.rul.datatypes import RULReport
        builder = GCSPanelBuilder()
        report = RULReport(
            p05=1.0, p50=2.0, p95=3.0,
            p_fail_before_mission_end=0.1,
            mission_end_time_s=3600.0,
            theta_index=1,
            theta_name="theta_comb",
            current_theta=0.90,
            threshold=0.80,
            particles_converged=True,
        )
        prog = builder.build_prognostics(report)
        assert prog.rul_p50 == 2.0
        assert prog.p_fail_before_mission == 0.1

    def test_build_mission(self) -> None:
        builder = GCSPanelBuilder()
        from replan_to_learn.planner.datatypes import MissionSimulationResult
        sim = MissionSimulationResult(
            risk="LOW",
            risk_probability=0.05,
            recommended_action="SAFE: proceed as planned",
            alternatives=[],
            margin_warnings=(),
        )
        mission = builder.build_mission(sim)
        assert mission.risk == "LOW"

    def test_build_fleet(self) -> None:
        builder = GCSPanelBuilder()
        fleet = builder.build_fleet(
            fleet_epoch=5,
            contributor_count=3,
            borrowed_regimes=(0, 1),
            r2=0.94,
            transfer_status="FLEET",
        )
        assert fleet.fleet_epoch == 5
        assert fleet.r2 == 0.94

    def test_render_text(self) -> None:
        builder = GCSPanelBuilder()
        health = builder.build_engine_health(
            z=np.zeros(9, dtype=np.float32),
            egt=(950.0, 955.0, 960.0, 965.0),
            cht=360.0,
            p_oil=3.5e5,
            t_oil=350.0,
            n_rpm=2800.0,
            mdot_f=0.05,
        )
        diagnosis = builder.build_diagnosis(
            verdict="NAMED", named=1, ambiguous_set=(),
            confidence=0.95, separable_set=("theta_comb",),
            cos_matrix=None, contributing_channels=(),
            attribution_basis="LOCAL",
        )
        from replan_to_learn.rul.datatypes import RULReport
        report = RULReport(p05=1.0, p50=2.0, p95=3.0, p_fail_before_mission_end=0.1,
                           mission_end_time_s=3600.0, theta_index=1, theta_name="theta_comb",
                           current_theta=0.90, threshold=0.80, particles_converged=True)
        prognostics = builder.build_prognostics(report)
        from replan_to_learn.planner.datatypes import MissionSimulationResult
        sim = MissionSimulationResult(risk="LOW", risk_probability=0.05,
                                      recommended_action="SAFE", alternatives=[], margin_warnings=())
        mission = builder.build_mission(sim)
        fleet = builder.build_fleet(None, 0, (), None, "DISCONNECTED")
        panel = builder.build(health, diagnosis, prognostics, mission, fleet)
        text = builder.render_text(panel)
        assert "GCS OPERATOR PANEL" in text
        assert "DIAGNOSIS" in text
