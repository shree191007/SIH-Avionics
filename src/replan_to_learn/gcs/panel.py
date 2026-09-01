"""
SIH26054 Replan to Learn: GCS Operator Panel Builder.
Implements Section 4 of 04_ml_rul_mission_probe.md.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from replan_to_learn.gcs.datatypes import (
    DiagnosisPanel,
    EngineHealthSummary,
    FleetPanel,
    GCSPanel,
    MissionPanel,
    PrognosticsPanel,
)


class GCSPanelBuilder:
    """
    Builds the operator-facing GCS panel.
    Enforces refusal state: never renders an ambiguous candidate as a definitive subsystem.
    """

    def __init__(self, theta_names: Optional[List[str]] = None) -> None:
        self.theta_names = theta_names or [
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

    def build_engine_health(
        self,
        z: np.ndarray,
        egt: Tuple[float, float, float, float],
        cht: float,
        p_oil: float,
        t_oil: float,
        n_rpm: float,
        mdot_f: float,
    ) -> EngineHealthSummary:
        health_score = 100
        max_resid = float(np.max(np.abs(z))) if z is not None else 0.0
        if max_resid > 3.0:
            health_score = max(0, 100 - int(max_resid * 20))
        elif max_resid > 2.0:
            health_score = max(0, 100 - int(max_resid * 10))

        if health_score >= 90:
            status = "HEALTHY"
        elif health_score >= 70:
            status = "DEGRADING"
        else:
            status = "ACTION REQUIRED"

        return EngineHealthSummary(
            health_percent=health_score,
            status=status,
            primary_issue=None,
            egt=egt,
            cht=cht,
            p_oil=p_oil,
            t_oil=t_oil,
            n_rpm=n_rpm,
            mdot_f=mdot_f,
        )

    def build_diagnosis(
        self,
        verdict: str,
        named: Optional[int],
        ambiguous_set: Tuple[int, ...],
        confidence: float,
        separable_set: Tuple[str, ...],
        cos_matrix: Optional[np.ndarray],
        contributing_channels: Tuple[int, ...],
        attribution_basis: str,
        tree_shap_top5: Optional[List[Tuple[str, float]]] = None,
    ) -> DiagnosisPanel:
        if tree_shap_top5 is None:
            tree_shap_top5 = []

        if verdict == "AMBIGUOUS":
            ambiguous_names = tuple(self.theta_names[i] for i in ambiguous_set if i < len(self.theta_names))
            cos_vals = ""
            if cos_matrix is not None and len(ambiguous_set) >= 2:
                i, j = ambiguous_set[0], ambiguous_set[1]
                if i < cos_matrix.shape[0] and j < cos_matrix.shape[1]:
                    cos_vals = f"|cos| = {cos_matrix[i, j]:.3f}"

            primary_issue = f"AMBIGUOUS — {' OR '.join(ambiguous_names)} cannot separate at cruise ({cos_vals})"
            confidence_display = 0.0
        elif verdict == "BORROWED":
            primary_issue = f"BORROWED — {self.theta_names[named] if named is not None and named < len(self.theta_names) else 'unknown'}"
            confidence_display = confidence
        elif verdict == "INVALID":
            primary_issue = "INVALID — model or input status prevents any claim"
            confidence_display = 0.0
        else:
            primary_issue = self.theta_names[named] if named is not None and named < len(self.theta_names) else "HEALTHY"
            confidence_display = confidence

        return DiagnosisPanel(
            verdict=verdict,
            named=named,
            ambiguous_set=ambiguous_set,
            confidence=float(confidence_display),
            separable_set=separable_set,
            cos_matrix=cos_matrix,
            contributing_channels=contributing_channels,
            attribution_basis=attribution_basis,
            tree_shap_top5=tree_shap_top5,
        )

    def build_prognostics(
        self,
        rul_report: Any,
    ) -> PrognosticsPanel:
        return PrognosticsPanel(
            rul_p05=float(rul_report.p05),
            rul_p50=float(rul_report.p50),
            rul_p95=float(rul_report.p95),
            p_fail_before_mission=float(rul_report.p_fail_before_mission_end),
            mission_end_time_s=float(rul_report.mission_end_time_s),
            degradation_trajectory=None,
        )

    def build_mission(self, sim_result: Any) -> MissionPanel:
        return MissionPanel(
            risk=str(sim_result.risk),
            risk_probability=float(sim_result.risk_probability),
            recommendation=str(sim_result.recommended_action),
            alternatives=[str(a) for a in sim_result.alternatives],
            mission_endurance_s=3600.0,
        )

    def build_fleet(
        self,
        fleet_epoch: Optional[int],
        contributor_count: int,
        borrowed_regimes: Tuple[int, ...],
        r2: Optional[float],
        transfer_status: str,
    ) -> FleetPanel:
        return FleetPanel(
            fleet_epoch=fleet_epoch,
            contributor_count=contributor_count,
            borrowed_regimes=borrowed_regimes,
            r2=r2,
            transfer_status=transfer_status,
        )

    def build(
        self,
        health: EngineHealthSummary,
        diagnosis: DiagnosisPanel,
        prognostics: PrognosticsPanel,
        mission: MissionPanel,
        fleet: FleetPanel,
    ) -> GCSPanel:
        return GCSPanel(
            engine_health=health,
            diagnosis=diagnosis,
            prognostics=prognostics,
            mission=mission,
            fleet=fleet,
        )

    def render_text(self, panel: GCSPanel) -> str:
        lines = [
            "=" * 60,
            "SIH26054 REPLAN TO LEARN — GCS OPERATOR PANEL",
            "=" * 60,
            "",
            "ENGINE HEALTH",
            f"  Status      : {panel.engine_health.status}",
            f"  Health %    : {panel.engine_health.health_percent}%",
            f"  EGT[1..4]   : {panel.engine_health.egt}",
            f"  CHT         : {panel.engine_health.cht:.1f} K",
            f"  p_oil       : {panel.engine_health.p_oil / 1e5:.2f} bar",
            f"  T_oil       : {panel.engine_health.t_oil:.1f} K",
            f"  N           : {panel.engine_health.n_rpm:.0f} rpm",
            "",
            "DIAGNOSIS",
            f"  Verdict     : {panel.diagnosis.verdict}",
            f"  Named       : {panel.diagnosis.named}",
            f"  Ambiguous   : {panel.diagnosis.ambiguous_set}",
            f"  Confidence  : {panel.diagnosis.confidence:.3f}",
            f"  Basis       : {panel.diagnosis.attribution_basis}",
            "",
            "PROGNOSTICS",
            f"  RUL p05/p50/p95: {panel.prognostics.rul_p05:.1f} / {panel.prognostics.rul_p50:.1f} / {panel.prognostics.rul_p95:.1f} h",
            f"  P(fail before mission): {panel.prognostics.p_fail_before_mission:.3f}",
            "",
            "MISSION",
            f"  Risk        : {panel.mission.risk}",
            f"  Probability : {panel.mission.risk_probability:.3f}",
            f"  Recommendation: {panel.mission.recommendation}",
            "",
            "FLEET",
            f"  Epoch       : {panel.fleet.fleet_epoch}",
            f"  Contributors: {panel.fleet.contributor_count}",
            f"  Borrowed    : {panel.fleet.borrowed_regimes}",
            f"  R2          : {panel.fleet.r2}",
            f"  Status      : {panel.fleet.transfer_status}",
            "=" * 60,
        ]
        return "\n".join(lines)
