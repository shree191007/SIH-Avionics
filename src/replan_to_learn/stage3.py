"""
SIH26054 Replan to Learn: Stage 3 Orchestrator.
Wires L5 ML, L6 RUL, L7 Planner, and GCS together end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from replan_to_learn.contracts.residuals import ResidualFrame
from replan_to_learn.contracts.regimes import REGIME_GRID_VERSION
from replan_to_learn.gate.datatypes import AttributionFrame, Verdict
from replan_to_learn.gate.identifiability import IdentifiabilityGate
from replan_to_learn.ml.classifier import MLClassifier
from replan_to_learn.ml.features import StreamingFeatureBuilder
from replan_to_learn.planner.datatypes import (
    CandidateManoeuvre,
    MissionSimulationResult,
    ProbeSelectionResult,
)
from replan_to_learn.planner.mission_simulator import MissionSimulator
from replan_to_learn.planner.probe_selector import ProbeSelector
from replan_to_learn.rul.datatypes import RULConfig, RULReport
from replan_to_learn.rul.particle_filter import RULParticleFilter
from replan_to_learn.gcs.panel import GCSPanelBuilder
from replan_to_learn.gcs.datatypes import (
    DiagnosisPanel,
    EngineHealthSummary,
    FleetPanel,
    GCSPanel,
    MissionPanel,
    PrognosticsPanel,
)


@dataclass(frozen=True)
class Stage3Config:
    ml_model_path: Optional[str] = None
    rul_config: Optional[RULConfig] = None
    mission_end_time_s: float = 3600.0
    current_fuel_kg: float = 50.0
    current_altitude_m: float = 1500.0
    return_margin_s: float = 600.0


class Stage3Orchestrator:
    """
    End-to-end Stage 3 orchestrator.

    Pipeline:
      ResidualFrame
        -> MLClassifier (L5)
        -> IdentifiabilityGate (L4)
        -> RULParticleFilter (L6)  [freezes on AMBIGUOUS]
        -> MissionSimulator (L7 nominal)
        -> ProbeSelector (L7 probe) [only if AMBIGUOUS and no borrow]
        -> GCSPanelBuilder
    """

    def __init__(
        self,
        physics_twin: Any,
        config: Optional[Stage3Config] = None,
        fleet_node: Optional[Any] = None,
    ) -> None:
        self.physics_twin = physics_twin
        self.config = config or Stage3Config()
        self.fleet_node = fleet_node

        ml_model_path = config.ml_model_path if config else None
        if ml_model_path is None:
            default_path = Path(__file__).resolve().parents[2] / "model" / "ml_classifier.pkl"
            if default_path.exists():
                ml_model_path = str(default_path)
        if ml_model_path:
            try:
                self.ml_classifier = MLClassifier.from_file(ml_model_path)
            except Exception:
                # No trained artifact available (e.g. before scripts/train_ml_classifier.py
                # has been run): fall back to an unfitted classifier.
                # attributions() calls on it will raise RuntimeError, caught below.
                self.ml_classifier = MLClassifier()
        else:
            self.ml_classifier = MLClassifier()

        self.feature_builder = StreamingFeatureBuilder()

        self.gate = IdentifiabilityGate(physics_twin=physics_twin, n_theta=15)
        self.mission_simulator = MissionSimulator(physics_twin)
        self.probe_selector = ProbeSelector(physics_twin)
        self.gcs_builder = GCSPanelBuilder()

        self.rul_filters: Dict[int, RULParticleFilter] = {}
        self._init_rul_filters()

        self._base_mission_profile = self._default_mission_profile()
        self._last_theta_hat: Optional[np.ndarray] = None

    def _init_rul_filters(self) -> None:
        """
        Failure thresholds are physics-derived (04_ml_rul_mission_probe.md
        Sec 2.4: "computed from the twin, not picked") via
        rul/failure_threshold_solver.py, which binary-searches each health
        parameter against a real Rotax 915iS operator's-manual limit
        (CHT/coolant 120C, EGT 950C, oil pressure 0.8bar below 3500rpm).
        Previously these were hardcoded round numbers (0.80, 0.70, 0.75,
        1.30) with no derivation -- a real, now-closed gap. Loaded from
        the precomputed model/rul_failure_thresholds.json cache (see
        scripts/solve_rul_failure_thresholds.py) where available, falling
        back to a live (slower) solve otherwise.
        """
        from replan_to_learn.rul.failure_threshold_solver import load_or_solve_failure_threshold

        theta_names = [
            "theta_vol", "theta_comb", "theta_cool",
            "theta_inj_1", "theta_inj_2", "theta_inj_3", "theta_inj_4",
            "theta_oilp", "theta_fric",
            "b_egt_1", "b_egt_2", "b_egt_3", "b_egt_4",
            "b_cht", "b_poil",
        ]
        solvable_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8}
        for i, name in enumerate(theta_names):
            if i in solvable_indices:
                threshold = load_or_solve_failure_threshold(self.physics_twin, i, name)
                self.rul_filters[i] = RULParticleFilter(
                    theta_index=i,
                    theta_name=name,
                    config=self.config.rul_config,
                    threshold=threshold,
                )

    def _default_mission_profile(self) -> List[Any]:
        from replan_to_learn.planner.mission_simulator import MissionState
        profile = []
        t = 0.0
        for _ in range(60):
            profile.append(MissionState(
                altitude_m=1500.0,
                ias_kt=120.0,
                rpm=2800.0,
                fuel_kg=self.config.current_fuel_kg - t * 0.1,
                elapsed_s=t,
                regime=4,
            ))
            t += 60.0
        return profile

    def process_residual_frame(self, rf: ResidualFrame) -> Tuple[GCSPanel, Dict[str, Any]]:
        """
        Process a single ResidualFrame through the full Stage 3 pipeline.

        L5 feature order matters: the gate (L4) is updated FIRST so its
        AttributionFrame (theta_hat/crlb/verdict) can be folded into the
        streaming feature builder's "estimator context" block AND passed
        to ml_classifier.attributions() for the G3.1 mask -- see
        ml/gate_bridge.py.
        """
        gate_attribution = self.gate.update(rf)
        self.feature_builder.ingest(rf, gate_attribution)
        feature_vector = self.feature_builder.compute()
        combined_verdict = gate_attribution.verdict

        if combined_verdict == Verdict.AMBIGUOUS and self.fleet_node is not None:
            borrow_result = self.gate.try_borrow(
                gate_attribution.named or 0,
                self.fleet_node.get_shape(gate_attribution.named or 0),
            )
            if borrow_result.admissible:
                combined_verdict = Verdict.BORROWED

        if combined_verdict == Verdict.INVALID:
            final_verdict = "INVALID"
        elif combined_verdict == Verdict.AMBIGUOUS:
            final_verdict = "AMBIGUOUS"
        elif combined_verdict == Verdict.BORROWED:
            final_verdict = "BORROWED"
        else:
            final_verdict = "NAMED"

        try:
            # gate_attribution (the REAL gate.datatypes.AttributionFrame) is
            # passed directly -- ml_classifier.attributions() derives its
            # own separable/ambiguous class sets from it internally (G3.1);
            # there is no override parameter here for stage3 to (mis)supply.
            ml_attribution = self.ml_classifier.attributions(
                feature_vector,
                gate_frame=gate_attribution,
                regime=int(rf.regime),
                flight_id=rf.flight_id,
                contributing_channels=tuple(
                    i for i, v in enumerate(rf.z) if not np.isnan(v) and abs(v) > 1.5
                ),
            )
        except RuntimeError:
            ml_attribution = None

        theta_hat = gate_attribution.theta_hat
        self._last_theta_hat = theta_hat

        for idx, pf in self.rul_filters.items():
            if idx < len(theta_hat):
                if not pf._initialized:
                    P_theta_jj = float(gate_attribution.crlb[idx]) if idx < len(gate_attribution.crlb) else 0.1
                    pf.initialize(
                        theta_init=float(theta_hat[idx]),
                        P_theta_jj=max(P_theta_jj ** 2, 1e-6),
                        seed=42 + idx,
                    )
                P_theta_jj = float(gate_attribution.crlb[idx]) if idx < len(gate_attribution.crlb) else 0.1
                pf.step(
                    theta_obs=float(theta_hat[idx]),
                    P_obs=max(P_theta_jj ** 2, 1e-6),
                    dt_fh=1.0 / 3600.0,
                    verdict=final_verdict,
                )
                pf.set_mission_end(self.config.mission_end_time_s)

        rul_reports = {idx: pf.estimate_rul() for idx, pf in self.rul_filters.items()}
        active_rul = next(iter(rul_reports.values())) if rul_reports else None

        sim_result = self.mission_simulator.simulate(
            theta=theta_hat if self._last_theta_hat is not None else np.ones(15),
            mission_profile=self._base_mission_profile,
            current_fuel_kg=self.config.current_fuel_kg,
        )

        probe_result = None
        if final_verdict == "AMBIGUOUS":
            fim_det = float(np.linalg.det(np.eye(15))) if theta_hat is not None else 1.0
            probe_result = self.probe_selector.select_probe(
                ambiguous_set=gate_attribution.ambiguous_set,
                theta=theta_hat if self._last_theta_hat is not None else np.ones(15),
                current_fim_det=fim_det * 0.8,
                post_probe_fim_det=fim_det,
                current_fuel_kg=self.config.current_fuel_kg,
                current_altitude_m=self.config.current_altitude_m,
                return_margin_s=self.config.return_margin_s,
            )

        health = self.gcs_builder.build_engine_health(
            z=np.array(rf.z, dtype=np.float32),
            egt=(float(rf.z[0]), float(rf.z[1]), float(rf.z[2]), float(rf.z[3])),
            cht=360.0,
            p_oil=3.5e5,
            t_oil=350.0,
            n_rpm=2800.0,
            mdot_f=0.05,
        )

        diagnosis = self.gcs_builder.build_diagnosis(
            verdict=final_verdict,
            named=gate_attribution.named,
            ambiguous_set=gate_attribution.ambiguous_set,
            confidence=float(gate_attribution.crlb[gate_attribution.named]) if gate_attribution.named is not None and gate_attribution.named < len(gate_attribution.crlb) else 0.0,
            separable_set=(),
            cos_matrix=gate_attribution.cos_matrix,
            contributing_channels=(),
            attribution_basis="LOCAL",
            tree_shap_top5=list(ml_attribution.tree_shap_top5) if ml_attribution is not None else [],
        )

        prognostics = self.gcs_builder.build_prognostics(active_rul) if active_rul else PrognosticsPanel(
            rul_p05=float('nan'), rul_p50=float('nan'), rul_p95=float('nan'),
            p_fail_before_mission=0.0, mission_end_time_s=self.config.mission_end_time_s,
            degradation_trajectory=None,
        )

        mission = self.gcs_builder.build_mission(sim_result)
        fleet = self.gcs_builder.build_fleet(
            fleet_epoch=None,
            contributor_count=0,
            borrowed_regimes=(),
            r2=None,
            transfer_status="DISCONNECTED",
        )

        panel = self.gcs_builder.build(health, diagnosis, prognostics, mission, fleet)
        metadata = {
            "ml_attribution": (
                {
                    "fault_probs": ml_attribution.fault_probs.to_dict(),
                    "separable_set": ml_attribution.separable_set,
                    "ambiguous_set": ml_attribution.ambiguous_set,
                    "gate_verdict": ml_attribution.gate_verdict,
                    "confidence": ml_attribution.confidence,
                    "is_novel": ml_attribution.is_novel,
                }
                if ml_attribution is not None else None
            ),
            "rul_reports": {k: v.to_dict() for k, v in rul_reports.items()},
            "probe_selection": probe_result.to_dict() if hasattr(probe_result, 'to_dict') else None,
            "gate_attribution": {
                "verdict": str(gate_attribution.verdict),
                "named": gate_attribution.named,
                "ambiguous_set": gate_attribution.ambiguous_set,
                "cos_matrix": gate_attribution.cos_matrix.tolist() if gate_attribution.cos_matrix is not None else None,
                "crlb": gate_attribution.crlb.tolist(),
            },
        }
        return panel, metadata
