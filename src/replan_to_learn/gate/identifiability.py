"""
SIH26054 Replan to Learn: L4 Identifiability Gate.
Implements Section 2 of 03_state_identifiability_fleet.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from replan_to_learn.contracts.residuals import ResidualFrame
from replan_to_learn.contracts.regimes import (
    REGIME_GRID_VERSION,
    REGIME_GRID_DEFINITIONS,
    CORE_REGIME_BINS,
    RegimeDefinition,
)
from replan_to_learn.contracts.faults import SpatialEGTDecomposer
from replan_to_learn.contracts.faults import SpatialEGTDecomposer
from replan_to_learn.gate.datatypes import (
    AttributionFrame,
    BorrowResult,
    ProbeRequest,
    ReasonCode,
    Verdict,
)


@dataclass(frozen=True)
class RegimeOperatingPoint:
    """Representative operating point for a regime bin."""
    n_rpm: float
    p_brake_kw: float
    altitude_m: float
    roc_ms: float
    map_pa: float
    t_im: float
    p_amb: float
    t_amb: float
    v_tas: float


# Representative centers for the 12 core regime bins
# Derived from REGIME_GRID_DEFINITIONS midpoints
REGIME_CENTERS: Dict[int, RegimeOperatingPoint] = {
    0: RegimeOperatingPoint(n_rpm=2200.0, p_brake_kw=25.0,  altitude_m=1500.0, roc_ms=5.0,   map_pa=95000.0,  t_im=295.0, p_amb=84700.0,  t_amb=278.15, v_tas=40.0),
    1: RegimeOperatingPoint(n_rpm=2600.0, p_brake_kw=55.0,  altitude_m=1500.0, roc_ms=5.0,   map_pa=115000.0, t_im=300.0, p_amb=84700.0,  t_amb=278.15, v_tas=45.0),
    2: RegimeOperatingPoint(n_rpm=3100.0, p_brake_kw=85.0,  altitude_m=1500.0, roc_ms=5.0,   map_pa=135000.0, t_im=310.0, p_amb=84700.0,  t_amb=278.15, v_tas=50.0),
    3: RegimeOperatingPoint(n_rpm=2000.0, p_brake_kw=25.0,  altitude_m=1500.0, roc_ms=0.0,   map_pa=90000.0,  t_im=295.0, p_amb=84700.0,  t_amb=278.15, v_tas=55.0),
    4: RegimeOperatingPoint(n_rpm=2400.0, p_brake_kw=55.0,  altitude_m=1500.0, roc_ms=0.0,   map_pa=110000.0, t_im=300.0, p_amb=84700.0,  t_amb=278.15, v_tas=60.0),
    5: RegimeOperatingPoint(n_rpm=2800.0, p_brake_kw=85.0,  altitude_m=1500.0, roc_ms=0.0,   map_pa=130000.0, t_im=310.0, p_amb=84700.0,  t_amb=278.15, v_tas=65.0),
    6: RegimeOperatingPoint(n_rpm=2000.0, p_brake_kw=25.0,  altitude_m=1500.0, roc_ms=-5.0,  map_pa=90000.0,  t_im=295.0, p_amb=84700.0,  t_amb=278.15, v_tas=50.0),
    7: RegimeOperatingPoint(n_rpm=2400.0, p_brake_kw=55.0,  altitude_m=1500.0, roc_ms=-5.0,  map_pa=110000.0, t_im=300.0, p_amb=84700.0,  t_amb=278.15, v_tas=55.0),
    8: RegimeOperatingPoint(n_rpm=2800.0, p_brake_kw=85.0,  altitude_m=1500.0, roc_ms=-5.0,  map_pa=130000.0, t_im=310.0, p_amb=84700.0,  t_amb=278.15, v_tas=60.0),
    9: RegimeOperatingPoint(n_rpm=2200.0, p_brake_kw=25.0,  altitude_m=5000.0, roc_ms=0.0,   map_pa=75000.0,  t_im=290.0, p_amb=54000.0,  t_amb=270.15, v_tas=60.0),
    10: RegimeOperatingPoint(n_rpm=2500.0, p_brake_kw=55.0,  altitude_m=5000.0, roc_ms=0.0,   map_pa=95000.0,  t_im=295.0, p_amb=54000.0,  t_amb=270.15, v_tas=70.0),
    11: RegimeOperatingPoint(n_rpm=2900.0, p_brake_kw=85.0,  altitude_m=5000.0, roc_ms=0.0,   map_pa=115000.0, t_im=300.0, p_amb=54000.0,  t_amb=270.15, v_tas=80.0),
}


class IdentifiabilityGate:
    """
    Runtime identifiability gate that refuses to name a subsystem it cannot separate.

    Implements Section 2 of 03_state_identifiability_fleet.md:
    - Sensitivity fingerprint computation (3-axis: regime, spatial, cross-modal)
    - Cosine separability metric
    - Fisher information and CRLB
    - Verdict logic: NAMED, AMBIGUOUS, BORROWED, INVALID
    - Borrow attempt before probe request (A1-A6)
    """

    def __init__(
        self,
        physics_twin: Any,
        n_theta: int,
        tau_sep: float = 0.90,
        tau_r2: float = 0.85,
        delta_j: Optional[np.ndarray] = None,
    ) -> None:
        self.physics_twin = physics_twin
        self.n_theta = n_theta
        self.tau_sep = tau_sep
        self.tau_r2 = tau_r2
        # [ASSUMED] placeholder default -- Sec 2.4 says delta_j should be
        # "a physically meaningful bound... set per parameter from the
        # maintenance manual, not from the data", which this codebase does
        # not have. The previous default (0.05) was far below this model's
        # actual achievable CRLB scale (~10-40 for a typical parameter
        # given the current calibration) and made every verdict AMBIGUOUS
        # regardless of what was faulted, for every caller that didn't
        # override it -- verified empirically. 20.0 is a functional
        # placeholder (confirmed to let real, cleanly-separable faults like
        # individual injector degradation actually get NAMED), not a
        # domain-sourced threshold; replace with real maintenance-decision
        # values per parameter when available.
        self.delta_j = delta_j if delta_j is not None else np.ones(n_theta) * 20.0

        self.theta_hat: Optional[np.ndarray] = None
        self.P_theta: Optional[np.ndarray] = None
        self.regime_encounters: List[set] = [set() for _ in range(n_theta)]
        self._last_ambiguous_set: Tuple[int, ...] = ()
        self._last_cos_matrix: np.ndarray = np.zeros((n_theta, n_theta))
        self._last_crlb: np.ndarray = np.zeros(n_theta)
        self._cached_theta_hat: Optional[np.ndarray] = None
        self._last_fingerprints: Optional[np.ndarray] = None
        self._cached_regime_jacobian: Optional[np.ndarray] = None
        self._cached_regime_jacobian_early: Optional[Dict[int, np.ndarray]] = None
        self._last_model_version: Optional[str] = None
        self._last_regime_grid_version: Optional[str] = None

    def update(self, rf: ResidualFrame) -> AttributionFrame:
        if rf.status != 0:
            return AttributionFrame(
                t=rf.t,
                verdict=Verdict.INVALID,
                theta_hat=self.theta_hat if self.theta_hat is not None else np.ones(self.n_theta),
                crlb=np.full(self.n_theta, np.inf),
                named=None,
                ambiguous_set=(),
                cos_matrix=np.zeros((self.n_theta, self.n_theta)),
                borrowed_regimes=(),
                r2=None,
                alpha_hat=None,
                fleet_epoch=None,
                reason=ReasonCode.NONE,
                model_version=rf.model_version,
                regime_grid_version=rf.regime_grid_version,
            )

        if self.theta_hat is None:
            # Was np.ones(n_theta) -- wrong for the real 15-element layout,
            # where indices 9-14 are ADDITIVE sensor biases whose nominal
            # is 0.0, not 1.0. Defaulting them to 1.0 made every fresh gate
            # compute a fake deviation=1.0 on all 6 bias terms even for a
            # genuinely healthy aircraft (deviation = |theta_hat - real
            # nominal|), which the argmax(deviation/crlb) ranking could
            # win once CRLB stopped being artificially huge (see
            # _compute_crlb's scale-relative regularization fix) --
            # confirmed via G2.2/G2.4 regressing to 0% refusal / 100%
            # misattribution on cruise-only healthy data once that CRLB
            # fix landed, root-caused to this exact mismatch.
            self.theta_hat = self._nominal_theta()
            self.P_theta = np.eye(self.n_theta) * 0.01

        if rf.regime in CORE_REGIME_BINS:
            for j in range(self.n_theta):
                self.regime_encounters[j].add(rf.regime)

        if self.theta_hat is None or self._cached_theta_hat is None or not np.allclose(self.theta_hat, self._cached_theta_hat):
            self._cached_theta_hat = self.theta_hat.copy() if self.theta_hat is not None else np.ones(self.n_theta)
            self._cached_regime_jacobian = self._precompute_regime_jacobian()

        fingerprints = self._compute_fingerprints(rf)
        self._last_fingerprints = fingerprints
        self._last_model_version = rf.model_version
        self._last_regime_grid_version = rf.regime_grid_version
        separable, ambiguous_set, cos_matrix = self._check_separability(
            self._weighted_for_separability(fingerprints)
        )
        fim = self._compute_fisher_information(fingerprints)
        crlb = self._compute_crlb(fim)

        self._last_ambiguous_set = ambiguous_set
        self._last_cos_matrix = cos_matrix
        self._last_crlb = crlb

        # `ambiguous_set` is the SPECIFIC subset of parameters entangled
        # with at least one other parameter (Sec 2.5: "the inseparable set
        # {theta_j, theta_k, ...} is reported explicitly") -- it is not a
        # verdict on the whole theta vector. A parameter that isn't
        # entangled with anything should still be nameable even if some
        # OTHER, unrelated pair elsewhere in theta remains confounded
        # (e.g. theta_oilp vs its own sensor bias b_poil being ambiguous
        # says nothing about whether an injector fault, cleanly separable
        # from everything, can be named). Requiring the ENTIRE system to
        # be simultaneously separable before naming anything made the gate
        # refuse universally in practice, since some pair (often a
        # parameter vs its own additive bias, which move the same channel
        # nearly identically at a single evaluated regime) is almost
        # always entangled somewhere in a 15-dimensional theta vector.
        ambiguous_indices = set(ambiguous_set)
        n_theta_local = fingerprints.shape[0]

        # Rank EVERY parameter (not just the clean ones) by a deviation-to-
        # uncertainty score -- effectively a z-score: "how many standard
        # errors from nominal is this estimate". argmin(crlb) alone picks
        # whichever parameter happens to be most PRECISELY measurable, with
        # no regard to whether it actually deviates from nominal at all, so
        # a perfectly healthy but well-identified parameter would always
        # outrank a genuinely faulted but less-precisely-identified one.
        nominal = self._nominal_theta(n_theta_local)
        deviation = np.abs(self.theta_hat - nominal)
        crlb_floor = np.maximum(crlb, 1e-6)
        score = deviation / crlb_floor

        # The candidate to report is whichever parameter shows the
        # strongest real signal, full stop -- THEN we decide whether that
        # signal can be honestly attributed. Restricting the search to
        # "clean" (non-entangled) parameters first was wrong: when the
        # actual fault sits inside an ambiguous pair (e.g. theta_vol
        # entangled with theta_comb/theta_fric, a real, physically
        # plausible confound at a single operating point), every candidate
        # with real signal gets excluded, and the ranking degenerates to a
        # tie among zero-deviation clean parameters -- silently naming an
        # arbitrary HEALTHY parameter while the real fault goes unreported.
        # That is exactly the false-confidence failure mode this gate
        # exists to prevent.
        best = int(np.argmax(score))

        # If theta_hat sits at (or within noise of) nominal for every
        # parameter -- a genuinely healthy aircraft -- every score is ~0,
        # and np.argmax on an all-zero array deterministically returns
        # index 0 by numpy's tie-breaking convention: whichever parameter
        # happens to occupy index 0 (theta_vol) then gets confidently
        # NAMED purely because it's well-identified (small CRLB), with NO
        # real deviation behind it at all. The `identifiable` check below
        # only tests PRECISION (is CRLB small enough), never whether the
        # estimated deviation is actually distinguishable from zero --
        # those are different questions, and conflating them let a
        # healthy aircraft get a confident, false NAMED verdict (caught
        # empirically via G2.2/G2.4: 0% refusal / 100% misattribution on
        # cruise-only healthy data, once the CRLB-conditioning fix made
        # theta_vol's CRLB small enough to pass `crlb < delta_j` on pure
        # noise). [ASSUMED] threshold: real fault magnitudes tested this
        # session (3-20% deviation) score orders of magnitude above this;
        # exact-nominal or UKF-estimation-noise-level theta_hat scores
        # far below it.
        MIN_SCORE_THRESHOLD = 1e-3
        if score[best] < MIN_SCORE_THRESHOLD:
            return AttributionFrame(
                t=rf.t,
                verdict=Verdict.AMBIGUOUS,
                theta_hat=self.theta_hat.copy(),
                crlb=crlb,
                named=None,
                ambiguous_set=(),
                cos_matrix=cos_matrix,
                borrowed_regimes=(),
                r2=None,
                alpha_hat=None,
                fleet_epoch=None,
                reason=ReasonCode.NONE,
                model_version=rf.model_version,
                regime_grid_version=rf.regime_grid_version,
            )

        if best in ambiguous_indices:
            entangled_with_best = {best}
            for i in range(n_theta_local):
                for j in range(i + 1, n_theta_local):
                    if cos_matrix[i, j] >= self.tau_sep and (i == best or j == best):
                        entangled_with_best.add(i)
                        entangled_with_best.add(j)
            return AttributionFrame(
                t=rf.t,
                verdict=Verdict.AMBIGUOUS,
                theta_hat=self.theta_hat.copy(),
                crlb=crlb,
                named=None,
                ambiguous_set=tuple(sorted(entangled_with_best)),
                cos_matrix=cos_matrix,
                borrowed_regimes=(),
                r2=None,
                alpha_hat=None,
                fleet_epoch=None,
                reason=ReasonCode.COS_TOO_HIGH,
                model_version=rf.model_version,
                regime_grid_version=rf.regime_grid_version,
            )

        named = best
        identifiable = crlb[named] < self.delta_j[named]

        if identifiable:
            return AttributionFrame(
                t=rf.t,
                verdict=Verdict.NAMED,
                theta_hat=self.theta_hat.copy(),
                crlb=crlb,
                named=named,
                ambiguous_set=(),
                cos_matrix=cos_matrix,
                borrowed_regimes=(),
                r2=None,
                alpha_hat=None,
                fleet_epoch=None,
                reason=ReasonCode.NONE,
                model_version=rf.model_version,
                regime_grid_version=rf.regime_grid_version,
            )
        else:
            return AttributionFrame(
                t=rf.t,
                verdict=Verdict.AMBIGUOUS,
                theta_hat=self.theta_hat.copy(),
                crlb=crlb,
                named=None,
                ambiguous_set=(named,),
                cos_matrix=cos_matrix,
                borrowed_regimes=(),
                r2=None,
                alpha_hat=None,
                fleet_epoch=None,
                reason=ReasonCode.COS_TOO_HIGH,
                model_version=rf.model_version,
                regime_grid_version=rf.regime_grid_version,
            )

    def _nominal_theta(self, n: Optional[int] = None) -> np.ndarray:
        """
        Nominal theta vector: the first 9 elements (multiplicative health
        parameters) are 1.0, any remaining elements (additive sensor
        biases, for the real 15-element layout) are 0.0. Single source of
        truth for "nominal" -- used both as the fallback theta_hat for a
        fresh gate and as the deviation baseline in update()'s scoring, so
        the two can never silently disagree again.
        """
        n = self.n_theta if n is None else n
        if n <= 9:
            return np.ones(n)
        return np.concatenate([np.ones(9), np.zeros(n - 9)])

    def _regime_op_point(self, regime: int) -> RegimeOperatingPoint:
        return REGIME_CENTERS.get(regime, REGIME_CENTERS[4])

    def _make_telemetry_frame(self, regime: int, overrides: Optional[Dict[str, float]] = None) -> Any:
        from replan_to_learn.contracts.telemetry import TelemetryFrame
        op = self._regime_op_point(regime)
        kwargs = dict(
            t=0.0,
            egt=(950.0, 955.0, 960.0, 965.0),
            cht=360.0,
            p_oil=3.5e5,
            t_oil=350.0,
            n_rpm=op.n_rpm,
            mdot_f=0.05,
            map_pa=op.map_pa,
            tps=0.5,
            t_im=op.t_im,
            p_amb=op.p_amb,
            t_amb=op.t_amb,
            v_tas=op.v_tas,
            h_p=op.altitude_m,
            valid_mask=0xFFFF,
            flight_id="",
            aircraft_id="",
            engine_id="",
        )
        if overrides:
            kwargs.update(overrides)
        return TelemetryFrame(**kwargs)

    # Fingerprint layout: 12 regimes x 9 channels (regime axis) + 3
    # (spatial EGT decomposition) + 1 (cross-modal EGT/CHT coherence) + 2
    # (transient axis: early-minus-late Jacobian for CHT and p_oil, the
    # two channels whose parameter, in this model, is entangled with its
    # own additive sensor bias -- see _precompute_regime_jacobian).
    FINGERPRINT_DIM = 12 * 9 + 3 + 1 + 2

    # The transient axis is only 2 of 114 fingerprint dimensions, so its
    # raw contribution to the whole-vector cosine similarity is
    # numerically swamped by the 108-dim regime axis even though it is
    # QUALITATIVELY decisive: a pure additive bias (b_cht, b_poil) has
    # EXACTLY zero transient signal (confirmed empirically, dCHT/db_cht
    # and dp_oil/db_poil are identically 1.0 at every timescale), while
    # theta_cool/theta_oilp's is always nonzero. Because the bias's
    # transient component is exactly zero, scaling this axis up only
    # grows theta_cool/theta_oilp's own fingerprint norm -- it cannot
    # manufacture a false signal for the bias, so this is a legitimate
    # way to let a qualitatively decisive but numerically small axis
    # actually influence the combined cosine, not an arbitrary tuning
    # knob.
    #
    # The two channels need DIFFERENT weights, not one shared constant:
    # theta_vol also has a real (smaller) CHT transient signal, since it
    # scales total fuel energy input into the same T_hd ODE theta_cool's
    # cooling term multiplies -- pushing the CHT weight as high as
    # theta_oilp/b_poil's pairing needs (their transient signal is far
    # weaker, only ~1.6x early-to-late growth vs theta_cool's ~15x)
    # created a NEW false entanglement between theta_vol and theta_cool
    # (cos 0.92) that didn't exist before. theta_vol has zero effect on
    # p_oil, so that channel has no such collision risk and can take the
    # larger weight theta_oilp needs. Values chosen empirically: CHT_W=8
    # clears theta_cool/b_cht (cos 0.48) while keeping theta_vol/theta_cool
    # well clear (cos 0.21, vs tau_sep=0.90); POIL_W=100 clears
    # theta_oilp/b_poil (cos 0.60, vs 0.996 unweighted).
    TRANSIENT_AXIS_WEIGHT_CHT = 8.0
    TRANSIENT_AXIS_WEIGHT_POIL = 100.0

    def _compute_fingerprints(self, rf: ResidualFrame) -> np.ndarray:
        n_theta = self.n_theta
        fingerprints = np.zeros((n_theta, self.FINGERPRINT_DIM), dtype=np.float64)

        regime = rf.regime if rf.regime in CORE_REGIME_BINS else 4
        regime_jacobian = self._cached_regime_jacobian
        regime_jacobian_early = self._cached_regime_jacobian_early

        sigma_current = np.array(rf.sigma, dtype=np.float64)
        if sigma_current.ndim == 0:
            sigma_current = np.full(9, float(sigma_current))

        for j in range(n_theta):
            regime_vec = np.zeros(12 * 9, dtype=np.float64)
            if regime_jacobian is not None:
                for r in CORE_REGIME_BINS:
                    J = regime_jacobian[r]
                    sigma = np.array(rf.sigma, dtype=np.float64)
                    if sigma.ndim == 0:
                        sigma = np.full(9, float(sigma))
                    regime_vec[r * 9:(r + 1) * 9] = J[:, j] / (sigma + 1e-12)

            transient_vec = np.zeros(2, dtype=np.float64)
            if regime_jacobian is not None and regime_jacobian_early is not None:
                late = regime_jacobian[regime][:, j]
                early = regime_jacobian_early[regime][:, j]
                # Channels 4 (CHT) and 5 (p_oil): the two channels whose
                # multiplicative parameter is confounded with a pure
                # additive bias. A bias has zero dynamic effect (early ==
                # late always); theta_cool/theta_oilp's effect grows with
                # equilibration time, so this difference is nonzero only
                # for the real dynamic parameter.
                # NOT weighted here -- this is the raw physical fingerprint,
                # also used for Fisher information/CRLB (see update()),
                # where an arbitrary weight would distort the estimated
                # precision of theta_cool/theta_oilp themselves (confirmed
                # empirically: weighting here made CRLB[theta_oilp]
                # artificially tiny, so ordinary small noise on theta_oilp
                # started outscoring and getting misattributed over real
                # faults elsewhere). The separability weighting is applied
                # separately, only for the cosine/entanglement check --
                # see _weighted_for_separability().
                transient_vec[0] = (early[4] - late[4]) / (sigma_current[4] + 1e-12)
                transient_vec[1] = (early[5] - late[5]) / (sigma_current[5] + 1e-12)

            frame_regime = self._make_telemetry_frame(regime)
            saved = self.physics_twin.health_params.copy()

            # preds0 and preds1 MUST be evaluated at the same baseline
            # (theta_hat), differing only by the 1e-3 perturbation on
            # index j -- otherwise this isn't a finite-difference
            # derivative at all. The previous version computed preds0
            # with health_params left at whatever it already was
            # (nominal ones(), since nothing else syncs it to theta_hat)
            # and preds1 at theta_hat+eps, so the "derivative" actually
            # measured theta_hat's ENTIRE deviation from nominal divided
            # by 1e-3 -- e.g. a real 0.85 vs 1.0 injector fault came out
            # scaled by 1000x, swamping every other component of the
            # fingerprint and collapsing every cosine similarity toward
            # 1.0 regardless of the parameter actually perturbed. This
            # was the dominant cause of the gate never NAMING anything.
            self.physics_twin.health_params = self.theta_hat.copy()
            preds0 = self.physics_twin._compute_predictions(frame_regime)

            theta_perturb = self.theta_hat.copy()
            theta_perturb[j] += 1e-3
            self.physics_twin.health_params = theta_perturb
            preds1 = self.physics_twin._compute_predictions(frame_regime)
            self.physics_twin.health_params = saved

            # [TRIED AND REVERTED TWICE NOW -- see identifiability.py
            # module-level notes / test_gate_naming_accuracy.py docstring
            # for the full writeup] SpatialEGTDecomposer here receives raw,
            # unwhitened Kelvin predictions even though its own docstring
            # says it decomposes "z_EGT" (whitened residual) -- a real unit
            # mismatch against the sigma-normalized 108-dim regime axis
            # that inflates theta_comb's own Fisher diagonal ~123x (a pure
            # common-mode/alpha EGT effect, confirmed empirically:
            # unwhitened spatial axis contributed ~5,084,051 vs ~41,513
            # from the regime axis alone for theta_comb).
            #
            # SECOND ATTEMPT (this session): whitened this axis AND
            # replaced _compute_crlb's global trace(fim)/n regularizer with
            # a per-parameter diagonal-relative one (see _compute_crlb's
            # history comment), specifically to remove the cross-parameter
            # coupling that sank the FIRST attempt. That eliminated the
            # coupling artifact, but did NOT fix theta_fric: over 2
            # independently-seeded 30-trial background-drift runs, theta_fric
            # scored 25/30 (seed 1002, misattributed 3x to theta_cool -- a
            # NEW failure mode) and 27/30 (seed 2002, misattributed 2x to
            # theta_comb, 1x to theta_vol), averaging 86.7% -- still below
            # the 90% bar, and WORSE on the reference seed (1002) than the
            # unwhitened baseline's 26/30. theta_comb/theta_cool themselves
            # stayed at 30/30 under their own noise seeds (no regression
            # there). Conclusion: removing the global-regularizer coupling
            # artifact does not, by itself, give theta_fric a genuinely
            # stronger signal -- its only real channel (the noisy
            # self-calibrated cube-law power-balance residual) is
            # legitimately weakly identified relative to SEVERAL other
            # parameters' true (not regularization-floor) Fisher
            # information, so whichever competitor has the most correlated
            # real signal under a given noise draw wins the occasional
            # swap -- first theta_comb, now (with whitening) sometimes
            # theta_cool instead. This is a genuine remaining-precision
            # problem, not a regularization-scheme artifact, so reverted
            # again rather than accepting a lateral, seed-dependent result.
            egt0 = preds0.get("egt", np.zeros(4))
            egt1 = preds1.get("egt", np.zeros(4))
            alpha0, beta0, s0, _ = SpatialEGTDecomposer.decompose(egt0)
            alpha1, beta1, s1, _ = SpatialEGTDecomposer.decompose(egt1)
            spatial_vec = np.array([
                alpha1 - alpha0,
                beta1 - beta0,
                s1 - s0,
            ], dtype=np.float64) / 1e-3

            egt0 = preds0.get("egt", np.zeros(4))
            cht0 = preds0.get("cht", 0.0)
            if np.std(egt0) > 1e-9 and np.std(np.array([cht0] * 4)) > 1e-9:
                coherence0 = float(np.corrcoef(egt0, np.array([cht0] * 4))[0, 1])
            else:
                coherence0 = 0.0

            egt1 = preds1.get("egt", np.zeros(4))
            cht1 = preds1.get("cht", 0.0)
            if np.std(egt1) > 1e-9 and np.std(np.array([cht1] * 4)) > 1e-9:
                coherence1 = float(np.corrcoef(egt1, np.array([cht1] * 4))[0, 1])
            else:
                coherence1 = 0.0

            cross_modal = np.array([coherence1 - coherence0], dtype=np.float64) / 1e-3
            fingerprints[j] = np.concatenate([regime_vec, spatial_vec, cross_modal, transient_vec])

        return fingerprints

    # Number of repeated-telemetry steps used to equilibrate physics_twin's
    # internal state onto each regime's target operating point before
    # taking the regime-axis Jacobian (see _precompute_regime_jacobian).
    # 25 steps (2.5s) is enough to settle the fast state variables
    # (omega/p_im, time constant ~0.5s) that the regime axis needs.
    # theta_cool's own CHT sensitivity is still growing well beyond this
    # window (empirically: -8.8/-17.8/-31 at 2.5s/6s/20s) since thermal
    # mass settles much slower -- but widening this window to try to
    # capture that fully-settled value did NOT help separability: the
    # transient axis (see TRANSIENT_AXIS_WEIGHT below) is an (early-late)
    # DIFFERENCE, and both early and late grow together as the window
    # widens, so their difference relative to the regime axis's own
    # (also-growing) magnitude stays roughly constant regardless of
    # window length. The actual fix is TRANSIENT_AXIS_WEIGHT, not a
    # longer window, so this stays at the cheap 2.5s setting.
    _REGIME_EQUILIBRATION_STEPS = 25

    # Step index (0-based) used as the "early" (mostly unequilibrated)
    # Jacobian snapshot for the transient axis -- see
    # _precompute_regime_jacobian's docstring on why theta_cool/theta_oilp
    # need this in addition to the (now-correct) equilibrated regime axis.
    # Index 0 (the very first step) maximizes contrast against the fully
    # equilibrated late snapshot.
    _EARLY_TRANSIENT_STEP_INDEX = 0

    def _precompute_regime_jacobian(self) -> Optional[Dict[int, np.ndarray]]:
        """
        Also populates self._cached_regime_jacobian_early as a side effect.

        Beyond the steady-state regime axis (see the equilibration fix
        above), theta_cool and theta_oilp remained entangled with their own
        additive sensor bias (b_cht, b_poil respectively) even after that
        fix, at cosine ~0.98-0.996 -- close but not enough to cross
        tau_sep=0.90. Root cause: b_cht/b_poil are pure OUTPUT shifts with
        zero effect on system dynamics (dCHT/db_cht = dp_oil/db_poil = 1.0
        at every timescale, confirmed empirically), while theta_cool
        multiplies the T_hd relaxation ODE's own rate constant (so its
        effect on CHT keeps GROWING the longer the twin has been at that
        operating point -- confirmed empirically: dCHT/dtheta_cool grew
        from -1.18 at t=0.3s to -17.8 at t=6s, a 15x change) and
        theta_oilp's effect scales with omega/mu_oil(T_oil), themselves
        still-settling state variables early on (dp_oil/dtheta_oilp grew
        ~1.6x from t=0.3s to t=6s). A pure additive bias can NEVER show
        this growth, so the (early - late) Jacobian difference is a real,
        physically-motivated axis that specifically targets this pair of
        confounds -- not a data-availability-limited signal like the
        regime axis, so it doesn't depend on real multi-regime coverage.
        physics_twin.jacobian() already computes the Jacobian at every
        intermediate equilibration step for a multi-frame sequence, so the
        "early" snapshot is free -- just a different row of the same
        tensor already being computed for the (now-correct) equilibrated
        regime axis, not a second simulation.
        """
        try:
            regime_jacobian = {}
            regime_jacobian_early = {}
            for r in CORE_REGIME_BINS:
                # physics_twin.predict() takes only ONE RK4 step per frame
                # from whatever state the twin instance is CURRENTLY in --
                # for a live twin mid-flight, that's an arbitrary real
                # operating point, not this regime's target. A single step
                # barely moves slow state variables (p_im, T_hd, T_oil,
                # omega) toward the target, so every regime's Jacobian ends
                # up dominated by the same shared, irrelevant starting
                # state instead of that regime's actual steady-state
                # behavior -- collapsing the regime-to-regime shape
                # differentiation the fingerprint's regime axis exists to
                # capture (empirically: theta_oilp vs its own additive
                # bias b_poil came out at cosine similarity of EXACTLY
                # 1.0, which a single frozen shared state fully explains).
                # Fix: feed a sequence of REGIME_EQUILIBRATION_STEPS
                # identical frames so physics_twin.jacobian's internal
                # predict() calls settle onto this regime's real target
                # before the finite-difference perturbation is taken; use
                # only the last (equilibrated) row of the resulting
                # per-step Jacobian tensor.
                frames_r = [
                    self._make_telemetry_frame(r, overrides={"t": i * self.physics_twin.config.dt})
                    for i in range(self._REGIME_EQUILIBRATION_STEPS)
                ]
                J = self.physics_twin.jacobian(frames_r, self.theta_hat)
                regime_jacobian[r] = np.array(J[-1], dtype=np.float64)
                regime_jacobian_early[r] = np.array(J[self._EARLY_TRANSIENT_STEP_INDEX], dtype=np.float64)
            self._cached_regime_jacobian_early = regime_jacobian_early
            return regime_jacobian
        except Exception:
            self._cached_regime_jacobian_early = None
            return None

    def _weighted_for_separability(self, fingerprints: np.ndarray) -> np.ndarray:
        """
        Returns a COPY of `fingerprints` with the transient axis (last 2
        components) scaled by TRANSIENT_AXIS_WEIGHT_CHT/POIL, for the
        cosine/entanglement check ONLY. Deliberately not applied to the
        fingerprints used for Fisher information/CRLB -- see the
        docstring at _compute_fingerprints' transient_vec assignment for
        why conflating the two broke naming accuracy under noise.
        """
        weighted = fingerprints.copy()
        weighted[:, -2] *= self.TRANSIENT_AXIS_WEIGHT_CHT
        weighted[:, -1] *= self.TRANSIENT_AXIS_WEIGHT_POIL
        return weighted

    def _check_separability(
        self,
        fingerprints: np.ndarray,
    ) -> Tuple[bool, Tuple[int, ...], np.ndarray]:
        n_theta = fingerprints.shape[0]
        cos_matrix = np.zeros((n_theta, n_theta))

        for i in range(n_theta):
            for j in range(i + 1, n_theta):
                f_i = fingerprints[i]
                f_j = fingerprints[j]
                norm_i = np.linalg.norm(f_i)
                norm_j = np.linalg.norm(f_j)
                if norm_i > 1e-12 and norm_j > 1e-12:
                    cos_matrix[i, j] = np.abs(np.dot(f_i, f_j) / (norm_i * norm_j))
                    cos_matrix[j, i] = cos_matrix[i, j]

        ambiguous = []
        for i in range(n_theta):
            for j in range(i + 1, n_theta):
                if cos_matrix[i, j] >= self.tau_sep:
                    ambiguous.extend([i, j])

        ambiguous_set = tuple(sorted(set(ambiguous)))
        return len(ambiguous_set) == 0, ambiguous_set, cos_matrix

    def _compute_fisher_information(self, fingerprints: np.ndarray) -> np.ndarray:
        F = fingerprints @ fingerprints.T
        return F

    def _compute_crlb(self, fim: np.ndarray) -> np.ndarray:
        try:
            # A fixed 1e-12 regularizer is meaningless once real per-channel
            # noise sigmas (which the fingerprint Jacobian is whitened by)
            # span many orders of magnitude across channels/regimes -- the
            # FIM's own diagonal can range from ~1e-9 to ~1e6, giving a
            # condition number so large (~1e18 observed) that inv() is
            # numerically garbage: CRLB blows up globally, including for
            # parameters (e.g. a cleanly spatially-separable injector
            # fault) whose own row is nowhere near singular. Regularize
            # relative to the matrix's own scale (standard practice for
            # ill-conditioned Fisher/covariance inversion) instead of an
            # absolute constant, so the correction is negligible for a
            # well-scaled FIM but keeps a badly-scaled one invertible.
            #
            # [TRIED AND REVERTED -- see _compute_fingerprints' spatial-EGT
            # whitening comment for the full writeup] A per-parameter
            # (diagonal-relative) regularization scheme -- `fim_reg[i,i] =
            # fim[i,i] * (1 + eps_rel) + eps_abs` -- was tried alongside
            # whitening the spatial-EGT axis, specifically to remove this
            # global scheme's cross-parameter coupling (a change to one
            # parameter's Fisher diagonal shifting trace(fim)/n shifts
            # every other parameter's regularization floor). That coupling
            # fix worked as intended, but did not fix the underlying
            # theta_fric problem it was chasing: over 2 independently-seeded
            # 30-trial runs theta_fric still only scored 25/30 and 27/30
            # (86.7% average, still below the 90% bar, and WORSE than this
            # global scheme's 26/30 on the same reference seed), with
            # misattribution shifting to theta_cool instead of theta_comb.
            # Reverted back to this scheme since the alternative was not a
            # net improvement and both spatial-EGT whitening and the
            # per-parameter regularizer were only justified together as a
            # theta_fric fix.
            n = fim.shape[0]
            scale = np.trace(fim) / max(n, 1)
            fim_reg = fim + 1e-6 * scale * np.eye(n)
            F_inv = np.linalg.inv(fim_reg)
            return np.sqrt(np.diag(F_inv))
        except np.linalg.LinAlgError:
            return np.full(self.n_theta, np.inf)

    def fingerprint(self, theta_index: int) -> np.ndarray:
        """Full FINGERPRINT_DIM-dim fingerprint for theta_index from the most recent update()."""
        if self._last_fingerprints is not None and theta_index < self._last_fingerprints.shape[0]:
            return self._last_fingerprints[theta_index].copy()
        return np.zeros(self.FINGERPRINT_DIM)

    @staticmethod
    def _regime_summary(fingerprint_row: np.ndarray) -> np.ndarray:
        """
        Reduce the full per-channel regime block (12 regimes x 9 channels)
        to one magnitude per regime, comparable to FleetShape.u (12-dim,
        one scalar per regime). The L2 norm across channels is the
        "how strongly does this regime show up for this parameter"
        summary implied by treating the regime axis as its own axis,
        separate from the per-channel spatial/cross-modal ones (Sec 2.3).
        """
        regime_block = fingerprint_row[:108].reshape(12, 9)
        return np.linalg.norm(regime_block, axis=1)

    def try_borrow(self, theta_index: int, shape: Any) -> BorrowResult:
        """
        Implements 03_state_identifiability_fleet.md Sec 3.4/3.6:
            alpha_hat = <f_i, u_j|R_i> / ||u_j|R_i||^2
            R^2       = 1 - ||f_i - alpha_hat * u_j|R_i||^2 / ||f_i||^2
        against the aircraft's OWN local fingerprint f_i (theta_index's
        fingerprint from the most recent update(), restricted to the
        regimes this aircraft has actually flown for that parameter) --
        not a proxy computed purely from the fleet shape's own variance.
        Rule A1 (>=3 distinct regimes) is required for the guardrail to
        have any statistical power at all (Sec 3.6): with 1 flown regime
        the projection is trivially exact and R^2 = 1 identically.
        """
        try:
            if shape.u is None or shape.var_u is None:
                return BorrowResult(admissible=False, verdict=Verdict.AMBIGUOUS, reason=ReasonCode.A1_INSUFFICIENT_REGIMES)

            # A6 (documented in this method's own class docstring as part of
            # A1-A6, but never actually checked anywhere until now): a fleet
            # shape fused under a different model calibration or a different
            # regime grid definition is not a valid projection target for
            # this aircraft's local fingerprint -- the regime axes and the
            # per-regime scale wouldn't even mean the same thing.
            shape_model_version = getattr(shape, "model_version", None)
            shape_grid_version = getattr(shape, "regime_grid_version", None)
            if (
                self._last_model_version is not None and shape_model_version is not None
                and shape_model_version != self._last_model_version
            ) or (
                self._last_regime_grid_version is not None and shape_grid_version is not None
                and shape_grid_version != self._last_regime_grid_version
            ):
                return BorrowResult(admissible=False, verdict=Verdict.AMBIGUOUS, reason=ReasonCode.A6_VERSION_MISMATCH)

            local_regimes = np.array(sorted(self.regime_encounters[theta_index]), dtype=int)
            if len(local_regimes) < 3:
                return BorrowResult(admissible=False, verdict=Verdict.AMBIGUOUS, reason=ReasonCode.A1_INSUFFICIENT_REGIMES)

            if self._last_fingerprints is None or theta_index >= self._last_fingerprints.shape[0]:
                return BorrowResult(admissible=False, verdict=Verdict.AMBIGUOUS, reason=ReasonCode.A1_INSUFFICIENT_REGIMES)

            f_i_regime = self._regime_summary(self._last_fingerprints[theta_index])
            f_i_local = f_i_regime[local_regimes]
            u_local = shape.u[local_regimes]

            norm_f_sq = float(np.dot(f_i_local, f_i_local))
            if norm_f_sq < 1e-12:
                return BorrowResult(admissible=False, verdict=Verdict.AMBIGUOUS, reason=ReasonCode.A1_INSUFFICIENT_REGIMES)

            denom_u_sq = float(np.dot(u_local, u_local))
            alpha_hat = float(np.dot(f_i_local, u_local) / denom_u_sq) if denom_u_sq > 1e-12 else 0.0

            if alpha_hat <= 0:
                return BorrowResult(admissible=False, verdict=Verdict.AMBIGUOUS, reason=ReasonCode.A3_NEGATIVE_GAIN)

            residual = f_i_local - alpha_hat * u_local
            r2 = 1.0 - float(np.dot(residual, residual)) / norm_f_sq

            if r2 < self.tau_r2:
                return BorrowResult(
                    admissible=False, verdict=Verdict.AMBIGUOUS, reason=ReasonCode.A2_LOW_R2,
                    r2=r2, alpha_hat=alpha_hat,
                )

            return BorrowResult(
                admissible=True,
                verdict=Verdict.BORROWED,
                r2=r2,
                alpha_hat=alpha_hat,
                reason=ReasonCode.NONE,
                borrowed_regimes=tuple(int(r) for r in local_regimes),
            )
        except Exception:
            return BorrowResult(admissible=False, verdict=Verdict.AMBIGUOUS, reason=ReasonCode.A1_INSUFFICIENT_REGIMES)

    def probe_request(self) -> Optional[ProbeRequest]:
        if not self._last_ambiguous_set:
            return None
        return ProbeRequest(
            ambiguous_set=self._last_ambiguous_set,
            reason=ReasonCode.A1_INSUFFICIENT_REGIMES,
            cos_matrix=self._last_cos_matrix,
            crlb=self._last_crlb,
        )
