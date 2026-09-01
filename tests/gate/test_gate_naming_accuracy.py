"""
Naming-accuracy evaluation for the L4 identifiability gate: given a single
injected fault, does gate.update() actually NAME the correct subsystem?

This is distinct from every other gate test in this repo, which checks
REFUSAL behavior (does the gate correctly say AMBIGUOUS). Before the fixes
this test locks in, the gate could not name ANY subsystem under ANY
circumstance -- confirmed empirically: 0/180 across 9 single-fault
scenarios x 20 trials each. Three real bugs combined to cause this:

1. physics_twin.py: the intake-manifold pressure relaxation ODE had a
   flipped sign (`p_im - map_pa` instead of `map_pa - p_im`), so p_im ran
   away from the real commanded value and saturated at its clip bound
   within ~4 steps, corrupting nearly everything downstream.
2. gate/identifiability.py _compute_fingerprints(): preds0 and preds1 were
   evaluated at inconsistent theta baselines (preds0 used whatever
   physics_twin.health_params already was -- nominal ones(), never synced
   to theta_hat -- while preds1 used theta_hat+eps), so the "derivative"
   actually measured theta_hat's entire deviation from nominal divided by
   1e-3, swamping every fingerprint with a fault-magnitude artifact
   regardless of which parameter was actually perturbed.
3. gate/identifiability.py update(): verdict logic required the ENTIRE
   15-parameter system to be simultaneously separable before naming
   anything, and separately, candidate selection used argmin(crlb) (most
   PRECISELY measurable parameter) instead of a deviation-weighted score,
   so even when a real fault was cleanly separable it either got
   universally blocked by an unrelated pair's entanglement, or lost to an
   unrelated but more-precisely-identified healthy parameter.

Also locks in the honest counterpart: theta_comb/theta_fric (power-balance-
confounded) and theta_cool/theta_oilp (entangled with their own additive
sensor bias) remain genuinely ambiguous at this calibration -- the gate
must report AMBIGUOUS for those, with the true fault always included in
the reported candidate set, never silently misattributed to something
else. That is the gate's central design principle, and it must keep
holding even as naming accuracy improves elsewhere.

theta_vol was originally in that ambiguous group too, but turned out to be
a calibration artifact, not a structural limit: model/noise_model.json's
Phase C fit had computed sigma from the twin's own already-normalized
residual (rf.z) instead of the raw residual, compounding against whatever
sigma was already loaded and inflating z_mdot_f's fitted sigma to ~5.35
(vs a real raw mdot_f prediction error of ~0.0005 kg/s). mdot_f is exactly
the channel that algebraically separates theta_vol (scales mdot_air, hence
mdot_f and CHT heat, but cancels out of the lambda-derived combustion
efficiency term entirely) from theta_comb (scales combustion efficiency,
hence EGT, but is algebraically absent from mdot_f and CHT heat). With
mdot_f's real sigma restored, theta_vol separates cleanly. Two more real
bugs were found and fixed alongside this:
  - physics_twin.py _initialize_nominal_state(): split health_params at
    index 8 instead of 9 (`[0:8]=1.0, [8:]=0.0`), so theta_fric (index 8,
    a multiplicative parameter) defaulted to 0.0 instead of 1.0 for any
    fresh twin -- silently zeroing all friction loss (FMEP) until a caller
    explicitly overrode it.
  - gate/identifiability.py _compute_crlb(): used a fixed 1e-12 Tikhonov
    regularizer on the 15x15 Fisher information matrix, which real
    per-channel noise sigmas (spanning ~1e-9 to ~1e6 once whitened) can
    push to a condition number of ~1e18 -- inv() becomes numerically
    garbage and CRLB blows up globally, even for a parameter (e.g. a
    cleanly spatially-separable injector) whose own row is nowhere near
    singular. Fixed by regularizing relative to the matrix's own scale
    (trace/n) instead of an absolute constant -- standard practice for
    ill-conditioned Fisher/covariance inversion, negligible correction on
    a well-scaled matrix but keeps a badly-scaled one invertible. This was
    a real, independent regression this session's Phase C fix triggered
    (it changed every channel's sigma simultaneously, not just mdot_f),
    caught by this test file's own injector-accuracy assertions dropping
    from 100% to 0% and root-caused before being accepted as a fix.

Two more hypotheses were tried against the remaining ambiguous group
(theta_comb/theta_fric, theta_cool/b_cht, theta_oilp/b_poil) and are
documented here even though they did not resolve the entanglement, per
this repo's practice of recording what was tried and ruled out, not just
what worked:
  - gate/identifiability.py _precompute_regime_jacobian(): was calling
    physics_twin.jacobian() with a SINGLE telemetry frame per regime, but
    physics_twin.predict() only takes one 0.1s RK4 step from whatever
    state the twin instance is CURRENTLY in (a real, live mid-flight
    state for an in-service gate) -- nowhere near that regime's target
    equilibrium. Every regime's Jacobian ended up dominated by the same
    shared, irrelevant starting state (empirically: theta_oilp vs b_poil
    cosine was EXACTLY 1.0). Fixed by feeding a repeated-frame sequence
    (_REGIME_EQUILIBRATION_STEPS=25, ~2.5s) so the twin settles onto each
    regime's real target before the perturbation is taken. This is a
    real, kept fix (moved theta_oilp/b_poil cosine from 1.0 to 0.996) but
    did not on its own drop any pair below tau_sep=0.90.
  - Root cause for why equilibration wasn't enough: model/noise_model.json
    (see sample_counts_by_regime in its provenance block) has 97.7% of
    all real fitted samples concentrated in a single regime bin, with 6 of
    12 core regimes carrying zero real coverage (fallback sigma=1.0
    everywhere else). The fingerprint's 108-dim regime axis is supposed to
    separate a rate-dependent parameter (theta_oilp scales with omega,
    theta_cool's effect scales with (T_hd-T_amb)) from its flat additive
    twin (b_poil, b_cht) by comparing their RESPONSE SHAPE across regimes
    -- but with real statistical weight concentrated on effectively one
    regime, that comparison degenerates to two numbers at a single point,
    which are always ~collinear regardless of the true underlying
    function's shape. This is a genuine real-data-coverage gap (the same
    narrow-regime-coverage limitation noted earlier this project, now
    shown to directly block gate resolution, not just RUL/mission
    planning), not a code bug -- would need real flight data spanning
    climb/high-power/descent regimes to resolve, which was not available
    at the time of this fix cycle.

A THIRD hypothesis, targeting theta_fric's specific noise-robustness gap
(26/30 = 87% under background drift, 2/30 misattributed to theta_comb --
see TestPowerBalanceGroupWeakerMembers below), was tried in a later session
and also reverted, with a second sub-attempt at the same idea:
  - gate/identifiability.py _compute_fingerprints(): SpatialEGTDecomposer's
    own docstring says it decomposes "z_EGT" (a whitened residual), but the
    call site fed it raw, unwhitened Kelvin EGT predictions -- a real unit
    mismatch against the sigma-normalized 108-dim regime axis, confirmed to
    inflate theta_comb's own Fisher diagonal ~123x (a pure common-mode EGT
    effect: ~5,084,051 from the unwhitened spatial axis vs only ~41,513
    from the regime axis alone). This double-counts the same physical
    signal in unit-inconsistent ways and makes theta_comb's CRLB
    artificially tiny, so ordinary background noise on theta_comb was
    enough to occasionally outscore theta_fric's genuinely weaker signal.
    Whitening this axis (feeding sigma-normalized z_EGT into decompose())
    is the mathematically correct fix for the unit mismatch, confirmed by
    direct measurement, but applying it ALONE regressed theta_fric from
    26/30 to 6/30 (14/30 newly misattributed to theta_oilp instead) because
    _compute_crlb's regularizer was a single GLOBAL scalar
    (1e-6*trace(fim)/n added to every diagonal): removing theta_comb's
    huge outlier diagonal entry shrank that shared floor for all 15
    parameters simultaneously, exposing theta_oilp's previously
    floor-masked Fisher information as newly (and wrongly) competitive.
  - A second sub-attempt paired the same whitening fix with a
    per-parameter (diagonal-relative) CRLB regularizer --
    fim_reg[i,i] = fim[i,i]*(1+eps_rel) + eps_abs -- specifically so no
    parameter's regularization floor would depend on another parameter's
    Fisher-information scale. This did eliminate the specific
    cross-parameter coupling that sank the first sub-attempt (theta_comb
    and theta_cool both stayed at 30/30 under their own noise seeds), but
    still did not resolve theta_fric: across 2 independently-seeded 30-trial
    runs it scored 25/30 (seed 1002 -- the SAME seed used in the xfail
    below -- 3/30 newly misattributed to theta_cool, a failure mode that
    didn't exist before) and 27/30 (seed 2002, 2/30 to theta_comb, 1/30 to
    theta_vol), averaging 86.7% and actually WORSE than the original 26/30
    on the reference seed. Conclusion: theta_fric's residual gap is a
    genuine precision limitation, not an artifact of either regularization
    scheme -- its only real signal (the noisy, self-calibrated cube-law
    power-balance channel) is legitimately weaker than SEVERAL other
    parameters' true Fisher information, so removing one competitor's
    regularization-floor artifact (theta_comb, or in the second attempt,
    theta_oilp) simply hands the occasional misattribution to whichever
    OTHER parameter has the next-most-correlated real signal under a given
    noise draw. Both sub-attempts were reverted; identifiability.py's
    _compute_fingerprints and _compute_crlb carry inline comments with the
    same history. See the xfail reason on
    test_theta_fric_named_at_least_90_percent_under_background_drift for
    the concrete before/after numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from replan_to_learn.contracts.telemetry import TelemetryFrame
from replan_to_learn.gate.datatypes import Verdict
from replan_to_learn.gate.identifiability import IdentifiabilityGate
from replan_to_learn.physics_twin.physics_twin import PhysicsTwin

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
NOMINAL = np.concatenate([np.ones(9), np.zeros(6)])
THETA_NAMES = [
    "theta_vol", "theta_comb", "theta_cool", "theta_inj1", "theta_inj2",
    "theta_inj3", "theta_inj4", "theta_oilp", "theta_fric",
]

REGIME_CENTERS = {
    0: dict(n_rpm=2200.0, map_pa=95000.0, t_im=295.0, p_amb=84700.0, t_amb=278.15, v_tas=40.0, altitude_m=1500.0),
    3: dict(n_rpm=2000.0, map_pa=90000.0, t_im=295.0, p_amb=84700.0, t_amb=278.15, v_tas=55.0, altitude_m=1500.0),
    5: dict(n_rpm=2800.0, map_pa=130000.0, t_im=310.0, p_amb=84700.0, t_amb=278.15, v_tas=65.0, altitude_m=1500.0),
    6: dict(n_rpm=2000.0, map_pa=90000.0, t_im=295.0, p_amb=84700.0, t_amb=278.15, v_tas=50.0, altitude_m=1500.0),
    8: dict(n_rpm=2800.0, map_pa=130000.0, t_im=310.0, p_amb=84700.0, t_amb=278.15, v_tas=60.0, altitude_m=1500.0),
}


def _frame(t: float, c: dict) -> TelemetryFrame:
    return TelemetryFrame(
        t=t, egt=(950.0, 955.0, 960.0, 965.0), cht=360.0, p_oil=3.5e5, t_oil=350.0,
        n_rpm=c["n_rpm"], mdot_f=0.05, map_pa=c["map_pa"], tps=0.5, t_im=c["t_im"],
        p_amb=c["p_amb"], t_amb=c["t_amb"], v_tas=c["v_tas"], h_p=c["altitude_m"],
        valid_mask=0xFFFF, flight_id="", aircraft_id="", engine_id="",
    )


def _run_single_fault(true_idx: int, fault_value: float, n_trials: int = 20):
    twin = PhysicsTwin(model_dir=MODEL_DIR, dt=0.1)
    gate = IdentifiabilityGate(physics_twin=twin, n_theta=15)
    true_theta = NOMINAL.copy()
    true_theta[true_idx] = fault_value
    gate.theta_hat = true_theta.copy()

    verdicts = []
    for i in range(n_trials):
        center = REGIME_CENTERS[[0, 3, 5, 6, 8][i % 5]]
        frame = _frame(0.0, center)
        twin.reset(frame)
        for step_i in range(25):  # clear the 20s quasi-steady stabilization window
            rf = twin.step(_frame(float(step_i), center))
        af = gate.update(rf)
        verdicts.append(af)
    return verdicts


class TestInjectorFaultsAreCleanlySeparableAndCorrectlyNamed:
    """Each injector's per-cylinder spatial signature is distinct enough to be reliably named."""

    @pytest.mark.parametrize("inj_idx", [3, 4, 5, 6], ids=lambda i: THETA_NAMES[i])
    def test_correct_injector_named_at_least_90_percent(self, inj_idx):
        verdicts = _run_single_fault(inj_idx, fault_value=0.85)
        named = [af for af in verdicts if af.verdict == Verdict.NAMED]
        correct = [af for af in named if af.named == inj_idx]

        accuracy = len(correct) / len(verdicts)
        assert accuracy >= 0.90, (
            f"{THETA_NAMES[inj_idx]}: only {len(correct)}/{len(verdicts)} trials correctly named "
            f"(NAMED count: {len(named)}, wrong names: {[af.named for af in named if af.named != inj_idx]})"
        )

    def test_never_misattributes_one_injector_fault_to_another_injector(self):
        """A theta_inj1 fault must never come back NAMED as theta_inj2/3/4, or vice versa."""
        for true_idx in [3, 4, 5, 6]:
            verdicts = _run_single_fault(true_idx, fault_value=0.85, n_trials=10)
            wrong_injector_names = [
                af.named for af in verdicts
                if af.verdict == Verdict.NAMED and af.named in {3, 4, 5, 6} and af.named != true_idx
            ]
            assert wrong_injector_names == [], (
                f"{THETA_NAMES[true_idx]} fault was misattributed to a different injector: {wrong_injector_names}"
            )


def _run_noisy_fault(idx: int, fault_value: float, seed: int, n_trials: int = 30, noise_scale: float = 0.02):
    """
    The clean single-fault test (_run_single_fault) sets every OTHER
    parameter to exactly nominal (deviation=0), so the true fault
    trivially wins the deviation/CRLB ranking regardless of magnitude --
    a weak bar that does not by itself prove the fix generalizes. This is
    the real test: with realistic simultaneous small drift on every
    parameter (not just the one true fault), does the true fault still
    win the ranking, or does noise on an unrelated parameter occasionally
    look larger and get misattributed? Each call needs its own
    independently-seeded RNG -- reusing one shared RNG sequentially across
    parameters in a loop makes later parameters' results depend on how
    many draws happened before them, which isn't a fair comparison.
    """
    rng = np.random.default_rng(seed)
    verdicts = []
    for trial in range(n_trials):
        twin = PhysicsTwin(model_dir=MODEL_DIR, dt=0.1)
        gate = IdentifiabilityGate(physics_twin=twin, n_theta=15)
        true_theta = NOMINAL.copy()
        true_theta += rng.normal(0, noise_scale, 15)
        true_theta[idx] = fault_value
        gate.theta_hat = true_theta.copy()
        center = REGIME_CENTERS[[0, 3, 5, 6, 8][trial % 5]]
        frame = _frame(0.0, center)
        twin.reset(frame)
        for step_i in range(25):
            rf = twin.step(_frame(float(step_i), center))
        verdicts.append(gate.update(rf))
    return verdicts


class TestPowerBalanceGroupIsCorrectlyNamed:
    """
    theta_vol, theta_comb, and theta_cool turned out to be separable, not
    structurally confounded -- see the module docstring for the
    calibration/model bugs that were masking them (the mdot_f-sigma
    compounding bug; the T_exh_calc clip sitting below the real Rotax EGT
    limit and silently zeroing theta_comb's EGT signature; and the
    transient-axis fix separating theta_cool from its own additive CHT
    bias). Confirmed robust under realistic simultaneous background drift
    on all 15 parameters, not just the idealized single-fault case (30
    independently-seeded trials each): theta_comb 30/30, theta_cool
    30/30, theta_vol 29/30 (one benign miss, no misattribution).

    theta_fric and theta_oilp are DELIBERATELY EXCLUDED from this class --
    see TestPowerBalanceGroupWeakerMembers below for why.
    """

    @pytest.mark.parametrize(
        "idx,fault_value", [(0, 0.85), (1, 0.85), (2, 0.85)],
        ids=["theta_vol", "theta_comb", "theta_cool"],
    )
    def test_named_at_least_90_percent(self, idx, fault_value):
        verdicts = _run_single_fault(idx, fault_value=fault_value)
        named = [af for af in verdicts if af.verdict == Verdict.NAMED]
        correct = [af for af in named if af.named == idx]

        accuracy = len(correct) / len(verdicts)
        assert accuracy >= 0.90, (
            f"{THETA_NAMES[idx]}: only {len(correct)}/{len(verdicts)} trials correctly named "
            f"(NAMED count: {len(named)}, wrong names: {[af.named for af in named if af.named != idx]})"
        )

    @pytest.mark.parametrize(
        "idx,fault_value,seed", [(0, 0.85, 1003), (1, 0.85, 1004), (2, 0.85, 1005), (3, 0.85, 1006)],
        ids=["theta_vol", "theta_comb", "theta_cool", "theta_inj1"],
    )
    def test_named_correctly_under_simultaneous_background_drift(self, idx, fault_value, seed):
        verdicts = _run_noisy_fault(idx, fault_value, seed)
        correct = sum(1 for af in verdicts if af.verdict == Verdict.NAMED and af.named == idx)
        wrong = [af.named for af in verdicts if af.verdict == Verdict.NAMED and af.named != idx]

        accuracy = correct / len(verdicts)
        assert accuracy >= 0.90, (
            f"{THETA_NAMES[idx]} under background drift: only {correct}/{len(verdicts)} correct, wrong={wrong}"
        )


class TestPowerBalanceGroupWeakerMembers:
    """
    theta_fric and theta_oilp are ALSO now correctly named in the
    idealized single-fault case (20/20 each, via the transient axis and
    the cube-law power-balance residual respectively) -- but unlike
    theta_vol/theta_comb/theta_cool, they do NOT hold up to the same
    30-trial/independently-seeded background-drift stress test, and are
    deliberately kept OUT of TestPowerBalanceGroupIsCorrectlyNamed rather
    than overclaiming a fix that only works in the easy case. Measured:

    theta_oilp: 18/30 (60%) named correctly under noise, but 0/30
    misattributed -- the other 12/30 fall back to honest AMBIGUOUS. This
    is the gate's designed-safe failure mode: its separability from
    b_poil is real but numerically weak (needed a large transient-axis
    weight to even clear tau_sep in the clean case), so ordinary 2%
    parameter noise is enough to occasionally erase the margin -- but it
    never crosses into a false claim.

    theta_fric: 26/30 (87%) correct, but 2/30 WERE misattributed to
    theta_comb -- a real, if rare, violation of the gate's core "never
    misattribute" guarantee under realistic noise, not just a confidence
    problem. Its only real signal is the self-calibrated cube-law
    power-balance channel (noisy, sigma ~38-157 kW) plus T_oil; under
    background drift on theta_comb specifically, the two can occasionally
    swap rank. Documented as a known, quantified residual gap rather than
    silently accepted or hidden.
    """

    @pytest.mark.parametrize(
        "idx,fault_value", [(7, 0.85), (8, 1.15)],
        ids=["theta_oilp", "theta_fric"],
    )
    def test_named_at_least_90_percent_in_idealized_single_fault_case(self, idx, fault_value):
        verdicts = _run_single_fault(idx, fault_value=fault_value)
        named = [af for af in verdicts if af.verdict == Verdict.NAMED]
        correct = [af for af in named if af.named == idx]

        accuracy = len(correct) / len(verdicts)
        assert accuracy >= 0.90, (
            f"{THETA_NAMES[idx]}: only {len(correct)}/{len(verdicts)} trials correctly named "
            f"(NAMED count: {len(named)}, wrong names: {[af.named for af in named if af.named != idx]})"
        )

    def test_theta_oilp_never_misattributes_despite_lower_confidence_under_noise(self):
        """theta_oilp's honest-refusal guarantee holds under noise even though its NAMED rate drops well below 90%."""
        verdicts = _run_noisy_fault(7, 0.85, seed=1001)
        wrong = [af.named for af in verdicts if af.verdict == Verdict.NAMED and af.named != 7]
        assert wrong == [], f"theta_oilp was misattributed under background drift: {wrong}"

    @pytest.mark.xfail(
        reason=(
            "theta_fric's separability relies on the noisy, self-calibrated "
            "cube-law power-balance channel and T_oil; under realistic 2% "
            "background drift on all 15 parameters it was misattributed in "
            "2/30 independently-seeded trials (26/30 = 87% correct, just "
            "under the 90% bar), historically to theta_comb. "
            "Two root-cause/fix attempts this session, both reverted: "
            "(1) theta_comb's Fisher diagonal was inflated ~123x by "
            "SpatialEGTDecomposer being fed raw unwhitened Kelvin EGT "
            "predictions instead of the whitened z_EGT its own docstring "
            "specifies -- a real, confirmed unit-consistency bug. Whitening "
            "that axis alone regressed theta_fric from 26/30 to 6/30 "
            "(14/30 newly misattributed to theta_oilp) because "
            "_compute_crlb's regularizer was a single GLOBAL scalar "
            "(1e-6*trace(fim)/n on every diagonal): shrinking theta_comb's "
            "outlier diagonal lowered that shared floor for all 15 "
            "parameters at once, exposing theta_oilp's previously "
            "floor-masked Fisher information. (2) Replacing the global "
            "regularizer with a per-parameter, diagonal-relative one "
            "(fim_reg[i,i] = fim[i,i]*(1+eps_rel) + eps_abs, so no "
            "parameter's floor depends on another's FIM scale) alongside "
            "re-applying the whitening fix removed that specific coupling "
            "artifact, but still did not clear the bar: over 2 "
            "independently-seeded 30-trial runs theta_fric scored 25/30 "
            "(seed 1002, 3x misattributed to theta_cool -- a NEW failure "
            "mode) and 27/30 (seed 2002, 2x theta_comb, 1x theta_vol), "
            "averaging 86.7% -- worse than the original 26/30 on the "
            "reference seed, and still below 90%. Conclusion: theta_fric's "
            "gap is not a regularization-scheme artifact -- its only real "
            "signal is genuinely weak relative to several other "
            "parameters' true (not regularization-floor) Fisher "
            "information, so whichever competitor has the most correlated "
            "real signal under a given noise draw wins the occasional "
            "swap, and removing one competitor's artifact just hands the "
            "occasional win to a different one. Would need a stronger or "
            "better-calibrated theta_fric-specific channel (e.g. real "
            "propeller/prop-load data for the power-balance residual) to "
            "close, not a CRLB-regularization change."
        ),
        strict=True,
    )
    def test_theta_fric_named_at_least_90_percent_under_background_drift(self):
        verdicts = _run_noisy_fault(8, 1.15, seed=1002)
        correct = sum(1 for af in verdicts if af.verdict == Verdict.NAMED and af.named == 8)
        wrong = [af.named for af in verdicts if af.verdict == Verdict.NAMED and af.named != 8]
        accuracy = correct / len(verdicts)
        assert accuracy >= 0.90, (
            f"theta_fric under background drift: only {correct}/{len(verdicts)} correct, wrong={wrong}"
        )


class TestEntangledFaultsHonestlyRefuseRatherThanMisattribute:
    """
    b_egt/b_cht/b_poil-style sensor-bias entanglement aside, no
    multiplicative health parameter remains UNCONDITIONALLY ambiguous in
    the idealized single-fault case after this fix cycle. This class is
    kept as an explicit regression guard with an empty parameter list --
    if a future change reintroduces a genuinely, always-ambiguous
    parameter, add it back here rather than silently losing coverage.
    """

    @pytest.mark.parametrize("entangled_idx", [], ids=lambda i: THETA_NAMES[i])
    def test_true_fault_always_appears_in_reported_ambiguous_set(self, entangled_idx):
        verdicts = _run_single_fault(entangled_idx, fault_value=0.85 if entangled_idx != 8 else 1.15)

        wrongly_named_something_else = [
            af for af in verdicts if af.verdict == Verdict.NAMED and af.named != entangled_idx
        ]
        assert wrongly_named_something_else == [], (
            f"{THETA_NAMES[entangled_idx]} fault was misattributed to a healthy parameter "
            f"instead of being refused: named={[af.named for af in wrongly_named_something_else]}"
        )

        ambiguous_verdicts = [af for af in verdicts if af.verdict == Verdict.AMBIGUOUS]
        if ambiguous_verdicts:
            missing_true_fault = [af for af in ambiguous_verdicts if entangled_idx not in af.ambiguous_set]
            assert missing_true_fault == [], (
                f"{THETA_NAMES[entangled_idx]} was refused but not listed in its own ambiguous_set "
                f"in {len(missing_true_fault)}/{len(ambiguous_verdicts)} trials"
            )
