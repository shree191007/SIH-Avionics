"""
G3.1 structural enforcement tests: "0 instances of L5 naming a subsystem L4
declared inseparable -- enforced structurally, not statistically."

The previous placeholder's actual gap: MLClassifier.attributions() took
separable_set/ambiguous_set as caller-supplied kwargs and trusted them at
face value -- a caller (or a bug in a caller) could pass an empty
ambiguous_set even while the real gate verdict was AMBIGUOUS, letting the
ML layer's raw posterior "win" a tie the gate had explicitly refused to
break.

The fix (ml/gate_bridge.py::mask_posterior_by_gate + ml/classifier.py::
MLClassifier.attributions): the mask is now computed unconditionally
inside attributions() from a REAL gate.datatypes.AttributionFrame's
.verdict/.ambiguous_set, type-checked so nothing else can stand in for it,
with no parameter through which a caller can override or bypass it.

These tests specifically try to defeat that: a classifier confidently
biased toward one class OUTSIDE an ambiguous pair must still have that
class's probability driven to exactly zero once masked.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest

from replan_to_learn.gate.datatypes import AttributionFrame as GateAttributionFrame
from replan_to_learn.gate.datatypes import ReasonCode, Verdict
from replan_to_learn.ml.classifier import MLClassifier
from replan_to_learn.ml.datatypes import CLASS_ORDER
from replan_to_learn.ml.gate_bridge import mask_posterior_by_gate


def _gate_af(verdict: Verdict, ambiguous_set=(), named=None) -> GateAttributionFrame:
    n = 15
    return GateAttributionFrame(
        t=0.0,
        verdict=verdict,
        theta_hat=np.ones(n),
        crlb=np.ones(n),
        named=named,
        ambiguous_set=tuple(ambiguous_set),
        cos_matrix=np.zeros((n, n)),
        borrowed_regimes=(),
        r2=None,
        alpha_hat=None,
        fleet_epoch=None,
        reason=ReasonCode.COS_TOO_HIGH,
        model_version="1.0.0",
        regime_grid_version="REGIME_GRID_V1",
    )


class _OverconfidentClassifier(MLClassifier):
    """A classifier deliberately, maximally biased toward a class OUTSIDE
    the ambiguous pair under test -- the exact "89% confidence" failure
    mode the spec calls out. If G3.1 were only a convention (caller-trusted
    kwargs), this subclass's bias would leak straight through. Overriding
    predict_proba_raw (not attributions) proves the mask lives in
    attributions() itself, not in something the raw-prediction path could
    have coincidentally provided."""

    def __init__(self, biased_class: str):
        super().__init__(artifact=None)
        self._biased_class = biased_class

    def predict_proba_raw(self, feature_vector: np.ndarray) -> Dict[str, float]:
        probs = {c: 0.001 for c in CLASS_ORDER}
        probs[self._biased_class] = 1.0 - 0.001 * (len(CLASS_ORDER) - 1)
        total = sum(probs.values())
        return {c: v / total for c, v in probs.items()}


class TestG31CannotBeDefeatedByAnOverconfidentModel:
    def test_mass_outside_ambiguous_set_is_driven_to_exactly_zero(self):
        # Gate says INJECTOR_1 vs INJECTOR_2 is an unbreakable tie (theta
        # indices 3, 4). The model is maximally confident in HEALTHY --
        # a class entirely outside that tie.
        gate_frame = _gate_af(Verdict.AMBIGUOUS, ambiguous_set=(3, 4))
        clf = _OverconfidentClassifier(biased_class="HEALTHY")

        result = clf.attributions(
            feature_vector=np.zeros(172),
            gate_frame=gate_frame,
            regime=4,
        )

        assert result.fault_probs.probs["HEALTHY"] == 0.0, (
            "G3.1 VIOLATION: an overconfident model's mass on a class outside "
            "the gate's declared ambiguous set leaked through the mask."
        )
        for c in CLASS_ORDER:
            if c not in ("INJECTOR_1", "INJECTOR_2"):
                assert result.fault_probs.probs[c] == 0.0
        assert result.gate_verdict == "AMBIGUOUS"
        assert set(result.ambiguous_set) == {"INJECTOR_1", "INJECTOR_2"}
        assert result.fault_probs.top_class() in ("INJECTOR_1", "INJECTOR_2")

    def test_confident_bias_toward_one_member_of_the_pair_is_allowed(self):
        """Ranking WITHIN the ambiguous set is legitimate (spec: 'can rank
        candidates within an ambiguous set -- that is useful and honest')."""
        gate_frame = _gate_af(Verdict.AMBIGUOUS, ambiguous_set=(3, 4))
        clf = _OverconfidentClassifier(biased_class="INJECTOR_1")

        result = clf.attributions(
            feature_vector=np.zeros(172),
            gate_frame=gate_frame,
            regime=4,
        )
        assert result.fault_probs.top_class() == "INJECTOR_1"
        assert result.fault_probs.probs["INJECTOR_1"] > result.fault_probs.probs["INJECTOR_2"]
        total_in_set = result.fault_probs.probs["INJECTOR_1"] + result.fault_probs.probs["INJECTOR_2"]
        assert total_in_set == pytest.approx(1.0, abs=1e-6)

    def test_named_verdict_does_not_mask_the_full_posterior(self):
        gate_frame = _gate_af(Verdict.NAMED, named=3)
        clf = _OverconfidentClassifier(biased_class="INJECTOR_1")
        result = clf.attributions(feature_vector=np.zeros(172), gate_frame=gate_frame, regime=4)
        assert result.gate_verdict == "NAMED"
        assert result.fault_probs.probs["INJECTOR_1"] > 0.9

    def test_ambiguous_set_mapping_to_zero_classes_fails_closed(self):
        """If the gate's ambiguous_set contains only theta indices this ML
        layer has no class mapping for, masking must fail CLOSED (all zero),
        never silently fall back to the unmasked posterior."""
        gate_frame = _gate_af(Verdict.AMBIGUOUS, ambiguous_set=(99,))  # not in THETA_TO_CLASS
        clf = _OverconfidentClassifier(biased_class="HEALTHY")
        result = clf.attributions(feature_vector=np.zeros(172), gate_frame=gate_frame, regime=4)
        assert all(v == 0.0 for v in result.fault_probs.probs.values())


class TestMaskPosteriorByGateTypeSafety:
    def test_rejects_non_gate_attribution_frame(self):
        """A caller cannot substitute a duck-typed stand-in with a spoofed
        verdict/ambiguous_set for the real gate output."""
        class FakeGateFrame:
            verdict = Verdict.NAMED
            ambiguous_set = ()

        raw = {c: 1.0 / len(CLASS_ORDER) for c in CLASS_ORDER}
        with pytest.raises(TypeError):
            mask_posterior_by_gate(raw, FakeGateFrame())

    def test_named_passes_through_unmasked(self):
        raw = {c: 1.0 / len(CLASS_ORDER) for c in CLASS_ORDER}
        gate_frame = _gate_af(Verdict.NAMED, named=0)
        masked, is_amb, sep, amb = mask_posterior_by_gate(raw, gate_frame)
        assert not is_amb
        assert masked == raw
        assert amb == ()

    def test_ambiguous_renormalizes_over_allowed_classes(self):
        raw = {c: 1.0 / len(CLASS_ORDER) for c in CLASS_ORDER}
        gate_frame = _gate_af(Verdict.AMBIGUOUS, ambiguous_set=(2, 13))  # COOLING, COOLING (theta_cool, b_cht)
        masked, is_amb, sep, amb = mask_posterior_by_gate(raw, gate_frame)
        assert is_amb
        assert amb == ("COOLING",)
        assert masked["COOLING"] == pytest.approx(1.0, abs=1e-9)
        assert sum(v for k, v in masked.items() if k != "COOLING") == pytest.approx(0.0, abs=1e-9)
