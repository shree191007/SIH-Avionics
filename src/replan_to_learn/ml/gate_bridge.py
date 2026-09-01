"""
SIH26054 Replan to Learn: L5/L4 structural bridge -- G3.1 enforcement.

Spec (04_ml_rul_mission_probe.md Sec 1.1):

    posterior = model.predict(features)
    posterior = mask(posterior, gate.separable_classes())
    if gate.verdict is AMBIGUOUS:
        report = AmbiguousReport(gate.ambiguous_set, posterior_over_set)

"The ML layer can rank candidates within an ambiguous set -- that is
useful and honest -- but it can never break a tie the physics says is
unbreakable."

This module is the ONLY place the mask is computed. ml/classifier.py's
`attributions()` calls `mask_posterior_by_gate()` unconditionally on every
call and has no parameter through which a caller can supply its own
separable_set/ambiguous_set to bypass it -- that was the previous
placeholder's actual G3.1 gap (predict_proba/attributions took
separable_set/ambiguous_set as caller-supplied kwargs, trusted at face
value). The mask now derives entirely from a REAL
`replan_to_learn.gate.datatypes.AttributionFrame` instance's `.verdict` and
`.ambiguous_set` fields (theta indices), type-checked so a duck-typed stand-in
cannot be substituted for the real gate output.
"""

from __future__ import annotations

from typing import Dict, Set, Tuple

from replan_to_learn.gate.datatypes import AttributionFrame as GateAttributionFrame
from replan_to_learn.gate.datatypes import Verdict
from replan_to_learn.ml.datatypes import CLASS_ORDER, THETA_TO_CLASS


def classes_for_theta_indices(indices: Tuple[int, ...]) -> Set[str]:
    return {THETA_TO_CLASS[i] for i in indices if i in THETA_TO_CLASS}


def mask_posterior_by_gate(
    raw_probs: Dict[str, float],
    gate_frame: GateAttributionFrame,
) -> Tuple[Dict[str, float], bool, Tuple[str, ...], Tuple[str, ...]]:
    """
    Structural G3.1 mask. Returns (masked_probs, is_ambiguous, separable_set,
    ambiguous_set) where separable_set/ambiguous_set are ML class names.

    gate_frame MUST be a real gate.datatypes.AttributionFrame (isinstance
    checked) -- this is what makes the mask non-bypassable: there is no code
    path in this module, or in ml/classifier.py, that accepts a
    caller-constructed stand-in for the gate's verdict.
    """
    if not isinstance(gate_frame, GateAttributionFrame):
        raise TypeError(
            "mask_posterior_by_gate requires a real gate.datatypes.AttributionFrame "
            f"instance (G3.1 structural constraint) -- got {type(gate_frame)!r}"
        )

    missing = [c for c in CLASS_ORDER if c not in raw_probs]
    if missing:
        raise ValueError(f"raw_probs missing classes required for masking: {missing}")

    if gate_frame.verdict == Verdict.AMBIGUOUS:
        allowed = classes_for_theta_indices(tuple(gate_frame.ambiguous_set))
        # UNKNOWN is never part of a named ambiguous_set (the gate does not
        # reason about novelty), but the ML layer is always permitted to
        # flag novelty independently -- that is added by the caller via
        # is_novel, not through this mask. Do not add UNKNOWN here.
        if not allowed:
            # Gate declared AMBIGUOUS but its ambiguous_set contained no
            # theta index this ML layer has a class for (e.g. entirely
            # bias/unmapped indices) -- fail closed: mask everything to
            # zero rather than silently falling back to the full posterior.
            masked = {c: 0.0 for c in CLASS_ORDER}
            return masked, True, (), ()

        masked = {c: (p if c in allowed else 0.0) for c, p in raw_probs.items()}
        total = sum(masked.values())
        if total > 1e-12:
            masked = {c: v / total for c, v in masked.items()}
        else:
            # Model placed ~zero mass on every allowed class: fail to a
            # uniform distribution restricted to the allowed set rather
            # than to an unmasked (and therefore G3.1-violating) posterior.
            masked = {c: (1.0 / len(allowed) if c in allowed else 0.0) for c in CLASS_ORDER}
        ambiguous_set = tuple(sorted(allowed))
        return masked, True, (), ambiguous_set

    # NAMED / BORROWED / INVALID: gate did not declare an unresolved tie for
    # this update, so the full posterior passes through unmasked. Note this
    # still never lets the ML layer NAME something the gate refused to --
    # it only ever ranks within a space the gate has already certified as
    # not containing an unbroken tie.
    separable_set = tuple(CLASS_ORDER)
    return dict(raw_probs), False, separable_set, ()
