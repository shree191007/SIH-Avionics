"""
SIH26054 Replan to Learn: Stage 5 evaluation harness metrics.
Implements 04_ml_rul_mission_probe.md Sec 5.2 / 05_gcs Sec 5.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from replan_to_learn.eval.datatypes import ArmMetrics, ArmVerdict, TestCase

# A claim is "over-confident" at the spec's stated threshold (Sec 5.2:
# "confidence >= 0.8"). Arms without real calibration (A, C) report a
# fixed high confidence for any non-refused claim by construction (see
# arms.py) -- that IS the honest behavior being measured, not a
# harness artifact.
OVER_CONFIDENT_THRESHOLD = 0.8


def compute_arm_metrics(arm: str, verdicts: List[ArmVerdict], cases_by_id: Dict[str, TestCase]) -> ArmMetrics:
    n = len(verdicts)
    if n == 0:
        return ArmMetrics(arm=arm, n_cases=0, over_confident_misattribution_rate=float("nan"),
                           refusal_rate=float("nan"), resolution_rate=float("nan"),
                           false_positive_rate_per_100h=None, precision_macro=None, recall_macro=None,
                           mean_latency_ms=float("nan"), p95_latency_ms=float("nan"))

    over_confident_wrong = 0
    refused = 0
    healthy_false_alerts = 0
    n_healthy = 0
    per_class_tp: Dict[int, int] = {}
    per_class_fp: Dict[int, int] = {}
    per_class_fn: Dict[int, int] = {}
    latencies = []

    for v in verdicts:
        case = cases_by_id[v.case_id]
        latencies.append(v.latency_ms)
        truth = case.true_fault_index

        if truth is None:
            n_healthy += 1
            if not v.refused and v.named_index is not None:
                healthy_false_alerts += 1

        if v.refused:
            refused += 1
            continue

        # A claim: named_index (possibly None for arm A's "unqualified alert").
        is_wrong = (truth is None) or (v.named_index is not None and v.named_index != truth) or case.is_genuinely_ambiguous
        if is_wrong and v.confidence >= OVER_CONFIDENT_THRESHOLD:
            over_confident_wrong += 1

        if truth is not None and v.named_index is not None:
            if v.named_index == truth:
                per_class_tp[truth] = per_class_tp.get(truth, 0) + 1
            else:
                per_class_fp[v.named_index] = per_class_fp.get(v.named_index, 0) + 1
                per_class_fn[truth] = per_class_fn.get(truth, 0) + 1

    over_confident_rate = over_confident_wrong / n
    refusal_rate = refused / n

    fp_rate_per_100h = None
    if n_healthy > 0:
        # Approximates "alerts per 100 healthy flight hours" using the
        # per-case false-alert rate scaled to a 100h-equivalent count;
        # a real flight-hour-weighted computation needs per-case duration,
        # which the harness records if TestCase.frames carries real
        # timestamps (see harness.py).
        fp_rate_per_100h = (healthy_false_alerts / n_healthy) * 100.0

    classes = set(per_class_tp) | set(per_class_fp) | set(per_class_fn)
    precisions, recalls = [], []
    for c in classes:
        tp = per_class_tp.get(c, 0)
        fp = per_class_fp.get(c, 0)
        fn = per_class_fn.get(c, 0)
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    precision_macro = float(np.mean(precisions)) if precisions else None
    recall_macro = float(np.mean(recalls)) if recalls else None

    lat = np.array(latencies) if latencies else np.array([0.0])
    return ArmMetrics(
        arm=arm, n_cases=n,
        over_confident_misattribution_rate=over_confident_rate,
        refusal_rate=refusal_rate,
        resolution_rate=0.0,  # arms without a probe/borrow stage never resolve a refusal; overridden by harness for E/F
        false_positive_rate_per_100h=fp_rate_per_100h,
        precision_macro=precision_macro, recall_macro=recall_macro,
        mean_latency_ms=float(np.mean(lat)), p95_latency_ms=float(np.percentile(lat, 95)),
    )
