"""
SIH26054 Replan to Learn: L5 ML classifier training pipeline.

Generates real training data (04_ml_rul_mission_probe.md Sec 1.4):
  - negative class (HEALTHY): real NGAFID healthy flights via
    tests/real_data/real_data_loader.RealFlightLoader, run through the
    calibrated PhysicsTwin at nominal health parameters.
  - positive classes: physics-injected faults. A "truth" PhysicsTwin runs
    with health_params set to a faulted theta vector (the same real
    fault-injection idiom used throughout tests/gate/test_gate_naming_
    accuracy.py's _run_single_fault/_run_noisy_fault); its predicted
    measurements (x_hat) are fed as the "measured" telemetry into a
    SEPARATE, nominal-parameter "estimator" PhysicsTwin, whose residuals
    (measured-faulted vs predicted-nominal) are therefore genuinely
    fault-driven -- not merely a copy of the injected theta into the
    estimator-context feature slot. gate.theta_hat is set to the same true
    theta on the estimator-side gate, simulating a converged L3 UKF
    estimate for the estimator-context (theta_hat/crlb) part of the
    feature vector, exactly as _run_single_fault does.

Split by aircraft/flight (never by sample) -- GroupKFold on a `groups`
array where each real flight and each synthetic fault trial is its own
group, so no adjacent-second leakage between train/test.

Trains a LightGBM multiclass GBT (depth<=6, <=300 trees), isotonic-
calibrates per class on a held-out fold, computes ECE/Brier (G3.2), fits
the UNKNOWN novelty detector on the held-out healthy fold, and caches the
resulting MLModelArtifact to model/ml_classifier.pkl.

Usage:
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
        scripts/train_ml_classifier.py [--quick]
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "real_data"))

from replan_to_learn.contracts.telemetry import TelemetryFrame  # noqa: E402
from replan_to_learn.gate.identifiability import IdentifiabilityGate, REGIME_CENTERS  # noqa: E402
from replan_to_learn.gate.datatypes import Verdict  # noqa: E402
from replan_to_learn.ml.datatypes import (  # noqa: E402
    CLASS_ORDER,
    NUM_CLASSES,
    THETA_TO_CLASS,
    MLModelArtifact,
)
from replan_to_learn.ml.features import StreamingFeatureBuilder, TOTAL_FEATURES  # noqa: E402
from replan_to_learn.physics_twin.physics_twin import PhysicsTwin  # noqa: E402
from real_data_loader import RealFlightLoader  # noqa: E402

MODEL_DIR = REPO_ROOT / "model"
DATA_ROOT = REPO_ROOT / "data"
ARTIFACT_PATH = MODEL_DIR / "ml_classifier.pkl"
TRAINING_DATA_CACHE = DATA_ROOT / "ml_training_features_v2.npz"

NOMINAL = np.concatenate([np.ones(9), np.zeros(6)])

# Fault magnitudes swept per theta index (spec Sec 1.4: "swept over
# magnitude, onset rate, and regime exposure"). Multiplicative parameters
# (idx 0-8, nominal 1.0) get relative-degradation magnitudes; additive
# sensor biases (idx 9-14, nominal 0.0) get Kelvin/Pa-scale offsets
# consistent with the twin's own noise/sigma scale for those channels.
MULT_MAGNITUDES = (0.70, 0.85)
FRIC_MAGNITUDES = (1.20, 1.35)  # theta_fric: higher = more friction loss
BIAS_MAGNITUDES = (20.0, 35.0)  # Kelvin (EGT/CHT) or Pa-equivalent bias offsets

REGIME_SCHEDULE_SEEDS = (7001, 7002)

N_STEPS_PER_TRIAL = 300
WARMUP_S = 90
EMIT_EVERY_S = 30

N_HEALTHY_FLIGHTS = 40
HEALTHY_MAX_ROWS = 600
HEALTHY_WARMUP_S = 60
HEALTHY_EMIT_EVERY_S = 20

# NOTE on runtime: IdentifiabilityGate._precompute_regime_jacobian (12
# regimes x ~25 equilibration steps each, via PhysicsTwin.jacobian's
# finite-difference predict() calls) costs several seconds and is cached
# per DISTINCT theta_hat value (gate/identifiability.py: invalidated only
# when theta_hat changes). To keep this script's runtime bounded, a single
# gate instance is reused across (a) every healthy flight -- theta_hat is
# always NOMINAL there -- and (b) every regime-schedule seed for a given
# (theta_idx, magnitude_slot) fault -- theta_hat is identical across
# seeds, only the regime dwell schedule differs. This is a real, measured
# runtime optimization (profiled: ~13s per fresh jacobian precompute),
# not a shortcut on the physics -- the twin's PHYSICAL state is still
# reset() between every flight/trial.


def _frame(t: float, c) -> TelemetryFrame:
    return TelemetryFrame(
        t=t, egt=(950.0, 955.0, 960.0, 965.0), cht=360.0, p_oil=3.5e5, t_oil=350.0,
        n_rpm=c.n_rpm, mdot_f=0.05, map_pa=c.map_pa, tps=0.5, t_im=c.t_im,
        p_amb=c.p_amb, t_amb=c.t_amb, v_tas=c.v_tas, h_p=c.altitude_m,
        valid_mask=0xFFFF, flight_id="", aircraft_id="", engine_id="",
    )


def _measured_frame_from_xhat(t: float, c, x_hat: np.ndarray) -> TelemetryFrame:
    return TelemetryFrame(
        t=t, egt=(float(x_hat[0]), float(x_hat[1]), float(x_hat[2]), float(x_hat[3])),
        cht=float(x_hat[4]), p_oil=float(x_hat[5]), t_oil=float(x_hat[6]),
        n_rpm=c.n_rpm, mdot_f=float(x_hat[7]), map_pa=c.map_pa, tps=0.5, t_im=c.t_im,
        p_amb=c.p_amb, t_amb=c.t_amb, v_tas=c.v_tas, h_p=c.altitude_m,
        valid_mask=0xFFFF, flight_id="", aircraft_id="", engine_id="",
    )


def _fault_value(idx: int, magnitude_slot: int) -> float:
    if idx == 8:
        return FRIC_MAGNITUDES[magnitude_slot]
    if idx in (9, 10, 11, 12, 13):
        return BIAS_MAGNITUDES[magnitude_slot]
    if idx == 14:
        # b_poil: pressure-scale bias (Pa); use a proportionally larger
        # magnitude since p_oil operates on a ~1e5 Pa scale vs EGT's ~1e3 K.
        return BIAS_MAGNITUDES[magnitude_slot] * 4000.0
    return MULT_MAGNITUDES[magnitude_slot]


def _regime_sequence(n_steps: int, seed: int, dwell_s: int = 90) -> List[int]:
    rng = np.random.default_rng(seed)
    regimes = list(REGIME_CENTERS.keys())
    seq = []
    t = 0
    while t < n_steps:
        r = int(rng.choice(regimes))
        seq.extend([r] * dwell_s)
        t += dwell_s
    return seq[:n_steps]


def generate_fault_trials_for_theta(theta_idx: int, magnitude_slot: int, seeds: Tuple[int, ...]) -> List[Tuple[np.ndarray, str, str]]:
    """One (theta_idx, magnitude_slot) pair: a single gate/jacobian is built
    once (theta_hat constant across seeds) and reused for every seed's
    regime-schedule trial."""
    label = THETA_TO_CLASS[theta_idx]
    true_theta = NOMINAL.copy()
    true_theta[theta_idx] = _fault_value(theta_idx, magnitude_slot)

    c0 = REGIME_CENTERS[0]
    twin_truth = PhysicsTwin(model_dir=MODEL_DIR, dt=0.1)
    twin_est = PhysicsTwin(model_dir=MODEL_DIR, dt=0.1)
    twin_est.reset(_frame(0.0, c0))
    gate = IdentifiabilityGate(physics_twin=twin_est, n_theta=15)
    gate.theta_hat = true_theta.copy()

    out: List[Tuple[np.ndarray, str, str]] = []
    for seed in seeds:
        regime_seq = _regime_sequence(N_STEPS_PER_TRIAL, seed + theta_idx * 100 + magnitude_slot * 10)
        c0 = REGIME_CENTERS[regime_seq[0]]
        twin_truth.reset(_frame(0.0, c0))
        twin_truth.health_params = true_theta.copy()
        twin_est.reset(_frame(0.0, c0))

        fb = StreamingFeatureBuilder(flight_id=f"fault_{theta_idx}_{magnitude_slot}_{seed}")
        rows = []
        for step_i in range(N_STEPS_PER_TRIAL):
            t = float(step_i)
            c = REGIME_CENTERS[regime_seq[step_i]]
            drive_frame = _frame(t, c)
            rf_truth = twin_truth.step(drive_frame)
            measured_frame = _measured_frame_from_xhat(t, c, rf_truth.x_hat_array)
            rf_est = twin_est.step(measured_frame)
            if rf_est.status != 0:
                continue
            af = gate.update(rf_est)
            fb.ingest(rf_est, af)
            if step_i >= WARMUP_S and step_i % EMIT_EVERY_S == 0:
                rows.append(fb.compute())
        if rows:
            out.append((np.stack(rows), label, f"fault_{theta_idx}_{magnitude_slot}_{seed}"))
    return out


def generate_healthy_samples(flight_ids: List[str], shared_gate: IdentifiabilityGate, shared_twin: PhysicsTwin) -> List[Tuple[np.ndarray, str, str]]:
    """theta_hat is always NOMINAL for healthy flights, so the gate/jacobian
    is built once by the caller and reused across every flight -- only the
    twin's physical state is reset() per flight."""
    loader = RealFlightLoader(DATA_ROOT)
    out: List[Tuple[np.ndarray, str, str]] = []
    for flight_id in flight_ids:
        df = loader.load_flight(flight_id)
        if df is None or df.shape[0] < HEALTHY_WARMUP_S + HEALTHY_EMIT_EVERY_S:
            continue
        frames = loader.to_telemetry_frames(df.head(HEALTHY_MAX_ROWS), flight_id)
        if len(frames) < HEALTHY_WARMUP_S + HEALTHY_EMIT_EVERY_S:
            continue

        shared_twin.reset(frames[0])
        fb = StreamingFeatureBuilder(flight_id=f"healthy_{flight_id}")

        rows = []
        for step_i, frame in enumerate(frames):
            try:
                rf = shared_twin.step(frame)
            except Exception:
                break
            if rf.status != 0:
                continue
            af = shared_gate.update(rf)
            fb.ingest(rf, af)
            if step_i >= HEALTHY_WARMUP_S and step_i % HEALTHY_EMIT_EVERY_S == 0:
                rows.append(fb.compute())
        if rows:
            out.append((np.stack(rows), "HEALTHY", f"healthy_{flight_id}"))
    return out


def build_training_data(quick: bool = False) -> Dict[str, np.ndarray]:
    import polars as pl

    loader = RealFlightLoader(DATA_ROOT)
    manifest = loader.load_manifest()
    all_ids = [str(x) for x in manifest["flight_id"].to_list()]

    rng = np.random.default_rng(42)
    n_flights = 15 if quick else N_HEALTHY_FLIGHTS
    candidate_ids = list(rng.choice(all_ids, size=min(n_flights * 4, len(all_ids)), replace=False))

    print(f"[healthy] scanning up to {len(candidate_ids)} candidate flights for {n_flights} usable ones...")
    healthy_rows: List[Tuple[np.ndarray, str, str]] = []
    shared_healthy_twin = PhysicsTwin(model_dir=MODEL_DIR, dt=0.1)
    shared_healthy_twin.reset(_frame(0.0, REGIME_CENTERS[0]))
    shared_healthy_gate = IdentifiabilityGate(physics_twin=shared_healthy_twin, n_theta=15)
    shared_healthy_gate.theta_hat = NOMINAL.copy()
    t0 = time.time()
    for fid in candidate_ids:
        if len(healthy_rows) >= n_flights:
            break
        path = DATA_ROOT / "processed" / f"flight_{fid}.parquet"
        if not path.exists():
            continue
        got = generate_healthy_samples([fid], shared_healthy_gate, shared_healthy_twin)
        healthy_rows.extend(got)
        if len(healthy_rows) % 10 == 0:
            print(f"[healthy] {len(healthy_rows)}/{n_flights} flights so far ({time.time()-t0:.0f}s elapsed)")
    print(f"[healthy] {len(healthy_rows)} flights yielded samples ({time.time()-t0:.0f}s elapsed)")

    theta_indices = list(THETA_TO_CLASS.keys())
    if quick:
        theta_indices = theta_indices[:3]
        magnitude_slots = [1]
        seeds = REGIME_SCHEDULE_SEEDS[:1]
    else:
        magnitude_slots = [0, 1]
        seeds = REGIME_SCHEDULE_SEEDS

    fault_rows: List[Tuple[np.ndarray, str, str]] = []
    t0 = time.time()
    n_pairs = len(theta_indices) * len(magnitude_slots)
    pair_i = 0
    for theta_idx in theta_indices:
        for mag_slot in magnitude_slots:
            pair_i += 1
            trials = generate_fault_trials_for_theta(theta_idx, mag_slot, seeds)
            fault_rows.extend(trials)
            print(f"[fault] (theta_idx={theta_idx}, mag_slot={mag_slot}) pair {pair_i}/{n_pairs} "
                  f"-> {len(trials)} trials ({time.time()-t0:.0f}s elapsed)")

    X_list, y_list, g_list = [], [], []
    group_counter = 0
    for X, label, gid in healthy_rows + fault_rows:
        X_list.append(X)
        y_list.extend([label] * X.shape[0])
        g_list.extend([group_counter] * X.shape[0])
        group_counter += 1

    X_all = np.concatenate(X_list, axis=0).astype(np.float32)
    y_all = np.array([CLASS_ORDER.index(c) for c in y_list], dtype=np.int64)
    groups_all = np.array(g_list, dtype=np.int64)

    print(f"[data] total samples={X_all.shape[0]}, groups={group_counter}, "
          f"class counts={ {CLASS_ORDER[i]: int(np.sum(y_all == i)) for i in range(NUM_CLASSES)} }")

    return {"X": X_all, "y": y_all, "groups": groups_all}


def train_and_calibrate(data: Dict[str, np.ndarray]) -> MLModelArtifact:
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import GroupKFold

    X, y, groups = data["X"], data["y"], data["groups"]
    present_classes = sorted(set(y.tolist()))
    print(f"[train] classes present in data: {[CLASS_ORDER[c] for c in present_classes]}")

    gkf = GroupKFold(n_splits=min(5, len(set(groups.tolist()))))
    splits = list(gkf.split(X, y, groups))
    train_idx, holdout_idx = splits[0]
    # Further split holdout into a calibration fold and a final test fold,
    # still respecting group boundaries.
    holdout_groups = groups[holdout_idx]
    unique_holdout_groups = np.unique(holdout_groups)
    rng = np.random.default_rng(0)
    rng.shuffle(unique_holdout_groups)
    half = max(1, len(unique_holdout_groups) // 2)
    cal_groups = set(unique_holdout_groups[:half].tolist())
    test_groups = set(unique_holdout_groups[half:].tolist())
    cal_mask = np.isin(groups, list(cal_groups))
    test_mask = np.isin(groups, list(test_groups))

    X_train, y_train = X[train_idx], y[train_idx]
    X_cal, y_cal = X[cal_mask], y[cal_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"[split] train={X_train.shape[0]} cal={X_cal.shape[0]} test={X_test.shape[0]} "
          f"(group-disjoint GroupKFold, by flight/trial)")

    booster = lgb.train(
        params=dict(
            objective="multiclass",
            num_class=NUM_CLASSES,
            max_depth=6,
            num_leaves=31,
            learning_rate=0.05,
            min_data_in_leaf=5,
            verbose=-1,
        ),
        train_set=lgb.Dataset(X_train, label=y_train),
        num_boost_round=300,
    )

    calibrators: Dict[str, bytes] = {}
    if X_cal.shape[0] > 0:
        raw_cal = booster.predict(X_cal)
        for ci, cname in enumerate(CLASS_ORDER):
            binary_labels = (y_cal == ci).astype(np.float64)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            try:
                iso.fit(raw_cal[:, ci], binary_labels)
            except ValueError:
                continue
            calibrators[cname] = pickle.dumps(iso)
    print(f"[calibrate] fit isotonic regressors for {len(calibrators)}/{NUM_CLASSES} classes")

    novelty_blob = None
    healthy_mask_train = y_train == CLASS_ORDER.index("HEALTHY")
    if np.sum(healthy_mask_train) >= 30:
        from replan_to_learn.ml.novelty import NoveltyDetector
        detector = NoveltyDetector.fit(X_train[healthy_mask_train])
        novelty_blob = pickle.dumps(detector)
        print(f"[novelty] fit on {int(np.sum(healthy_mask_train))} healthy samples")

    artifact = MLModelArtifact(
        booster_model_str=booster.model_to_string(),
        class_order=CLASS_ORDER,
        feature_names=tuple(f"f{i}" for i in range(TOTAL_FEATURES)),
        calibrators=calibrators,
        novelty_detector=novelty_blob,
        training_provenance={
            "n_train": int(X_train.shape[0]),
            "n_cal": int(X_cal.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_groups_total": int(len(set(groups.tolist()))),
            "split": "GroupKFold(by flight/trial), never by sample",
        },
    )

    # Evaluate on the truly held-out test fold (real numbers only).
    from replan_to_learn.ml.classifier import MLClassifier
    clf = MLClassifier(artifact)
    if X_test.shape[0] > 0:
        probs_test = np.array([
            [clf.predict_proba_raw(X_test[i])[c] for c in CLASS_ORDER]
            for i in range(X_test.shape[0])
        ])
        pred_idx = np.argmax(probs_test, axis=1)
        acc = float(np.mean(pred_idx == y_test))
        confidences = np.max(probs_test, axis=1)
        correct = (pred_idx == y_test).astype(np.float64)
        ece_report = MLClassifier.compute_ece(confidences, correct)
        brier = MLClassifier.compute_brier(probs_test, y_test)
        print(f"[eval] held-out test accuracy={acc:.4f}  ECE={ece_report.ece:.4f} "
              f"(threshold 0.05, passed={ece_report.passed})  Brier={brier:.4f}")
        artifact.training_provenance["test_accuracy"] = acc
        artifact.training_provenance["test_ece"] = ece_report.ece
        artifact.training_provenance["test_ece_passed"] = bool(ece_report.passed)
        artifact.training_provenance["test_brier"] = brier
        artifact.training_provenance["test_reliability_data"] = ece_report.reliability_data
    else:
        print("[eval] WARNING: empty test fold, no held-out metrics computed")

    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Small smoke-test run, not for real metrics.")
    parser.add_argument("--use-cache", action="store_true", help="Reuse cached training features if present.")
    args = parser.parse_args()

    if args.use_cache and TRAINING_DATA_CACHE.exists():
        print(f"[data] loading cached training features from {TRAINING_DATA_CACHE}")
        npz = np.load(TRAINING_DATA_CACHE)
        data = {"X": npz["X"], "y": npz["y"], "groups": npz["groups"]}
    else:
        data = build_training_data(quick=args.quick)
        if not args.quick:
            np.savez_compressed(TRAINING_DATA_CACHE, **data)
            print(f"[data] cached training features to {TRAINING_DATA_CACHE}")

    artifact = train_and_calibrate(data)
    if not args.quick:
        MODEL_DIR.mkdir(exist_ok=True)
        artifact.save(str(ARTIFACT_PATH))
        print(f"[save] artifact written to {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
