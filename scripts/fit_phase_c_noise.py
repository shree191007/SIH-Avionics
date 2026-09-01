"""
Phase C noise-model recompute: per-channel, per-regime residual std from
real flight data, computed from RAW residuals (telemetry - prediction)
directly -- not from the twin's own already-normalized rf.z output, which
would divide by whatever sigma is already loaded and compound across runs.

Rerun this whenever a formula affecting one of the 9 residual channels
changes (most recently: _compute_power_balance_residual's propeller
constant self-calibration, changed from a single-sample ratio to a
running least-squares fit -- see physics_twin.py). Uses the real NTSB
915iS fixture plus the same 6 real NGAFID flights used elsewhere this
session (tests/real_data/build_ngafid_fleet_fixture.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "real_data"))

from ntsb_915is_loader import load_ntsb_915is_fixture  # noqa: E402
from real_data_loader import RealFlightLoader  # noqa: E402
from replan_to_learn.physics_twin.physics_twin import PhysicsTwin  # noqa: E402

MODEL_DIR = ROOT / "model"
DATA_ROOT = ROOT / "data"
N_CHANNELS = 9
N_REGIME = 12
CHANNEL_NAMES = ["z_EGT_1", "z_EGT_2", "z_EGT_3", "z_EGT_4", "z_CHT", "z_p_oil", "z_t_oil", "z_mdot_f", "z_N"]
FLIGHT_IDS = ["23517", "27142", "15379", "30799", "28763", "30252"]

samples_by_regime = {r: [[] for _ in range(N_CHANNELS)] for r in range(N_REGIME)}


def compute_raw_e(twin: PhysicsTwin, frame) -> np.ndarray:
    preds = twin._compute_predictions(frame)
    raw_e = np.zeros(9, dtype=np.float64)
    raw_e[0:4] = np.array(frame.egt) - np.array(preds["egt"])
    raw_e[4] = frame.cht - preds["cht"]
    raw_e[5] = frame.p_oil - preds["p_oil"]
    raw_e[6] = frame.t_oil - preds["t_oil"]
    raw_e[7] = frame.mdot_f - preds["mdot_f"]
    raw_e[8] = twin._compute_power_balance_residual(frame, preds["p_brake"])
    return raw_e


def process_flight(twin: PhysicsTwin, frames: list) -> None:
    twin.reset(frames[0])
    for frame in frames:
        try:
            rf = twin.step(frame)
        except Exception:
            continue
        if rf.status != 0:
            continue
        r = int(rf.regime)
        if r < 0 or r >= N_REGIME:
            continue
        raw_e = compute_raw_e(twin, frame)
        for c in range(N_CHANNELS):
            v = float(raw_e[c])
            if np.isfinite(v):
                samples_by_regime[r][c].append(v)


def main() -> None:
    twin = PhysicsTwin(model_dir=MODEL_DIR, dt=0.1)
    ntsb_frames = load_ntsb_915is_fixture()
    print(f"NTSB 915iS fixture: {len(ntsb_frames)} frames")
    process_flight(twin, ntsb_frames)

    loader = RealFlightLoader(DATA_ROOT)
    for fid in FLIGHT_IDS:
        df = loader.load_flight(fid)
        if df is None:
            continue
        frames = loader.to_telemetry_frames(df, fid)
        if len(frames) < 500:
            continue
        twin2 = PhysicsTwin(model_dir=MODEL_DIR, dt=0.1)
        process_flight(twin2, frames)
        print(f"  processed {fid}: {len(frames)} frames")

    sigma_grid = {ch: [None] * N_REGIME for ch in CHANNEL_NAMES}
    counts_grid = {r: [len(samples_by_regime[r][c]) for c in range(N_CHANNELS)] for r in range(N_REGIME)}
    MIN_SAMPLES = 20

    # A uniform FALLBACK_SIGMA=1.0 across all 9 channels is dimensionally
    # meaningless: real per-channel scales span many orders of magnitude
    # (mdot_f sigma ~0.0003-0.001 kg/s vs p_oil sigma ~10000-30000 Pa).
    # Confirmed this session as a real, exploitable bug: real flight data
    # is overwhelmingly concentrated in one regime bin (regime 3, see
    # sample_counts_by_regime below) -- when a DIFFERENT regime happens to
    # have just under MIN_SAMPLES real samples (observed: regime 0 with
    # only 2), every channel there fell back to sigma=1.0, which for
    # p_oil is ~10,000-30,000x too small. The gate's Fisher information is
    # proportional to 1/sigma^2, so this made theta_oilp look almost
    # perfectly determined at that one regime -- enough to occasionally
    # outscore a real theta_fric fault under realistic background noise
    # (empirically: theta_fric noise-robustness regressed from 26/30 to
    # 11/30, root-caused directly to this exact mechanism at regime 0).
    # Fix: fall back to the SAME channel's sigma from whichever regime has
    # the most real samples (a real, physically-grounded estimate of that
    # channel's noise scale) instead of an arbitrary dimensionless
    # constant -- honest uncertainty about an under-sampled regime, not
    # false precision.
    for c, ch in enumerate(CHANNEL_NAMES):
        counts_per_regime = [len(samples_by_regime[r][c]) for r in range(N_REGIME)]
        best_regime = int(np.argmax(counts_per_regime))
        if counts_per_regime[best_regime] >= MIN_SAMPLES:
            channel_fallback = float(np.std(samples_by_regime[best_regime][c], ddof=1))
        else:
            channel_fallback = 1.0  # genuinely no real data for this channel anywhere; last-resort constant
        for r in range(N_REGIME):
            vals = samples_by_regime[r][c]
            sigma_grid[ch][r] = float(np.std(vals, ddof=1)) if len(vals) >= MIN_SAMPLES else channel_fallback

    noise_model_path = MODEL_DIR / "noise_model.json"
    nm = json.loads(noise_model_path.read_text())
    nm["provenance"] = {
        "procedure": (
            "Phase C (raw-residual, re-run after propeller-constant self-calibration "
            "was changed from single-sample to running least-squares, and after fixing "
            "a real bug where an under-sampled regime fell back to a dimensionless "
            "sigma=1.0 for every channel regardless of that channel's real physical "
            "scale): sigma computed from raw_e = telemetry - prediction directly."
        ),
        "source": "[FIT]",
        "fit_sources": ["NTSB WPR22LA211"] + [f"NGAFID {fid}" for fid in FLIGHT_IDS],
        "min_samples_for_fit": MIN_SAMPLES,
        "fallback_sigma": "per-channel: that channel's real sigma from its best-sampled regime, not a shared constant (see procedure)",
        "sample_counts_by_regime": counts_grid,
    }
    nm["noise_std"]["values"] = sigma_grid
    noise_model_path.write_text(json.dumps(nm, indent=2))
    print(f"\nWrote {noise_model_path}")
    for r in range(N_REGIME):
        print(f"  regime {r}: {sum(counts_grid[r])} samples")
    print("\nz_mdot_f sigma per regime:", [round(v, 6) if isinstance(v, float) else v for v in sigma_grid["z_mdot_f"]])
    print("z_N sigma per regime:", [round(v, 4) if isinstance(v, float) else v for v in sigma_grid["z_N"]])


if __name__ == "__main__":
    main()
