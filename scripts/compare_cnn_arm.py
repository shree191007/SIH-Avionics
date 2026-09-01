"""
SIH26054 Replan to Learn: L5 1D-CNN comparison arm (spec Sec 1.3, P1/secondary).

"A 1D-CNN over the residual window is implemented as a comparison arm
only -- if it does not beat GBT by a margin that survives the variance
across seeds, GBT ships."

This is deliberately a MINIMAL, honest comparison, not a second production
model: a small 1D-CNN over the reshaped 172-feature vector (treated as a
single "window" of engineered features rather than a raw residual
time-series, since the streaming feature builder -- not a raw window
buffer -- is this layer's actual real-time representation; see
ml/features.py), trained on the SAME cached group-disjoint split as the
GBT (data/ml_training_features_v2.npz, produced by
scripts/train_ml_classifier.py), evaluated across multiple random seeds to
report mean +/- std accuracy, compared honestly against the GBT's own
held-out accuracy from model/ml_classifier.pkl's training_provenance.

Usage (after scripts/train_ml_classifier.py has produced the cached
features and artifact):
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
        scripts/compare_cnn_arm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from replan_to_learn.ml.datatypes import CLASS_ORDER, MLModelArtifact, NUM_CLASSES  # noqa: E402

DATA_CACHE = REPO_ROOT / "data" / "ml_training_features_v2.npz"
ARTIFACT_PATH = REPO_ROOT / "model" / "ml_classifier.pkl"
SEEDS = (0, 1, 2)


def main() -> None:
    if not DATA_CACHE.exists():
        print(f"No cached training features at {DATA_CACHE} -- run scripts/train_ml_classifier.py first.")
        sys.exit(1)

    import torch
    import torch.nn as nn
    from sklearn.model_selection import GroupKFold

    npz = np.load(DATA_CACHE)
    X, y, groups = npz["X"], npz["y"], npz["groups"]
    present = sorted(set(y.tolist()))
    print(f"[cnn] classes present: {[CLASS_ORDER[c] for c in present]}")

    gkf = GroupKFold(n_splits=min(5, len(set(groups.tolist()))))
    train_idx, test_idx = next(gkf.split(X, y, groups))
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    class TinyCNN(nn.Module):
        def __init__(self, n_features: int, n_classes: int):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
            self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(32, n_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = x.unsqueeze(1)  # (B, 1, F)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
            return self.fc(x)

    accuracies = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = TinyCNN(X.shape[1], NUM_CLASSES)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        Xt = torch.tensor(X_train_n, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.long)
        Xv = torch.tensor(X_test_n, dtype=torch.float32)
        yv = torch.tensor(y_test, dtype=torch.long)

        model.train()
        for epoch in range(30):
            opt.zero_grad()
            out = model(Xt)
            loss = loss_fn(out, yt)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(Xv).argmax(dim=1)
            acc = float((pred == yv).float().mean())
        accuracies.append(acc)
        print(f"[cnn] seed={seed} test_accuracy={acc:.4f}")

    cnn_mean, cnn_std = float(np.mean(accuracies)), float(np.std(accuracies))
    print(f"[cnn] mean={cnn_mean:.4f} std={cnn_std:.4f} across seeds {SEEDS}")

    if ARTIFACT_PATH.exists():
        artifact = MLModelArtifact.load(str(ARTIFACT_PATH))
        gbt_acc = artifact.training_provenance.get("test_accuracy")
        if gbt_acc is not None:
            print(f"[compare] GBT held-out accuracy={gbt_acc:.4f} vs CNN mean={cnn_mean:.4f} (+/-{cnn_std:.4f})")
            margin = cnn_mean - gbt_acc
            if margin > cnn_std:
                print(f"[compare] CNN beats GBT by {margin:.4f}, which survives its own seed variance ({cnn_std:.4f}) -- consider CNN.")
            else:
                print(f"[compare] CNN does NOT beat GBT by a margin surviving seed variance -- GBT ships (per spec Sec 1.3).")
        else:
            print("[compare] GBT artifact has no test_accuracy in provenance -- cannot compare.")
    else:
        print(f"[compare] No GBT artifact at {ARTIFACT_PATH} to compare against.")


if __name__ == "__main__":
    main()
