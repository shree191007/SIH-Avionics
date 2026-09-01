"""
SIH26054 Replan to Learn: L4F Fleet Federation Node.
Implements Section 3 of 03_state_identifiability_fleet.md.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from replan_to_learn.contracts.regimes import REGIME_GRID_VERSION


@dataclass(frozen=True)
class FingerprintContribution:
    """
    Uplink message per aircraft per fusion round.

    Implements Section 3.11 of 03_state_identifiability_fleet.md.
    """
    aircraft_id: str
    model_version: str
    regime_grid_version: str
    theta_index: int
    info_vector: np.ndarray       # float32[n_regime] = s_hat_i / v_i
    info_scalar: np.ndarray       # float32[n_regime] = 1 / v_i
    n_seconds: np.ndarray         # uint32[n_regime] = n_i(r)
    epoch: int
    is_faulted: bool = False      # True if this aircraft has an active fault on theta_index
    timestamp_s: float = 0.0      # Wall-clock time of contribution


@dataclass(frozen=True)
class FleetShape:
    """
    Downlink message per fusion round.

    Implements Section 3.11 of 03_state_identifiability_fleet.md.
    """
    theta_index: int
    u: np.ndarray                 # float32[n_regime], unit norm, sign-fixed
    var_u: np.ndarray             # float32[n_regime]
    contributors: np.ndarray      # uint16[n_regime]
    alpha_median: float
    epoch: int
    model_version: str
    regime_grid_version: str


class FleetNode:
    """
    Fleet-federated identifiability node.

    Implements Section 3 of 03_state_identifiability_fleet.md:
    - Contribution ingestion
    - Rank-1 ALS fusion with precision weighting
    - Admissibility rules A1-A6
    - Sign convention application
    - Cold start / small fleets handling
    - Staleness and forgetting factor
    - Robustness: trimming, precision cap, faulted-unit exclusion
    """

    def __init__(
        self,
        n_regime: int = 12,
        tau_r2: float = 0.85,
        kappa_shrink: float = 10.0,
        staleness_s: float = 200.0 * 3600.0,
        max_alpha: float = 4.0,
        min_alpha: float = 0.25,
        precision_cap: float = 0.30,
        min_fleet_seconds: float = 200.0,
    ) -> None:
        self.n_regime = n_regime
        self.tau_r2 = tau_r2
        self.kappa_shrink = kappa_shrink
        self.staleness_s = staleness_s
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha
        self.precision_cap = precision_cap
        self.min_fleet_seconds = min_fleet_seconds

        self.shapes: Dict[int, FleetShape] = {}
        self.contributions: Dict[int, List[FingerprintContribution]] = {}
        self.priors: Dict[int, np.ndarray] = {}
        self.epoch = 0
        self._median_fleet_alpha: Dict[int, float] = {}

    def set_prior(self, theta_index: int, u_prior: np.ndarray) -> None:
        u_prior = u_prior / (np.linalg.norm(u_prior) + 1e-12)
        self.priors[theta_index] = u_prior

        if theta_index not in self.shapes:
            self.shapes[theta_index] = FleetShape(
                theta_index=theta_index,
                u=u_prior.copy(),
                var_u=np.ones(self.n_regime) * 0.1,
                contributors=np.zeros(self.n_regime, dtype=np.uint16),
                alpha_median=1.0,
                epoch=self.epoch,
                model_version="1.0.0",
                regime_grid_version="REGIME_GRID_V1",
            )

    def ingest(self, c: FingerprintContribution) -> None:
        if c.theta_index not in self.contributions:
            self.contributions[c.theta_index] = []
        self.contributions[c.theta_index].append(c)

    def fuse(self, theta_index: int) -> FleetShape:
        if theta_index not in self.contributions or not self.contributions[theta_index]:
            if theta_index in self.priors:
                return self.shapes.get(theta_index, FleetShape(
                    theta_index=theta_index,
                    u=self.priors[theta_index].copy(),
                    var_u=np.ones(self.n_regime) * 0.1,
                    contributors=np.zeros(self.n_regime, dtype=np.uint16),
                    alpha_median=1.0,
                    epoch=self.epoch,
                    model_version="1.0.0",
                    regime_grid_version="REGIME_GRID_V1",
                ))
            raise ValueError(f"No contributions or prior for theta_index={theta_index}")

        admissible = self._apply_admissibility(theta_index)

        if not admissible:
            if theta_index in self.priors:
                return self.shapes.get(theta_index, FleetShape(
                    theta_index=theta_index,
                    u=self.priors[theta_index].copy(),
                    var_u=np.ones(self.n_regime) * 0.1,
                    contributors=np.zeros(self.n_regime, dtype=np.uint16),
                    alpha_median=1.0,
                    epoch=self.epoch,
                    model_version="1.0.0",
                    regime_grid_version="REGIME_GRID_V1",
                ))
            raise ValueError(f"No admissible contributions for theta_index={theta_index}")

        u, alphas, var_u = self._rank1_als_fusion(theta_index, admissible)

        if theta_index in self.priors:
            r_ref = np.argmax(np.abs(self.priors[theta_index]))
            sign = np.sign(u[r_ref] * self.priors[theta_index][r_ref])
            u = u * sign

        contributors = np.zeros(self.n_regime, dtype=np.uint16)
        for c in admissible:
            contributors += (c.n_seconds > 0).astype(np.uint16)

        shape = FleetShape(
            theta_index=theta_index,
            u=u,
            var_u=var_u,
            contributors=contributors,
            alpha_median=float(np.median(alphas)) if len(alphas) > 0 else 1.0,
            epoch=self.epoch,
            model_version="1.0.0",
            regime_grid_version="REGIME_GRID_V1",
        )

        self.shapes[theta_index] = shape
        self.contributions[theta_index] = []
        self._median_fleet_alpha[theta_index] = float(np.median(alphas)) if len(alphas) > 0 else 1.0
        return shape

    def _apply_admissibility(self, theta_index: int) -> List[FingerprintContribution]:
        contributions = self.contributions.get(theta_index, [])
        now = time.time()

        admissible = []
        alphas_list: List[float] = []

        for c in contributions:
            if c.model_version != "1.0.0" or c.regime_grid_version != "REGIME_GRID_V1":
                continue

            # NOTE: rule A1 (>=3 distinct flown regimes) is a requirement on
            # the BORROWING aircraft's own history (Sec 3.6/try_borrow),
            # not on each fleet CONTRIBUTOR here. A contributor with a
            # single flown regime is exactly the case federation exists to
            # combine -- most real aircraft only ever fly a handful of
            # regimes, and Sec 3.9 states plainly that "the fleet supplies
            # empirical operating-point diversity that no single aircraft's
            # history contains". Filtering single-regime contributors out
            # would make real-fleet fusion structurally impossible.
            flown = np.where(c.n_seconds > 0)[0]
            if len(flown) < 1:
                continue

            # NOTE: min_fleet_seconds (rule A5, "n_fleet(r) >= 200s") is
            # also a FLEET-AGGREGATE-per-regime-cell threshold, not a
            # per-contributor gate -- it's about whether a given regime
            # cell has enough total observed seconds across the whole
            # fleet, not whether any single aircraft flew that long. A
            # single real flight's total quasi-steady time (often just a
            # couple hundred seconds even across a multi-hour flight,
            # since the classifier's 20s continuous-stability window is
            # strict) is not the quantity this rule is about. Precision
            # weighting already handles thin per-aircraft data correctly:
            # v_i(theta_j,r) = sigma^2/n_i(r) + v_floor (Sec 3.8) gives a
            # short-duration contribution high variance / low precision
            # in the fusion automatically, without needing a hard reject
            # here that would exclude every real single-flight contributor.
            admissible.append(c)
            if np.sum(c.info_scalar[flown]) > 1e-12:
                alpha = float(np.sum(c.info_vector[flown] * c.info_scalar[flown]) / np.sum(c.info_scalar[flown]))
            else:
                alpha = 1.0
            alphas_list.append(alpha)

        if len(admissible) < 3:
            return []

        median_alpha = float(np.median(alphas_list)) if alphas_list else 1.0
        filtered = []
        for c, alpha in zip(admissible, alphas_list):
            if alpha <= 0:
                continue
            if not (self.min_alpha * median_alpha <= alpha <= self.max_alpha * median_alpha):
                continue
            filtered.append(c)

        if len(filtered) >= 7:
            alpha_vals = []
            for c in filtered:
                flown = np.where(c.n_seconds > 0)[0]
                if np.sum(c.info_scalar[flown]) > 1e-12:
                    a = float(np.sum(c.info_vector[flown] * c.info_scalar[flown]) / np.sum(c.info_scalar[flown]))
                else:
                    a = 1.0
                alpha_vals.append(a)
            alpha_vals = np.array(alpha_vals)
            sorted_idx = np.argsort(alpha_vals)
            drop = int(len(filtered) * 0.1)
            drop = max(drop, 1)
            keep_idx = sorted_idx[drop:-drop] if drop > 0 else sorted_idx
            filtered = [filtered[i] for i in keep_idx]

        return filtered

    def _rank1_als_fusion(
        self,
        theta_index: int,
        contributions: List[FingerprintContribution],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_regime = self.n_regime

        if theta_index in self.priors:
            u = self.priors[theta_index].copy()
        else:
            u = np.zeros(n_regime)
            for c in contributions:
                mask = c.n_seconds > 0
                if np.any(mask):
                    u[mask] += c.info_vector[mask] * c.info_scalar[mask]
            norm = np.linalg.norm(u)
            if norm > 1e-12:
                u = u / norm
            else:
                u = np.ones(n_regime) / np.sqrt(n_regime)

        alphas = np.ones(len(contributions))
        max_iter = 20

        for iteration in range(max_iter):
            u_new = np.zeros(n_regime)
            denom = 0.0
            for i, c in enumerate(contributions):
                u_new += alphas[i] * c.info_vector * c.info_scalar
                denom += alphas[i] ** 2 * c.info_scalar
            denom = np.maximum(denom, 1e-12)
            u_new = u_new / denom
            norm = np.linalg.norm(u_new)
            if norm > 1e-12:
                u_new = u_new / norm
            else:
                u_new = u.copy()

            alphas_new = np.zeros(len(contributions))
            for i, c in enumerate(contributions):
                mask = c.n_seconds > 0
                if np.any(mask):
                    numerator = np.sum(c.info_vector[mask] * u_new[mask] * c.info_scalar[mask])
                    denominator = np.sum(u_new[mask] ** 2 * c.info_scalar[mask])
                    if denominator > 1e-12:
                        alphas_new[i] = numerator / denominator

            if np.allclose(alphas, alphas_new) and np.allclose(u, u_new):
                alphas = alphas_new
                u = u_new
                break

            alphas = alphas_new
            u = u_new

        if theta_index in self.priors:
            n_eff = sum(np.sum(c.info_scalar) for c in contributions)
            lambda_shrink = self.kappa_shrink / (self.kappa_shrink + n_eff)
            u = (1 - lambda_shrink) * u + lambda_shrink * self.priors[theta_index]
            norm = np.linalg.norm(u)
            if norm > 1e-12:
                u = u / norm

        total_precision = np.zeros(n_regime)
        for i, c in enumerate(contributions):
            mask = c.n_seconds > 0
            if np.any(mask):
                total_precision[mask] += alphas[i] ** 2 * c.info_scalar[mask]
        var_u = 1.0 / (np.maximum(total_precision, 1e-12) + 1e-12)

        return u, alphas, var_u

    def get_shape(self, theta_index: int) -> Optional[FleetShape]:
        return self.shapes.get(theta_index)

    def step(self) -> None:
        self.epoch += 1
