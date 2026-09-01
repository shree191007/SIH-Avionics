"""
SIH26054 Replan to Learn: L6 Particle-Filter RUL.
Implements Section 2 of 04_ml_rul_mission_probe.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from replan_to_learn.rul.datatypes import FailureThreshold, RULConfig, RULReport


@dataclass
class _Particle:
    theta_j: float
    rho: float
    weight: float


class RULParticleFilter:
    """
    Particle filter for remaining useful life estimation.

    State per particle: d_k = [theta_j, rho]
    - theta_j: health parameter value
    - rho: drift rate (per flight hour)

    Mean-reverting drift prevents runaway extrapolation.
    Freeze policy: no update while verdict == AMBIGUOUS.
    """

    def __init__(
        self,
        theta_index: int,
        theta_name: str,
        config: Optional[RULConfig] = None,
        threshold: Optional[FailureThreshold] = None,
    ) -> None:
        self.theta_index = theta_index
        self.theta_name = theta_name
        self.config = config or RULConfig()
        self.threshold = threshold

        self.particles: List[_Particle] = []
        self.theta_hat: float = 1.0
        self.rho_hat: float = 0.0
        self._initialized = False
        self._frozen = False
        self._mission_end_time_s: float = 0.0

    def initialize(
        self,
        theta_init: float,
        P_theta_jj: float,
        rho_init: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        n = self.config.n_particles
        self.particles = []
        for _ in range(n):
            self.particles.append(_Particle(
                theta_j=float(rng.normal(theta_init, math.sqrt(max(P_theta_jj, 1e-9)))),
                rho=float(rng.normal(rho_init, 1e-4)),
                weight=1.0 / n,
            ))
        self.theta_hat = float(theta_init)
        self.rho_hat = float(rho_init)
        self._initialized = True
        self._frozen = False

    def freeze(self) -> None:
        self._frozen = True

    def unfreeze(self) -> None:
        self._frozen = False

    def set_mission_end(self, mission_end_time_s: float) -> None:
        self._mission_end_time_s = float(mission_end_time_s)

    def _propose_rho(self, rho: float, rng: np.random.Generator) -> float:
        a = self.config.rho_mean_reversion
        return a * rho + (1.0 - a) * self.rho_hat + float(rng.normal(0, self.config.rho_drift_std))

    def _propagate(self, dt_fh: float, rng: np.random.Generator) -> None:
        theta_drift_std = self.config.theta_drift_std * math.sqrt(max(dt_fh, 1e-6))
        for p in self.particles:
            p.rho = self._propose_rho(p.rho, rng)
            p.theta_j = p.theta_j + p.rho * dt_fh + float(rng.normal(0, theta_drift_std))

    def _update(self, theta_obs: float, P_obs: float) -> None:
        if P_obs <= 0:
            P_obs = 1e-6
        for p in self.particles:
            p.weight *= math.exp(-0.5 * ((p.theta_j - theta_obs) ** 2) / P_obs)
        total_w = sum(p.weight for p in self.particles)
        if total_w <= 0:
            for p in self.particles:
                p.weight = 1.0 / len(self.particles)
        else:
            for p in self.particles:
                p.weight /= total_w

    def _resample(self, rng: np.random.Generator) -> None:
        n = len(self.particles)
        weights = np.array([p.weight for p in self.particles])
        ess = 1.0 / np.sum(weights ** 2)
        if ess < self.config.ess_threshold_frac * n:
            cumsum = np.cumsum(weights)
            cumsum[-1] = 1.0
            idx = np.searchsorted(cumsum, rng.random(n))
            new_particles = []
            for i in idx:
                i = int(np.clip(i, 0, n - 1))
                p = self.particles[i]
                new_particles.append(_Particle(
                    theta_j=p.theta_j + float(rng.normal(0, self.config.roughening_eps * 0.1)),
                    rho=p.rho + float(rng.normal(0, self.config.roughening_eps * 0.01)),
                    weight=1.0 / n,
                ))
            self.particles = new_particles

    def step(
        self,
        theta_obs: Optional[float],
        P_obs: float,
        dt_fh: float,
        verdict: str = "NAMED",
    ) -> None:
        if not self._initialized:
            raise RuntimeError("Particle filter not initialized")

        rng = np.random.default_rng()

        if self.config.freeze_on_ambiguous and verdict == "AMBIGUOUS":
            self._frozen = True
        else:
            self._frozen = False

        self._propagate(dt_fh, rng)

        if not self._frozen and theta_obs is not None and not math.isnan(theta_obs):
            self._update(theta_obs, P_obs)
            self._resample(rng)

        weights = np.array([p.weight for p in self.particles])
        self.theta_hat = float(np.sum(weights * np.array([p.theta_j for p in self.particles])))
        self.rho_hat = float(np.sum(weights * np.array([p.rho for p in self.particles])))

    def estimate_rul(self) -> RULReport:
        if self.threshold is None:
            raise RuntimeError("Failure threshold not set")

        particles = self.particles
        n = len(particles)
        if n == 0:
            return RULReport(
                p05=float('nan'),
                p50=float('nan'),
                p95=float('nan'),
                p_fail_before_mission_end=1.0,
                mission_end_time_s=self._mission_end_time_s,
                theta_index=self.theta_index,
                theta_name=self.theta_name,
                current_theta=float(self.theta_hat),
                threshold=self.threshold.threshold_value,
                particles_converged=False,
            )

        first_passage = []
        rng = np.random.default_rng(42)
        # dt_fh=0.1/max_steps=100000 (a 10,000 flight-hour search horizon,
        # ~500 particles x 9 RUL filters x 100000-step while-loop, RE-RUN
        # on every single process_residual_frame() call) was a real,
        # severe performance bug: for a healthy/near-nominal particle
        # (rho close to 0, no real degradation trend), theta_j barely
        # moves and the loop runs to the full max_steps before giving up
        # -- confirmed via a live stack sample of a hung API request,
        # which showed the overwhelming majority of wall-clock time spent
        # inside this loop's rng.normal() calls. No real mission-planning
        # decision needs a 10,000-flight-hour horizon (Stage3Config's
        # default mission_end_time_s is 3600s = 1 hour; even a generous
        # maintenance-planning horizon is measured in hundreds of flight
        # hours, not thousands). A coarser dt_fh for this specific Monte
        # Carlo horizon search (not the real state propagation step,
        # which stays at its own real per-call dt_fh) covers a still-
        # generous 2000 flight-hour horizon in 50x fewer iterations.
        dt_fh = 1.0
        max_steps = 2000
        # direction="below" (the historical hardcoded assumption, still
        # correct for every efficiency-loss parameter): failure = theta
        # DECREASING through threshold_value, so keep simulating while
        # theta_j is still ABOVE it. direction="above" (theta_fric only):
        # failure = theta INCREASING through threshold_value (threshold
        # ~1.53, above the 1.0 nominal starting point) -- a healthy
        # theta_fric particle starts BELOW threshold, so the loop must run
        # while theta_j is still BELOW it, not above. The un-directional
        # version of this check (hardcoded "below" always, before this
        # fix) made a perfectly healthy theta_fric report first-passage-
        # time=0 ("already failed") immediately, since 1.0 > 1.53 is
        # already False on the very first check.
        is_below = getattr(self.threshold, "direction", "below") != "above"
        # Vectorized across all particles simultaneously (was: a Python
        # `for p in particles:` loop with a per-particle scalar
        # rng.normal() call each step -- up to 500 particles x 2000 steps
        # x 9 RUL filters = up to 9,000,000 individual scalar RNG calls,
        # each paying full Python/numpy call overhead). Drawing one batch
        # of n_particles random increments per step instead collapses
        # that to at most max_steps (2000) batched calls per filter.
        # Only particles still short of the threshold ("active") are
        # updated each step; a particle's first-passage time is recorded
        # once when it crosses, and then it's excluded from further
        # updates -- reproducing the original per-particle stopping
        # semantics without the per-particle loop.
        threshold_value = self.threshold.threshold_value
        theta_arr = np.array([p.theta_j for p in particles], dtype=np.float64)
        rho_arr = np.array([p.rho for p in particles], dtype=np.float64)
        a_coef = self.config.rho_mean_reversion
        rho_drift_std = self.config.rho_drift_std
        rho_hat = self.rho_hat

        active = (theta_arr > threshold_value) if is_below else (theta_arr < threshold_value)
        first_passage_arr = np.where(active, np.inf, 0.0)
        step = 0
        while step < max_steps and np.any(active):
            noise = rng.normal(0.0, rho_drift_std, size=n)
            new_rho = a_coef * rho_arr + (1.0 - a_coef) * rho_hat + noise
            new_theta = theta_arr + new_rho * dt_fh
            rho_arr = np.where(active, new_rho, rho_arr)
            theta_arr = np.where(active, new_theta, theta_arr)
            step += 1
            crossed = active & ((theta_arr <= threshold_value) if is_below else (theta_arr >= threshold_value))
            first_passage_arr = np.where(crossed, step * dt_fh, first_passage_arr)
            active = active & ~crossed
        first_passage = first_passage_arr.tolist()

        first_passage_arr = np.array(first_passage, dtype=np.float64)
        finite_mask = np.isfinite(first_passage_arr)
        if not np.any(finite_mask):
            return RULReport(
                p05=float('inf'),
                p50=float('inf'),
                p95=float('inf'),
                p_fail_before_mission_end=0.0,
                mission_end_time_s=self._mission_end_time_s,
                theta_index=self.theta_index,
                theta_name=self.theta_name,
                current_theta=float(self.theta_hat),
                threshold=self.threshold.threshold_value,
                particles_converged=True,
            )

        p05 = float(np.percentile(first_passage_arr[finite_mask], 5))
        p50 = float(np.percentile(first_passage_arr[finite_mask], 50))
        p95 = float(np.percentile(first_passage_arr[finite_mask], 95))

        mission_end = self._mission_end_time_s
        p_fail = float(np.mean(first_passage_arr[finite_mask] < mission_end)) if mission_end > 0 else 0.0

        return RULReport(
            p05=p05,
            p50=p50,
            p95=p95,
            p_fail_before_mission_end=p_fail,
            mission_end_time_s=mission_end,
            theta_index=self.theta_index,
            theta_name=self.theta_name,
            current_theta=float(self.theta_hat),
            threshold=self.threshold.threshold_value,
            particles_converged=True,
        )
