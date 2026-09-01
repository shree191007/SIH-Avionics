"""
SIH26054 Replan to Learn: Unscented Kalman Filter for joint state/parameter estimation.
Implements L3 from 03_state_identifiability_fleet.md Section 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from replan_to_learn.contracts.residuals import ResidualFrame


@dataclass
class UKFConfig:
    """Configuration for Unscented Kalman Filter."""
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0
    use_square_root: bool = True
    theta_drift_std: float = 1e-4
    constraint_tol: float = 1e-6
    freeze_regimes: tuple = (13, 14)
    # Robustness: gate per-channel normalized innovation to at most this many
    # measurement-noise standard deviations before applying the Kalman gain.
    # Without this, a single badly-mismatched measurement (e.g. a physics
    # model evaluated far outside its calibrated envelope) can inject a huge
    # state jump, which on the next predict() step gets evaluated by a
    # nonlinear process model far outside any sane operating region -- the
    # resulting NaN/Inf covariance is what actually produces the "Matrix is
    # not positive definite" Cholesky failures seen on real, badly-fit data.
    # Gating trades a small bias (clipped information) for boundedness.
    innovation_gate_sigma: float = 8.0
    enable_innovation_gating: bool = True
    # Sanity bound on |x|: NaN/Inf checks alone miss a slow explosion where
    # every individual step is finite but the magnitude compounds far past
    # anything physically meaningful (state is Pa/K/rad-s scale, theta is
    # O(1)) before finally overflowing. Anything past this is treated the
    # same as non-finite: roll back and inflate rather than trust it.
    state_sanity_bound: float = 1.0e7


class UKF:
    """
    Unscented Kalman Filter for joint state/parameter estimation.

    Implements the contract from Section 1 of 03_state_identifiability_fleet.md:
    - Augmented state: [x; theta]
    - Theta as slow random walk
    - Square-root formulation for numerical stability
    - Constraint handling by projection with covariance inflation
    - Freeze policy for TRANSIENT and MODEL_SATURATED samples
    """

    def __init__(
        self,
        n_state: int,
        n_theta: int,
        n_meas: int,
        config: Optional[UKFConfig] = None,
    ) -> None:
        self.n_state = n_state
        self.n_theta = n_theta
        self.n_aug = n_state + n_theta
        self.n_meas = n_meas
        self.config = config or UKFConfig()

        self.x = np.zeros(self.n_aug)
        self.P = np.eye(self.n_aug) * 1.0

        if self.config.use_square_root:
            self.S = np.linalg.cholesky(self.P)
        else:
            self.S = None

        self.Q = np.eye(self.n_aug) * 1e-4
        self.Q[n_state:, n_state:] *= self.config.theta_drift_std ** 2

        self.R = np.eye(self.n_meas) * 1.0
        self._regime_R: Optional[np.ndarray] = None

        self._theta_lower, self._theta_upper = self._build_theta_bounds(n_theta)

        self._compute_weights()
        self._frozen = False

    @staticmethod
    def _build_theta_bounds(n_theta: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Per-parameter admissible ranges for the theta vector.

        For the real 15-element physical layout (theta_vol, theta_comb,
        theta_cool, theta_inj[1..4], theta_oilp, theta_fric, then 6 additive
        sensor biases), use the actual admissible ranges from
        02_physics_twin_residuals.md Sec 2.2 for the multiplicative
        parameters, and a wide symmetric range for the additive biases.
        Clamping bias terms to the same [0.75, 1.3] range as the
        multiplicative parameters (the previous behavior) makes it
        mathematically impossible for a bias -- whose true nominal is 0.0 --
        to ever be estimated correctly, since the clip range doesn't even
        contain zero.

        For any other n_theta (tests/callers using a reduced fingerprint
        dimension that doesn't correspond to this physical layout), fall
        back to the old uniform [0.75, 1.3] range applied to every element,
        which is a reasonable generic default when the parameter identities
        aren't known.
        """
        if n_theta == 15:
            lower = np.array([0.80, 0.80, 0.70, 0.80, 0.80, 0.80, 0.80, 0.75, 0.95,
                               -50.0, -50.0, -50.0, -50.0, -50.0, -50.0])
            upper = np.array([1.05, 1.05, 1.10, 1.10, 1.10, 1.10, 1.10, 1.05, 1.30,
                               50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
            return lower, upper
        return np.full(n_theta, 0.75), np.full(n_theta, 1.3)

    def set_regime_conditioned_R(self, sigma: np.ndarray) -> None:
        self._regime_R = np.diag(sigma ** 2)

    def freeze(self) -> None:
        self._frozen = True

    def unfreeze(self) -> None:
        self._frozen = False

    def _compute_weights(self) -> None:
        n = self.n_aug
        alpha = self.config.alpha
        beta = self.config.beta
        kappa = self.config.kappa

        lambda_ = alpha ** 2 * (n + kappa) - n

        self.W_m = np.zeros(2 * n + 1)
        self.W_c = np.zeros(2 * n + 1)

        self.W_m[0] = lambda_ / (n + lambda_)
        self.W_c[0] = lambda_ / (n + lambda_) + (1 - alpha ** 2 + beta)

        for i in range(1, 2 * n + 1):
            self.W_m[i] = 1.0 / (2 * (n + lambda_))
            self.W_c[i] = 1.0 / (2 * (n + lambda_))

    def _sigma_points(self, x: np.ndarray, S: Optional[np.ndarray] = None) -> np.ndarray:
        n = self.n_aug
        alpha = self.config.alpha
        kappa = self.config.kappa
        lambda_ = alpha ** 2 * (n + kappa) - n

        if S is None:
            P = self.P
            L = np.linalg.cholesky(P)
        else:
            L = S

        sigma = np.zeros((2 * n + 1, n))
        sigma[0] = x

        for i in range(n):
            offset = np.sqrt(n + lambda_) * L[:, i]
            sigma[i + 1] = x + offset
            sigma[i + n + 1] = x - offset

        return sigma

    def _project_constraints(self, theta: np.ndarray) -> np.ndarray:
        return np.clip(theta, self._theta_lower, self._theta_upper)

    def _is_sane(self, *arrays: np.ndarray) -> bool:
        """True iff every array is finite AND within state_sanity_bound magnitude."""
        for a in arrays:
            if a is None:
                continue
            if not np.all(np.isfinite(a)):
                return False
            if np.max(np.abs(a)) > self.config.state_sanity_bound:
                return False
        return True

    def _gate_state_update(self, dx: np.ndarray, P_prior: np.ndarray) -> np.ndarray:
        """
        Scale down the proposed state update K @ innovation if it would move
        any component by more than innovation_gate_sigma standard deviations
        of the filter's OWN pre-update uncertainty (diag(P_prior)) -- not of
        R. Gating against R directly is fragile: R is very often a rough
        placeholder (e.g. R = I, mixing Pa/K/kg-s scales with no real
        per-channel calibration), so a fixed absolute bound on the raw
        innovation either does nothing on badly-scaled R or clips away
        genuinely small, correct corrections. Gating the state jump against
        the filter's own belief in itself is self-consistent regardless of
        how R happens to be scaled, and still catches the actual failure
        mode: a state jump many orders of magnitude past what the filter
        currently believes plausible, which is what pushes the next
        nonlinear predict() into NaN territory.
        Uniform (not per-component) scaling preserves the update direction.
        """
        if not self.config.enable_innovation_gating:
            return dx
        std = np.sqrt(np.maximum(np.diag(P_prior), 1e-12))
        bound = self.config.innovation_gate_sigma * std
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.max(np.abs(dx) / bound)
        if np.isfinite(ratio) and ratio > 1.0:
            dx = dx / ratio
        return dx

    def _finite_or_rollback(
        self, x_prior: np.ndarray, P_prior: np.ndarray, S_prior: Optional[np.ndarray], innovation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Last-resort guard: if the update produced a non-finite state or
        covariance (e.g. because the process/measurement model was
        evaluated far outside its calibrated envelope), discard it and
        return the pre-update estimate with covariance inflated -- the same
        "missed information, stay honest about growing uncertainty" policy
        already used for frozen/transient samples -- instead of propagating
        NaN into a Cholesky factorization and crashing.
        """
        if self._is_sane(self.x, self.P, self.S if S_prior is not None else None):
            return self.x.copy(), self.P.copy(), innovation.copy()

        self.x = x_prior.copy()
        self.P = P_prior.copy()
        if S_prior is not None:
            self.S = S_prior.copy()
        self._inflate_covariance()
        return self.x.copy(), self.P.copy(), np.zeros_like(innovation)

    def _inflate_covariance(self) -> None:
        self.P *= 1.01
        self.P += 1e-6 * np.eye(self.n_aug)
        if self.config.use_square_root:
            try:
                self.S = np.linalg.cholesky(self.P)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(self.P)
                eigvals = np.maximum(eigvals, 1e-6)
                self.S = eigvecs @ np.diag(np.sqrt(eigvals))

    def predict(self, f: Any, u: Any, theta: Optional[np.ndarray] = None) -> None:
        if self.config.use_square_root:
            self._predict_square_root(f, u)
        else:
            self._predict_standard(f, u)

    def _predict_standard(self, f: Any, u: Any) -> None:
        """Standard UKF prediction."""
        x_prior, P_prior = self.x.copy(), self.P.copy()

        sigma = self._sigma_points(self.x)
        sigma_pred = np.zeros_like(sigma)
        for i, s in enumerate(sigma):
            x_next = f(s[:self.n_state], u, s[self.n_state:])
            theta_next = s[self.n_state:]
            sigma_pred[i] = np.concatenate([x_next, theta_next])

        self.x = self.W_m @ sigma_pred
        self.P = self.W_c[:, None] * (sigma_pred - self.x).T @ (sigma_pred - self.x) + self.Q
        self.P += 1e-12 * np.eye(self.n_aug)

        self.x[self.n_state:] = self._project_constraints(self.x[self.n_state:])

        if not self._is_sane(self.x, self.P):
            self.x, self.P = x_prior, P_prior
            self._inflate_covariance()

    def _predict_square_root(self, f: Any, u: Any) -> None:
        """Square-root UKF prediction."""
        x_prior, P_prior, S_prior = self.x.copy(), self.P.copy(), self.S.copy()

        sigma = self._sigma_points(self.x, self.S)
        sigma_pred = np.zeros_like(sigma)
        for i, s in enumerate(sigma):
            x_next = f(s[:self.n_state], u, s[self.n_state:])
            theta_next = s[self.n_state:]
            sigma_pred[i] = np.concatenate([x_next, theta_next])

        self.x = self.W_m @ sigma_pred
        diff = sigma_pred - self.x

        P_pred = diff.T @ (self.W_c[:, None] * diff) + self.Q
        P_pred += 1e-3 * np.eye(self.n_aug)
        self.P = P_pred
        try:
            self.S = np.linalg.cholesky(P_pred)
        except np.linalg.LinAlgError:
            try:
                eigvals, eigvecs = np.linalg.eigh(P_pred)
                eigvals = np.maximum(eigvals, 1e-3)
                self.S = eigvecs @ np.diag(np.sqrt(eigvals))
            except (np.linalg.LinAlgError, ValueError):
                # P_pred itself is too corrupted (e.g. NaN) for either
                # factorization to work at all -- don't try a third
                # factorization on the same bad input, just fall through to
                # the sanity check below, which will roll back to the prior
                # (known-good) state/covariance instead of crashing.
                self.S = np.full((self.n_aug, self.n_aug), np.nan)

        self.x[self.n_state:] = self._project_constraints(self.x[self.n_state:])

        if not self._is_sane(self.x, self.P, self.S):
            self.x, self.P, self.S = x_prior, P_prior, S_prior
            self._inflate_covariance()

    def update(self, h: Any, z: np.ndarray, R: Optional[np.ndarray] = None, regime: Optional[int] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._frozen:
            innovation = np.zeros(self.n_meas)
            return self.x.copy(), self.P.copy(), innovation

        R = R if R is not None else self.R
        if regime is not None and self._regime_R is not None:
            R = self._regime_R

        if self.config.use_square_root:
            return self._update_square_root(h, z, R)
        else:
            return self._update_standard(h, z, R)

    def _update_standard(self, h: Any, z: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_prior, P_prior = self.x.copy(), self.P.copy()

        sigma = self._sigma_points(self.x)
        sigma_meas = np.array([h(s[:self.n_state], s[self.n_state:]) for s in sigma])

        y_pred = self.W_m @ sigma_meas
        S_yy = self.W_c[:, None] * (sigma_meas - y_pred).T @ (sigma_meas - y_pred) + R
        S_xy = self.W_c[:, None] * (sigma - self.x).T @ (sigma_meas - y_pred)

        K = S_xy @ np.linalg.inv(S_yy)
        innovation = z - y_pred
        self.x = self.x + self._gate_state_update(K @ innovation, P_prior)
        self.P = self.P - K @ S_yy @ K.T

        theta_new = self._project_constraints(self.x[self.n_state:])
        if not np.allclose(theta_new, self.x[self.n_state:], atol=self.config.constraint_tol):
            self.x[self.n_state:] = theta_new
            self._inflate_covariance()

        return self._finite_or_rollback(x_prior, P_prior, None, innovation)

    def _update_square_root(self, h: Any, z: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_prior, P_prior, S_prior = self.x.copy(), self.P.copy(), self.S.copy()

        sigma = self._sigma_points(self.x, self.S)
        sigma_meas = np.array([h(s[:self.n_state], s[self.n_state:]) for s in sigma])

        y_pred = self.W_m @ sigma_meas
        diff = sigma_meas - y_pred

        sqrt_Wc = np.sqrt(np.abs(self.W_c))
        aug = np.vstack([sqrt_Wc[:, None] * diff, np.linalg.cholesky(R)])

        Q, S_yy = np.linalg.qr(aug)
        S_yy = S_yy[: self.n_meas, :]

        diff_state = sigma - self.x
        S_xy = (sqrt_Wc[:, None] * diff_state).T @ (sqrt_Wc[:, None] * diff)

        K = S_xy @ np.linalg.inv(S_yy)
        innovation = z - y_pred
        self.x = self.x + self._gate_state_update(K @ innovation, P_prior)

        U = K @ S_yy
        Phi = np.hstack([self.S, U])
        _, R_qr = np.linalg.qr(Phi)
        self.S = R_qr[:, : self.n_aug]
        self.P = self.S @ self.S.T

        theta_new = self._project_constraints(self.x[self.n_state:])
        if not np.allclose(theta_new, self.x[self.n_state:], atol=self.config.constraint_tol):
            self.x[self.n_state:] = theta_new
            self._inflate_covariance()

        return self._finite_or_rollback(x_prior, P_prior, S_prior, innovation)
