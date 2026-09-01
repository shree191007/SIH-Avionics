"""
SIH26054 Replan to Learn: L4 Identifiability Gate data types.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

import numpy as np


class Verdict(IntEnum):
    """Identifiability gate verdict."""
    NAMED = 0
    AMBIGUOUS = 1
    BORROWED = 2
    INVALID = 3


class ReasonCode(IntEnum):
    """Reason codes for gate refusals and borrow rejections."""
    NONE = 0
    A1_INSUFFICIENT_REGIMES = 1
    A2_LOW_R2 = 2
    A3_NEGATIVE_GAIN = 3
    A4_GAIN_OUTLIER = 4
    A5_FLEET_TOO_SMALL = 5
    A6_VERSION_MISMATCH = 6
    FIM_SINGULAR = 7
    COS_TOO_HIGH = 8


@dataclass(frozen=True)
class ProbeRequest:
    """Emitted when verdict == AMBIGUOUS and no admissible borrow exists."""
    ambiguous_set: Tuple[int, ...]
    reason: ReasonCode
    cos_matrix: np.ndarray
    crlb: np.ndarray


@dataclass(frozen=True)
class BorrowResult:
    """Result of attempting to borrow from fleet shape."""
    admissible: bool
    verdict: Verdict
    r2: Optional[float] = None
    alpha_hat: Optional[float] = None
    reason: ReasonCode = ReasonCode.NONE
    borrowed_regimes: Tuple[int, ...] = ()
    crlb_inflated: Optional[np.ndarray] = None


@dataclass(frozen=True)
class AttributionFrame:
    """
    Stable attribution output consumed by Stages 3 and 7.
    """
    t: float
    verdict: Verdict
    theta_hat: np.ndarray
    crlb: np.ndarray
    named: Optional[int]
    ambiguous_set: Tuple[int, ...]
    cos_matrix: np.ndarray
    borrowed_regimes: Tuple[int, ...]
    r2: Optional[float]
    alpha_hat: Optional[float]
    fleet_epoch: Optional[int]
    reason: ReasonCode
    model_version: str
    regime_grid_version: str
