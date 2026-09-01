"""
SIH26054 Replan to Learn: Fault Injection Harness (Section 7).
Provides the inject() function specified in 02_physics_twin_residuals.md §7.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from replan_to_learn.contracts.faults import FaultSpec, FaultInjectionHarness
from replan_to_learn.contracts.health import HealthParameters


def inject(
    theta_nom: Union[HealthParameters, np.ndarray],
    fault: FaultSpec,
    t: float,
) -> np.ndarray:
    """
    Apply fault injection specification to nominal health parameters at time t.

    Implements the contract from Section 7 of 02_physics_twin_residuals.md:
        def inject(theta_nom, fault: FaultSpec, t: float) -> np.ndarray

    Returns a flat numpy array of length 15 representing the health parameter
    vector theta in the order:
        [theta_vol, theta_comb, theta_cool, theta_inj[0..3],
         theta_oilp, theta_fric, b_egt[0..3], b_cht, b_poil]

    Args:
        theta_nom: Nominal health parameters (HealthParameters or np.ndarray of length 15)
        fault: FaultSpec defining fault class, magnitude, profile
        t: Flight time in seconds (used to evaluate ramp/step profile)

    Returns:
        Degraded health parameters as np.ndarray of length 15
    """
    if isinstance(theta_nom, np.ndarray):
        hp = HealthParameters.from_array(theta_nom)
    else:
        hp = theta_nom

    degraded = FaultInjectionHarness.inject(hp, fault, t)
    return degraded.to_array()


def inject_theta_cyl(
    theta_inj_nom: np.ndarray,
    cylinder: int,
    amplitude: float,
    t: float,
    ramp_hours: float = 5.0,
) -> np.ndarray:
    """
    Inject per-cylinder injector degradation: theta_inj[i] : 1.0 -> 1.0 - a over ramp_hours.

    Args:
        theta_inj_nom: Nominal injector health parameters, shape (4,)
        cylinder: Cylinder index 0-3
        amplitude: Final degradation magnitude a in [0, 1]
        t: Flight time in seconds
        ramp_hours: Ramp duration in hours (5-40 per spec)

    Returns:
        Modified theta_inj array
    """
    result = theta_inj_nom.copy()
    ramp_s = ramp_hours * 3600.0
    progress = min(1.0, t / ramp_s)
    result[cylinder] = 1.0 - amplitude * progress
    return result


def inject_cooling_nonuniform(
    theta_cool_nom: float,
    amplitude: float,
    t: float,
    ramp_hours: float = 10.0,
    w_spread: float = 0.35,
) -> tuple[float, np.ndarray]:
    """
    Inject non-uniform cooling degradation: theta_cool : 1.0 -> 1.0 - a,
    with a front/rear weighting w (deliberately non-uniform per spec §7).

    Args:
        theta_cool_nom: Nominal cooling health parameter
        amplitude: Final degradation magnitude a
        t: Flight time in seconds
        ramp_hours: Ramp duration in hours
        w_spread: Cylinder spread magnitude (>= 0.35 per spec)

    Returns:
        Tuple of (degraded theta_cool, per-cylinder asymmetry factors)
    """
    ramp_s = ramp_hours * 3600.0
    progress = min(1.0, t / ramp_s)
    new_cool = theta_cool_nom - amplitude * progress
    asymmetry = np.array([-w_spread, -w_spread / 2.0, w_spread / 2.0, w_spread])
    return new_cool, asymmetry