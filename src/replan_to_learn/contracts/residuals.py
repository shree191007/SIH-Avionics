"""
SIH26054 Replan to Learn: Foundation Data Contracts & Schemas.
Residual Vector Contract & Programmatic Residual Suppression Engine.

Enforces the core rule:
Exogenous inputs are model drivers, NOT health signals. Missing exogenous inputs
MUST suppress affected residuals (NaN / masked updates) rather than computing
from guessed/imputed values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple
import math
import numpy as np
import polars as pl
import pyarrow as pa

from replan_to_learn.contracts.telemetry import (
    FLAG_CHT_VALID,
    FLAG_EGT1_VALID,
    FLAG_EGT2_VALID,
    FLAG_EGT3_VALID,
    FLAG_EGT4_VALID,
    FLAG_HP_VALID,
    FLAG_MAP_VALID,
    FLAG_MDF_VALID,
    FLAG_PAMB_VALID,
    FLAG_POIL_VALID,
    FLAG_RPM_VALID,
    FLAG_TAMB_VALID,
    FLAG_TIM_VALID,
    FLAG_TOIL_VALID,
    FLAG_TPS_VALID,
    FLAG_VTAS_VALID,
    FULL_VALID_MASK,
    FrameStatus,
    STATUS_DEGRADED_INPUT,
    STATUS_INVALID_SENSOR,
    STATUS_MODEL_SATURATED,
    STATUS_OK,
    TelemetryFrame,
    TelemetryValidityFlag,
)

# Residual Channel Identifiers and Indices
RESIDUAL_CHANNELS = [
    "z_egt_1",
    "z_egt_2",
    "z_egt_3",
    "z_egt_4",
    "z_cht",
    "z_oil_pressure",
    "z_oil_temperature",
    "z_fuel_flow",
    "z_power_balance",
]
NUM_RESIDUAL_CHANNELS = 9

IDX_Z_EGT1 = 0
IDX_Z_EGT2 = 1
IDX_Z_EGT3 = 2
IDX_Z_EGT4 = 3
IDX_Z_CHT = 4
IDX_Z_POIL = 5
IDX_Z_TOIL = 6
IDX_Z_MDF = 7
IDX_Z_POWER = 8


# ============================================================================
# 1. ResidualFrame Frozen Dataclass
# ============================================================================

@dataclass(frozen=True)
class ResidualFrame:
    """
    Normalised residual vector and diagnostic metadata emitted from L1/L2 Physics Twin.
    Immutable contract for downstream estimators (L3 UKF, L4 ML, L7 Planner).
    """
    t: float                                         # Monotonic aircraft clock timestamp (s)
    z: Tuple[float, float, float, float, float, float, float, float, float]  # float32[9] normalised residuals
    regime: int                                      # Index into REGIME_GRID_V1 (0..11) or TRANSIENT(13)/INVALID(14)
    regime_stable_s: float                           # Duration current quasi-steady regime sustained (s)
    spatial: Tuple[float, float, float, int]         # EGT spatial decomposition: (alpha, beta, norm_inf_s, dominant_cyl)
    sigma: Tuple[float, float, float, float, float, float, float, float, float]  # Noise std applied
    x_hat: Tuple[float, float, float, float, float, float, float, float, float]  # Internal physical model state vector
    status: int                                      # Frame status: OK(0), DEGRADED_INPUT(1), MODEL_SATURATED(2), INVALID(3)
    model_version: str = "1.0.0"                     # Calibrated model semantic version
    regime_grid_version: str = "REGIME_GRID_V1"      # Permanent regime grid version
    flight_id: str = ""                              # Unique flight identifier

    def __post_init__(self) -> None:
        if len(self.z) != NUM_RESIDUAL_CHANNELS:
            raise ValueError(f"Residual vector z must contain {NUM_RESIDUAL_CHANNELS} elements, got {len(self.z)}")
        if len(self.sigma) != NUM_RESIDUAL_CHANNELS:
            raise ValueError(f"Sigma vector must contain {NUM_RESIDUAL_CHANNELS} elements, got {len(self.sigma)}")
        if len(self.x_hat) != NUM_RESIDUAL_CHANNELS:
            raise ValueError(f"State vector x_hat must contain {NUM_RESIDUAL_CHANNELS} elements, got {len(self.x_hat)}")
        if len(self.spatial) != 4:
            raise ValueError(f"Spatial tuple must contain 4 elements (alpha, beta, norm_inf, dominant_cyl), got {len(self.spatial)}")

    @property
    def z_array(self) -> np.ndarray:
        """Return residual vector z as a NumPy array (float32)."""
        return np.array(self.z, dtype=np.float32)

    @property
    def sigma_array(self) -> np.ndarray:
        """Return sigma vector as a NumPy array (float32)."""
        return np.array(self.sigma, dtype=np.float32)

    @property
    def x_hat_array(self) -> np.ndarray:
        """Return state vector x_hat as a NumPy array (float32)."""
        return np.array(self.x_hat, dtype=np.float32)

    def is_suppressed(self, channel_idx: int) -> bool:
        """Check if a specific residual channel is suppressed (NaN)."""
        val = self.z[channel_idx]
        return math.isnan(val)

    def count_valid_channels(self) -> int:
        """Count number of unsuppressed (non-NaN) residual channels."""
        return sum(1 for val in self.z if not math.isnan(val))

    def to_dict(self) -> Dict[str, object]:
        """Convert frame to dictionary format matching schema."""
        return {
            "timestamp": float(self.t),
            "z_egt_1": None if math.isnan(self.z[0]) else float(self.z[0]),
            "z_egt_2": None if math.isnan(self.z[1]) else float(self.z[1]),
            "z_egt_3": None if math.isnan(self.z[2]) else float(self.z[2]),
            "z_egt_4": None if math.isnan(self.z[3]) else float(self.z[3]),
            "z_cht": None if math.isnan(self.z[4]) else float(self.z[4]),
            "z_oil_pressure": None if math.isnan(self.z[5]) else float(self.z[5]),
            "z_oil_temperature": None if math.isnan(self.z[6]) else float(self.z[6]),
            "z_fuel_flow": None if math.isnan(self.z[7]) else float(self.z[7]),
            "z_power_balance": None if math.isnan(self.z[8]) else float(self.z[8]),
            "regime_bin": int(self.regime),
            "regime_stable_duration_s": float(self.regime_stable_s),
            "spatial_alpha_common": float(self.spatial[0]),
            "spatial_beta_gradient": float(self.spatial[1]),
            "spatial_s_infinity_norm": float(self.spatial[2]),
            "spatial_dominant_cylinder": int(self.spatial[3]),
            "status_code": int(self.status),
            "model_version": str(self.model_version),
            "regime_grid_version": str(self.regime_grid_version),
            "flight_id": str(self.flight_id),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> ResidualFrame:
        """Construct ResidualFrame from dictionary."""
        def parse_float(val: object) -> float:
            if val is None:
                return float("nan")
            return float(val)

        z_tuple = (
            parse_float(data.get("z_egt_1")),
            parse_float(data.get("z_egt_2")),
            parse_float(data.get("z_egt_3")),
            parse_float(data.get("z_egt_4")),
            parse_float(data.get("z_cht")),
            parse_float(data.get("z_oil_pressure")),
            parse_float(data.get("z_oil_temperature")),
            parse_float(data.get("z_fuel_flow")),
            parse_float(data.get("z_power_balance")),
        )
        spatial_tuple = (
            float(data.get("spatial_alpha_common", 0.0)),
            float(data.get("spatial_beta_gradient", 0.0)),
            float(data.get("spatial_s_infinity_norm", 0.0)),
            int(data.get("spatial_dominant_cylinder", 0)),
        )
        sigma_raw = data.get("sigma", (1.0,) * NUM_RESIDUAL_CHANNELS)
        if isinstance(sigma_raw, (list, tuple)) and len(sigma_raw) == NUM_RESIDUAL_CHANNELS:
            sigma_tuple = tuple(float(x) for x in sigma_raw)
        else:
            sigma_tuple = (1.0,) * NUM_RESIDUAL_CHANNELS

        x_hat_raw = data.get("x_hat", (0.0,) * NUM_RESIDUAL_CHANNELS)
        if isinstance(x_hat_raw, (list, tuple)) and len(x_hat_raw) == NUM_RESIDUAL_CHANNELS:
            x_hat_tuple = tuple(float(x) for x in x_hat_raw)
        else:
            x_hat_tuple = (0.0,) * NUM_RESIDUAL_CHANNELS

        return cls(
            t=float(data.get("timestamp", data.get("t", 0.0))),
            z=z_tuple,
            regime=int(data.get("regime_bin", data.get("regime", 13))),
            regime_stable_s=float(data.get("regime_stable_duration_s", data.get("regime_stable_s", 0.0))),
            spatial=spatial_tuple,
            sigma=sigma_tuple,
            x_hat=x_hat_tuple,
            status=int(data.get("status_code", data.get("status", STATUS_OK))),
            model_version=str(data.get("model_version", "1.0.0")),
            regime_grid_version=str(data.get("regime_grid_version", "REGIME_GRID_V1")),
            flight_id=str(data.get("flight_id", "")),
        )


# ============================================================================
# 2. Polars & PyArrow Schemas
# ============================================================================

RESIDUAL_POLARS_SCHEMA = pl.Schema([
    ("timestamp", pl.Float64),
    ("z_egt_1", pl.Float32),
    ("z_egt_2", pl.Float32),
    ("z_egt_3", pl.Float32),
    ("z_egt_4", pl.Float32),
    ("z_cht", pl.Float32),
    ("z_oil_pressure", pl.Float32),
    ("z_oil_temperature", pl.Float32),
    ("z_fuel_flow", pl.Float32),
    ("z_power_balance", pl.Float32),
    ("regime_bin", pl.Int32),
    ("regime_stable_duration_s", pl.Float32),
    ("spatial_alpha_common", pl.Float32),
    ("spatial_beta_gradient", pl.Float32),
    ("spatial_s_infinity_norm", pl.Float32),
    ("spatial_dominant_cylinder", pl.Int32),
    ("status_code", pl.UInt8),
    ("model_version", pl.String),
    ("regime_grid_version", pl.String),
    ("flight_id", pl.String),
])

RESIDUAL_ARROW_SCHEMA = pa.schema([
    ("timestamp", pa.float64(), False),
    ("z_egt_1", pa.float32(), True),
    ("z_egt_2", pa.float32(), True),
    ("z_egt_3", pa.float32(), True),
    ("z_egt_4", pa.float32(), True),
    ("z_cht", pa.float32(), True),
    ("z_oil_pressure", pa.float32(), True),
    ("z_oil_temperature", pa.float32(), True),
    ("z_fuel_flow", pa.float32(), True),
    ("z_power_balance", pa.float32(), True),
    ("regime_bin", pa.int32(), False),
    ("regime_stable_duration_s", pa.float32(), False),
    ("spatial_alpha_common", pa.float32(), False),
    ("spatial_beta_gradient", pa.float32(), False),
    ("spatial_s_infinity_norm", pa.float32(), False),
    ("spatial_dominant_cylinder", pa.int32(), False),
    ("status_code", pa.uint8(), False),
    ("model_version", pa.string(), False),
    ("regime_grid_version", pa.string(), False),
    ("flight_id", pa.string(), False),
])


# ============================================================================
# 3. Residual Suppression Engine (Strict Zero-Imputation Enforcement)
# ============================================================================

class ResidualSuppressionEngine:
    """
    Programmatic Residual Suppression Engine.

    Contract: Exogenous inputs are model drivers, NOT health signals.
    When any exogenous input or primary sensor is invalid or missing, the affected
    residuals are deterministically suppressed (set to NaN). Imputing or guessing
    exogenous values is strictly prohibited to prevent hallucinated physics baselines
    and false degradation signatures.
    """

    # Physical Dependency Mapping:
    # Key: Sensor validity bitmask flag
    # Value: Set of residual channel indices that depend on this sensor
    DEPENDENCY_MAP: Dict[int, Set[int]] = {
        # Gas Path / Induction dependencies (MAP, T_im)
        FLAG_MAP_VALID: {IDX_Z_EGT1, IDX_Z_EGT2, IDX_Z_EGT3, IDX_Z_EGT4, IDX_Z_MDF, IDX_Z_POWER},
        FLAG_TIM_VALID: {IDX_Z_EGT1, IDX_Z_EGT2, IDX_Z_EGT3, IDX_Z_EGT4, IDX_Z_MDF, IDX_Z_POWER},
        # Convective Cooling dependencies (V_TAS, p_amb, T_amb)
        FLAG_VTAS_VALID: {IDX_Z_CHT, IDX_Z_TOIL, IDX_Z_POWER},
        FLAG_PAMB_VALID: {IDX_Z_CHT, IDX_Z_TOIL, IDX_Z_POWER},
        FLAG_TAMB_VALID: {IDX_Z_CHT, IDX_Z_TOIL, IDX_Z_POWER},
        # Oil viscosity temperature dependency
        FLAG_TOIL_VALID: {IDX_Z_TOIL, IDX_Z_POIL},
        # Engine speed (RPM) is the fundamental master clock & power balance driver
        FLAG_RPM_VALID: {
            IDX_Z_EGT1, IDX_Z_EGT2, IDX_Z_EGT3, IDX_Z_EGT4,
            IDX_Z_CHT, IDX_Z_POIL, IDX_Z_TOIL, IDX_Z_MDF, IDX_Z_POWER
        },
        # Direct primary sensor dependencies
        FLAG_EGT1_VALID: {IDX_Z_EGT1},
        FLAG_EGT2_VALID: {IDX_Z_EGT2},
        FLAG_EGT3_VALID: {IDX_Z_EGT3},
        FLAG_EGT4_VALID: {IDX_Z_EGT4},
        FLAG_CHT_VALID: {IDX_Z_CHT},
        FLAG_POIL_VALID: {IDX_Z_POIL},
        FLAG_MDF_VALID: {IDX_Z_MDF},
    }

    @staticmethod
    def is_imputation_prohibited() -> bool:
        """Returns True to attest that zero-imputation policy is strictly enforced."""
        return True

    @classmethod
    def get_suppressed_channel_indices(cls, valid_mask: int) -> Set[int]:
        """
        Given a 32-bit channel validity bitmask, determine all residual indices
        that MUST be suppressed.
        """
        suppressed_indices: Set[int] = set()

        for flag, affected_channels in cls.DEPENDENCY_MAP.items():
            # If the required flag is NOT set in valid_mask, suppress affected channels
            if not (valid_mask & flag):
                suppressed_indices.update(affected_channels)

        return suppressed_indices

    @classmethod
    def evaluate_suppression(
        cls,
        raw_z: Sequence[float] | np.ndarray,
        valid_mask: int,
        is_saturated: bool = False,
    ) -> Tuple[Tuple[float, ...], int, List[str]]:
        """
        Apply suppression rules to raw residuals based on validity bitmask.

        Args:
            raw_z: Raw computed residual vector (len 9).
            valid_mask: 32-bit validity bitmask.
            is_saturated: Flag indicating physical model saturation clamp.

        Returns:
            Tuple of:
                - Suppressed residual tuple (with NaN for suppressed channels)
                - Frame status code (STATUS_OK, STATUS_DEGRADED_INPUT, STATUS_MODEL_SATURATED, STATUS_INVALID_SENSOR)
                - List of suppressed channel names
        """
        if len(raw_z) != NUM_RESIDUAL_CHANNELS:
            raise ValueError(f"Expected {NUM_RESIDUAL_CHANNELS} residuals, got {len(raw_z)}")

        suppressed_indices = cls.get_suppressed_channel_indices(valid_mask)
        z_out = list(raw_z)

        # Apply NaN suppression
        for idx in suppressed_indices:
            z_out[idx] = float("nan")

        suppressed_names = [RESIDUAL_CHANNELS[i] for i in sorted(suppressed_indices)]

        # Determine frame status
        if is_saturated:
            status = STATUS_MODEL_SATURATED
        elif len(suppressed_indices) > 0:
            # Check if suppression was caused by primary vs exogenous
            exogenous_flags_missing = False
            for flag in [
                FLAG_MAP_VALID, FLAG_TPS_VALID, FLAG_TIM_VALID,
                FLAG_PAMB_VALID, FLAG_TAMB_VALID, FLAG_VTAS_VALID, FLAG_HP_VALID
            ]:
                if not (valid_mask & flag):
                    exogenous_flags_missing = True
                    break

            primary_flags_missing = False
            for flag in [
                FLAG_EGT1_VALID, FLAG_EGT2_VALID, FLAG_EGT3_VALID, FLAG_EGT4_VALID,
                FLAG_CHT_VALID, FLAG_POIL_VALID, FLAG_TOIL_VALID, FLAG_RPM_VALID, FLAG_MDF_VALID
            ]:
                if not (valid_mask & flag):
                    primary_flags_missing = True
                    break

            if exogenous_flags_missing:
                status = STATUS_DEGRADED_INPUT
            elif primary_flags_missing:
                status = STATUS_INVALID_SENSOR
            else:
                status = STATUS_DEGRADED_INPUT
        else:
            status = STATUS_OK

        return tuple(float(x) for x in z_out), status, suppressed_names

    @classmethod
    def apply_suppression_to_frame(
        cls,
        telemetry: TelemetryFrame,
        raw_z: Sequence[float] | np.ndarray,
        regime: int,
        regime_stable_s: float,
        spatial: Tuple[float, float, float, int],
        sigma: Sequence[float] | np.ndarray,
        x_hat: Sequence[float] | np.ndarray,
        is_saturated: bool = False,
        model_version: str = "1.0.0",
        regime_grid_version: str = "REGIME_GRID_V1",
    ) -> ResidualFrame:
        """
        Evaluate TelemetryFrame validity and produce a strictly compliant ResidualFrame.
        """
        valid_mask = telemetry.compute_validity_mask()
        z_suppressed, status, _ = cls.evaluate_suppression(raw_z, valid_mask, is_saturated=is_saturated)

        sigma_tuple = tuple(float(s) for s in sigma)
        x_hat_tuple = tuple(float(x) for x in x_hat)

        return ResidualFrame(
            t=telemetry.t,
            z=z_suppressed,  # type: ignore[arg-type]
            regime=regime,
            regime_stable_s=regime_stable_s,
            spatial=spatial,
            sigma=sigma_tuple,  # type: ignore[arg-type]
            x_hat=x_hat_tuple,  # type: ignore[arg-type]
            status=status,
            model_version=model_version,
            regime_grid_version=regime_grid_version,
            flight_id=telemetry.flight_id,
        )


# ============================================================================
# 4. Conversion & Frame Validation Utilities
# ============================================================================

def residual_frames_to_polars(frames: Sequence[ResidualFrame]) -> pl.DataFrame:
    """Convert a sequence of ResidualFrame dataclasses to a validated Polars DataFrame."""
    if not frames:
        return pl.DataFrame(schema=RESIDUAL_POLARS_SCHEMA)
    data = [frame.to_dict() for frame in frames]
    return pl.DataFrame(data, schema=RESIDUAL_POLARS_SCHEMA)


def polars_to_residual_frames(df: pl.DataFrame) -> List[ResidualFrame]:
    """Convert a Polars DataFrame conforming to RESIDUAL_POLARS_SCHEMA to ResidualFrame list."""
    if df.is_empty():
        return []
    records = df.to_dicts()
    return [ResidualFrame.from_dict(row) for row in records]


def residual_frames_to_arrow(frames: Sequence[ResidualFrame]) -> pa.Table:
    """Convert a sequence of ResidualFrame dataclasses to an Apache Arrow Table."""
    df = residual_frames_to_polars(frames)
    return df.to_arrow()


def arrow_to_residual_frames(table: pa.Table) -> List[ResidualFrame]:
    """Convert an Apache Arrow Table to a list of ResidualFrame dataclasses."""
    df = pl.from_arrow(table)
    if isinstance(df, pl.Series):
        df = df.to_frame()
    return polars_to_residual_frames(df)


def validate_residual_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Validate residual DataFrame against schema and ensure correct nullable types.
    """
    for col_name in RESIDUAL_POLARS_SCHEMA.names():
        if col_name not in df.columns:
            raise ValueError(f"Missing required residual column: {col_name}")

    frames = polars_to_residual_frames(df)
    return residual_frames_to_polars(frames)
