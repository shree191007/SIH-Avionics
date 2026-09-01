"""Shared constants and flag definitions for SIH26054 test suites."""

from __future__ import annotations

# ============================================================================
# Telemetry Bitmask Flags (uint32)
# ============================================================================

FLAG_EGT1_VALID = 1 << 0
FLAG_EGT2_VALID = 1 << 1
FLAG_EGT3_VALID = 1 << 2
FLAG_EGT4_VALID = 1 << 3
FLAG_CHT_VALID = 1 << 4
FLAG_POIL_VALID = 1 << 5
FLAG_TOIL_VALID = 1 << 6
FLAG_RPM_VALID = 1 << 7
FLAG_MDF_VALID = 1 << 8
FLAG_MAP_VALID = 1 << 9
FLAG_TPS_VALID = 1 << 10
FLAG_TIM_VALID = 1 << 11
FLAG_PAMB_VALID = 1 << 12
FLAG_TAMB_VALID = 1 << 13
FLAG_VTAS_VALID = 1 << 14
FLAG_HP_VALID = 1 << 15

ALL_CHANNELS_VALID_MASK = (1 << 16) - 1

# ============================================================================
# Status Codes
# ============================================================================

STATUS_OK = 0
STATUS_DEGRADED_INPUT = 1
STATUS_MODEL_SATURATED = 2
STATUS_INVALID_SENSOR = 3

# ============================================================================
# Version Strings
# ============================================================================

REGIME_GRID_VERSION_V1 = "REGIME_GRID_V1"
TELEMETRY_SCHEMA_VERSION_V1 = "TELEMETRY_SCHEMA_V1"
RESIDUAL_SCHEMA_VERSION_V1 = "RESIDUAL_FRAME_V1"
