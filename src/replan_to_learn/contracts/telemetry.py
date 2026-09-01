"""
SIH26054 Replan to Learn: Foundation Data Contracts & Schemas.
Primary Telemetry Channels & Exogenous Inputs Contract.

Defines the 6 primary telemetry channels (across 9 scalar signals), 7 exogenous
model drivers, physical validity bounds, unit conversion utilities, bitmask flags,
Polars and PyArrow schemas, and the frozen TelemetryFrame contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import Dict, List, Optional, Sequence, Tuple
import math
import numpy as np
import polars as pl
import pyarrow as pa


# ============================================================================
# 1. Bitmask Flags & Status Enumerations
# ============================================================================

class TelemetryValidityFlag(IntFlag):
    """Channel validity bitmask flags (uint32)."""
    NONE = 0
    EGT1_VALID = 1 << 0
    EGT2_VALID = 1 << 1
    EGT3_VALID = 1 << 2
    EGT4_VALID = 1 << 3
    CHT_VALID = 1 << 4
    POIL_VALID = 1 << 5
    TOIL_VALID = 1 << 6
    RPM_VALID = 1 << 7
    MDF_VALID = 1 << 8
    MAP_VALID = 1 << 9
    TPS_VALID = 1 << 10
    TIM_VALID = 1 << 11
    PAMB_VALID = 1 << 12
    TAMB_VALID = 1 << 13
    VTAS_VALID = 1 << 14
    HP_VALID = 1 << 15
    ALL_PRIMARY = (
        (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) |
        (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8)
    )
    ALL_EXOGENOUS = (
        (1 << 9) | (1 << 10) | (1 << 11) | (1 << 12) |
        (1 << 13) | (1 << 14) | (1 << 15)
    )
    FULL = ALL_PRIMARY | ALL_EXOGENOUS


FLAG_EGT1_VALID = TelemetryValidityFlag.EGT1_VALID.value
FLAG_EGT2_VALID = TelemetryValidityFlag.EGT2_VALID.value
FLAG_EGT3_VALID = TelemetryValidityFlag.EGT3_VALID.value
FLAG_EGT4_VALID = TelemetryValidityFlag.EGT4_VALID.value
FLAG_CHT_VALID = TelemetryValidityFlag.CHT_VALID.value
FLAG_POIL_VALID = TelemetryValidityFlag.POIL_VALID.value
FLAG_TOIL_VALID = TelemetryValidityFlag.TOIL_VALID.value
FLAG_RPM_VALID = TelemetryValidityFlag.RPM_VALID.value
FLAG_MDF_VALID = TelemetryValidityFlag.MDF_VALID.value
FLAG_MAP_VALID = TelemetryValidityFlag.MAP_VALID.value
FLAG_TPS_VALID = TelemetryValidityFlag.TPS_VALID.value
FLAG_TIM_VALID = TelemetryValidityFlag.TIM_VALID.value
FLAG_PAMB_VALID = TelemetryValidityFlag.PAMB_VALID.value
FLAG_TAMB_VALID = TelemetryValidityFlag.TAMB_VALID.value
FLAG_VTAS_VALID = TelemetryValidityFlag.VTAS_VALID.value
FLAG_HP_VALID = TelemetryValidityFlag.HP_VALID.value
FULL_VALID_MASK = TelemetryValidityFlag.FULL.value


class FrameStatus(IntEnum):
    """Telemetry and residual frame processing status."""
    STATUS_OK = 0               # All required inputs valid, residuals clean
    STATUS_DEGRADED_INPUT = 1   # 1+ exogenous inputs missing; affected residuals suppressed
    STATUS_MODEL_SATURATED = 2  # Physical states clamped to bounds; step invalidated
    STATUS_INVALID_SENSOR = 3   # Primary sensor out-of-bounds or disconnect


STATUS_OK = FrameStatus.STATUS_OK.value
STATUS_DEGRADED_INPUT = FrameStatus.STATUS_DEGRADED_INPUT.value
STATUS_MODEL_SATURATED = FrameStatus.STATUS_MODEL_SATURATED.value
STATUS_INVALID_SENSOR = FrameStatus.STATUS_INVALID_SENSOR.value


# ============================================================================
# 2. Sensor Physical Bounds, Noise, and Disconnect Specifications
# ============================================================================

@dataclass(frozen=True)
class SensorSpec:
    """Physical specification and operational limits for a telemetry channel."""
    name: str
    symbol: str
    engineering_unit: str
    si_unit: str
    nominal_min: float
    nominal_max: float
    admissible_min: float
    admissible_max: float
    disconnect_min: float
    disconnect_max: float
    nominal_sigma: float
    description: str

    def is_in_admissible_range(self, value: float) -> bool:
        """Check if reading is within admissible operating limits."""
        if value is None or math.isnan(value) or math.isinf(value):
            return False
        return self.admissible_min <= value <= self.admissible_max

    def is_disconnected_or_failed(self, value: float) -> bool:
        """Check if reading triggers a hardware sensor failure/disconnect."""
        if value is None or math.isnan(value) or math.isinf(value):
            return True
        return value < self.disconnect_min or value > self.disconnect_max


# Sensor specifications in SI units
PRIMARY_SENSOR_SPECS: Dict[str, SensorSpec] = {
    "egt_1": SensorSpec(
        name="Exhaust Gas Temp Cyl 1",
        symbol="EGT[1]",
        engineering_unit="°C",
        si_unit="K",
        nominal_min=953.15,      # 680 °C
        nominal_max=1153.15,    # 880 °C
        admissible_min=473.15,   # 200 °C
        admissible_max=1323.15,  # 1050 °C
        disconnect_min=273.15,   # 0 °C
        disconnect_max=1423.15,  # 1150 °C
        nominal_sigma=8.0,       # 8.0 K
        description="Rotax 915 iS cylinder 1 exhaust gas temperature probe."
    ),
    "egt_2": SensorSpec(
        name="Exhaust Gas Temp Cyl 2",
        symbol="EGT[2]",
        engineering_unit="°C",
        si_unit="K",
        nominal_min=953.15,
        nominal_max=1153.15,
        admissible_min=473.15,
        admissible_max=1323.15,
        disconnect_min=273.15,
        disconnect_max=1423.15,
        nominal_sigma=8.0,
        description="Rotax 915 iS cylinder 2 exhaust gas temperature probe."
    ),
    "egt_3": SensorSpec(
        name="Exhaust Gas Temp Cyl 3",
        symbol="EGT[3]",
        engineering_unit="°C",
        si_unit="K",
        nominal_min=953.15,
        nominal_max=1153.15,
        admissible_min=473.15,
        admissible_max=1323.15,
        disconnect_min=273.15,
        disconnect_max=1423.15,
        nominal_sigma=8.0,
        description="Rotax 915 iS cylinder 3 exhaust gas temperature probe."
    ),
    "egt_4": SensorSpec(
        name="Exhaust Gas Temp Cyl 4",
        symbol="EGT[4]",
        engineering_unit="°C",
        si_unit="K",
        nominal_min=953.15,
        nominal_max=1153.15,
        admissible_min=473.15,
        admissible_max=1323.15,
        disconnect_min=273.15,
        disconnect_max=1423.15,
        nominal_sigma=8.0,
        description="Rotax 915 iS cylinder 4 exhaust gas temperature probe."
    ),
    "cht": SensorSpec(
        name="Cylinder Head / Coolant Node Temp",
        symbol="CHT",
        engineering_unit="°C",
        si_unit="K",
        nominal_min=348.15,      # 75 °C
        nominal_max=388.15,      # 115 °C
        admissible_min=243.15,   # -30 °C
        admissible_max=423.15,   # 150 °C
        disconnect_min=233.15,   # -40 °C
        disconnect_max=438.15,   # 165 °C
        nominal_sigma=2.5,       # 2.5 K
        description="Liquid-cooled cylinder head coolant node temperature."
    ),
    "oil_pressure": SensorSpec(
        name="Oil Pressure",
        symbol="p_oil",
        engineering_unit="bar",
        si_unit="Pa",
        nominal_min=200000.0,    # 2.0 bar
        nominal_max=500000.0,    # 5.0 bar
        admissible_min=50000.0,  # 0.5 bar
        admissible_max=800000.0, # 8.0 bar
        disconnect_min=20000.0,  # 0.2 bar
        disconnect_max=950000.0, # 9.5 bar
        nominal_sigma=12000.0,   # 0.12 bar (12 kPa)
        description="Main lubrication gallery oil pressure transducer."
    ),
    "oil_temperature": SensorSpec(
        name="Oil Temperature",
        symbol="T_oil",
        engineering_unit="°C",
        si_unit="K",
        nominal_min=343.15,      # 70 °C
        nominal_max=383.15,      # 110 °C
        admissible_min=243.15,   # -30 °C
        admissible_max=418.15,   # 145 °C
        disconnect_min=233.15,   # -40 °C
        disconnect_max=433.15,   # 160 °C
        nominal_sigma=1.5,       # 1.5 K
        description="Oil sump / reservoir temperature sensor."
    ),
    "engine_speed_rpm": SensorSpec(
        name="Engine Crankshaft Speed",
        symbol="N",
        engineering_unit="rpm",
        si_unit="rpm",
        nominal_min=1400.0,
        nominal_max=5800.0,
        admissible_min=0.0,
        admissible_max=6200.0,
        disconnect_min=0.0,
        disconnect_max=6500.0,
        nominal_sigma=4.0,       # 4.0 rpm
        description="Crankshaft optical/Hall-effect tachometer (10 Hz downsampled to 1 Hz)."
    ),
    "fuel_flow_kg_s": SensorSpec(
        name="Fuel Mass Flow Rate",
        symbol="ṁ_f",
        engineering_unit="L/h",
        si_unit="kg/s",
        nominal_min=0.0012,      # ~6.0 L/h (1.2 g/s)
        nominal_max=0.0085,      # ~42.5 L/h (8.5 g/s)
        admissible_min=0.0,
        admissible_max=0.0150,   # 15.0 g/s (75 L/h)
        disconnect_min=0.0,
        disconnect_max=0.01875,  # 90 L/h
        nominal_sigma=0.00012,   # 0.12 g/s (0.6 L/h)
        description="EMS total fuel delivery mass flow rate."
    ),
}

EXOGENOUS_SENSOR_SPECS: Dict[str, SensorSpec] = {
    "map_pa": SensorSpec(
        name="Manifold Absolute Pressure",
        symbol="MAP",
        engineering_unit="kPa / inHg",
        si_unit="Pa",
        nominal_min=60000.0,     # 60 kPa (17.7 inHg)
        nominal_max=165000.0,    # 165 kPa (48.7 inHg)
        admissible_min=30000.0,  # 30 kPa
        admissible_max=220000.0, # 220 kPa
        disconnect_min=20000.0,
        disconnect_max=250000.0,
        nominal_sigma=500.0,     # 0.5 kPa (500 Pa)
        description="Intake plenum manifold absolute pressure sensor."
    ),
    "throttle_position": SensorSpec(
        name="Throttle Position Sensor",
        symbol="TPS",
        engineering_unit="%",
        si_unit="ratio",
        nominal_min=0.0,
        nominal_max=1.0,
        admissible_min=0.0,
        admissible_max=1.0,
        disconnect_min=-0.05,
        disconnect_max=1.05,
        nominal_sigma=0.005,     # 0.5 %
        description="Electronic throttle valve angle ratio [0, 1]."
    ),
    "intake_temp_k": SensorSpec(
        name="Intake Manifold Air Temp",
        symbol="T_im",
        engineering_unit="°C",
        si_unit="K",
        nominal_min=283.15,      # 10 °C
        nominal_max=338.15,      # 65 °C
        admissible_min=233.15,   # -40 °C
        admissible_max=378.15,   # 105 °C
        disconnect_min=223.15,   # -50 °C
        disconnect_max=393.15,   # 120 °C
        nominal_sigma=1.0,       # 1.0 K
        description="Airbox / post-intercooler intake charge temperature."
    ),
    "ambient_pressure_pa": SensorSpec(
        name="Ambient Static Pressure",
        symbol="p_amb",
        engineering_unit="hPa / inHg",
        si_unit="Pa",
        nominal_min=45000.0,     # 450 hPa
        nominal_max=102500.0,    # 1025 hPa
        admissible_min=30000.0,  # 300 hPa
        admissible_max=110000.0, # 1100 hPa
        disconnect_min=25000.0,
        disconnect_max=120000.0,
        nominal_sigma=80.0,      # 0.8 hPa (80 Pa)
        description="Pitot-static barometric ambient static pressure."
    ),
    "ambient_temp_k": SensorSpec(
        name="Ambient Static Temperature",
        symbol="T_amb",
        engineering_unit="°C",
        si_unit="K",
        nominal_min=243.15,      # -30 °C
        nominal_max=318.15,      # +45 °C
        admissible_min=218.15,   # -55 °C
        admissible_max=333.15,   # +60 °C
        disconnect_min=203.15,   # -70 °C
        disconnect_max=348.15,   # +75 °C
        nominal_sigma=0.8,       # 0.8 K
        description="Outside air temperature (OAT) static thermal probe."
    ),
    "true_airspeed_ms": SensorSpec(
        name="True Airspeed",
        symbol="V_TAS",
        engineering_unit="kts",
        si_unit="m/s",
        nominal_min=20.57,       # 40 kts
        nominal_max=82.31,       # 160 kts
        admissible_min=0.0,
        admissible_max=110.0,    # ~214 kts
        disconnect_min=-1.0,
        disconnect_max=130.0,
        nominal_sigma=0.5,       # 0.5 m/s (~1 kt)
        description="Air data computer true airspeed."
    ),
    "pressure_altitude_m": SensorSpec(
        name="Pressure Altitude",
        symbol="h_p",
        engineering_unit="ft",
        si_unit="m",
        nominal_min=0.0,
        nominal_max=7010.4,      # 23,000 ft
        admissible_min=-304.8,   # -1,000 ft
        admissible_max=8534.4,   # 28,000 ft
        disconnect_min=-600.0,
        disconnect_max=10000.0,
        nominal_sigma=5.0,       # 5.0 m (~16 ft)
        description="Barometric pressure altitude standard atmosphere datum."
    ),
}

ALL_SENSOR_SPECS: Dict[str, SensorSpec] = {
    **PRIMARY_SENSOR_SPECS,
    **EXOGENOUS_SENSOR_SPECS,
}


# ============================================================================
# 3. Engineering Unit Conversions
# ============================================================================

STANDARD_GAS_DENSITY_KG_L = 0.720  # Avgas / Mogas nominal density at 15 °C (kg/L)


def celsius_to_kelvin(temp_c: float) -> float:
    """Convert temperature from Celsius (°C) to Kelvin (K)."""
    return float(temp_c + 273.15)


def kelvin_to_celsius(temp_k: float) -> float:
    """Convert temperature from Kelvin (K) to Celsius (°C)."""
    return float(temp_k - 273.15)


def bar_to_pa(p_bar: float) -> float:
    """Convert pressure from bar to Pascal (Pa)."""
    return float(p_bar * 100_000.0)


def pa_to_bar(p_pa: float) -> float:
    """Convert pressure from Pascal (Pa) to bar."""
    return float(p_pa / 100_000.0)


def kpa_to_pa(p_kpa: float) -> float:
    """Convert pressure from kiloPascal (kPa) to Pascal (Pa)."""
    return float(p_kpa * 1000.0)


def pa_to_kpa(p_pa: float) -> float:
    """Convert pressure from Pascal (Pa) to kiloPascal (kPa)."""
    return float(p_pa / 1000.0)


def inhg_to_pa(p_inhg: float) -> float:
    """Convert pressure from inches of mercury (inHg) to Pascal (Pa)."""
    return float(p_inhg * 3386.3886666667)


def pa_to_inhg(p_pa: float) -> float:
    """Convert pressure from Pascal (Pa) to inches of mercury (inHg)."""
    return float(p_pa / 3386.3886666667)


def hpa_to_pa(p_hpa: float) -> float:
    """Convert pressure from hectoPascal (hPa) to Pascal (Pa)."""
    return float(p_hpa * 100.0)


def pa_to_hpa(p_pa: float) -> float:
    """Convert pressure from Pascal (Pa) to hectoPascal (hPa)."""
    return float(p_pa / 100.0)


def rpm_to_rad_s(rpm: float) -> float:
    """Convert rotational speed from rpm to radians per second (rad/s)."""
    return float(rpm * (2.0 * math.pi / 60.0))


def rad_s_to_rpm(rad_s: float) -> float:
    """Convert rotational speed from radians per second (rad/s) to rpm."""
    return float(rad_s * (60.0 / (2.0 * math.pi)))


def liters_per_hour_to_kg_s(lph: float, density_kg_l: float = STANDARD_GAS_DENSITY_KG_L) -> float:
    """Convert fuel volumetric flow (L/h) to mass flow (kg/s)."""
    return float((lph * density_kg_l) / 3600.0)


def kg_s_to_liters_per_hour(kg_s: float, density_kg_l: float = STANDARD_GAS_DENSITY_KG_L) -> float:
    """Convert fuel mass flow (kg/s) to volumetric flow (L/h)."""
    return float((kg_s * 3600.0) / density_kg_l)


def knots_to_m_s(kts: float) -> float:
    """Convert speed from knots (kts) to meters per second (m/s)."""
    return float(kts * 0.5144444444444445)


def m_s_to_knots(ms: float) -> float:
    """Convert speed from meters per second (m/s) to knots (kts)."""
    return float(ms / 0.5144444444444445)


def feet_to_meters(ft: float) -> float:
    """Convert altitude from feet (ft) to meters (m)."""
    return float(ft * 0.3048)


def meters_to_feet(m: float) -> float:
    """Convert altitude from meters (m) to feet (ft)."""
    return float(m / 0.3048)


# ============================================================================
# 4. TelemetryFrame Frozen Dataclass
# ============================================================================

@dataclass(frozen=True)
class TelemetryFrame:
    """
    Immutable raw telemetry contract at 1 Hz.
    Contains 6 primary channels (9 scalar signals) and 7 exogenous model drivers.
    All physical quantities are represented strictly in SI units.
    """
    t: float                                    # Aircraft monotonic clock timestamp (s)
    egt: Tuple[float, float, float, float]      # Exhaust Gas Temps Cyl 1..4 (K)
    cht: float                                  # Cylinder Head / Coolant Node Temp (K)
    p_oil: float                                # Engine Oil Pressure (Pa)
    t_oil: float                                # Engine Oil Temperature (K)
    n_rpm: float                                # Engine Speed (rpm)
    mdot_f: float                               # Fuel Mass Flow Rate (kg/s)
    # Exogenous Model Drivers
    map_pa: float                               # Manifold Absolute Pressure (Pa)
    tps: float                                  # Throttle Position Sensor ratio [0.0, 1.0]
    t_im: float                                 # Intake Manifold Air Temp (K)
    p_amb: float                                # Ambient Static Pressure (Pa)
    t_amb: float                                # Ambient Static Temperature (K)
    v_tas: float                                # True Airspeed (m/s)
    h_p: float                                  # Pressure Altitude (m)
    # Provenance and Health Metadata
    valid_mask: int = FULL_VALID_MASK           # 32-bit channel validity bitmask
    flight_id: str = ""                         # Unique flight identifier
    aircraft_id: str = ""                       # Tail number / aircraft ID
    engine_id: str = ""                         # Engine serial number

    def __post_init__(self) -> None:
        """Validate shapes and invariants."""
        if len(self.egt) != 4:
            raise ValueError(f"EGT must contain exactly 4 cylinders, got {len(self.egt)}")

    @property
    def egt_1(self) -> float:
        return self.egt[0]

    @property
    def egt_2(self) -> float:
        return self.egt[1]

    @property
    def egt_3(self) -> float:
        return self.egt[2]

    @property
    def egt_4(self) -> float:
        return self.egt[3]

    def is_valid(self, flag: TelemetryValidityFlag | int) -> bool:
        """Check if a specific validity flag is set in valid_mask."""
        flag_val = flag.value if isinstance(flag, TelemetryValidityFlag) else flag
        return bool(self.valid_mask & flag_val)

    def are_primary_sensors_valid(self) -> bool:
        """Check if all 9 primary scalar sensor channels are marked valid."""
        primary_mask = TelemetryValidityFlag.ALL_PRIMARY.value
        return (self.valid_mask & primary_mask) == primary_mask

    def are_exogenous_inputs_valid(self) -> bool:
        """Check if all 7 exogenous model driver channels are marked valid."""
        exo_mask = TelemetryValidityFlag.ALL_EXOGENOUS.value
        return (self.valid_mask & exo_mask) == exo_mask

    def compute_validity_mask(self) -> int:
        """
        Evaluate sensor values against admissible physical limits and disconnect triggers.
        Returns the computed 32-bit valid_mask.
        """
        mask = 0
        # Check EGT 1..4
        for i, flag in enumerate([
            FLAG_EGT1_VALID, FLAG_EGT2_VALID, FLAG_EGT3_VALID, FLAG_EGT4_VALID
        ]):
            spec = PRIMARY_SENSOR_SPECS[f"egt_{i+1}"]
            val = self.egt[i]
            if spec.is_in_admissible_range(val) and not spec.is_disconnected_or_failed(val):
                mask |= flag

        # CHT
        if (PRIMARY_SENSOR_SPECS["cht"].is_in_admissible_range(self.cht) and
                not PRIMARY_SENSOR_SPECS["cht"].is_disconnected_or_failed(self.cht)):
            mask |= FLAG_CHT_VALID

        # Oil pressure
        if (PRIMARY_SENSOR_SPECS["oil_pressure"].is_in_admissible_range(self.p_oil) and
                not PRIMARY_SENSOR_SPECS["oil_pressure"].is_disconnected_or_failed(self.p_oil)):
            mask |= FLAG_POIL_VALID

        # Oil temp
        if (PRIMARY_SENSOR_SPECS["oil_temperature"].is_in_admissible_range(self.t_oil) and
                not PRIMARY_SENSOR_SPECS["oil_temperature"].is_disconnected_or_failed(self.t_oil)):
            mask |= FLAG_TOIL_VALID

        # RPM
        if (PRIMARY_SENSOR_SPECS["engine_speed_rpm"].is_in_admissible_range(self.n_rpm) and
                not PRIMARY_SENSOR_SPECS["engine_speed_rpm"].is_disconnected_or_failed(self.n_rpm)):
            mask |= FLAG_RPM_VALID

        # Fuel flow
        if (PRIMARY_SENSOR_SPECS["fuel_flow_kg_s"].is_in_admissible_range(self.mdot_f) and
                not PRIMARY_SENSOR_SPECS["fuel_flow_kg_s"].is_disconnected_or_failed(self.mdot_f)):
            mask |= FLAG_MDF_VALID

        # Exogenous inputs
        if (EXOGENOUS_SENSOR_SPECS["map_pa"].is_in_admissible_range(self.map_pa) and
                not EXOGENOUS_SENSOR_SPECS["map_pa"].is_disconnected_or_failed(self.map_pa)):
            mask |= FLAG_MAP_VALID

        if (EXOGENOUS_SENSOR_SPECS["throttle_position"].is_in_admissible_range(self.tps) and
                not EXOGENOUS_SENSOR_SPECS["throttle_position"].is_disconnected_or_failed(self.tps)):
            mask |= FLAG_TPS_VALID

        if (EXOGENOUS_SENSOR_SPECS["intake_temp_k"].is_in_admissible_range(self.t_im) and
                not EXOGENOUS_SENSOR_SPECS["intake_temp_k"].is_disconnected_or_failed(self.t_im)):
            mask |= FLAG_TIM_VALID

        if (EXOGENOUS_SENSOR_SPECS["ambient_pressure_pa"].is_in_admissible_range(self.p_amb) and
                not EXOGENOUS_SENSOR_SPECS["ambient_pressure_pa"].is_disconnected_or_failed(self.p_amb)):
            mask |= FLAG_PAMB_VALID

        if (EXOGENOUS_SENSOR_SPECS["ambient_temp_k"].is_in_admissible_range(self.t_amb) and
                not EXOGENOUS_SENSOR_SPECS["ambient_temp_k"].is_disconnected_or_failed(self.t_amb)):
            mask |= FLAG_TAMB_VALID

        if (EXOGENOUS_SENSOR_SPECS["true_airspeed_ms"].is_in_admissible_range(self.v_tas) and
                not EXOGENOUS_SENSOR_SPECS["true_airspeed_ms"].is_disconnected_or_failed(self.v_tas)):
            mask |= FLAG_VTAS_VALID

        if (EXOGENOUS_SENSOR_SPECS["pressure_altitude_m"].is_in_admissible_range(self.h_p) and
                not EXOGENOUS_SENSOR_SPECS["pressure_altitude_m"].is_disconnected_or_failed(self.h_p)):
            mask |= FLAG_HP_VALID

        return mask

    def to_dict(self) -> Dict[str, object]:
        """Convert frame to dictionary format matching schema."""
        return {
            "timestamp": float(self.t),
            "egt_1": float(self.egt[0]),
            "egt_2": float(self.egt[1]),
            "egt_3": float(self.egt[2]),
            "egt_4": float(self.egt[3]),
            "cht": float(self.cht),
            "oil_pressure": float(self.p_oil),
            "oil_temperature": float(self.t_oil),
            "engine_speed_rpm": float(self.n_rpm),
            "fuel_flow_kg_s": float(self.mdot_f),
            "map_pa": float(self.map_pa),
            "throttle_position": float(self.tps),
            "intake_temp_k": float(self.t_im),
            "ambient_pressure_pa": float(self.p_amb),
            "ambient_temp_k": float(self.t_amb),
            "true_airspeed_ms": float(self.v_tas),
            "pressure_altitude_m": float(self.h_p),
            "validity_bitmask": int(self.valid_mask),
            "flight_id": str(self.flight_id),
            "aircraft_id": str(self.aircraft_id),
            "engine_id": str(self.engine_id),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> TelemetryFrame:
        """Create TelemetryFrame from dictionary."""
        egt_tuple = (
            float(data.get("egt_1", 0.0)),
            float(data.get("egt_2", 0.0)),
            float(data.get("egt_3", 0.0)),
            float(data.get("egt_4", 0.0)),
        )
        return cls(
            t=float(data.get("timestamp", data.get("t", 0.0))),
            egt=egt_tuple,
            cht=float(data.get("cht", 0.0)),
            p_oil=float(data.get("oil_pressure", data.get("p_oil", 0.0))),
            t_oil=float(data.get("oil_temperature", data.get("t_oil", 0.0))),
            n_rpm=float(data.get("engine_speed_rpm", data.get("n_rpm", 0.0))),
            mdot_f=float(data.get("fuel_flow_kg_s", data.get("mdot_f", 0.0))),
            map_pa=float(data.get("map_pa", 0.0)),
            tps=float(data.get("throttle_position", data.get("tps", 0.0))),
            t_im=float(data.get("intake_temp_k", data.get("t_im", 0.0))),
            p_amb=float(data.get("ambient_pressure_pa", data.get("p_amb", 0.0))),
            t_amb=float(data.get("ambient_temp_k", data.get("t_amb", 0.0))),
            v_tas=float(data.get("true_airspeed_ms", data.get("v_tas", 0.0))),
            h_p=float(data.get("pressure_altitude_m", data.get("h_p", 0.0))),
            valid_mask=int(data.get("validity_bitmask", data.get("valid_mask", FULL_VALID_MASK))),
            flight_id=str(data.get("flight_id", "")),
            aircraft_id=str(data.get("aircraft_id", "")),
            engine_id=str(data.get("engine_id", "")),
        )


# ============================================================================
# 5. Multi-Format Schema Definitions (Polars & PyArrow)
# ============================================================================

TELEMETRY_POLARS_SCHEMA = pl.Schema([
    ("timestamp", pl.Float64),
    ("egt_1", pl.Float64),
    ("egt_2", pl.Float64),
    ("egt_3", pl.Float64),
    ("egt_4", pl.Float64),
    ("cht", pl.Float64),
    ("oil_pressure", pl.Float64),
    ("oil_temperature", pl.Float64),
    ("engine_speed_rpm", pl.Float64),
    ("fuel_flow_kg_s", pl.Float64),
    ("map_pa", pl.Float64),
    ("throttle_position", pl.Float64),
    ("intake_temp_k", pl.Float64),
    ("ambient_pressure_pa", pl.Float64),
    ("ambient_temp_k", pl.Float64),
    ("true_airspeed_ms", pl.Float64),
    ("pressure_altitude_m", pl.Float64),
    ("validity_bitmask", pl.UInt32),
    ("flight_id", pl.String),
    ("aircraft_id", pl.String),
    ("engine_id", pl.String),
])

TELEMETRY_ARROW_SCHEMA = pa.schema([
    ("timestamp", pa.float64(), False),
    ("egt_1", pa.float64(), False),
    ("egt_2", pa.float64(), False),
    ("egt_3", pa.float64(), False),
    ("egt_4", pa.float64(), False),
    ("cht", pa.float64(), False),
    ("oil_pressure", pa.float64(), False),
    ("oil_temperature", pa.float64(), False),
    ("engine_speed_rpm", pa.float64(), False),
    ("fuel_flow_kg_s", pa.float64(), False),
    ("map_pa", pa.float64(), False),
    ("throttle_position", pa.float64(), False),
    ("intake_temp_k", pa.float64(), False),
    ("ambient_pressure_pa", pa.float64(), False),
    ("ambient_temp_k", pa.float64(), False),
    ("true_airspeed_ms", pa.float64(), False),
    ("pressure_altitude_m", pa.float64(), False),
    ("validity_bitmask", pa.uint32(), False),
    ("flight_id", pa.string(), False),
    ("aircraft_id", pa.string(), False),
    ("engine_id", pa.string(), False),
])


# ============================================================================
# 6. Conversion & Frame Validation Utilities
# ============================================================================

def telemetry_frames_to_polars(frames: Sequence[TelemetryFrame]) -> pl.DataFrame:
    """Convert a sequence of TelemetryFrame dataclasses to a validated Polars DataFrame."""
    if not frames:
        return pl.DataFrame(schema=TELEMETRY_POLARS_SCHEMA)
    data = [frame.to_dict() for frame in frames]
    return pl.DataFrame(data, schema=TELEMETRY_POLARS_SCHEMA)


def polars_to_telemetry_frames(df: pl.DataFrame) -> List[TelemetryFrame]:
    """Convert a Polars DataFrame conforming to TELEMETRY_POLARS_SCHEMA to TelemetryFrame list."""
    if df.is_empty():
        return []
    records = df.to_dicts()
    return [TelemetryFrame.from_dict(row) for row in records]


def telemetry_frames_to_arrow(frames: Sequence[TelemetryFrame]) -> pa.Table:
    """Convert a sequence of TelemetryFrame dataclasses to an Apache Arrow Table."""
    df = telemetry_frames_to_polars(frames)
    return df.to_arrow()


def arrow_to_telemetry_frames(table: pa.Table) -> List[TelemetryFrame]:
    """Convert an Apache Arrow Table to a list of TelemetryFrame dataclasses."""
    df = pl.from_arrow(table)
    if isinstance(df, pl.Series):
        df = df.to_frame()
    return polars_to_telemetry_frames(df)


def validate_telemetry_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Validate telemetry DataFrame against schema and recompute validity bitmasks.
    Ensures correct types and flags out-of-bounds / disconnected channels.
    """
    # Verify required column existence
    for col_name in TELEMETRY_POLARS_SCHEMA.names():
        if col_name not in df.columns:
            raise ValueError(f"Missing required telemetry column: {col_name}")

    frames = polars_to_telemetry_frames(df)
    validated_frames = []
    for frame in frames:
        computed_mask = frame.compute_validity_mask()
        # Create updated frame with recomputed validity mask
        validated_frame = TelemetryFrame(
            t=frame.t,
            egt=frame.egt,
            cht=frame.cht,
            p_oil=frame.p_oil,
            t_oil=frame.t_oil,
            n_rpm=frame.n_rpm,
            mdot_f=frame.mdot_f,
            map_pa=frame.map_pa,
            tps=frame.tps,
            t_im=frame.t_im,
            p_amb=frame.p_amb,
            t_amb=frame.t_amb,
            v_tas=frame.v_tas,
            h_p=frame.h_p,
            valid_mask=computed_mask,
            flight_id=frame.flight_id,
            aircraft_id=frame.aircraft_id,
            engine_id=frame.engine_id,
        )
        validated_frames.append(validated_frame)

    return telemetry_frames_to_polars(validated_frames)
