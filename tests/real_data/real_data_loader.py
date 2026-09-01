"""
SIH26054 Replan to Learn: Real Data Loader.
Maps processed flight parquet files to TelemetryFrame for Stage 3 testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from replan_to_learn.contracts.telemetry import TelemetryFrame, FrameStatus
from replan_to_learn.contracts.telemetry import (
    celsius_to_kelvin,
    knots_to_m_s,
    bar_to_pa,
    pa_to_bar,
    liters_per_hour_to_kg_s,
)


REAL_COLUMN_MAP = {
    "E1 EGT1": "egt1",
    "E1 EGT2": "egt2",
    "E1 EGT3": "egt3",
    "E1 EGT4": "egt4",
    "E1 CHT1": "cht1",
    "E1 CHT2": "cht2",
    "E1 CHT3": "cht3",
    "E1 CHT4": "cht4",
    "E1 OilP": "p_oil",
    "E1 OilT": "t_oil",
    "E1 RPM": "n_rpm",
    "E1 FFlow": "mdot_f",
    "OAT": "t_amb",
    "IAS": "ias_kt",
    "AltMSL": "h_p",
}


def _fahrenheit_to_kelvin(temp_f: float) -> float:
    """NGAFID EDM temperature channels (EGT/CHT/OilT) are reported in degrees Fahrenheit."""
    return (temp_f - 32.0) * 5.0 / 9.0 + 273.15


def _safe_float(value, default: float) -> float:
    """Safely convert a value to float, replacing None/NaN with default."""
    if value is None:
        return float(default)
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return float(default)
        return v
    except (TypeError, ValueError):
        return float(default)


class RealFlightLoader:
    """
    Loads processed flight parquet files and maps to TelemetryFrame sequence.

    Data source units (from processed parquet; NGAFID EDM export convention):
    - E1 EGT1..4: Fahrenheit
    - E1 CHT1..4: Fahrenheit
    - E1 OilP: PSI (pounds per square inch)
    - E1 OilT: Fahrenheit
    - E1 RPM: rpm
    - E1 FFlow: gallons per hour (GPH)
    - OAT: Celsius
    - IAS: knots
    - AltMSL: feet
    """

    OILP_PSI_TO_PA = 6894.757293168361
    FUEL_DENSITY_KG_L = 0.72

    def __init__(self, data_root: Path) -> None:
        self.data_root = Path(data_root)
        self.processed_dir = self.data_root / "processed"
        self.manifest_path = self.data_root / "flight_manifest.parquet"

    def load_manifest(self) -> pl.DataFrame:
        return pl.read_parquet(self.manifest_path)

    def load_flight(self, flight_id: str) -> Optional[pl.DataFrame]:
        path = self.processed_dir / f"flight_{flight_id}.parquet"
        if not path.exists():
            return None
        return pl.read_parquet(path)

    def to_telemetry_frames(self, flight_df: pl.DataFrame, flight_id: str) -> list:
        frames = []
        df = flight_df.to_pandas()
        n_rows = len(df)

        for idx in range(n_rows):
            row = df.iloc[idx]
            t = float(row.get("timestep", idx)) if row.get("timestep", idx) is not None else float(idx)

            egt_raw = (
                _safe_float(row.get("E1 EGT1"), 680.0),
                _safe_float(row.get("E1 EGT2"), 680.0),
                _safe_float(row.get("E1 EGT3"), 680.0),
                _safe_float(row.get("E1 EGT4"), 680.0),
            )
            egt = (
                _fahrenheit_to_kelvin(egt_raw[0]),
                _fahrenheit_to_kelvin(egt_raw[1]),
                _fahrenheit_to_kelvin(egt_raw[2]),
                _fahrenheit_to_kelvin(egt_raw[3]),
            )

            cht_values = [
                _safe_float(row.get("E1 CHT1"), 200.0),
                _safe_float(row.get("E1 CHT2"), 200.0),
                _safe_float(row.get("E1 CHT3"), 200.0),
                _safe_float(row.get("E1 CHT4"), 200.0),
            ]
            cht_k_values = [_fahrenheit_to_kelvin(v) for v in cht_values]
            cht = float(np.mean(cht_k_values))

            oilp_psi = _safe_float(row.get("E1 OilP"), 50.0)
            p_oil = oilp_psi * self.OILP_PSI_TO_PA

            oilt_f = _safe_float(row.get("E1 OilT"), 190.0)
            t_oil = _fahrenheit_to_kelvin(oilt_f)

            n_rpm = _safe_float(row.get("E1 RPM"), 2000.0)

            fflow_gph = _safe_float(row.get("E1 FFlow"), 5.0)
            fflow_lph = fflow_gph * 3.78541
            mdot_f = liters_per_hour_to_kg_s(fflow_lph, self.FUEL_DENSITY_KG_L)

            oat_c = _safe_float(row.get("OAT"), 15.0)
            t_amb = celsius_to_kelvin(oat_c)

            ias_kt = _safe_float(row.get("IAS"), 0.0)
            v_tas = knots_to_m_s(ias_kt)
            v_tas = max(0.1, v_tas)

            h_p_ft = _safe_float(row.get("AltMSL"), 3000.0)
            h_p = h_p_ft * 0.3048

            map_pa = 101325.0 * (1.0 - h_p / 44330.0) ** 5.2561
            map_pa = max(20000.0, min(120000.0, map_pa))

            tps = 0.5
            t_im = t_amb + 5.0

            valid_mask = 0xFFFF
            for i, egt_val in enumerate(egt):
                if egt_val < 273.15 or egt_val > 1423.15 or np.isnan(egt_val):
                    valid_mask &= ~(1 << i)
            if cht < 233.15 or cht > 438.15 or np.isnan(cht):
                valid_mask &= ~0x0010
            if p_oil < 20000.0 or p_oil > 950000.0 or np.isnan(p_oil):
                valid_mask &= ~0x0020
            if t_oil < 233.15 or t_oil > 433.15 or np.isnan(t_oil):
                valid_mask &= ~0x0040
            if n_rpm < 0.0 or n_rpm > 6500.0 or np.isnan(n_rpm):
                valid_mask &= ~0x0080
            if mdot_f < 0.0 or mdot_f > 0.01875 or np.isnan(mdot_f):
                valid_mask &= ~0x0100
            if np.isnan(map_pa) or map_pa < 25000.0 or map_pa > 120000.0:
                valid_mask &= ~0x0200
            if np.isnan(t_im) or t_im < 223.15 or t_im > 393.15:
                valid_mask &= ~0x0800
            if np.isnan(t_amb) or t_amb < 203.15 or t_amb > 348.15:
                valid_mask &= ~0x2000
            if v_tas < 0.0 or v_tas > 130.0 or np.isnan(v_tas):
                valid_mask &= ~0x4000
            if h_p < -600.0 or h_p > 10000.0 or np.isnan(h_p):
                valid_mask &= ~0x8000

            status = FrameStatus.STATUS_OK if valid_mask == 0xFFFF else FrameStatus.STATUS_DEGRADED_INPUT

            frames.append(TelemetryFrame(
                t=t,
                egt=egt,
                cht=cht,
                p_oil=p_oil,
                t_oil=t_oil,
                n_rpm=n_rpm,
                mdot_f=mdot_f,
                map_pa=map_pa,
                tps=tps,
                t_im=t_im,
                p_amb=map_pa,
                t_amb=t_amb,
                v_tas=v_tas,
                h_p=h_p,
                valid_mask=valid_mask,
                flight_id=flight_id,
                aircraft_id=f"AC_{flight_id}",
                engine_id=f"ENG_{flight_id}",
            ))
        return frames
