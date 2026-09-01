"""
Loader for the real Rotax 915 iS telemetry fixture at
data/ntsb_915is/wpr22la211_n46jh_telemetry.parquet (see the README next to
it for provenance). Unlike real_data_loader.py (NGAFID, cross-engine), this
is genuine 915 iS Dynon CAN-bus data, already converted to TelemetryFrame
units.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import polars as pl

from replan_to_learn.contracts.telemetry import TelemetryFrame

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "data" / "ntsb_915is" / "wpr22la211_n46jh_telemetry.parquet"


def load_ntsb_915is_fixture(path: Path = FIXTURE_PATH) -> List[TelemetryFrame]:
    df = pl.read_parquet(path)
    frames = []
    for row in df.iter_rows(named=True):
        frames.append(TelemetryFrame(
            t=row["t"], egt=(row["egt1"], row["egt2"], row["egt3"], row["egt4"]),
            cht=row["cht"], p_oil=row["p_oil"], t_oil=row["t_oil"], n_rpm=row["n_rpm"],
            mdot_f=row["mdot_f"], map_pa=row["map_pa"], tps=row["tps"], t_im=row["t_im"],
            p_amb=row["p_amb"], t_amb=row["t_amb"], v_tas=row["v_tas"], h_p=row["h_p"],
            valid_mask=row["valid_mask"], flight_id="WPR22LA211", aircraft_id="N46JH",
            engine_id="915iS_SN13785",
        ))
    return frames
