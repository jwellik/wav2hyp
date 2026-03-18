"""
Notebook helpers for St. Helens analysis.

Goals:
- Keep `sthelens_analysis.ipynb` small and clean.
- Centralize data loading from `results/sthelens/*` artifacts.
- Avoid importing `wav2hyp` (which may pull heavyweight ML deps).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import yaml
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.filesystem.sds import Client as SDSClient

from vdapseisutils import VCatalog


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_repo_root() -> Path:
    """
    Find the repo root by looking for `results/sthelens`.
    """
    # Prefer locating from this module's file path (works regardless of notebook CWD).
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "results" / "sthelens").exists():
            return parent

    # Fallback: try current working directory.
    cwd = Path.cwd()
    if (cwd / "results" / "sthelens").exists():
        return cwd

    # Final fallback: assume CWD is repo root (may be wrong if notebook run from a subdir).
    return cwd


def sthelens_paths(base_results_dir: str | Path = None) -> dict:
    """
    Return commonly used paths for the notebook.
    """
    repo_root = infer_repo_root() if base_results_dir is None else Path(base_results_dir).resolve().parent.parent
    base_results_dir = repo_root / "results" / "sthelens"
    return {
        "base_dir": base_results_dir,
        # EQT outputs
        "eqt_picks_h5": base_results_dir / "1_picks" / "eqt-volpick.h5",
        "eqt_picks_h5_alt": base_results_dir / "1_picks" / "eqt-volpick.h5",
        # PyOcto outputs (not required yet by this notebook)
        "pyocto_h5": base_results_dir / "2_associations" / "pyocto.h5",
        # NLL outputs
        "nll_h5": base_results_dir / "3_locations" / "nll.h5",
        # output artifacts
        "output_dir": repo_root / "analysis_local",
    }


def eqt_h5_path(paths: dict) -> Path:
    """
    Support both `results/sthelens/1_picks/...` and `results/sthelens/../1_picks/...`.
    """
    p = paths.get("eqt_picks_h5_alt")
    if p and p.exists():
        return Path(p)
    p = paths.get("eqt_picks_h5")
    if p and p.exists():
        return Path(p)
    raise FileNotFoundError("Could not find EQT picks HDF5 under results/sthelens/1_picks/")


def load_nll_catalog_from_catalog_table(
    nll_h5_path: str | Path,
    t1: Optional[str | UTCDateTime] = None,
    t2: Optional[str | UTCDateTime] = None,
) -> VCatalog:
    """
    Build a lightweight VCatalog from `nll.h5` `catalog_table`.

    This is used for map/cross-section plotting where pick-by-pick detail
    is not required.
    """
    nll_h5_path = Path(nll_h5_path)
    df = pd.read_hdf(nll_h5_path, key="catalog_table")
    # Normalize to tz-aware UTC for robust comparisons.
    df["origin_time"] = pd.to_datetime(df["origin_time"], utc=True)

    if t1 is not None:
        t1_utc = pd.Timestamp(UTCDateTime(t1).datetime, tz="UTC")
        df = df[df["origin_time"] >= t1_utc]
    if t2 is not None:
        t2_utc = pd.Timestamp(UTCDateTime(t2).datetime, tz="UTC")
        df = df[df["origin_time"] <= t2_utc]

    # Ensure required columns exist
    required = ["origin_time", "latitude", "longitude", "depth_km"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"catalog_table missing columns: {missing}. Have: {list(df.columns)}")

    # VolcanoFigure prefers magnitude-based sizing unless overridden; keep it numeric and finite.
    if "mag" not in df.columns:
        df["mag"] = 1.0
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce").fillna(1.0).astype(float)

    # VCatalog.from_txyzm expects: time, latitude, longitude, depth, mag
    cat_df = pd.DataFrame(
        {
            "time": pd.to_datetime(df["origin_time"], utc=True).dt.tz_convert("UTC"),
            "latitude": df["latitude"].astype(float),
            "longitude": df["longitude"].astype(float),
            "depth": df["depth_km"].astype(float),
            "mag": df["mag"].astype(float),
        }
    )

    return VCatalog.from_txyzm(cat_df)


def load_eqt_picks_and_detections(
    eqt_h5_path: str | Path,
    t1: Optional[str | UTCDateTime] = None,
    t2: Optional[str | UTCDateTime] = None,
):
    """
    Load EQTransformer picks/detections directly as pandas tables.

    Returns
    -------
    picks_df, detections_df
    """
    eqt_h5_path = Path(eqt_h5_path)
    picks_df = pd.read_hdf(eqt_h5_path, key="picks").sort_values("peak_time")
    detections_df = pd.read_hdf(eqt_h5_path, key="detections").sort_values("start_time")

    if t1 is not None:
        t1_dt = UTCDateTime(t1).datetime if not hasattr(t1, "datetime") else t1.datetime
        picks_df = picks_df[picks_df["peak_time"] >= pd.Timestamp(t1_dt)]
        detections_df = detections_df[detections_df["start_time"] >= pd.Timestamp(t1_dt)]
    if t2 is not None:
        t2_dt = UTCDateTime(t2).datetime if not hasattr(t2, "datetime") else t2.datetime
        picks_df = picks_df[picks_df["peak_time"] <= pd.Timestamp(t2_dt)]
        detections_df = detections_df[detections_df["start_time"] <= pd.Timestamp(t2_dt)]

    return picks_df, detections_df


def eqt_trace_id(network: str, station: str) -> str:
    """
    EQT trace_id format in your HDF5: e.g. `UW.HSR.`
    """
    return f"{network}.{station}."


def filter_eqt_for_station_phase(
    picks_df: pd.DataFrame,
    network: str,
    station: str,
    phase: str,
    t1: Optional[str | UTCDateTime] = None,
    t2: Optional[str | UTCDateTime] = None,
) -> list[UTCDateTime]:
    """
    Return pick times (UTCDateTime) for a given station + phase.
    """
    trace_id = eqt_trace_id(network, station)
    df = picks_df[(picks_df["trace_id"] == trace_id) & (picks_df["phase"] == phase)].copy()

    if t1 is not None:
        t1_dt = UTCDateTime(t1).datetime if not hasattr(t1, "datetime") else t1.datetime
        df = df[df["peak_time"] >= pd.Timestamp(t1_dt)]
    if t2 is not None:
        t2_dt = UTCDateTime(t2).datetime if not hasattr(t2, "datetime") else t2.datetime
        df = df[df["peak_time"] <= pd.Timestamp(t2_dt)]

    df.sort_values("peak_time", inplace=True)
    return [UTCDateTime(t.to_pydatetime()) if hasattr(t, "to_pydatetime") else UTCDateTime(t) for t in df["peak_time"].values]


def detections_intervals_for_station(
    detections_df: pd.DataFrame,
    network: str,
    station: str,
    t1: Optional[str | UTCDateTime] = None,
    t2: Optional[str | UTCDateTime] = None,
) -> list[tuple[UTCDateTime, UTCDateTime]]:
    """
    Return (start, end) intervals for Helicorder.highlight().
    """
    trace_id = eqt_trace_id(network, station)
    df = detections_df[detections_df["trace_id"] == trace_id].copy()

    if t1 is not None:
        t1_dt = UTCDateTime(t1).datetime if not hasattr(t1, "datetime") else t1.datetime
        df = df[df["start_time"] >= pd.Timestamp(t1_dt)]
    if t2 is not None:
        t2_dt = UTCDateTime(t2).datetime if not hasattr(t2, "datetime") else t2.datetime
        df = df[df["start_time"] <= pd.Timestamp(t2_dt)]

    df.sort_values("start_time", inplace=True)
    intervals = []
    for start, end in zip(df["start_time"].values, df["end_time"].values):
        intervals.append((UTCDateTime(start), UTCDateTime(end)))
    return intervals


def located_pick_times_from_nll(
    nll_h5_path: str | Path,
    station: str,
    phase: str,
    t1: Optional[str | UTCDateTime] = None,
    t2: Optional[str | UTCDateTime] = None,
) -> list[UTCDateTime]:
    """
    Extract located pick times for a given station + phase from NLL arrivals_table.

    Reads only the HDF5 arrivals_table (no QuakeML). Raises if the file or
    arrivals_table is missing.
    """
    nll_h5_path = Path(nll_h5_path)
    if not nll_h5_path.exists():
        raise FileNotFoundError(f"nll.h5 not found: {nll_h5_path}")
    if t1 is not None:
        t1 = UTCDateTime(t1)
    if t2 is not None:
        t2 = UTCDateTime(t2)

    phase_u = str(phase).strip().upper()
    if phase_u.startswith("P"):
        phase_u = "P"
    elif phase_u.startswith("S"):
        phase_u = "S"

    def _iso_time(t: UTCDateTime) -> str:
        return pd.Timestamp(t.datetime, tz="UTC").isoformat()

    where_parts = []
    if t1 is not None and t2 is not None:
        where_parts.append(
            f"(arrival_time >= '{_iso_time(t1)}') & (arrival_time <= '{_iso_time(t2)}')"
        )
    elif t1 is not None:
        where_parts.append(f"(arrival_time >= '{_iso_time(t1)}')")
    elif t2 is not None:
        where_parts.append(f"(arrival_time <= '{_iso_time(t2)}')")

    where_phase = f"(phase == '{phase_u}')"
    where = f"{where_phase}" if not where_parts else f"({' & '.join(where_parts)}) & {where_phase}"

    try:
        df = pd.read_hdf(nll_h5_path, key="arrivals_table", where=where)
    except KeyError:
        raise KeyError(f"arrivals_table not found in {nll_h5_path}") from None

    if len(df) == 0:
        return []

    if "trace_id" in df.columns:
        sta_codes = df["trace_id"].astype(str).str.split(".").str[1]
        mask = sta_codes == station
    elif "station_id" in df.columns:
        sta_codes = df["station_id"].astype(str).str.split(".").str[1]
        mask = sta_codes == station
    else:
        mask = pd.Series([False] * len(df))

    out_times: list[UTCDateTime] = []
    for ts in df.loc[mask, "arrival_time"].values:
        if hasattr(ts, "to_pydatetime"):
            out_times.append(UTCDateTime(ts.to_pydatetime()))
        else:
            out_times.append(UTCDateTime(ts))

    out_times.sort()
    return out_times


# Backward-compat alias (nll.h5 is now table-only; no QuakeML)
located_pick_times_from_nll_quakeml = located_pick_times_from_nll


def load_hr_stream_from_sds(
    datasource_root: str | Path,
    network: str,
    station: str,
    starttime: UTCDateTime,
    endtime: UTCDateTime,
) -> "object":
    """
    Load HSR waveforms from the SDS datasource and return an ObsPy Stream.
    """
    client = SDSClient(str(datasource_root))
    # Match example notebooks: ?H? channels (HHZ/HHE/HHN etc)
    st = client.get_waveforms(network, station, "*", "?H?", starttime=starttime, endtime=endtime)
    return st.merge()


@dataclass(frozen=True)
class HelicoorderStyle:
    """
    Centralize marker styling for helicorder overlays.
    """
    all_p_color: str = "white"
    all_p_edge: str = "red"
    located_p_color: str = "red"
    located_p_edge: str = "blue"
    all_s_color: str = "white"
    all_s_edge: str = "blue"
    located_s_color: str = "blue"
    located_s_edge: str = "red"
    marker_size: int = 5
    located_marker_size: int = 6

