"""
Build `station_summary.txt` by joining per-stage station-summary slices stored in HDF5.

The WAV2HYP pipeline writes station-summary *stage slices* during processing:
- picker stage: pick/detection metrics per (date, trace_id)
- associator stage: association metrics per (date, trace_id)
- locator stage: event-count metrics per (date, trace_id)

This tool loads those slices from the stage H5 files and merges them into a single
CSV text file with columns matching `wav2hyp.utils.summary.STATION_SUMMARY_HEADER`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from obspy import UTCDateTime

from wav2hyp.utils.summary import (
    STATION_SUMMARY_HEADER,
    STATION_SUMMARY_H5_ROOT,
)

from wav2hyp.utils.io import _parse_summary_date

_log = logging.getLogger("wav2hyp")


def _load_stage_slices(
    h5_path: str,
    step: str,
    t1: Optional[str] = None,
    t2: Optional[str] = None,
) -> pd.DataFrame:
    path = Path(h5_path)
    if not path.exists():
        return pd.DataFrame(columns=STATION_SUMMARY_HEADER)

    prefix = f"{STATION_SUMMARY_H5_ROOT}/{step}/"
    with pd.HDFStore(str(path), mode="r") as store:
        keys = [k for k in store.keys() if k.startswith(prefix)]

        if not keys:
            return pd.DataFrame(columns=STATION_SUMMARY_HEADER)

        out_frames: list[pd.DataFrame] = []
        ts_lo = None
        ts_hi = None
        if t1 is not None and t2 is not None:
            ts_lo = pd.Timestamp(UTCDateTime(t1).datetime, tz="UTC")
            ts_hi = pd.Timestamp(UTCDateTime(t2).datetime, tz="UTC")

        for key in keys:
            df = store.get(key)
            if df is None or len(df) == 0:
                continue

            # Filter by date range if requested.
            if ts_lo is not None and ts_hi is not None and "date" in df.columns:
                mask = []
                for d in df["date"]:
                    ts = _parse_summary_date(d)
                    mask.append(ts is not None and ts_lo <= ts <= ts_hi)
                df = df.loc[mask]
                if len(df) == 0:
                    continue

            out_frames.append(df)

    if not out_frames:
        return pd.DataFrame(columns=STATION_SUMMARY_HEADER)
    return pd.concat(out_frames, ignore_index=True)


def write_station_summary_txt_from_stage_slices(
    picker_h5_path: str,
    associator_h5_path: str,
    locator_h5_path: str,
    output_txt_path: str,
    t1: Optional[str] = None,
    t2: Optional[str] = None,
) -> Path:
    """
    Merge stage slices into a single station-summary CSV text file.

    Parameters
    ----------
    picker_h5_path, associator_h5_path, locator_h5_path:
        Stage HDF5 paths written by the WAV2HYP pipeline.
    output_txt_path:
        Where to write the merged `station_summary.txt` (CSV).
    t1, t2:
        Optional UTCDateTime-compatible time range strings to limit output.

    Returns
    -------
    Path
        The written output file path.
    """
    picker_df = _load_stage_slices(picker_h5_path, "picker", t1=t1, t2=t2)
    associator_df = _load_stage_slices(associator_h5_path, "associator", t1=t1, t2=t2)
    locator_df = _load_stage_slices(locator_h5_path, "locator", t1=t1, t2=t2)

    # Ensure required join columns exist.
    for df in (picker_df, associator_df, locator_df):
        if "date" not in df.columns:
            df["date"] = []
        if "trace_id" not in df.columns:
            df["trace_id"] = []

    merged = pd.DataFrame(columns=["date", "trace_id"])
    if len(picker_df) > 0:
        merged = picker_df[["date", "trace_id", "nsamples", "ncha", "np", "ns", "nd"]].copy()

    if len(associator_df) > 0:
        assoc_cols = [c for c in ["date", "trace_id", "nassign", "nassoc"] if c in associator_df.columns]
        if len(assoc_cols) >= 4:
            merged = merged.merge(
                associator_df[assoc_cols],
                on=["date", "trace_id"],
                how="outer",
            )
        else:
            merged = pd.DataFrame(columns=["date", "trace_id"])
            merged = merged.merge(
                associator_df[assoc_cols],
                on=["date", "trace_id"],
                how="outer",
            )

    if len(locator_df) > 0:
        loc_cols = [c for c in ["date", "trace_id", "nevents"] if c in locator_df.columns]
        merged = merged.merge(
            locator_df[loc_cols],
            on=["date", "trace_id"],
            how="outer",
        )

    # Fill missing numeric fields with 0 and enforce integer dtype.
    numeric_cols = ["nsamples", "ncha", "np", "ns", "nd", "nassign", "nassoc", "nevents"]
    for c in numeric_cols:
        if c not in merged.columns:
            merged[c] = 0
        merged[c] = merged[c].fillna(0).astype(int)

    merged = merged[STATION_SUMMARY_HEADER].sort_values(["date", "trace_id"]).reset_index(drop=True)

    out_path = Path(output_txt_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    _log.info("Wrote station summary to %s (%d rows)", out_path, len(merged))
    return out_path


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge stage slices into station_summary.txt",
    )
    p.add_argument("--picker-h5", required=True, help="Path to picker HDF5 (eqt-volpick.h5)")
    p.add_argument("--associator-h5", required=True, help="Path to associator HDF5 (pyocto.h5)")
    p.add_argument("--locator-h5", required=True, help="Path to locator HDF5 (nll.h5)")
    p.add_argument("output_txt_path", help="Output station_summary CSV/text file path")
    p.add_argument("--t1", default=None, help="Optional UTC start time to filter output")
    p.add_argument("--t2", default=None, help="Optional UTC end time to filter output")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logging.getLogger("wav2hyp").setLevel(logging.DEBUG)

    write_station_summary_txt_from_stage_slices(
        picker_h5_path=args.picker_h5,
        associator_h5_path=args.associator_h5,
        locator_h5_path=args.locator_h5,
        output_txt_path=args.output_txt_path,
        t1=args.t1,
        t2=args.t2,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

