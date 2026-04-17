"""
Export an earthquake catalog from a PyOcto or NLL HDF5 file to a simple CSV.

Output columns:

    datetime,lat,lon,depth,mag

- For **NLL** H5 files: reads ``catalog_table`` which already contains
  ``origin_time``, ``latitude``, ``longitude``, ``depth_km``, and ``mag``.
- For **PyOcto** H5 files: reads ``events`` which has ``time`` (Unix seconds),
  ``x``, ``y``, ``z`` in local km coordinates. Latitude/longitude are included
  only if the events DataFrame already contains them; otherwise they are written
  as empty.

Usage:
  python -m wav2hyp.tools.catalog_to_csv <input_h5_file> <output_csv_file> \\
      [--t1 START] [--t2 END]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from obspy import UTCDateTime

_log = logging.getLogger("wav2hyp")

_PYOCTO_KEY = "events"
_NLL_KEY = "catalog_table"


def _detect_h5_type(h5_path: str) -> str:
    with pd.HDFStore(h5_path, mode="r") as store:
        keys = {k.lstrip("/") for k in store.keys()}
    if _NLL_KEY in keys:
        return "nll"
    if _PYOCTO_KEY in keys:
        return "pyocto"
    raise ValueError(f"Cannot detect H5 type: expected '{_NLL_KEY}' or '{_PYOCTO_KEY}' key in {h5_path}")


def _catalog_from_nll(
    h5_path: str,
    t1: Optional[str] = None,
    t2: Optional[str] = None,
) -> pd.DataFrame:
    from wav2hyp.utils.io import NLLOutput
    nll = NLLOutput(h5_path)
    cat = nll.read_catalog_table(t1=t1, t2=t2)
    if len(cat) == 0:
        return pd.DataFrame(columns=["datetime", "lat", "lon", "depth", "mag"])
    out = pd.DataFrame()
    out["datetime"] = pd.to_datetime(cat["origin_time"], utc=True, errors="coerce")
    out["lat"] = cat["latitude"]
    out["lon"] = cat["longitude"]
    out["depth"] = cat["depth_km"]
    out["mag"] = cat.get("mag", pd.Series([None] * len(cat)))
    return out.sort_values("datetime").reset_index(drop=True)


def _catalog_from_pyocto(
    h5_path: str,
    t1: Optional[str] = None,
    t2: Optional[str] = None,
) -> pd.DataFrame:
    events = pd.read_hdf(h5_path, key=_PYOCTO_KEY)
    events["_dt"] = events["time"].apply(lambda t: datetime.fromtimestamp(t, tz=timezone.utc))
    if t1 is not None:
        lo = UTCDateTime(t1).datetime.replace(tzinfo=timezone.utc)
        events = events[events["_dt"] >= lo]
    if t2 is not None:
        hi = UTCDateTime(t2).datetime.replace(tzinfo=timezone.utc)
        events = events[events["_dt"] <= hi]
    if len(events) == 0:
        return pd.DataFrame(columns=["datetime", "lat", "lon", "depth", "mag"])

    out = pd.DataFrame()
    out["datetime"] = events["_dt"].values
    out["lat"] = events["latitude"] if "latitude" in events.columns else None
    out["lon"] = events["longitude"] if "longitude" in events.columns else None
    out["depth"] = events["z"] if "z" in events.columns else None
    out["mag"] = events["mag"] if "mag" in events.columns else None
    return out.sort_values("datetime").reset_index(drop=True)


def export_catalog_csv(
    h5_path: str,
    output_path: str,
    t1: Optional[str] = None,
    t2: Optional[str] = None,
) -> Path:
    """
    Read event catalog from *h5_path* and write a CSV with columns
    ``datetime,lat,lon,depth,mag``.

    Parameters
    ----------
    h5_path : str
        Path to a PyOcto or NLL HDF5 file.
    output_path : str
        Destination CSV file.
    t1, t2 : str, optional
        UTCDateTime-compatible time bounds on origin time.
    """
    h5_type = _detect_h5_type(h5_path)
    _log.info("Detected H5 type: %s", h5_type)

    if h5_type == "nll":
        catalog = _catalog_from_nll(h5_path, t1=t1, t2=t2)
    else:
        catalog = _catalog_from_pyocto(h5_path, t1=t1, t2=t2)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out, index=False)

    _log.info("Wrote %d events to %s", len(catalog), out)
    print(f"Wrote {len(catalog)} events to {out}")
    return out


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export earthquake catalog from a PyOcto or NLL HDF5 file to CSV.",
        epilog=(
            "Example:\n"
            "  python -m wav2hyp.tools.catalog_to_csv nll.h5 catalog.csv "
            "--t1 2023-01-01 --t2 2023-02-01"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input_h5_file", help="Path to PyOcto or NLL HDF5 file")
    p.add_argument("output_csv_file", help="Path to output CSV file")
    p.add_argument("--t1", default=None, help="Start time (UTC) to filter events")
    p.add_argument("--t2", default=None, help="End time (UTC) to filter events")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logging.getLogger("wav2hyp").setLevel(logging.DEBUG)

    h5_path = Path(args.input_h5_file)
    if not h5_path.exists():
        print(f"Error: HDF5 file not found: {h5_path}", file=sys.stderr)
        return 1

    export_catalog_csv(
        h5_path=str(h5_path),
        output_path=args.output_csv_file,
        t1=args.t1,
        t2=args.t2,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
