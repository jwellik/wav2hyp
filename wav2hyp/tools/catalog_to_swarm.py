"""
Export events from a PyOcto or NLL HDF5 file to Swarm tagger format.

For each earthquake event the output is one line:

    yyyy-mm-dd HH:MM:SS.ff,STA CHA NET LOC,Earthquake

The SCNL (Station Channel Network Location) must be provided via ``--scnl``
because event catalogs do not carry a single canonical station identifier.

The tool auto-detects whether the input is a PyOcto H5 (has ``events`` key with
Unix-timestamp ``time`` column) or an NLL H5 (has ``catalog_table`` key with
``origin_time`` column).

Usage:
  python -m wav2hyp.tools.catalog_to_swarm <input_h5_file> <output_swarm_file> \\
      --scnl "STA CHA NET LOC" [--t1 START] [--t2 END]
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
    """Return ``'pyocto'`` or ``'nll'`` depending on which HDF5 keys exist."""
    with pd.HDFStore(h5_path, mode="r") as store:
        keys = {k.lstrip("/") for k in store.keys()}
    if _NLL_KEY in keys:
        return "nll"
    if _PYOCTO_KEY in keys:
        return "pyocto"
    raise ValueError(f"Cannot detect H5 type: expected '{_NLL_KEY}' or '{_PYOCTO_KEY}' key in {h5_path}")


def _read_event_times_pyocto(
    h5_path: str,
    t1: Optional[str] = None,
    t2: Optional[str] = None,
) -> list[datetime]:
    events_df = pd.read_hdf(h5_path, key=_PYOCTO_KEY)
    events_df["_dt"] = events_df["time"].apply(lambda t: datetime.fromtimestamp(t, tz=timezone.utc))
    if t1 is not None:
        lo = UTCDateTime(t1).datetime.replace(tzinfo=timezone.utc)
        events_df = events_df[events_df["_dt"] >= lo]
    if t2 is not None:
        hi = UTCDateTime(t2).datetime.replace(tzinfo=timezone.utc)
        events_df = events_df[events_df["_dt"] <= hi]
    return sorted(events_df["_dt"].tolist())


def _read_event_times_nll(
    h5_path: str,
    t1: Optional[str] = None,
    t2: Optional[str] = None,
) -> list[datetime]:
    from wav2hyp.utils.io import NLLOutput
    nll = NLLOutput(h5_path)
    cat_df = nll.read_catalog_table(t1=t1, t2=t2)
    if len(cat_df) == 0:
        return []
    times = pd.to_datetime(cat_df["origin_time"], utc=True, errors="coerce").dropna()
    return sorted(times.dt.to_pydatetime().tolist())


def _format_event_time(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:23]


def export_events_to_swarm(
    h5_path: str,
    output_path: str,
    scnl: str,
    t1: Optional[str] = None,
    t2: Optional[str] = None,
) -> Path:
    """
    Read event origin times from *h5_path* and write a Swarm tagger file.

    Parameters
    ----------
    h5_path : str
        Path to a PyOcto or NLL HDF5 file.
    output_path : str
        Destination file for the Swarm tagger output.
    scnl : str
        Station Channel Network Location string (space-separated), e.g. ``"MSH BHZ UW --"``.
    t1, t2 : str, optional
        UTCDateTime-compatible time bounds on origin time.
    """
    h5_type = _detect_h5_type(h5_path)
    _log.info("Detected H5 type: %s", h5_type)

    if h5_type == "pyocto":
        event_times = _read_event_times_pyocto(h5_path, t1=t1, t2=t2)
    else:
        event_times = _read_event_times_nll(h5_path, t1=t1, t2=t2)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        for dt in event_times:
            f.write(f"{_format_event_time(dt)},{scnl},Earthquake\n")

    _log.info("Wrote %d events to %s", len(event_times), out)
    print(f"Wrote {len(event_times)} events to {out}")
    return out


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export events from a PyOcto or NLL HDF5 file to Swarm tagger format.",
        epilog=(
            "Example:\n"
            '  python -m wav2hyp.tools.catalog_to_swarm pyocto.h5 events.tag '
            '--scnl "MSH BHZ UW --" --t1 2023-01-01 --t2 2023-02-01'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input_h5_file", help="Path to PyOcto or NLL HDF5 file")
    p.add_argument("output_swarm_file", help="Path to output Swarm tagger file")
    p.add_argument("--scnl", required=True, help='Station Channel Network Location, e.g. "MSH BHZ UW --"')
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

    export_events_to_swarm(
        h5_path=str(h5_path),
        output_path=args.output_swarm_file,
        scnl=args.scnl,
        t1=args.t1,
        t2=args.t2,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
