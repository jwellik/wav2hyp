"""
Export picks from a WAV2HYP picks HDF5 file to Swarm tagger format.

Reads the 'picks' table (trace_id, peak_time, peak_value, phase) and writes
one line per pick in the Swarm tagger CSV format:

    yyyy-mm-dd HH:MM:SS.ff,STA CHA NET LOC,P|S|D

trace_id is stored as NET.STA.LOC.CHA and is re-ordered to Swarm's SCNL
convention (STA CHA NET LOC, space-separated).

Usage:
  python -m wav2hyp.tools.picks_to_swarm <input_h5_file> <output_swarm_file> \\
      [--t1 START] [--t2 END] [--thresh 0.3] [--pthresh 0.5] [--sthresh 0.3]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from obspy import UTCDateTime

_log = logging.getLogger("wav2hyp")


def trace_id_to_scnl(trace_id: str) -> str:
    """Convert FDSN trace_id ``NET.STA.LOC.CHA`` to Swarm ``STA CHA NET LOC``."""
    parts = str(trace_id).split(".")
    if len(parts) == 4:
        net, sta, loc, cha = parts
    elif len(parts) == 3:
        net, sta, cha = parts
        loc = "--"
    elif len(parts) == 2:
        net, sta = parts
        cha, loc = "--", "--"
    else:
        return trace_id
    return f"{sta} {cha} {net} {loc}"


def _format_pick_time(ts) -> str:
    """Format a pandas Timestamp / datetime to ``yyyy-mm-dd HH:MM:SS.ff``."""
    dt = pd.Timestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:23]


def _swarm_phase_label(phase: str) -> str:
    """Map internal phase names to Swarm tag labels (P, S, or D for detections)."""
    p = str(phase).strip().upper()
    if p.startswith("P"):
        return "P"
    if p.startswith("S"):
        return "S"
    return "D"


def export_picks_to_swarm(
    h5_path: str,
    output_path: str,
    t1: Optional[str] = None,
    t2: Optional[str] = None,
    thresh: Optional[float] = None,
    pthresh: Optional[float] = None,
    sthresh: Optional[float] = None,
    dthresh: Optional[float] = None,
) -> Path:
    """
    Read the picks table from *h5_path* and write a Swarm tagger file.

    Parameters
    ----------
    h5_path : str
        Path to a WAV2HYP picks HDF5 file (e.g. ``eqt-volpick.h5``).
    output_path : str
        Destination file for the Swarm tagger output.
    t1, t2 : str, optional
        UTCDateTime-compatible time bounds on ``peak_time``.
    thresh : float, optional
        Global minimum ``peak_value`` threshold applied to all phases.
    pthresh, sthresh, dthresh : float, optional
        Per-phase thresholds (override *thresh* for the corresponding phase).
    """
    where = None
    clauses: list[str] = []
    if t1 is not None:
        ts1 = pd.Timestamp(UTCDateTime(t1).datetime, tz="UTC")
        clauses.append(f"peak_time >= '{ts1}'")
    if t2 is not None:
        ts2 = pd.Timestamp(UTCDateTime(t2).datetime, tz="UTC")
        clauses.append(f"peak_time <= '{ts2}'")
    if thresh is not None:
        clauses.append(f"peak_value >= {thresh}")
    if clauses:
        where = " & ".join(clauses)

    try:
        if where:
            try:
                picks = pd.read_hdf(h5_path, key="picks", where=where)
            except (TypeError, ValueError):
                picks = pd.read_hdf(h5_path, key="picks")
                if t1 is not None:
                    picks = picks[picks["peak_time"] >= pd.Timestamp(UTCDateTime(t1).datetime, tz="UTC")]
                if t2 is not None:
                    picks = picks[picks["peak_time"] <= pd.Timestamp(UTCDateTime(t2).datetime, tz="UTC")]
                if thresh is not None:
                    picks = picks[picks["peak_value"] >= thresh]
        else:
            picks = pd.read_hdf(h5_path, key="picks")
    except KeyError:
        print(f"Error: no 'picks' table in {h5_path}", file=sys.stderr)
        return Path(output_path)

    if pthresh is not None:
        mask_p = picks["phase"].str.upper().str.startswith("P")
        picks = picks[~mask_p | (picks["peak_value"] >= pthresh)]
    if sthresh is not None:
        mask_s = picks["phase"].str.upper().str.startswith("S")
        picks = picks[~mask_s | (picks["peak_value"] >= sthresh)]
    if dthresh is not None:
        mask_d = ~(picks["phase"].str.upper().str.startswith("P") | picks["phase"].str.upper().str.startswith("S"))
        picks = picks[~mask_d | (picks["peak_value"] >= dthresh)]

    picks = picks.sort_values("peak_time").reset_index(drop=True)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        for _, row in picks.iterrows():
            tid = row.get("trace_id", row.get("channel_id", row.get("station_id", "")))
            scnl = trace_id_to_scnl(str(tid))
            time_str = _format_pick_time(row["peak_time"])
            phase = _swarm_phase_label(row.get("phase", "D"))
            f.write(f"{time_str},{scnl},{phase}\n")

    _log.info("Wrote %d picks to %s", len(picks), out)
    print(f"Wrote {len(picks)} picks to {out}")
    return out


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export picks from a WAV2HYP HDF5 file to Swarm tagger format.",
        epilog=(
            "Example:\n"
            "  python -m wav2hyp.tools.picks_to_swarm eqt-volpick.h5 picks.tag "
            "--t1 2023-01-01 --t2 2023-02-01 --pthresh 0.5 --sthresh 0.3"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input_h5_file", help="Path to picks HDF5 file (e.g. eqt-volpick.h5)")
    p.add_argument("output_swarm_file", help="Path to output Swarm tagger file")
    p.add_argument("--t1", default=None, help="Start time (UTC) to filter picks")
    p.add_argument("--t2", default=None, help="End time (UTC) to filter picks")
    p.add_argument("--thresh", type=float, default=None, help="Global min peak_value threshold for all phases")
    p.add_argument("--pthresh", type=float, default=None, help="Min peak_value threshold for P picks")
    p.add_argument("--sthresh", type=float, default=None, help="Min peak_value threshold for S picks")
    p.add_argument("--dthresh", type=float, default=None, help="Min peak_value threshold for D (detection) picks")
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

    export_picks_to_swarm(
        h5_path=str(h5_path),
        output_path=args.output_swarm_file,
        t1=args.t1,
        t2=args.t2,
        thresh=args.thresh,
        pthresh=args.pthresh,
        sthresh=args.sthresh,
        dthresh=args.dthresh,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
