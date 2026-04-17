"""
Export an earthquake catalog from a PyOcto or NLL HDF5 file to QuakeML.

- For **NLL** H5 files: uses ``NLLOutput.export_catalog()`` which rebuilds full
  ObsPy Event/Origin objects (and optionally Picks/Arrivals) from the HDF5 tables.
- For **PyOcto** H5 files: uses ``PyOctoOutput.read()`` which returns a VCatalog
  built from the events and assignments DataFrames.

By default only origins are included. Pass ``--include-arrivals`` to attach
picks and arrivals to each event.

Usage:
  python -m wav2hyp.tools.catalog_to_quakeml <input_h5_file> <output_quakeml> \\
      [--t1 START] [--t2 END] [--include-arrivals]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

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


def _export_nll(
    h5_path: str,
    output_path: str,
    t1: Optional[str] = None,
    t2: Optional[str] = None,
    include_arrivals: bool = False,
) -> str:
    from wav2hyp.utils.io import NLLOutput
    nll = NLLOutput(h5_path)
    catalog, _ = nll.read(t1=t1, t2=t2, include_arrivals=include_arrivals)
    if catalog is None or len(catalog) == 0:
        print("No events found; nothing to export.", file=sys.stderr)
        return output_path
    catalog.write(output_path, format="QUAKEML")
    return output_path


def _export_pyocto(
    h5_path: str,
    output_path: str,
    t1: Optional[str] = None,
    t2: Optional[str] = None,
    include_arrivals: bool = False,
) -> str:
    from wav2hyp.utils.io import PyOctoOutput
    pyocto = PyOctoOutput(h5_path)
    catalog, events_df, assignments_df, metadata = pyocto.read(t1=t1, t2=t2)
    if catalog is None or len(catalog) == 0:
        print("No events found; nothing to export.", file=sys.stderr)
        return output_path

    if not include_arrivals:
        for ev in catalog:
            for origin in ev.origins:
                origin.arrivals.clear()
            ev.picks.clear()

    catalog.write(output_path, format="QUAKEML")
    return output_path


def export_catalog_quakeml(
    h5_path: str,
    output_path: str,
    t1: Optional[str] = None,
    t2: Optional[str] = None,
    include_arrivals: bool = False,
) -> Path:
    """
    Read event catalog from *h5_path* and write QuakeML.

    Parameters
    ----------
    h5_path : str
        Path to a PyOcto or NLL HDF5 file.
    output_path : str
        Destination QuakeML file.
    t1, t2 : str, optional
        UTCDateTime-compatible time bounds on origin time.
    include_arrivals : bool
        If True, attach picks and arrivals to each event.
    """
    h5_type = _detect_h5_type(h5_path)
    _log.info("Detected H5 type: %s", h5_type)

    if h5_type == "nll":
        _export_nll(h5_path, output_path, t1=t1, t2=t2, include_arrivals=include_arrivals)
    else:
        _export_pyocto(h5_path, output_path, t1=t1, t2=t2, include_arrivals=include_arrivals)

    out = Path(output_path)
    print(f"Wrote QuakeML to {out}")
    return out


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export earthquake catalog from a PyOcto or NLL HDF5 file to QuakeML.",
        epilog=(
            "Example:\n"
            "  python -m wav2hyp.tools.catalog_to_quakeml nll.h5 catalog.xml "
            "--t1 2023-01-01 --t2 2023-02-01 --include-arrivals"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input_h5_file", help="Path to PyOcto or NLL HDF5 file")
    p.add_argument("output_quakeml", help="Path to output QuakeML file")
    p.add_argument("--t1", default=None, help="Start time (UTC) to filter events")
    p.add_argument("--t2", default=None, help="End time (UTC) to filter events")
    p.add_argument(
        "--include-arrivals",
        action="store_true",
        default=False,
        help="Attach picks and arrivals to each event (default: origins only)",
    )
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

    export_catalog_quakeml(
        h5_path=str(h5_path),
        output_path=args.output_quakeml,
        t1=args.t1,
        t2=args.t2,
        include_arrivals=args.include_arrivals,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
