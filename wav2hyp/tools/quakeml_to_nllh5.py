"""
Convert a QuakeML file into the `NLLOutput` lightweight `nll.h5` format.

Goal:
- Normalize QuakeML catalogs from heterogeneous sources into a consistent
  `catalog_table` + `arrivals_table` representation for easy comparison.

Usage:
  python -m wav2hyp.tools.quakeml_to_nllh5 --input /path/to/catalog.xml --output /path/to/nll.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from obspy import read_events

from wav2hyp.utils.io import NLLOutput


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert QuakeML -> wav2hyp NLLOutput nll.h5")
    p.add_argument("--input", "-i", required=True, help="Input QuakeML file (XML)")
    p.add_argument("--output", "-o", required=False, help="Output nll.h5 path (default: <input>.h5)")
    p.add_argument(
        "--metadata-json",
        default=None,
        help="Optional JSON string with metadata written into NLLOutput 'metadata' group.",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    return p.parse_args(argv)


def _load_metadata(metadata_json: Optional[str], verbose: bool) -> dict[str, Any]:
    base = {
        "method": "QUAKEML_IMPORT",
        "event_date": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
    }
    if not metadata_json:
        return base
    try:
        extra = json.loads(metadata_json)
        if isinstance(extra, dict):
            base.update(extra)
        else:
            if verbose:
                print("--metadata-json was not a JSON object; ignoring.")
    except json.JSONDecodeError:
        if verbose:
            print("WARNING: --metadata-json is not valid JSON; ignoring.")
    return base


def quakeml_to_nllh5(
    input_path: str,
    output_path: Optional[str] = None,
    metadata_json: Optional[str] = None,
    verbose: bool = False,
) -> Path:
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input QuakeML not found: {in_path}")

    if output_path is None:
        output_path = str(in_path.with_suffix(".h5"))
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Reading QuakeML: {in_path}")

    catalog = read_events(str(in_path))

    metadata = _load_metadata(metadata_json, verbose=verbose)
    if verbose:
        print(f"Writing NLLOutput: {out_path}")
        print(f"Metadata: {metadata}")

    nll = NLLOutput(str(out_path))
    nll.write(catalog, metadata=metadata)

    return out_path


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    quakeml_to_nllh5(
        input_path=args.input,
        output_path=args.output,
        metadata_json=args.metadata_json,
        verbose=bool(args.verbose),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

