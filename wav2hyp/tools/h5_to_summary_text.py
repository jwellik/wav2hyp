"""
Write step summary text files from WAV2HYP HDF5 files (picks, associations, or locations).

Reads the 'summary' table from an H5 file, optionally resamples to a coarser period,
and writes a CSV summary file. Use this to (re)generate *_picker_summary.txt,
*_associator_summary.txt, or *_locator_summary.txt from existing H5.

Usage:
  python -m wav2hyp.tools.h5_to_summary_text <input_h5_file> <output_text_file> [--period '1d']
  python -m wav2hyp.tools.h5_to_summary_text picks/eqt-volpick.h5 sthelens_picker_summary.txt --period 1d
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from wav2hyp.utils.summary import (
    infer_step_from_hdf5,
    write_summary_txt_from_hdf5,
)

_log = logging.getLogger("wav2hyp")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Write step summary text file from WAV2HYP HDF5 file (with optional resampling).",
        epilog="Example: python -m wav2hyp.tools.h5_to_summary_text picks/eqt-volpick.h5 out.txt --period 1d",
    )
    p.add_argument("input_h5_file", help="Path to HDF5 file (e.g. picks/eqt-volpick.h5, associations/pyocto.h5)")
    p.add_argument("output_text_file", help="Path to output summary CSV file")
    p.add_argument(
        "--step",
        choices=["picker", "associator", "locator"],
        default=None,
        help="Processing step (inferred from H5 'summary' table if omitted)",
    )
    p.add_argument(
        "--period",
        default="1d",
        help="Resampling period for output (e.g. '1d', '1h'). Default: 1d",
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

    step = args.step
    if step is None:
        try:
            step = infer_step_from_hdf5(str(h5_path))
            _log.info("Inferred step: %s", step)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    _log.info("Loaded summary from %s", h5_path)
    write_summary_txt_from_hdf5(
        str(h5_path),
        args.output_text_file,
        step,
        summary_text_period=args.period,
    )
    _log.info("Wrote summary to %s", args.output_text_file)
    print(f"Summary written to {args.output_text_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
