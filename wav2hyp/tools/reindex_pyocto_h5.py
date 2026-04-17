"""
Retrofit PyOcto HDF5 files with indexed data_columns.

Usage examples
--------------
python -m wav2hyp.tools.reindex_pyocto_h5 --roots results results_local
python -m wav2hyp.tools.reindex_pyocto_h5 --files results/sthelens/2_associations/pyocto.h5
python -m wav2hyp.tools.reindex_pyocto_h5 --roots results --dry-run
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import h5py
import pandas as pd

from wav2hyp.utils.io import PyOctoOutput


def _collect_paths(roots: list[str], files: list[str]) -> list[Path]:
    out: set[Path] = set()
    for p in files:
        pp = Path(p).resolve()
        if pp.is_file() and pp.name == "pyocto.h5":
            out.add(pp)
    for root in roots:
        rr = Path(root).resolve()
        if not rr.exists():
            continue
        for p in rr.rglob("pyocto.h5"):
            if p.is_file():
                out.add(p.resolve())
    return sorted(out)


def _reindex_one(path: Path, dry_run: bool = False) -> tuple[bool, str]:
    try:
        events = pd.read_hdf(path, key="events")
        assignments = pd.read_hdf(path, key="assignments")
    except Exception as exc:
        return False, f"read failed: {exc}"

    summary = None
    try:
        summary = pd.read_hdf(path, key="summary")
    except Exception:
        summary = None

    metadata_raw = None
    try:
        with h5py.File(path, "r") as f:
            metadata_raw = f.attrs.get("metadata", None)
    except Exception:
        metadata_raw = None

    if dry_run:
        return True, f"would rewrite (events={len(events)}, assignments={len(assignments)})"

    parent = path.parent
    with tempfile.NamedTemporaryFile(prefix=".pyocto_reindex_", suffix=".h5", dir=parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        events.to_hdf(
            tmp_path,
            key="events",
            mode="w",
            format="table",
            data_columns=PyOctoOutput.EVENTS_DATA_COLUMNS,
        )
        assignments.to_hdf(
            tmp_path,
            key="assignments",
            mode="a",
            format="table",
            data_columns=PyOctoOutput.ASSIGNMENTS_DATA_COLUMNS,
        )
        if summary is not None and len(summary) > 0:
            summary.to_hdf(tmp_path, key="summary", mode="a", format="table")
        if metadata_raw is not None:
            with h5py.File(tmp_path, "a") as f:
                f.attrs["metadata"] = metadata_raw
        os.replace(tmp_path, path)
        return True, f"rewritten (events={len(events)}, assignments={len(assignments)})"
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return False, f"rewrite failed: {exc}"


def main():
    parser = argparse.ArgumentParser(description="Retrofit pyocto.h5 files with indexed data_columns.")
    parser.add_argument(
        "--roots",
        nargs="*",
        default=["results", "results_local"],
        help="Directories to recursively scan for pyocto.h5 (default: results results_local).",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="Specific pyocto.h5 files to rewrite.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List files and actions without rewriting.")
    args = parser.parse_args()

    paths = _collect_paths(args.roots, args.files)
    if not paths:
        print("No pyocto.h5 files found.")
        return

    print(f"Found {len(paths)} pyocto.h5 file(s).")
    ok = 0
    bad = 0
    for p in paths:
        success, msg = _reindex_one(p, dry_run=args.dry_run)
        if success:
            ok += 1
            print(f"[OK]   {p}: {msg}")
        else:
            bad += 1
            print(f"[FAIL] {p}: {msg}")

    print(f"Done. ok={ok}, failed={bad}, dry_run={args.dry_run}")


if __name__ == "__main__":
    main()

