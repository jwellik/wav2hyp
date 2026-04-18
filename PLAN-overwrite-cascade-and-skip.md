# Plan: Overwrite cascade and early skip

## Goal

- Run **cascading overwrite cleanup** once per time chunk in `_process_timespan` when `--overwrite` is set: `remove_range` on affected stage HDF5 files, optional station-summary slice reset, optional CSV summary row removal, and `clear_is_associated_in_range` on picks when re-associating without re-picking.
- Move **skip vs run** decisions to the **start** of `_run_picker`, `_run_associator`, and `_run_locator`: without overwrite, if the chunk is already processed, **load from HDF5** and return (no heavy work).
- Pass **`summary_stats`** into `EQTOutput` / `PyOctoOutput` / `NLLOutput` `write()` so HDF5 `summary` tables match documented skip behavior.
- **Log** cascade operations with removal counts; add **comments** in each `_run_*` that cascade already ran in `_process_timespan` when overwriting.

## References

- Implementation: [`wav2hyp/core.py`](wav2hyp/core.py), [`wav2hyp/utils/io.py`](wav2hyp/utils/io.py), [`wav2hyp/utils/summary.py`](wav2hyp/utils/summary.py)
- Behavior spec: [`docs/workflow.md`](docs/workflow.md)

## Status

Implemented on branch `fix/duplicate-p-s-picks` (see `docs/branch/BRANCH_SUMMARY_fix_duplicate-p-s-picks.md`).
