# Branch summary: `fix/duplicate-p-s-picks`

## Metadata

- **Branch**: fix/duplicate-p-s-picks
- **Status**: open
- **Opened on**: 2026-04-18
- **Closed on**: -
- **Merged into**: -
- **Merge strategy**: -
- **Last updated**: 2026-04-18

## Accomplishments

- Opened a dedicated branch to fix duplicate / ambiguous P and S picks (channel identity and SeisBench `trace_id` behavior).
- Scaffolded branch documentation and repo-local git hooks per `docs/branch` workflow.
- Implemented overwrite cascade cleanup and early skip for picker/associator/locator (`PLAN-overwrite-cascade-and-skip.md`): `_cascade_overwrite_cleanup` in `_process_timespan`, HDF5 `summary_stats` on writes, `EQTOutput.clear_is_associated_in_range` for re-association without re-pick.

## Planned work

- Preserve or recover channel (or full NSLC) through picking, association, and catalog export so P/S picks are not collapsed to station-only identifiers.
- Align `EQTOutput` / docs with actual `trace_id` contents (NET.STA.LOC vs NSLC).
- Extend or adjust `VCatalog.from_pyocto` (vdapseisutils) so ObsPy picks carry location/channel when available.

## Executed work

- Created branch `fix/duplicate-p-s-picks` and added `scripts/branch_docs.py` plus `.githooks` for timeline and summary maintenance.
- Core pipeline: cascading `remove_range` with INFO summaries, station-summary slice reset, per-stage CSV row drops, early load-from-HDF5 skips using `date_already_processed_for_stage` (plus legacy data fallback), and `summary_stats` passed into stage HDF5 writes.

## Back-and-forth / iteration notes

- Scope for this branch is limited to fixing duplicate P/S handling and related metadata; implementation work follows in subsequent commits.
- Overwrite cascade logging and `_run_*` comments document that cleanup runs only in `_process_timespan`.

## Problems + resolutions

- `scripts/branch_docs.py` was referenced by existing timeline text but missing; added a minimal implementation for `regenerate-timeline` and hook helpers.
- PyTables could not store a mixed bool/`pd.NA` `is_associated` column; cascade clear uses boolean `False` for cleared picks.

## Validation

- `python3 scripts/branch_docs.py regenerate-timeline` run after adding the branch summary.
- `pytest tests/test_overwrite_skip.py` (with `vdapseisutils` on `PYTHONPATH` if needed).

## Final changelog-style outcome

- _(To be filled when the branch is closed.)_
