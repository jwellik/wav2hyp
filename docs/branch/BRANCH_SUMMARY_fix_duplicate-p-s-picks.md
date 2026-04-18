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

## Planned work

- Preserve or recover channel (or full NSLC) through picking, association, and catalog export so P/S picks are not collapsed to station-only identifiers.
- Align `EQTOutput` / docs with actual `trace_id` contents (NET.STA.LOC vs NSLC).
- Extend or adjust `VCatalog.from_pyocto` (vdapseisutils) so ObsPy picks carry location/channel when available.

## Executed work

- Created branch `fix/duplicate-p-s-picks` and added `scripts/branch_docs.py` plus `.githooks` for timeline and summary maintenance.

## Back-and-forth / iteration notes

- Scope for this branch is limited to fixing duplicate P/S handling and related metadata; implementation work follows in subsequent commits.

## Problems + resolutions

- `scripts/branch_docs.py` was referenced by existing timeline text but missing; added a minimal implementation for `regenerate-timeline` and hook helpers.

## Validation

- `python3 scripts/branch_docs.py regenerate-timeline` run after adding the branch summary.

## Final changelog-style outcome

- _(To be filled when the branch is closed.)_
