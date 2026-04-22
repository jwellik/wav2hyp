# Plan: catalog comparison visualizations (Canonical vs Test)

**Status: decisions locked; implementation in progress** — see `wav2hyp.viz.catalog_comparer` and `notebooks/catalog_comparer.ipynb`.

## Resolved decisions (2026-04-22)

| Topic | Choice |
|-------|--------|
| Polar scatter color | **Δ depth (km)** = `depth_km_test − depth_km_canonical` (diverging colormap; matches `origin-delta.png` style). |
| Arrival-time scatter | **One series per station** (not channel). **Two subplots, side by side:** P-waves only, S-waves only. X = `arrival_time_delta` (s); Y = EQT **peak_value** *when that column is available* (see *Data: peak_value* below). |
| EQT / eqt-volpick | **Do not** open a separate `eqt-volpick` file; use **only columns present in locator `arrivals_table`** for the Y metric. If `peak_value` is absent, fall back to an available metric (see below) and **adjust axis title/ylabel** accordingly. |
| All figures | **Display inline in Jupyter** and **save** under `notebooks/<volcano>/` with stable filenames. |
| Code organization | **Small module** `wav2hyp/viz/catalog_comparer.py` (+ `plot_styles` additions); **notebook stays thin** (wiring + CONFIG + display/save). |

## Data: peak_value vs `arrivals_table` (important)

The **documented** WAV2HYP locator `arrivals_table` schema in `NLLOutput.ARRIVALS_TABLE_COLUMNS` and `docs/data-structures.md` does **not** list **EQT `peak_value`**. Picks in `eqt-volpick.h5` carry `peak_value`; the locator stores `trace_id`, `station_id`, `phase`, `arrival_time`, NLL `residual` / `weight`, etc.

**Implication:** For a stock pipeline output, the scatter “peak_value” Y cannot come from a column that is not on disk. **Implementation behavior:**

- If **`peak_value` exists** on arrival rows (optional column, future or custom H5s), use it; titles use **"wav2hyp (peak_value)"** as below.
- Else if **`weight`** (NLL time weight) is present, use that and label the y-axis *truthfully* (NLL time weight, not EQT).
- Else fall back to **`residual`** (absolute residual) with clear labeling.
- The notebook’s markdown will briefly note this so expectations match the on-disk schema.

*If* the project later adds `peak_value` to `arrivals_table` at write time, the same plotting code will pick it up with no change.

## Goals (unchanged, refined)

1. **Three meta-catalogs:** **Canonical only**, **Test only**, **Both** (1:1 matched pairs after de-dupe).
2. **Time series:** binned event **rate** for the **full** Canonical and **full** Test catalogs (stairs / step; configurable bin width, e.g. 1h).
3. **VolcanoFigure, no TimeSeries column:** one figure, **overlay** the three meta-catalogs (distinct colors/markers + legend). `ts_axis_type=None` like the single-event explorer figure.
4. **Polar plot (Both only):** for each match, great-circle **distance (km)** and **azimuth** from **Canonical** → **Test**; color = **Δ depth (test − canonical)**.
5. **Shared-arrivals table** on **(station_id, phase_hint)** with `arrival_time_delta` = `arrival_time_test − arrival_time_canonical` (s); **negative** → Canonical’s arrival time is **earlier** (canonical “first” / smaller epoch time).
6. **Plots on shared arrivals:** (a) Y vs `arrival_time_delta` with **per-station** scatters, P and S in **two subplots**; (b) horizontal P/S boxplot by station.

## Efficiency vs naive “per-event loop” (recommended)

| Step | Naive | Preferred |
|------|--------|------------|
| Event matching | Nested loops O(N×M) | `pd.merge_asof` on sorted `origin_time` + **dedupe** to enforce 1:1 on canonical and test event_ids |
| Load arrivals | `read_arrivals` per event | **One** (or two) time-bounded `read_arrivals(event_ids=[])` for `[t1,t2]`, then `isin(matched event ids)` — avoids huge PyTables `OR` chains (same pattern as `pick_quality_explorer_utils`) |
| Shared arrivals | For-loop each pair | `merge` pairs into both arrival tables, normalize `(station, phase_hint)`, `merge` on pair keys with inner join; `drop_duplicates` with policy when multiple rows share a key |
| Polar offsets | row-wise Python | Vectorized `gps2dist_azimuth` on float arrays of lat/lon; θ/r arrays passed to one `scatter` |
| Event-rate bins | reindex in loop | `value_counts` / `Grouper` on aligned DatetimeIndex (fill 0) for both series |

## Output paths

- `notebook_export_root` = repository-relative `notebooks/<volcano_slug>/` with `<volcano_slug>` = `CONFIG_PATH` stem (same idea as `catalog_explorer`).
- All figures: `fig.savefig(out_dir / <name>__<volcano>.png, dpi=..., bbox_inches="tight")` plus `display` in the notebook. Filenames in code as constants (e.g. `catalog_comparer_figure_names`).

## Implementation file map

| Piece | Where |
|-------|--------|
| partition + shared arrivals + 1:1 match | `wav2hyp/viz/catalog_comparer.py` |
| Volcano 3-way overlay (no TS) | `make_catalog_comparer_volcano_figure` in `catalog_comparer.py` (uses `VolcanoFigure`, `ts_axis_type=None`, reuses `_cat_df_to_obspy_catalog` from `catalog_explorer_utils`) |
| Colors, default scatter/box style hooks | `wav2hyp/viz/plot_styles.py` (appended) |
| Wiring | `notebooks/catalog_comparer.ipynb` |

## Open items (post-v1, optional)

- [ ] If `weight` is used for Y, consider optional **volpick** join in a *debug* function only.
- [ ] Tighter **1:1** using global assignment (e.g. linear assignment) if de-dupe after `merge_asof` is insufficient in dense regions.

---

*Last updated: 2026-04-22 (decisions + efficiency + data correction).*
