# Program Workflow

The WAV2HYP pipeline runs in three steps: **phase picking**, **event association**, and **earthquake location**. Each step reads from config and (when applicable) from the previous step’s output, and writes to HDF5/ASDF files under `results/<target>/`. The path placeholder `<target>` comes from your config (e.g. the locator `config_name` or output base directory name).

When you omit a step (e.g. you run only `-a -l` without `-p`), the pipeline reads existing data from the corresponding HDF5/ASDF file for that step instead of recomputing it.

---

## Step 1 – Phase picking

**Inputs**

- Config: inventory path, waveform client (datasource, optional client_type), picker model, P/S/D thresholds
- Time range: `(t1, t2)` for the window to process

**Data in**

- Continuous waveforms from **VClient** (FDSN, SDS, Earthworm, or SeedLink)
- Station **inventory** (StationXML) from config

**Outputs**

- **File:** `results/<target>/picks/eqt-volpick.h5`  
  Contains: `picks`, `detections`, `summary`, `pick_peak_histogram` (per-station peak_value histograms; see [Data structures](data-structures.md)).
- **In-memory:** `PickListX` (picks), `Stream` (optional), `DetectionListX` (detections)

---

## Step 2 – Event association

**Inputs**

- **Picks:** from the picker step or read from `results/<target>/picks/eqt-volpick.h5` when the picker was not run
- Station **inventory**
- Config: associator section (velocity, tolerance, depth limits, min P/S picks, etc.)

**Outputs**

- **File:** `results/<target>/associations/pyocto.h5`  
  Contains: `events`, `assignments`, `summary` (see [Data structures](data-structures.md)).
- **Side effect:** Updates the `is_associated` column in `eqt-volpick.h5` for the same time range
- **In-memory:** `VCatalog` (events), assignments DataFrame

---

## Step 3 – Location

**Inputs**

- **Catalog** from the associator step (events + assignments)
- Station **inventory**
- Config: locator section (`nll_home`, `config_name`, station_format, optional `velocity_model_file`)

**Data in**

- NLL observation files under `nll_home`, NLL control file, and grid/time runs (Vel2Grid, Grid2Time, NLLoc). If `velocity_model_file` is set in the locator config, it must be a path to a NonLinLoc LAYER-format 1D velocity model file; that file is read and its layers are passed to NLLpy so the generated control file uses your model instead of the default.
- `wav2hyp` always requests that NLLPy run its grid steps (Vel2Grid, Grid2Time) as part of location, and will also pass the global `--overwrite` flag to NLLPy **when the installed NLLPy version exposes an `overwrite` parameter on `run_nlloc`**. In older NLLPy versions, `overwrite` only affects wav2hyp’s own output files, while grid reuse/recomputation behavior is handled internally by NLLPy/NonLinLoc.

**Outputs**

- **File:** `results/<target>/locations/nll.h5`  
  ASDF with QuakeML events, `metadata` group, and `summary` table. Fallback: `nll.xml` and `nll_metadata.json`.
- **Directory:** `results/<target>/locations/nll_<config_name>/loc-YYYY-MM-DD/`  
  NonLinLoc working files and `.hyp` results.
- **In-memory:** `VCatalog` of located events

---

## Summary diagram

```
Config + time range (t1, t2)
         │
         ▼
┌─────────────────────┐     results/<target>/picks/eqt-volpick.h5
│  1. Phase picking    │ ─────────────────────────────────────────►
│  (EQTransformer)     │     PickListX, DetectionListX
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐     results/<target>/associations/pyocto.h5
│  2. Event association│ ─────────────────────────────────────────►
│  (PyOcto)            │     + updates is_associated in picks HDF5
└──────────┬──────────┘     VCatalog, assignments_df
           │
           ▼
┌─────────────────────┐     results/<target>/locations/nll.h5
│  3. Location         │ ─────────────────────────────────────────►
│  (NonLinLoc)         │     + nll_<config_name>/loc-YYYY-MM-DD/
└─────────────────────┘     VCatalog (located events)
```

---

## Overwrite and skip behavior

**Without `-o` / `--overwrite` (default):** For each time chunk and for each stage (picker, associator, locator), the pipeline checks whether that stage has already been processed by reading the stage’s HDF5 `summary` table and looking for any row whose parsed `date` falls within the chunk’s \([t1, t2]\) range. (Per-stage summary `.txt` files are derived from HDF5 and are not the source of truth for skip decisions.)

- If the stage **was already processed** for the chunk, the stage is **skipped** and results are **loaded from HDF5** for that time range (no waveform download, no EQTransformer / PyOcto / NonLinLoc run for that stage).
- If the stage **was not processed** for the chunk, the stage **runs** and **writes** outputs normally.
- If a stage is **not requested**, its existing outputs are read from HDF5 for the time range.

**With `-o` / `--overwrite`:** Before running any requested stages for a chunk, the pipeline performs a **cascade cleanup**: it removes existing data in \([t1, t2]\) for any stage that is being run **and all downstream stages**.

- If **picker** is run with overwrite → cleanup **picker + associator + locator**
- If **associator** is run with overwrite (picker not run) → cleanup **associator + locator**
- If **locator only** is run with overwrite (e.g. `-l -o`) → cleanup **locator**

Cleanup details:

- Stage output HDF5 files are cleaned by calling the stage output object’s `remove_range(t1, t2)`.
- Per-stage summary `.txt` files (when configured) have rows in-range removed.
- Station summary stage slices are cleaned by deleting only the affected HDF5 nodes for the period range in each cleanup stage (`/station_summary/<step>/<period_id>`). This avoids reading/re-writing a monolithic station summary text file.

After cleanup, the pipeline runs requested stages as usual and writes fresh outputs for the chunk. Station summary text (`output.station_summary`) is generated once per `run()` by joining stage slices from picker/associator/locator HDF5 outputs. See [Data structures](data-structures.md) for stage summary columns and station-summary slice layout.
