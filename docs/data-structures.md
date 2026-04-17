# Data Structures

WAV2HYP stores results in HDF5 (picker, associator, and locator). This page describes the layout of each file and how to read data, including efficient queries using indexed columns where available.

**Export to text/CSV:** The `summary` tables in picker, associator, and locator HDF5 files are written automatically to CSV when the pipeline is run with summary output paths (e.g. `*_picker_summary.txt`). Summary table granularity is set by `output.summary_table_period` (default `1h`); CSV export is resampled by `output.summary_text_period` (default `1d`). You can regenerate summary CSV from H5 with `python -m wav2hyp.tools.h5_to_summary_text <input_h5> <output.txt> --period 1d`. Station summary is now stored as **stage slices in HDF5** and merged to text when requested: `python -m wav2hyp.tools.station_summary_from_stage_slices --picker-h5 <eqt-volpick.h5> --associator-h5 <pyocto.h5> --locator-h5 <nll.h5> <station_summary.txt>`. Station-summary granularity is `output.station_summary_period` (default `1d`) with one row per (`date`, `trace_id`) in the merged output. Picks and detections can be exported via `EQTOutput(...).read()` then `.to_dataframe().to_csv(...)`; the locator catalog via `NLLOutput(path).read()` or `NLLOutput(path).read_catalog_table()` then `.to_csv(...)`.

**Last execution (summary tables):** Each stage’s `summary` table includes a “last execution” timestamp: `t_updated_pick` (picker), `t_updated_assoc` (associator), `t_update_loc` (locator). These are set when the stage writes results (e.g. to `%Y/%m/%dT%H:%M:%S` UTC). The pipeline uses the presence of a summary row for a given date to decide whether that date has already been processed for that stage. Without `--overwrite`, a stage is skipped for a chunk if the summary has a row for that date; with `--overwrite`, data for that date is removed from the run stage and downstream stages before re-running. See [Program workflow](workflow.md) for overwrite and skip behavior.

## Picker: `results/<target>/picks/eqt-volpick.h5`

**Format:** HDF5 with pandas tables (format `table`). The `picks` and `detections` tables use **indexed data_columns** so that `pd.read_hdf(..., where=...)` can filter efficiently without loading the full table.

**Indexed data_columns (queryable with `where=`):** On `picks`: **peak_time**, **is_associated**, **peak_value**. On `detections`: **start_time**, **peak_value**. These columns are marked in bold in the column lists below.

**Keys**


| Key                   | Description                                                        |
| --------------------- | ------------------------------------------------------------------ |
| `picks`               | Phase picks (P and S) from EQTransformer                           |
| `detections`          | Detection windows (no phase label)                                 |
| `summary`             | One row per summary period (default 1h; configurable via `output.summary_table_period`) — counts, thresholds, timing |
| `pick_peak_histogram` | Daily histogram of peak_value bins per station and phase (P, S, D) |
| `station_summary/picker/<period_id>` | Picker station-summary slice for one period (`date`,`trace_id`,`nsamples`,`ncha`,`np`,`ns`,`nd`) |


**Columns** (bold = indexed data_column)

**picks**

- trace_id — Trace identifier (NET.STA.LOC.CHA)
- start_time — Start of pick window
- end_time — End of pick window
- **peak_time** — Time of maximum probability
- **peak_value** — Maximum probability (0–1)
- phase — Phase label (P or S)
- **is_associated** — Whether pick is associated to an event

**detections**

- trace_id — Trace identifier
- **start_time** — Window start
- end_time — Window end
- **peak_value** — Peak detection score

**summary** *(one row per period; CSV export resampled by `output.summary_text_period`)*

- date — Period start (e.g. YYYY/MM/DD or YYYY/MM/DDTHH)
- config — Config identifier
- ncha, nsamp — Channel/sample counts
- pick_model — Model name
- np, ns — P and S pick counts
- npicks, ndetections — Total pick and detection counts
- p_thresh, s_thresh, d_thresh — Thresholds used
- t_exec_pick, t_updated_pick — Timing (execution, last update)

**pick_peak_histogram**

- date — Processing date
- station_id — Station code
- phase — P, S, or D
- peak_value_bin — Histogram bin
- count — Count in bin

**Attributes:** Root group has `metadata` (JSON string) with method, model, and thresholds.

### Retrieval examples

**High-level (time, confidence, and/or association filter):** Uses indexed `peak_time` (picks), `start_time` (detections), `peak_value` (both), and `is_associated` (picks only) under the hood.

```python
from wav2hyp.utils.io import EQTOutput

eqt = EQTOutput("results/sthelens/picks/eqt-volpick.h5")
picks, detections, metadata = eqt.read()                                    # full file
picks, detections, metadata = eqt.read(t1="2004-09-23", t2="2004-09-25")    # time window
picks, detections, metadata = eqt.read(min_peak_value=0.6)                  # confidence >= 0.6
picks, detections, metadata = eqt.read(is_associated=True)                  # only associated picks
picks, detections, metadata = eqt.read(t1="2004-09-23", t2="2004-09-25", min_peak_value=0.6, is_associated=True)
```

**Full table (no filter):**

```python
import pandas as pd
path = "results/sthelens/picks/eqt-volpick.h5"
df_picks = pd.read_hdf(path, key='picks')
df_summary = pd.read_hdf(path, key='summary')
```

**Time range (between two dates):** Use nanosecond timestamps for PyTables `where=`.

For **picks**, filter on `peak_time`; for **detections**, filter on `start_time`:

```python
import pandas as pd
path = "results/sthelens/picks/eqt-volpick.h5"
t1, t2 = "2004-09-23", "2004-09-25"
ts_lo = pd.Timestamp(t1).value
ts_hi = pd.Timestamp(t2).value

# Picks in date range
where_picks = f"(peak_time >= {ts_lo}) & (peak_time <= {ts_hi})"
picks_subset = pd.read_hdf(path, key='picks', where=where_picks)

# Detections in date range
where_det = f"(start_time >= {ts_lo}) & (start_time <= {ts_hi})"
det_subset = pd.read_hdf(path, key='detections', where=where_det)
```

**peak_value threshold:** Because `peak_value` is an indexed data_column on both tables:

```python
# High-confidence picks only (e.g. >= 0.7)
picks_high = pd.read_hdf(path, key='picks', where="(peak_value >= 0.7)")
```

**Partial table (time + peak_value):** Combine conditions to load only rows in a time window with confidence above a threshold:

```python
import pandas as pd
path = "results/sthelens/picks/eqt-volpick.h5"
t1, t2 = "2004-09-23", "2004-09-25"
ts_lo = pd.Timestamp(t1).value
ts_hi = pd.Timestamp(t2).value
where = f"(peak_time >= {ts_lo}) & (peak_time <= {ts_hi}) & (peak_value >= 0.6)"
picks_subset = pd.read_hdf(path, key='picks', where=where)
```

---

## Associator: `results/<target>/associations/pyocto.h5`

**Indexed data_columns (new files):**
- On `events`: **time**, **residual_rms**, **n_picks**, **n_p_picks**, **n_s_picks**
- On `assignments`: **event_idx**, **station_id**, **phase**, **pick_time**

New `pyocto.h5` files are written with indexed `data_columns` so `where=` queries on these columns can avoid full-table scans.
For older files created before this change, run the retrofit tool:

```bash
python -m wav2hyp.tools.reindex_pyocto_h5 --roots results results_local
```

**Keys**


| Key           | Description                                                                |
| ------------- | -------------------------------------------------------------------------- |
| `events`      | Associated events (origin time, location, pick counts, residual)           |
| `assignments` | Pick-to-event assignments (event_idx, station_id, phase, residual, weight) |
| `summary`     | One row per processing day (nassignments, nevents, timing)                 |
| `station_summary/associator/<period_id>` | Associator station-summary slice for one period (`date`,`trace_id`,`nassign`,`nassoc`) |


**Columns**

**events**

- time — Origin time (Unix timestamp)
- x, y, z — Local coordinates (km)
- latitude, longitude, depth — Geographic origin
- residual_rms — Origin residual RMS
- n_picks, n_p_picks, n_s_picks — Pick counts (total, P, S)

**assignments**

- event_idx — Index into events table
- station_id — Station code
- phase — P or S
- pick_time — Pick time
- residual, weight — Residual and weight from associator

**summary** *(one row per period; CSV export resampled by `output.summary_text_period`)*

- date — Period start
- config — Config identifier
- assoc_method — Association method name
- nassignments, nevents — Counts
- t_exec_assoc, t_updated_assoc — Timing (execution, last update)

**Attributes:** Root group has `metadata` (JSON string).

### Retrieval examples

**High-level (time filter on event times):**

```python
from wav2hyp.utils.io import PyOctoOutput

pyocto = PyOctoOutput("results/sthelens/associations/pyocto.h5")
catalog, events_df, assignments_df, metadata = pyocto.read(t1="2004-09-23", t2="2004-09-25")
```

**Fast event-table read (recommended for notebooks/rates):**

```python
from wav2hyp.utils.io import PyOctoOutput

pyocto = PyOctoOutput("results/sthelens/associations/pyocto.h5")
events_df = pyocto.read_events_table(t1="2004-09-23", t2="2004-09-25")
```

**Direct pandas (full tables):**

```python
import pandas as pd
path = "results/sthelens/associations/pyocto.h5"
events_df = pd.read_hdf(path, key='events')
assignments_df = pd.read_hdf(path, key='assignments')
```

---

## Locator: `results/<target>/locations/nll.h5`

**Format:** HDF5 only. No ASDF QuakeML blob, no sidecar nll.xml or nll_metadata.json. The file contains a metadata group (attrs), optional summary table, catalog_table, and arrivals_table. NLLOutput.read() builds ObsPy Event/Origin (and Picks/Arrivals when requested) from these tables.

**Keys / content**


| Key / content      | Description                                                                   |
| ------------------ | ----------------------------------------------------------------------------- |
| (ASDF) `events`    | Full QuakeML catalog (PyASDF/ObsPy)                                           |
| (group) `metadata` | Attributes: method, nll_home, target, event_date, nll_directory, last_updated |
| `summary`          | One row per processing day (nlocations, timing)                               |
| `catalog_table`    | One row per located event; pandas table with indexed columns for fast query   |
| `arrivals_table`   | One row per origin arrival (phase pick linkage) for fast arrival queries        |
| `station_summary/locator/<period_id>` | Locator station-summary slice for one period (`date`,`trace_id`,`nevents`) |


**Indexed data_columns (queryable with `where=`):**
- On `catalog_table` only: **event_id**, **origin_time**, **mag**, **residual_rms**, **azimuthal_gap**
- On `arrivals_table`: **event_id**, **arrival_time**, **station_id**, **phase**

**Columns** (bold = indexed on `catalog_table`)

**catalog_table** *(can be exported via `pd.read_hdf(path, key="catalog_table").to_csv(...)`)*

- **event_id** — Event identifier (matches QuakeML resource_id where available)
- **origin_time** — Event origin time (UTC)
- latitude, longitude — Epicentral coordinates (deg)
- depth_km — Depth in km (positive downward)
- x, y, z — Local coordinates in km (optional)
- **mag** — Preferred magnitude value (if present)
- mag_type — Preferred magnitude type (if present)
- **residual_rms** — Origin standard error / RMS residual (if available)
- n_picks, n_p_picks, n_s_picks — Associated pick counts (total, P, S)
- **azimuthal_gap** — Azimuthal gap in degrees (if from NonLinLoc/QuakeML)

arrivals_table *(can be exported via `pd.read_hdf(path, key="arrivals_table").to_csv(...)`)*

Columns:
- **event_id** — Event identifier (matches QuakeML resource_id where available)
- **origin_time** — Event origin time (UTC)
- trace_id — Full waveform seed string when available (NET.STA.LOC.CHA)
- **station_id** — Station identifier derived from `trace_id` (best-effort NET.STA)
- **phase** — Canonical phase label (`P`/`S`)
- pick_id — Referenced pick resource identifier (used by ObsPy `Arrival`)
- **arrival_time** — Arrival/pick time (UTC), derived from referenced pick time
- residual, weight — Best-effort residual and association weight (if present in QuakeML/ObsPy)
- residual_uncert, weight_uncert — Uncertainties when present (otherwise NA)

**summary** *(one row per period; CSV export resampled by `output.summary_text_period`)*

- date — Period start
- config — Config identifier
- loc_method — Location method name
- nlocations — Number of located events
- t_exec_loc, t_update_loc — Timing (execution, last update)

### Retrieval examples

**High-level (time filter on origin times):**

```python
from wav2hyp.utils.io import NLLOutput

nll = NLLOutput("results/sthelens/locations/nll.h5")
catalog, metadata = nll.read(t1="2004-09-23", t2="2004-09-25")
```

**Fast event-level view via `catalog_table`:**

```python
import pandas as pd
path = "results/sthelens/locations/nll.h5"

# Full table
cat_df = pd.read_hdf(path, key="catalog_table")

# Time window and basic quality filters
ts_lo = pd.Timestamp("2004-09-23", tz="UTC").isoformat()
ts_hi = pd.Timestamp("2004-09-25", tz="UTC").isoformat()
where = (
    f"(origin_time >= '{ts_lo}') & "
    f"(origin_time <= '{ts_hi}') & "
    f"(mag >= 1.5) & "
    f"(residual_rms <= 0.5) & "
    f"(azimuthal_gap <= 180)"
)
cat_subset = pd.read_hdf(path, key="catalog_table", where=where)

# Fast arrivals subset (phase + time)
where_arr = (
    f"(arrival_time >= '{ts_lo}') & "
    f"(arrival_time <= '{ts_hi}') & "
    f"(phase == 'P')"
)
arr_subset = pd.read_hdf(path, key="arrivals_table", where=where_arr)
```

**Direct ASDF (full catalog):**

```python
import pyasdf
with pyasdf.ASDFDataSet("results/sthelens/locations/nll.h5", mode='r') as ds:
    catalog = ds.events
```

**Metadata and summary only:**

```python
import h5py
import pandas as pd
path = "results/sthelens/locations/nll.h5"
with h5py.File(path, 'r') as f:
    meta = dict(f['metadata'].attrs)
summary_df = pd.read_hdf(path, key='summary')
```

---

## Station Summary (H5 stage slices + merged text)

Station summary is written as stage-specific slices in each stage HDF5 file and later merged into `station_summary.txt` on demand (or once at the end of `run()` when `output.station_summary` is configured).

**Stage-slice columns by key**

- `station_summary/picker/<period_id>`: `date`, `trace_id`, `nsamples`, `ncha`, `np`, `ns`, `nd`
- `station_summary/associator/<period_id>`: `date`, `trace_id`, `nassign`, `nassoc`
- `station_summary/locator/<period_id>`: `date`, `trace_id`, `nevents`

**Merged `station_summary.txt` columns**

- `date` — Period start string (format depends on `output.station_summary_period`)
- `trace_id` — Channel id (`NET.STA.LOC.CHA`)
- `nsamples` — Samples for this trace in the period (picker)
- `ncha` — Channel count marker (picker; typically 1 per trace row)
- `np`, `ns`, `nd` — P picks, S picks, detections (picker)
- `nassign` — Number of assignments linked to the trace (associator)
- `nassoc` — Number of distinct associated events for the trace (associator)
- `nevents` — Number of located events with arrivals on the trace (locator)

Missing stage values are filled with `0` in the merged text output.

