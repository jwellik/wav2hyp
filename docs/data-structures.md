# Data Structures

WAV2HYP stores results in HDF5 (picker and associator) and ASDF (locator). This page describes the layout of each file and how to read data, including efficient queries using indexed columns where available.

**Export to text/CSV:** The `summary` tables in picker, associator, and locator HDF5 files are written automatically to CSV when the pipeline is run with summary output paths (e.g. `*_picker_summary.txt`). Picks and detections can be exported via `EQTOutput(...).read()` then `.to_dataframe().to_csv(...)`; the locator `catalog_table` via `pd.read_hdf(path, key="catalog_table").to_csv(...)`.

---

## Picker: `results/<target>/picks/eqt-volpick.h5`

**Format:** HDF5 with pandas tables (format `table`). The `picks` and `detections` tables use **indexed data_columns** so that `pd.read_hdf(..., where=...)` can filter efficiently without loading the full table.

**Indexed data_columns (queryable with `where=`):** On `picks`: **peak_time**, **is_associated**, **peak_value**. On `detections`: **start_time**, **peak_value**. These columns are marked in bold in the column lists below.

**Keys**


| Key                   | Description                                                        |
| --------------------- | ------------------------------------------------------------------ |
| `picks`               | Phase picks (P and S) from EQTransformer                           |
| `detections`          | Detection windows (no phase label)                                 |
| `summary`             | One row per processing day (counts, thresholds, timing)            |
| `pick_peak_histogram` | Daily histogram of peak_value bins per station and phase (P, S, D) |


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

**summary** *(exported as CSV when pipeline is run with summary output paths)*

- date — Processing date
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

**Indexed data_columns:** None in the current implementation. The `events` and `assignments` tables are written without `data_columns`, so efficient `where=` queries are not available; retrieval is full-table or via the wav2hyp reader.

**Keys**


| Key           | Description                                                                |
| ------------- | -------------------------------------------------------------------------- |
| `events`      | Associated events (origin time, location, pick counts, residual)           |
| `assignments` | Pick-to-event assignments (event_idx, station_id, phase, residual, weight) |
| `summary`     | One row per processing day (nassignments, nevents, timing)                 |


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

**summary** *(exported as CSV when pipeline is run with summary output paths)*

- date — Processing date
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

**Direct pandas (full tables):**

```python
import pandas as pd
path = "results/sthelens/associations/pyocto.h5"
events_df = pd.read_hdf(path, key='events')
assignments_df = pd.read_hdf(path, key='assignments')
```

---

## Locator: `results/<target>/locations/nll.h5`

**Format:** ASDF (HDF5-based). Fallback when PyASDF is not available: `nll.xml` (QuakeML) and `nll_metadata.json`. In addition to the full QuakeML catalog inside ASDF, a lightweight tabular catalog view is stored as a pandas table for fast event-level queries.

**Keys / content**


| Key / content      | Description                                                                   |
| ------------------ | ----------------------------------------------------------------------------- |
| (ASDF) `events`    | Full QuakeML catalog (PyASDF/ObsPy)                                           |
| (group) `metadata` | Attributes: method, nll_home, target, event_date, nll_directory, last_updated |
| `summary`          | One row per processing day (nlocations, timing)                               |
| `catalog_table`    | One row per located event; pandas table with indexed columns for fast query   |


**Indexed data_columns (queryable with `where=`):** On `catalog_table` only: **event_id**, **origin_time**, **mag**, **residual_rms**, **azimuthal_gap**. The underlying QuakeML catalog is accessed via PyASDF/ObsPy.

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

**summary** *(exported as CSV when pipeline is run with summary output paths)*

- date — Processing date
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
ts_lo = pd.Timestamp("2004-09-23").value
ts_hi = pd.Timestamp("2004-09-25").value
where = (
    f"(origin_time >= {ts_lo}) & "
    f"(origin_time <= {ts_hi}) & "
    f"(mag >= 1.5) & "
    f"(residual_rms <= 0.5) & "
    f"(azimuthal_gap <= 180)"
)
cat_subset = pd.read_hdf(path, key="catalog_table", where=where)
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

