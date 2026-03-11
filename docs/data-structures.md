# Data Structures

WAV2HYP stores results in HDF5 (picker and associator) and ASDF (locator). This page describes the layout of each file and how to read data, including efficient queries using indexed columns where available.

---

## Picker: `results/<target>/picks/eqt-volpick.h5`

**Format:** HDF5 with pandas tables (format `table`). The `picks` and `detections` tables use **indexed data_columns** so that `pd.read_hdf(..., where=...)` can filter efficiently without loading the full table.

**Indexed data_columns (queryable with `where=`):**

- **picks:** `peak_time`, `is_associated`, `peak_value`
- **detections:** `start_time`, `peak_value`

**Keys**

| Key | Description |
|-----|-------------|
| `picks` | Phase picks (P and S) from EQTransformer |
| `detections` | Detection windows (no phase label) |
| `summary` | One row per processing day (counts, thresholds, timing) |
| `pick_peak_histogram` | Daily histogram of peak_value bins per phase (P, S, D) |

**Columns**

- **picks:** `trace_id`, `start_time`, `end_time`, `peak_time`, `peak_value`, `phase`, `is_associated`
- **detections:** `trace_id`, `start_time`, `end_time`, `peak_value`
- **summary:** `date`, `config`, `ncha`, `nsamp`, `pick_model`, `np`, `ns`, `npicks`, `ndetections`, `p_thresh`, `s_thresh`, `d_thresh`, `t_exec_pick`, `t_updated_pick`
- **pick_peak_histogram:** `date`, `phase`, `peak_value_bin`, `count`

**Attributes:** Root group has `metadata` (JSON string) with method, model, and thresholds.

### Retrieval examples

**High-level (time filter only):** Uses indexed `peak_time` (picks) and `start_time` (detections) under the hood.

```python
from wav2hyp.utils.io import EQTOutput

eqt = EQTOutput("results/sthelens/picks/eqt-volpick.h5")
picks, detections, metadata = eqt.read()                              # full file
picks, detections, metadata = eqt.read(t1="2004-09-23", t2="2004-09-25")  # time window
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

| Key | Description |
|-----|-------------|
| `events` | Associated events (origin time, location, pick counts, residual) |
| `assignments` | Pick-to-event assignments (event_idx, station_id, phase, residual, weight) |
| `summary` | One row per processing day (nassignments, nevents, timing) |

**Columns**

- **events:** `time` (Unix timestamp), `x`, `y`, `z`, `latitude`, `longitude`, `depth`, `residual_rms`, `n_picks`, `n_p_picks`, `n_s_picks`
- **assignments:** `event_idx`, `station_id`, `phase`, `pick_time`, `residual`, `weight`
- **summary:** `date`, `config`, `assoc_method`, `nassignments`, `nevents`, `t_exec_assoc`, `t_updated_assoc`

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

**Format:** ASDF (HDF5-based). Fallback when PyASDF is not available: `nll.xml` (QuakeML) and `nll_metadata.json`. There are no indexed data_columns for the main event store or the summary table.

**Content**

- **Events:** QuakeML catalog stored via PyASDF
- **HDF5 group `metadata`:** Attributes such as `method`, `nll_home`, `target`, `event_date`, `nll_directory`, `last_updated`
- **Pandas table `summary`:** `date`, `config`, `loc_method`, `nlocations`, `t_exec_loc`, `t_update_loc`

### Retrieval examples

**High-level (time filter on origin times):**

```python
from wav2hyp.utils.io import NLLOutput

nll = NLLOutput("results/sthelens/locations/nll.h5")
catalog, metadata = nll.read(t1="2004-09-23", t2="2004-09-25")
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
