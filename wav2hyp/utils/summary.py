"""
Summary export functionality for WAV2HYP pipeline.

This module provides real-time and retrospective summary generation for processing
statistics and timing information. Summary text files can be written from HDF5
summary tables (single source of truth).
"""

import logging
import os
import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

from obspy import UTCDateTime

from .io import PickListX, DetectionListX, _parse_summary_date
from .stations import station_from_trace_id

_log = logging.getLogger("wav2hyp")

# Stages in pipeline order; used for overwrite cascade (cleanup downstream only).
STAGES_ORDER = ("picker", "associator", "locator")


def date_already_processed_for_stage(hdf5_path: str, start_time, end_time) -> bool:
    """
    Return True if the stage's HDF5 summary table has at least one row whose date falls in [start_time, end_time].

    Used to decide whether to skip running a stage for a chunk when --overwrite is not set.
    The summary table is the single source of truth (summary .txt is derived from it).

    Parameters
    ----------
    hdf5_path : str
        Path to the stage's HDF5 file (picker, associator, or locator).
    start_time, end_time
        UTCDateTime or str. Chunk time range; summary rows are parsed and compared to this range.

    Returns
    -------
    bool
        True if any summary row date falls within [start_time, end_time], else False.
    """
    path = Path(hdf5_path)
    if not path.exists():
        return False
    try:
        df = pd.read_hdf(hdf5_path, key="summary")
    except (KeyError, FileNotFoundError):
        return False
    if df is None or len(df) == 0 or "date" not in df.columns:
        return False
    ts_lo = pd.Timestamp(UTCDateTime(start_time).datetime, tz="UTC")
    ts_hi = pd.Timestamp(UTCDateTime(end_time).datetime, tz="UTC")
    for _, row in df.iterrows():
        ts = _parse_summary_date(row.get("date"))
        if ts is not None and ts_lo <= ts <= ts_hi:
            return True
    return False


def drop_summary_txt_rows_in_range(txt_path: str, t1, t2) -> int:
    """
    Remove rows whose 'date' falls in [t1, t2] from a per-stage summary .txt (CSV) file.
    Used by overwrite cleanup. Rewrites the file with remaining rows.

    Parameters
    ----------
    txt_path : str
        Path to the summary .txt CSV file.
    t1, t2
        UTCDateTime or str. Time range; rows with date in this range are removed.

    Returns
    -------
    int
        Number of rows removed (for logging).
    """
    path = Path(txt_path)
    if not path.exists():
        return 0
    try:
        df = pd.read_csv(path)
    except Exception:
        return 0
    if df is None or len(df) == 0 or "date" not in df.columns:
        return 0
    ts_lo = pd.Timestamp(UTCDateTime(t1).datetime, tz="UTC")
    ts_hi = pd.Timestamp(UTCDateTime(t2).datetime, tz="UTC")
    n_before = len(df)
    keep = []
    for idx, row in df.iterrows():
        ts = _parse_summary_date(row.get("date"))
        if ts is None:
            keep.append(idx)
        elif ts < ts_lo or ts > ts_hi:
            keep.append(idx)
    df_out = df.loc[keep].copy() if keep else pd.DataFrame(columns=df.columns)
    df_out.to_csv(path, index=False)
    removed = n_before - len(df_out)
    _log.info("Removed %d rows for range %s to %s from summary text %s", removed, t1, t2, txt_path)
    return removed


def station_summary_reset_for_overwrite(
    base_output_dir: str,
    summary_filename: str,
    cleanup_stages: set,
    t1,
    t2,
) -> dict:
    """
    For overwrite cleanup: reset or remove station summary rows for the given date range.
    - If 'picker' in cleanup_stages: remove entire rows for that date.
    - Else if 'associator' in cleanup_stages: for rows with that date, set nassign, nassoc, nevents to 0.
    - Else if 'locator' in cleanup_stages: for rows with that date, set nevents to 0.

    Parameters
    ----------
    base_output_dir : str
        Base output directory (same as append_station_summary_rows).
    summary_filename : str
        Station summary filename (e.g. from config output.station_summary).
    cleanup_stages : set of str
        Subset of {'picker', 'associator', 'locator'} (stages being overwritten + downstream).
    t1, t2
        UTCDateTime or str. Date range to affect.

    Returns
    -------
    dict
        For logging: rows_removed (when picker in set), or rows_reset (count of rows whose columns were zeroed).
    """
    path = Path(base_output_dir) / summary_filename
    if not path.exists():
        return {"rows_removed": 0, "rows_reset": 0}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {"rows_removed": 0, "rows_reset": 0}
    if df is None or len(df) == 0 or "date" not in df.columns:
        return {"rows_removed": 0, "rows_reset": 0}
    ts_lo = pd.Timestamp(UTCDateTime(t1).datetime, tz="UTC")
    ts_hi = pd.Timestamp(UTCDateTime(t2).datetime, tz="UTC")
    mask_in_range = []
    for _, row in df.iterrows():
        ts = _parse_summary_date(row.get("date"))
        in_range = ts is not None and ts_lo <= ts <= ts_hi
        mask_in_range.append(in_range)
    mask_in_range = pd.Series(mask_in_range, index=df.index)
    n_in_range = mask_in_range.sum()
    if n_in_range == 0:
        return {"rows_removed": 0, "rows_reset": 0}
    if "picker" in cleanup_stages:
        df_out = df[~mask_in_range].copy()
        df_out.to_csv(path, index=False)
        _log.info("Station summary: removed %d rows for range %s to %s", n_in_range, t1, t2)
        return {"rows_removed": int(n_in_range), "rows_reset": 0}
    cols_to_zero = []
    if "associator" in cleanup_stages:
        cols_to_zero.extend(["nassign", "nassoc", "nevents"])
    elif "locator" in cleanup_stages:
        cols_to_zero.append("nevents")
    if not cols_to_zero:
        return {"rows_removed": 0, "rows_reset": 0}
    df = df.copy()
    for c in cols_to_zero:
        if c in df.columns:
            df.loc[mask_in_range, c] = 0
    df.to_csv(path, index=False)
    _log.info("Station summary: reset %s for %d rows in range %s to %s", cols_to_zero, n_in_range, t1, t2)
    return {"rows_removed": 0, "rows_reset": int(n_in_range)}


def infer_step_from_hdf5(hdf5_path: str) -> str:
    """
    Infer processing step (picker, associator, locator) from the 'summary' table columns.

    Parameters
    ----------
    hdf5_path : str
        Path to the HDF5 file.

    Returns
    -------
    str
        One of 'picker', 'associator', 'locator'.

    Raises
    ------
    ValueError
        If the file has no 'summary' table or columns do not match any known step.
    """
    path = Path(hdf5_path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    try:
        df = pd.read_hdf(hdf5_path, key='summary')
    except (KeyError, FileNotFoundError) as e:
        raise ValueError(f"HDF5 file has no 'summary' table or could not be read: {e}") from e
    if df is None or len(df) == 0:
        raise ValueError("HDF5 'summary' table is empty")
    cols = set(df.columns)
    if 't_exec_pick' in cols or 't_updated_pick' in cols:
        return 'picker'
    if 't_exec_assoc' in cols or 't_updated_assoc' in cols:
        return 'associator'
    if 't_exec_loc' in cols or 't_update_loc' in cols:
        return 'locator'
    raise ValueError(
        "Could not infer step from HDF5 'summary' columns. "
        "Use --step picker|associator|locator to specify."
    )


def write_summary_txt_from_hdf5(
    hdf5_path: str,
    summary_txt_path: str,
    step: str,
    summary_text_period: Optional[str] = None,
) -> None:
    """
    Read the 'summary' table from an HDF5 file and write it to a CSV summary text file.
    Optionally resample to a coarser period (e.g. '1d') before writing.

    Parameters
    ----------
    hdf5_path : str
        Path to the HDF5 file (picks, associations, or locations).
    summary_txt_path : str
        Path to the output CSV summary file.
    step : str
        One of 'picker', 'associator', 'locator' (determines column order).
    summary_text_period : str, optional
        Time string for resampling (e.g. '1d', '1h'). If None, no resampling.
    """
    path = Path(hdf5_path)
    if not path.exists():
        return
    try:
        df = pd.read_hdf(hdf5_path, key='summary')
    except (KeyError, FileNotFoundError):
        return
    if df is None or len(df) == 0:
        return
    _log.info("Loaded summary from %s, %d rows", hdf5_path, len(df))

    if step == 'picker':
        cols = ['date', 'config', 'ncha', 'nsamp', 'pick_model', 'np', 'ns', 'npicks', 'ndetections',
                'p_thresh', 's_thresh', 'd_thresh', 't_exec_pick', 't_updated_pick']
        sum_cols = ['ncha', 'nsamp', 'np', 'ns', 'npicks', 'ndetections']
        first_cols = ['config', 'pick_model', 'p_thresh', 's_thresh', 'd_thresh']
        last_cols = ['t_updated_pick']
    elif step == 'associator':
        cols = ['date', 'config', 'assoc_method', 'nassignments', 'nevents', 't_exec_assoc', 't_updated_assoc']
        sum_cols = ['nassignments', 'nevents']
        first_cols = ['config', 'assoc_method']
        last_cols = ['t_updated_assoc']
    elif step == 'locator':
        cols = ['date', 'config', 'loc_method', 'nlocations', 't_exec_loc', 't_update_loc']
        sum_cols = ['nlocations']
        first_cols = ['config', 'loc_method']
        last_cols = ['t_update_loc']
    else:
        raise ValueError(f"Unknown step: {step}")

    if summary_text_period:
        from ..core import parse_time_string
        period_seconds = parse_time_string(summary_text_period)
        df = df.copy()
        df['_ts'] = df['date'].apply(_parse_summary_date)
        df = df.dropna(subset=['_ts'])
        if len(df) == 0:
            out = Path(summary_txt_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            cols_out = [c for c in cols if c in df.columns]
            df[cols_out].to_csv(out, index=False) if cols_out else out.write_text("")
            _log.info("Wrote summary to %s, 0 rows", summary_txt_path)
            return
        df = df.set_index('_ts')
        agg_map = {}
        for c in sum_cols:
            if c in df.columns:
                agg_map[c] = 'sum'
        for c in first_cols:
            if c in df.columns and c not in agg_map:
                agg_map[c] = 'first'
        for c in last_cols:
            if c in df.columns and c not in agg_map:
                agg_map[c] = 'last'
        if step == 'picker' and 't_exec_pick' in df.columns:
            agg_map['t_exec_pick'] = 'sum'
        elif step == 'associator' and 't_exec_assoc' in df.columns:
            agg_map['t_exec_assoc'] = 'sum'
        elif step == 'locator' and 't_exec_loc' in df.columns:
            agg_map['t_exec_loc'] = 'sum'
        resampled = df.resample(pd.Timedelta(seconds=period_seconds)).agg(agg_map)
        resampled = resampled.dropna(how='all')
        if len(resampled) == 0:
            df = resampled.reset_index(drop=True)
        else:
            if period_seconds >= 86400:
                resampled['date'] = [t.strftime('%Y/%m/%d') for t in resampled.index]
            elif period_seconds >= 3600:
                resampled['date'] = [t.strftime('%Y/%m/%dT%H') for t in resampled.index]
            else:
                resampled['date'] = [t.strftime('%Y/%m/%dT%H:%M:%S') for t in resampled.index]
            df = resampled.reset_index(drop=True)
        _log.info("Resampled summary to period %s, %d rows", summary_text_period, len(df))

    cols = [c for c in cols if c in df.columns]
    out = Path(summary_txt_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(out, index=False)
    _log.info("Wrote summary to %s", summary_txt_path)


def append_station_summary_rows(
    base_output_dir: str,
    summary_filename: str,
    date_str: str,
    stream,
    picks,
    detections,
    assignments_df: Optional[pd.DataFrame],
    arrivals_df: Optional[pd.DataFrame] = None,
    step: str = "picker",
) -> None:
    """
    Write or update one row per (date_str, trace_id) in station summary file.
    Columns: date, trace_id (NET.STA.LOC.CHA), nsamples, ncha, np, ns, nd, nassign, nassoc, nevents.
    step: 'picker' | 'associator' | 'locator'. Only picker requires stream; associator/locator merge into existing rows.
    For locator, nevents = unique event count from arrivals_df per trace_id (when arrivals_df given).
    """
    header = ['date', 'trace_id', 'nsamples', 'ncha', 'np', 'ns', 'nd', 'nassign', 'nassoc', 'nevents']
    path = Path(base_output_dir) / summary_filename
    path.parent.mkdir(parents=True, exist_ok=True)

    # Collect trace_ids: from stream, or from picks/assignments/arrivals when stream is None
    trace_ids = set()
    if stream is not None and len(stream) > 0:
        trace_ids.update(tr.id for tr in stream)
    if picks is not None and len(picks) > 0:
        try:
            pick_df = PickListX(picks).to_dataframe()
            if 'trace_id' in pick_df.columns:
                trace_ids.update(pick_df['trace_id'].astype(str).unique())
        except Exception:
            pass
    if assignments_df is not None and len(assignments_df) > 0 and 'trace_id' in assignments_df.columns:
        trace_ids.update(assignments_df['trace_id'].astype(str).unique())
    if assignments_df is not None and len(assignments_df) > 0 and 'station_id' in assignments_df.columns and not trace_ids:
        for sta in assignments_df['station_id'].unique():
            trace_ids.add(str(sta))
    if arrivals_df is not None and len(arrivals_df) > 0 and 'trace_id' in arrivals_df.columns:
        trace_ids.update(arrivals_df['trace_id'].astype(str).unique())
    if not trace_ids:
        return

    trace_ids = sorted(trace_ids)

    nsamples_per_ch = {}
    if stream is not None and len(stream) > 0:
        nsamples_per_ch = {tr.id: sum(len(t.data) for t in stream if t.id == tr.id) for tr in stream}
    np_per_ch = {}
    ns_per_ch = {}
    nd_per_ch = {}
    if picks is not None and len(picks) > 0:
        try:
            pick_df = PickListX(picks).to_dataframe()
            for tid in pick_df['trace_id'].astype(str).unique():
                sub = pick_df[pick_df['trace_id'].astype(str) == tid]
                np_per_ch[tid] = int((sub['phase'] == 'P').sum())
                ns_per_ch[tid] = int((sub['phase'] == 'S').sum())
        except Exception:
            pass
    if detections is not None and len(detections) > 0:
        try:
            det_df = DetectionListX(detections).to_dataframe()
            for tid in det_df['trace_id'].astype(str).unique():
                nd_per_ch[tid] = int(len(det_df[det_df['trace_id'].astype(str) == tid]))
        except Exception:
            pass
    nassign_per_trace = {}
    nassoc_per_trace = {}
    nevents_per_trace = {}
    if assignments_df is not None and len(assignments_df) > 0:
        if 'trace_id' in assignments_df.columns and 'event_idx' in assignments_df.columns:
            for tid in assignments_df['trace_id'].astype(str).unique():
                sub = assignments_df[assignments_df['trace_id'].astype(str) == tid]
                nassign_per_trace[tid] = int(len(sub))
                nassoc_per_trace[tid] = int(sub['event_idx'].nunique())
                nevents_per_trace[tid] = int(sub['event_idx'].nunique())
        elif 'station_id' in assignments_df.columns and 'event_idx' in assignments_df.columns:
            for sta in assignments_df['station_id'].unique():
                sub = assignments_df[assignments_df['station_id'] == sta]
                nassign = int(len(sub))
                nevents = int(sub['event_idx'].nunique())
                for tid in trace_ids:
                    if station_from_trace_id(tid) == str(sta):
                        nassign_per_trace[tid] = nassign
                        nassoc_per_trace[tid] = nevents
                        nevents_per_trace[tid] = nevents
    if arrivals_df is not None and len(arrivals_df) > 0 and 'trace_id' in arrivals_df.columns and 'event_id' in arrivals_df.columns:
        for tid in arrivals_df['trace_id'].astype(str).unique():
            nevents_per_trace[tid] = int(arrivals_df[arrivals_df['trace_id'].astype(str) == tid]['event_id'].nunique())

    new_rows = []
    for tid in trace_ids:
        sta = station_from_trace_id(tid)
        row = {
            'date': date_str,
            'trace_id': tid,
            'nsamples': nsamples_per_ch.get(tid, 0),
            'ncha': 1 if stream else 0,
            'np': np_per_ch.get(tid, 0),
            'ns': ns_per_ch.get(tid, 0),
            'nd': nd_per_ch.get(tid, 0),
            'nassign': nassign_per_trace.get(tid, 0),
            'nassoc': nassoc_per_trace.get(tid, 0),
            'nevents': nevents_per_trace.get(tid, 0),
        }
        new_rows.append(row)

    existing_df = None
    if path.exists():
        try:
            existing_df = pd.read_csv(path)
            _log.info("Loaded station summary from %s, %d rows", path, len(existing_df))
            if 'trace_id' not in existing_df.columns and 'channel_id' in existing_df.columns:
                existing_df['trace_id'] = existing_df['channel_id']
        except Exception:
            pass

    rows_by_key = {}
    if existing_df is not None and len(existing_df) > 0 and 'date' in existing_df.columns and ('trace_id' in existing_df.columns or 'channel_id' in existing_df.columns):
        tid_col = 'trace_id' if 'trace_id' in existing_df.columns else 'channel_id'
        for _, r in existing_df.iterrows():
            k = (str(r['date']), str(r[tid_col]))
            row_dict = {c: r.get(c, 0) for c in header if c in r}
            if 'trace_id' not in row_dict and tid_col in r:
                row_dict['trace_id'] = str(r[tid_col])
            rows_by_key[k] = row_dict
    for row in new_rows:
        k = (str(row['date']), str(row['trace_id']))
        if k in rows_by_key and step != 'picker':
            for col in ['nassign', 'nassoc', 'nevents']:
                rows_by_key[k][col] = row[col]
        else:
            rows_by_key[k] = row
    out_df = pd.DataFrame(list(rows_by_key.values()), columns=header)
    out_df = out_df.sort_values(['date', 'trace_id']).reset_index(drop=True)
    out_df.to_csv(path, index=False)
    _log.info("Wrote station summary to %s, %d rows", path, len(out_df))


class SummaryExporter:
    """
    Thread-safe summary exporter for WAV2HYP processing statistics.
    
    Handles real-time updates to separate CSV summary files for each processing step
    (picker, associator, locator) with processing statistics and timing information.
    """
    
    def __init__(self, base_output_dir: str, config_name: str, step: str, summary_filename: Optional[str] = None):
        """
        Initialize summary exporter for a specific processing step.
        
        Parameters
        ----------
        base_output_dir : str
            Base output directory for results
        config_name : str
            Configuration name (used for default filename)
        step : str
            Processing step ('picker', 'associator', 'locator')
        summary_filename : str, optional
            Custom summary filename. If None, uses {config_name}_{step}_summary.txt
        """
        self.base_output_dir = Path(base_output_dir)
        self.config_name = config_name
        self.step = step
        
        # Set summary file path
        if summary_filename is None:
            summary_filename = f"{config_name}_{step}_summary.txt"
        self.summary_file = self.base_output_dir / summary_filename
        
        # Thread lock for safe concurrent access
        self._lock = threading.Lock()
        
        # Ensure output directory exists
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize summary file with header if needed
        self._ensure_header()
    
    def _ensure_header(self):
        """Ensure summary file exists with proper CSV header."""
        try:
            if not self.summary_file.exists():
                self._write_header()
        except Exception as e:
            # Fail gracefully if we can't create the file
            print(f"Warning: Could not create summary file {self.summary_file}: {e}")
    
    def _write_header(self):
        """Write CSV header to summary file based on processing step."""
        if self.step == 'picker':
            header = ['date', 'config', 'ncha', 'nsamp', 'pick_model', 'np', 'ns', 'npicks', 'ndetections', 'p_thresh', 's_thresh', 'd_thresh', 't_exec_pick', 't_updated_pick']
        elif self.step == 'associator':
            header = ['date', 'config', 'assoc_method', 'nassignments', 'nevents', 't_exec_assoc', 't_updated_assoc']
        elif self.step == 'locator':
            header = ['date', 'config', 'loc_method', 'nlocations', 't_exec_loc', 't_update_loc']
        else:
            raise ValueError(f"Unknown step: {self.step}")
        
        with open(self.summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    
    def update_entry(self, date: str, stats: Dict[str, Any], timing: float):
        """
        Update or create summary entry for a specific date.
        
        Parameters
        ----------
        date : str
            Date in format 'YYYY/MM/DD'
        stats : dict
            Statistics dictionary containing relevant metrics
        timing : float
            Processing time in seconds
        """
        with self._lock:
            try:
                self._update_entry_threadsafe(date, stats, timing)
            except Exception as e:
                # Fail gracefully - don't crash the main processing
                print(f"Warning: Could not update summary file: {e}")
    
    def _update_entry_threadsafe(self, date: str, stats: Dict[str, Any], timing: float):
        """Thread-safe implementation of entry update."""
        # Read existing data
        existing_data = self._read_existing_data()
        
        # Find or create entry for this date
        entry = self._get_or_create_entry(existing_data, date)
        
        # Update entry based on step
        if self.step == 'picker':
            self._update_picker_data(entry, stats, timing)
        elif self.step == 'associator':
            self._update_associator_data(entry, stats, timing)
        elif self.step == 'locator':
            self._update_locator_data(entry, stats, timing)
        
        # Update step-specific timestamp
        ts = datetime.now().strftime('%Y/%m/%dT%H:%M:%S')
        if self.step == 'picker':
            entry['t_updated_pick'] = ts
        elif self.step == 'associator':
            entry['t_updated_assoc'] = ts
        elif self.step == 'locator':
            entry['t_update_loc'] = ts

        # Write updated data back to file
        self._write_data(existing_data)
    
    def _read_existing_data(self) -> List[Dict[str, Any]]:
        """Read existing summary data from CSV file."""
        if not self.summary_file.exists():
            return []
        
        try:
            df = pd.read_csv(self.summary_file)
            return df.to_dict('records')
        except Exception:
            # If file is corrupted, return empty list
            return []
    
    def _write_data(self, data: List[Dict[str, Any]]):
        """Write data back to CSV file."""
        if not data:
            return
        
        df = pd.DataFrame(data)
        df.to_csv(self.summary_file, index=False)
    
    def _get_or_create_entry(self, existing_data: List[Dict[str, Any]], date: str) -> Dict[str, Any]:
        """Get existing entry for date or create new one."""
        for entry in existing_data:
            if entry.get('date') == date:
                return entry
        
        # Create new entry based on step
        if self.step == 'picker':
            new_entry = {
                'date': date,
                'config': '',
                'ncha': 0,
                'nsamp': 0,
                'pick_model': '',
                'np': 0,
                'ns': 0,
                'npicks': 0,
                'ndetections': 0,
                'p_thresh': 0.0,
                's_thresh': 0.0,
                'd_thresh': 0.0,
                't_exec_pick': 0.0,
                't_updated_pick': ''
            }
        elif self.step == 'associator':
            new_entry = {
                'date': date,
                'config': '',
                'assoc_method': '',
                'nassignments': 0,
                'nevents': 0,
                't_exec_assoc': 0.0,
                't_updated_assoc': ''
            }
        elif self.step == 'locator':
            new_entry = {
                'date': date,
                'config': '',
                'loc_method': '',
                'nlocations': 0,
                't_exec_loc': 0.0,
                't_update_loc': ''
            }
        else:
            raise ValueError(f"Unknown step: {self.step}")
        
        existing_data.append(new_entry)
        return new_entry
    
    def _update_picker_data(self, entry: Dict[str, Any], stats: Dict[str, Any], timing: float):
        """Update entry with picker statistics."""
        entry['config'] = stats.get('config', '')
        entry['ncha'] = stats.get('ncha', 0)
        entry['nsamp'] = stats.get('nsamp', 0)
        entry['pick_model'] = stats.get('pick_model', '')
        entry['np'] = stats.get('np', 0)
        entry['ns'] = stats.get('ns', 0)
        entry['npicks'] = stats.get('npicks', 0)
        entry['ndetections'] = stats.get('ndetections', 0)
        entry['p_thresh'] = stats.get('p_thresh', 0.0)
        entry['s_thresh'] = stats.get('s_thresh', 0.0)
        entry['d_thresh'] = stats.get('d_thresh', 0.0)
        entry['t_exec_pick'] = timing

    def _update_associator_data(self, entry: Dict[str, Any], stats: Dict[str, Any], timing: float):
        """Update entry with associator statistics."""
        entry['config'] = stats.get('config', '')
        entry['assoc_method'] = stats.get('assoc_method', '')
        entry['nassignments'] = stats.get('nassignments', 0)
        entry['nevents'] = stats.get('nevents', 0)
        entry['t_exec_assoc'] = timing

    def _update_locator_data(self, entry: Dict[str, Any], stats: Dict[str, Any], timing: float):
        """Update entry with locator statistics."""
        entry['config'] = stats.get('config', '')
        entry['loc_method'] = stats.get('loc_method', '')
        entry['nlocations'] = stats.get('nlocations', 0)
        entry['t_exec_loc'] = timing


