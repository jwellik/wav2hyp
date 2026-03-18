"""
Input/Output Module for WAV2HYP
===============================

This module contains the main data classes used by the WAV2HYP program for handling input and output
operations. It provides extended functionality for seismic data processing workflows including:

- Enhanced pick and detection list classes with DataFrame conversion capabilities
- Output classes for different processing stages (EQTransformer, PyOcto, NonLinLoc)
- Time-based data management and cleanup functionality
- Support for HDF5, ASDF, XML, and JSON file formats

The module handles three main types of data:
1. Phase picks and detections from EQTransformer
2. Event associations from PyOcto
3. Earthquake locations from NonLinLoc

Classes
-------
PickListX : Extended seisbench PickList with DataFrame conversion
DetectionListX : Extended seisbench DetectionList with DataFrame conversion  
EQTOutput : Handles EQTransformer output (picks and detections)
PyOctoOutput : Handles PyOcto output (events and associations)
NLLOutput : Handles NonLinLoc output (earthquake catalog)

Author: WAV2HYP Development Team
"""

from obspy import UTCDateTime

# `seisbench` is used for EQTransformer pick/detection annotations.
# The locator/arrivals logic in this module should still be importable even
# when `seisbench` isn't installed (e.g. during lightweight analysis/tests).
try:
    from seisbench.util.annotations import PickList as SBPickList, DetectionList as SBDetectionList  # type: ignore
    from seisbench.util.annotations import Pick as _SeisBenchPick  # type: ignore
    from seisbench.util.annotations import Detection as _SeisBenchDetection  # type: ignore

    _SEISBENCH_AVAILABLE = True
except ModuleNotFoundError:
    _SEISBENCH_AVAILABLE = False

    class _StubPick:
        def __init__(
            self,
            trace_id: str,
            start_time: UTCDateTime,
            end_time: UTCDateTime,
            peak_time: UTCDateTime,
            peak_value: float,
            phase: str,
        ):
            self.trace_id = trace_id
            self.start_time = start_time
            self.end_time = end_time
            self.peak_time = peak_time
            self.peak_value = float(peak_value)
            self.phase = phase

    class _StubDetection:
        def __init__(
            self,
            trace_id: str,
            start_time: UTCDateTime,
            end_time: UTCDateTime,
            peak_value: float,
        ):
            self.trace_id = trace_id
            self.start_time = start_time
            self.end_time = end_time
            self.peak_value = float(peak_value)

    class SBPickList:
        def __init__(self, picks):
            self._picks = list(picks)

        def __iter__(self):
            return iter(self._picks)

        def __len__(self):
            return len(self._picks)

    class SBDetectionList:
        def __init__(self, detections):
            self._detections = list(detections)

        def __iter__(self):
            return iter(self._detections)

        def __len__(self):
            return len(self._detections)
import pandas as pd
import numpy as np
import os
import h5py
import json


class PickListX(SBPickList):
    """
    Extended PickList class with DataFrame conversion capabilities.
    
    Inherits from seisbench.util.annotations.PickList and adds methods for converting
    to/from Pandas DataFrames for easier data manipulation and export.
    
    Parameters
    ----------
    picks : list
        List of seisbench Pick objects
        
    Methods
    -------
    to_dataframe() : Convert picks to Pandas DataFrame
    from_dataframe(pick_df) : Create PickListX from Pandas DataFrame
    """
    
    def __init__(self, picks):
        super().__init__(picks)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the pick list to a Pandas DataFrame.
        
        This is useful for exporting picks to various formats. For example, to export 
        picks as CSV: `picks.to_dataframe().to_csv("picks.csv")`
        
        Returns
        -------
        pd.DataFrame
            DataFrame with the following columns:
            
            - **trace_id** (str): Seismic trace identifier in format "NET.STA.LOC.CHA"
            - **start_time** (datetime): Start time of the pick time window
            - **end_time** (datetime): End time of the pick time window  
            - **peak_time** (datetime): Time of maximum probability/peak detection
            - **peak_value** (float): Maximum probability value (0.0-1.0)
            - **phase** (str): Seismic phase type ("P" or "S")
            - **is_associated** (object): True/False if known, else pd.NA (for association status)
            - **index** (int): Sequential row index (added after sorting)
            
        Notes
        -----
        The DataFrame is automatically sorted by start_time and reset with a new index.
        All datetime columns are in UTC timezone as Python datetime objects.
        """
        pick_df = []
        for p in self:
            pick_df.append(
                {
                    "trace_id": p.trace_id,
                    "start_time": p.start_time.datetime,
                    "end_time": p.end_time.datetime,
                    "peak_time": p.peak_time.datetime,
                    "peak_value": p.peak_value,
                    "phase": p.phase,
                    "is_associated": pd.NA,
                }
            )
        pick_df = pd.DataFrame(pick_df)

        if len(pick_df) > 0:
            pick_df.sort_values("start_time", inplace=True)
            pick_df.reset_index(inplace=True)

        return pick_df

    @classmethod
    def from_dataframe(cls, pick_df):
        """
        Create PickListX from a Pandas DataFrame.
        
        Parameters
        ----------
        pick_df : pd.DataFrame
            DataFrame containing pick data with required columns:
            - trace_id, start_time, end_time, peak_time, peak_value, phase
            
        Returns
        -------
        PickListX
            New PickListX instance containing the picks from the DataFrame
            
        Notes
        -----
        Datetime columns can be provided as datetime objects or strings that
        can be parsed by ObsPy's UTCDateTime constructor.
        Optional column is_associated is ignored when building Pick objects
        (backward compatible with files that do not have this column).
        """
        # Convert back to seisbench PickList format when available; otherwise
        # fall back to lightweight stubs (enough for summary/histogram tests).
        Pick = _SeisBenchPick if _SEISBENCH_AVAILABLE else _StubPick

        picks = []
        for _, row in pick_df.iterrows():
            pick = Pick(
                trace_id=row['trace_id'],
                start_time=UTCDateTime(row['start_time']),
                end_time=UTCDateTime(row['end_time']),
                peak_time=UTCDateTime(row['peak_time']),
                peak_value=row['peak_value'],
                phase=row['phase']
            )
            picks.append(pick)

        return cls(picks)


class DetectionListX(SBDetectionList):
    """
    Extended DetectionList class with DataFrame conversion capabilities.
    
    Inherits from seisbench.util.annotations.DetectionList and adds methods for 
    converting to/from Pandas DataFrames for easier data manipulation and export.
    
    Parameters
    ----------
    detections : list
        List of seisbench Detection objects
        
    Methods
    -------
    to_dataframe() : Convert detections to Pandas DataFrame
    from_dataframe(detection_df) : Create DetectionListX from Pandas DataFrame
    """
    
    def __init__(self, detections):
        super().__init__(detections)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the detection list to a Pandas DataFrame.
        
        This is useful for exporting detections to various formats. For example, to export 
        detections as CSV: `detections.to_dataframe().to_csv("detections.csv")`
        
        Returns
        -------
        pd.DataFrame
            DataFrame with the following columns:
            
            - **trace_id** (str): Seismic trace identifier in format "NET.STA.LOC.CHA"
            - **start_time** (datetime): Start time of the detection time window
            - **end_time** (datetime): End time of the detection time window
            - **peak_value** (float): Maximum detection probability value (0.0-1.0)
            - **index** (int): Sequential row index (added after sorting)
            
        Notes
        -----
        The DataFrame is automatically sorted by start_time and reset with a new index.
        All datetime columns are in UTC timezone as Python datetime objects.
        """
        detection_df = []
        for p in self:
            detection_df.append(
                {
                    "trace_id": p.trace_id,
                    "start_time": p.start_time.datetime,
                    "end_time": p.end_time.datetime,
                    "peak_value": p.peak_value,
                }
            )
        detection_df = pd.DataFrame(detection_df)

        if len(detection_df) > 0:
            detection_df.sort_values("start_time", inplace=True)
            detection_df.reset_index(inplace=True)

        return detection_df

    @classmethod
    def from_dataframe(cls, detection_df):
        """
        Create DetectionListX from a Pandas DataFrame.
        
        Parameters
        ----------
        detection_df : pd.DataFrame
            DataFrame containing detection data with required columns:
            - trace_id, start_time, end_time, peak_value
            
        Returns
        -------
        DetectionListX
            New DetectionListX instance containing the detections from the DataFrame
            
        Notes
        -----
        Datetime columns can be provided as datetime objects or strings that
        can be parsed by ObsPy's UTCDateTime constructor.
        """
        Detection = _SeisBenchDetection if _SEISBENCH_AVAILABLE else _StubDetection
        detections = []
        for _, row in detection_df.iterrows():
            det = Detection(
                trace_id=row['trace_id'],
                start_time=UTCDateTime(row['start_time']),
                end_time=UTCDateTime(row['end_time']),
                peak_value=row['peak_value'],
            )
            detections.append(det)

        return cls(detections)


class EQTOutput:
    """
    Handles input/output operations for EQTransformer (EQT) phase picking results.
    
    This class manages the storage and retrieval of seismic phase picks and detections
    generated by the EQTransformer neural network model. Data is stored in HDF5 format
    with automatic time-based data cleanup capabilities.
    
    Parameters
    ----------
    output_path : str
        Path to the HDF5 output file
    time_range : tuple of (UTCDateTime, UTCDateTime), optional
        Time range (t1, t2) for automatic cleanup of existing data when writing
        
    Attributes
    ----------
    output_path : str
        Path to the HDF5 file
    time_range : tuple or None
        Time range for data cleanup operations
        
    Methods
    -------
    write(pick_list, detection_list, metadata) : Write EQT results to file
    read(t1=None, t2=None) : Read EQT results from file with optional time filtering
    
    File Format
    -----------
    The HDF5 file contains two main datasets:
    
    **'picks' dataset** - DataFrame with columns:
    - trace_id (str): Seismic trace identifier "NET.STA.LOC.CHA"
    - start_time (datetime): Pick time window start
    - end_time (datetime): Pick time window end  
    - peak_time (datetime): Time of maximum probability
    - peak_value (float): Maximum probability (0.0-1.0)
    - phase (str): Seismic phase ("P" or "S")
    
    **'detections' dataset** - DataFrame with columns:
    - trace_id (str): Seismic trace identifier "NET.STA.LOC.CHA"
    - start_time (datetime): Detection time window start
    - end_time (datetime): Detection time window end
    - peak_value (float): Maximum detection probability (0.0-1.0)
    
    **File attributes**:
    - metadata (JSON string): Processing parameters and configuration
    """
    
    # Indexed data_columns for fast where= queries
    PICKS_DATA_COLUMNS = ['peak_time', 'is_associated', 'peak_value']
    DETECTIONS_DATA_COLUMNS = ['start_time', 'peak_value']

    def __init__(self, output_path, time_range=None):
        self.output_path = output_path
        self.time_range = time_range  # (t1, t2) tuple for cleaning existing data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    @staticmethod
    def _where_time_bounds(time_col, t1, t2):
        """Build a PyTables where string for time column between t1 and t2 (inclusive).
        t1, t2 can be UTCDateTime or str. Returns None if both t1 and t2 are None.
        Uses nanosecond timestamps for reliable comparison with stored datetime64.
        """
        if t1 is None and t2 is None:
            return None
        if isinstance(t1, str):
            t1 = UTCDateTime(t1)
        if isinstance(t2, str):
            t2 = UTCDateTime(t2)
        ts_lo = pd.Timestamp(t1.datetime).value if t1 is not None else None
        ts_hi = pd.Timestamp(t2.datetime).value if t2 is not None else None
        if ts_lo is not None and ts_hi is not None:
            return f"({time_col} >= {ts_lo}) & ({time_col} <= {ts_hi})"
        if ts_lo is not None:
            return f"({time_col} >= {ts_lo})"
        return f"({time_col} <= {ts_hi})"

    @staticmethod
    def _where_outside_time_bounds(time_col, t1, t2):
        """Build a PyTables where string for time column outside [t1, t2] (exclusive)."""
        if t1 is None and t2 is None:
            return None
        if isinstance(t1, str):
            t1 = UTCDateTime(t1)
        if isinstance(t2, str):
            t2 = UTCDateTime(t2)
        ts_lo = pd.Timestamp(t1.datetime).value if t1 is not None else None
        ts_hi = pd.Timestamp(t2.datetime).value if t2 is not None else None
        if ts_lo is not None and ts_hi is not None:
            return f"({time_col} < {ts_lo}) | ({time_col} > {ts_hi})"
        if ts_lo is not None:
            return f"({time_col} < {ts_lo})"
        return f"({time_col} > {ts_hi})"

    def write(self, pick_list, detection_list, metadata, summary_stats=None):
        """
        Write EQTransformer picks and detections to HDF5 file.
        
        Automatically cleans existing data within the specified time range before
        writing new data. Data is merged with existing data outside the time range.
        Optionally writes a daily summary table and pick_peak_histogram table.

        Parameters
        ----------
        pick_list : PickListX or seisbench.PickList
            List of seismic phase picks
        detection_list : DetectionListX or seisbench.DetectionList  
            List of earthquake detections
        metadata : dict
            Processing metadata to store as file attributes
        summary_stats : dict, optional
            If provided, one row is appended to the 'summary' table and
            histogram rows to 'pick_peak_histogram'. Keys: date, config, ncha,
            nsamp, pick_model, np, ns, npicks, ndetections, p_thresh, s_thresh,
            d_thresh, t_exec_pick, t_updated_pick (t_updated_pick optional).

        Notes
        -----
        If time_range was specified during initialization, any existing data
        within that time range will be removed before writing new data.
        """
        print(f"Writing EQT output to {self.output_path}")

        # If we will overwrite the file (time-range replacement), preserve existing summary/histogram
        existing_summary_df = None
        existing_hist_df = None
        if summary_stats is not None and self.time_range is not None and os.path.exists(self.output_path):
            try:
                existing_summary_df = pd.read_hdf(self.output_path, key='summary')
            except (KeyError, FileNotFoundError):
                pass
            try:
                existing_hist_df = pd.read_hdf(self.output_path, key='pick_peak_histogram')
            except (KeyError, FileNotFoundError):
                pass

        # Convert (SeisBench) PickList to (Pandas) DataFrame
        picks_df = PickListX(pick_list).to_dataframe()
        detections_df = DetectionListX(detection_list).to_dataframe()

        # Ensure is_associated exists for picks (for schema consistency)
        if 'is_associated' not in picks_df.columns:
            picks_df['is_associated'] = pd.NA

        # Handle time-based data replacement
        if self.time_range is not None and os.path.exists(self.output_path):
            # Read existing data and filter out the time range
            existing_picks_filtered, existing_detections_filtered = self._get_filtered_existing_data(
                self.time_range[0], self.time_range[1]
            )
            
            # Combine existing (filtered) data with new data
            if len(existing_picks_filtered) > 0:
                if 'is_associated' not in existing_picks_filtered.columns:
                    existing_picks_filtered = existing_picks_filtered.copy()
                    existing_picks_filtered['is_associated'] = pd.NA
                picks_df = pd.concat([existing_picks_filtered, picks_df], ignore_index=True)
            if len(existing_detections_filtered) > 0:
                detections_df = pd.concat([existing_detections_filtered, detections_df], ignore_index=True)
            
            # Write combined data (overwrite file to avoid schema conflicts)
            if len(picks_df) > 0:
                picks_df.to_hdf(
                    self.output_path, key='picks', mode='w', format='table',
                    data_columns=self.PICKS_DATA_COLUMNS
                )
            if len(detections_df) > 0:
                detections_df.to_hdf(
                    self.output_path, key='detections', mode='a', format='table',
                    data_columns=self.DETECTIONS_DATA_COLUMNS
                )
        else:
            # No time range specified or file doesn't exist, append normally
            if len(picks_df) > 0:
                picks_df.to_hdf(
                    self.output_path, key='picks', mode='a', append=True, format='table',
                    data_columns=self.PICKS_DATA_COLUMNS
                )
            if len(detections_df) > 0:
                detections_df.to_hdf(
                    self.output_path, key='detections', mode='a', append=True, format='table',
                    data_columns=self.DETECTIONS_DATA_COLUMNS
                )

        with h5py.File(self.output_path, 'a') as f:
            f.attrs['metadata'] = json.dumps(metadata)

        # Write summary and pick_peak_histogram tables when summary_stats provided
        if summary_stats is not None:
            self._write_picker_summary_and_histogram(
                summary_stats, picks_df, detections_df,
                existing_summary_df=existing_summary_df,
                existing_hist_df=existing_hist_df,
            )

    def _write_picker_summary_and_histogram(self, summary_stats, picks_df, detections_df,
                                           existing_summary_df=None, existing_hist_df=None):
        """Append one row to summary table and histogram rows to pick_peak_histogram."""
        from datetime import datetime
        date_str = summary_stats.get('date', '')
        if not date_str:
            return
        if 't_updated_pick' not in summary_stats:
            summary_stats = {**summary_stats, 't_updated_pick': datetime.utcnow().strftime('%Y/%m/%dT%H:%M:%S')}
        summary_row = pd.DataFrame([{
            'date': summary_stats.get('date'),
            'config': summary_stats.get('config', ''),
            'ncha': summary_stats.get('ncha', 0),
            'nsamp': summary_stats.get('nsamp', 0),
            'pick_model': summary_stats.get('pick_model', ''),
            'np': summary_stats.get('np', 0),
            'ns': summary_stats.get('ns', 0),
            'npicks': summary_stats.get('npicks', 0),
            'ndetections': summary_stats.get('ndetections', 0),
            'p_thresh': summary_stats.get('p_thresh', 0.0),
            's_thresh': summary_stats.get('s_thresh', 0.0),
            'd_thresh': summary_stats.get('d_thresh', 0.0),
            't_exec_pick': summary_stats.get('t_exec_pick', 0.0),
            't_updated_pick': summary_stats.get('t_updated_pick', ''),
        }])
        if existing_summary_df is not None:
            existing = existing_summary_df[existing_summary_df['date'].astype(str) != str(date_str)]
            summary_df = pd.concat([existing, summary_row], ignore_index=True)
        else:
            try:
                existing = pd.read_hdf(self.output_path, key='summary')
                existing = existing[existing['date'].astype(str) != str(date_str)]
                summary_df = pd.concat([existing, summary_row], ignore_index=True)
            except (KeyError, FileNotFoundError):
                summary_df = summary_row
        with pd.HDFStore(self.output_path, 'a') as store:
            if 'summary' in store:
                del store['summary']
            store.append('summary', summary_df, format='table')
        # Histogram: peak_value binned to 2 decimals, per station and phase (P, S, D)
        from .stations import station_from_trace_id

        hist_frames = []

        # Picks: P and S phases
        if len(picks_df) and 'peak_value' in picks_df.columns and 'trace_id' in picks_df.columns:
            picks_tmp = picks_df.copy()
            if 'phase' in picks_tmp.columns:
                picks_tmp = picks_tmp[picks_tmp['phase'].isin(['P', 'S'])]
            if len(picks_tmp):
                picks_tmp = picks_tmp.copy()
                picks_tmp['station_id'] = picks_tmp['trace_id'].astype(str).apply(station_from_trace_id)
                picks_tmp['peak_value_bin'] = picks_tmp['peak_value'].round(2)
                picks_tmp = picks_tmp.dropna(subset=['peak_value_bin'])
                if len(picks_tmp):
                    picks_hist = (
                        picks_tmp
                        .groupby(['station_id', 'phase', 'peak_value_bin'])
                        .size()
                        .reset_index(name='count')
                    )
                    picks_hist['date'] = date_str
                    hist_frames.append(picks_hist)

        # Detections: treated as phase 'D'
        if len(detections_df) and 'peak_value' in detections_df.columns and 'trace_id' in detections_df.columns:
            det_tmp = detections_df.copy()
            det_tmp['station_id'] = det_tmp['trace_id'].astype(str).apply(station_from_trace_id)
            det_tmp['peak_value_bin'] = det_tmp['peak_value'].round(2)
            det_tmp = det_tmp.dropna(subset=['peak_value_bin'])
            if len(det_tmp):
                det_hist = (
                    det_tmp
                    .groupby(['station_id', 'peak_value_bin'])
                    .size()
                    .reset_index(name='count')
                )
                det_hist['phase'] = 'D'
                det_hist['date'] = date_str
                hist_frames.append(det_hist)

        if hist_frames:
            new_hist_df = pd.concat(hist_frames, ignore_index=True)
            new_hist_df = new_hist_df[['date', 'station_id', 'phase', 'peak_value_bin', 'count']]
        else:
            new_hist_df = pd.DataFrame(columns=['date', 'station_id', 'phase', 'peak_value_bin', 'count'])
        if existing_hist_df is not None:
            existing_hist = existing_hist_df[existing_hist_df['date'].astype(str) != str(date_str)]
            hist_df = pd.concat([existing_hist, new_hist_df], ignore_index=True)
        else:
            try:
                existing_hist = pd.read_hdf(self.output_path, key='pick_peak_histogram')
                existing_hist = existing_hist[existing_hist['date'].astype(str) != str(date_str)]
                hist_df = pd.concat([existing_hist, new_hist_df], ignore_index=True)
            except (KeyError, FileNotFoundError):
                hist_df = new_hist_df
        with pd.HDFStore(self.output_path, 'a') as store:
            if 'pick_peak_histogram' in store:
                del store['pick_peak_histogram']
            store.append('pick_peak_histogram', hist_df, format='table')

    def _get_filtered_existing_data(self, t1, t2):
        """
        Get existing data filtered to exclude the specified time range.
        
        This private method returns existing data outside the specified time range,
        allowing new data to replace only the data within the time range.
        Uses where= to avoid loading the full tables into memory.
        
        Parameters
        ----------
        t1 : UTCDateTime or str
            Start time for data removal
        t2 : UTCDateTime or str
            End time for data removal
            
        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame)
            Filtered picks and detections DataFrames excluding the time range
        """
        print(f"Filtering existing EQT data to exclude {t1} to {t2}")
        
        where_picks = self._where_outside_time_bounds('peak_time', t1, t2)
        where_detections = self._where_outside_time_bounds('start_time', t1, t2)

        try:
            if where_picks is not None:
                try:
                    picks_filtered = pd.read_hdf(self.output_path, key='picks', where=where_picks)
                except (TypeError, ValueError):
                    picks_df = pd.read_hdf(self.output_path, key='picks')
                    t1_dt = UTCDateTime(t1).datetime if isinstance(t1, str) else t1.datetime
                    t2_dt = UTCDateTime(t2).datetime if isinstance(t2, str) else t2.datetime
                    picks_filtered = picks_df[~((picks_df['peak_time'] >= t1_dt) & (picks_df['peak_time'] <= t2_dt))]
            else:
                picks_filtered = pd.read_hdf(self.output_path, key='picks')
            if where_detections is not None:
                try:
                    detections_filtered = pd.read_hdf(self.output_path, key='detections', where=where_detections)
                except (TypeError, ValueError):
                    detections_df = pd.read_hdf(self.output_path, key='detections')
                    t1_dt = UTCDateTime(t1).datetime if isinstance(t1, str) else t1.datetime
                    t2_dt = UTCDateTime(t2).datetime if isinstance(t2, str) else t2.datetime
                    detections_filtered = detections_df[~((detections_df['start_time'] >= t1_dt) & (detections_df['start_time'] <= t2_dt))]
            else:
                detections_filtered = pd.read_hdf(self.output_path, key='detections')
        except (KeyError, FileNotFoundError):
            return pd.DataFrame(), pd.DataFrame()

        # Get total row counts for logging (without loading full tables)
        try:
            with pd.HDFStore(self.output_path, 'r') as store:
                n_picks_total = store.get_storer('picks').table.nrows
                n_det_total = store.get_storer('detections').table.nrows
        except Exception:
            n_picks_total = len(picks_filtered)
            n_det_total = len(detections_filtered)
        picks_removed = n_picks_total - len(picks_filtered) if where_picks else 0
        detections_removed = n_det_total - len(detections_filtered) if where_detections else 0

        print(f"Removing {picks_removed} picks and {detections_removed} detections within time range. "
              f"Keeping {len(picks_filtered)} picks and {len(detections_filtered)} detections outside time range.")
        return picks_filtered, detections_filtered

    def _append_peak_value_where(self, where_expr, min_peak_value):
        """Append (peak_value >= min_peak_value) to a where expression. peak_value is an indexed data_column."""
        if min_peak_value is None:
            return where_expr
        pv_cond = f"(peak_value >= {float(min_peak_value)})"
        if where_expr is None:
            return pv_cond
        return f"({where_expr}) & {pv_cond}"

    def _append_is_associated_where(self, where_expr, is_associated):
        """Append (is_associated == 1) or (is_associated == 0) to a where expression. Picks table only; is_associated is an indexed data_column."""
        if is_associated is None:
            return where_expr
        # PyTables stores bool as 0/1
        val = 1 if is_associated else 0
        cond = f"(is_associated == {val})"
        if where_expr is None:
            return cond
        return f"({where_expr}) & {cond}"

    def read(self, t1=None, t2=None, min_peak_value=None, is_associated=None):
        """
        Read EQTransformer results from HDF5 file with optional time, confidence, and association filtering.
        
        Uses where= on indexed columns: peak_time (picks), start_time (detections),
        peak_value (both), and is_associated (picks only) when the corresponding
        arguments are provided, so only matching rows are loaded from disk.
        
        Parameters
        ----------
        t1 : UTCDateTime or str, optional
            Start time for filtering results
        t2 : UTCDateTime or str, optional
            End time for filtering results
        min_peak_value : float, optional
            Minimum peak_value (confidence) to include; 0.0-1.0. Uses indexed
            data_column for efficient filtering.
        is_associated : bool, optional
            If True, return only picks that have been associated to an event.
            If False, return only picks not yet associated. None (default) means
            no filter. Uses indexed data_column (picks table only; detections are unchanged).
            
        Returns
        -------
        tuple of (PickListX, DetectionListX, dict)
            - PickListX: Seismic phase picks
            - DetectionListX: Earthquake detections  
            - dict: Processing metadata
            
        Notes
        -----
        Time filtering is applied to peak_time for picks and start_time for detections.
        When min_peak_value is set, peak_value is used in the where= clause (indexed).
        When is_associated is True or False, is_associated is used in the where= clause for picks (indexed).
        Prints summary statistics of loaded data.
        """
        print(f"Reading EQT output from {self.output_path}")
        where_picks = self._where_time_bounds('peak_time', t1, t2)
        where_detections = self._where_time_bounds('start_time', t1, t2)
        where_picks = self._append_peak_value_where(where_picks, min_peak_value)
        where_detections = self._append_peak_value_where(where_detections, min_peak_value)
        where_picks = self._append_is_associated_where(where_picks, is_associated)

        try:
            if where_picks is not None:
                try:
                    picks_df = pd.read_hdf(self.output_path, key='picks', where=where_picks)
                except (TypeError, ValueError):
                    picks_df = pd.read_hdf(self.output_path, key='picks')
                    if t1 is not None:
                        picks_df = picks_df[picks_df['peak_time'] >= pd.Timestamp(UTCDateTime(t1).datetime)]
                    if t2 is not None:
                        picks_df = picks_df[picks_df['peak_time'] <= pd.Timestamp(UTCDateTime(t2).datetime)]
                    if min_peak_value is not None:
                        picks_df = picks_df[picks_df['peak_value'] >= min_peak_value]
                    if is_associated is not None:
                        if 'is_associated' in picks_df.columns:
                            picks_df = picks_df[picks_df['is_associated'] == is_associated]
                        else:
                            picks_df = picks_df.iloc[0:0]
            else:
                picks_df = pd.read_hdf(self.output_path, key='picks')
            if where_detections is not None:
                try:
                    detections_df = pd.read_hdf(self.output_path, key='detections', where=where_detections)
                except (TypeError, ValueError):
                    detections_df = pd.read_hdf(self.output_path, key='detections')
                    if t1 is not None:
                        detections_df = detections_df[detections_df['start_time'] >= pd.Timestamp(UTCDateTime(t1).datetime)]
                    if t2 is not None:
                        detections_df = detections_df[detections_df['start_time'] <= pd.Timestamp(UTCDateTime(t2).datetime)]
                    if min_peak_value is not None:
                        detections_df = detections_df[detections_df['peak_value'] >= min_peak_value]
            else:
                detections_df = pd.read_hdf(self.output_path, key='detections')
        except (KeyError, FileNotFoundError) as e:
            print(f"WARNING: Could not read EQT output: {e}")
            raise

        if t1 is not None or t2 is not None:
            print(f"Time filtered from {t1} to {t2}")
        if min_peak_value is not None:
            print(f"Peak value filter: >= {min_peak_value}")
        if is_associated is not None:
            print(f"Association filter: is_associated == {is_associated}")

        # Backward compatibility: ensure is_associated column exists (old files may not have it)
        if 'is_associated' not in picks_df.columns:
            picks_df = picks_df.copy()
            picks_df['is_associated'] = pd.NA

        try:
            with h5py.File(self.output_path, 'r') as f:
                metadata = json.loads(f.attrs['metadata'])
        except Exception:
            print("WARNING: Could not load metadata from HDF5 file. Return None.")
            metadata = None

        # Count P and S picks separately
        p_picks_count = len(picks_df[picks_df['phase'] == 'P']) if len(picks_df) > 0 else 0
        s_picks_count = len(picks_df[picks_df['phase'] == 'S']) if len(picks_df) > 0 else 0
        
        print(f"Detections: {len(detections_df):5d} | P picks: {p_picks_count:5d} | S picks: {s_picks_count:5d}\n")

        return PickListX.from_dataframe(picks_df), DetectionListX.from_dataframe(detections_df), metadata

    def update_is_associated(self, t1, t2, assignments_df):
        """
        Update the is_associated column for picks in [t1, t2] based on assignments.
        Picks that appear in assignments_df are set True; others False.
        Uses where= to read only the affected time range and the rest, then rewrites the picks table.

        Parameters
        ----------
        t1 : UTCDateTime or str
            Start of time range that was just associated
        t2 : UTCDateTime or str
            End of time range
        assignments_df : pd.DataFrame
            Must have columns station_id, phase, pick_time (Unix float)
        """
        if assignments_df is None or len(assignments_df) == 0:
            return
        if 'station_id' not in assignments_df.columns or 'phase' not in assignments_df.columns or 'pick_time' not in assignments_df.columns:
            return
        if not os.path.exists(self.output_path):
            return

        where_in = self._where_time_bounds('peak_time', t1, t2)
        where_out = self._where_outside_time_bounds('peak_time', t1, t2)
        if where_in is None:
            return

        # Build set of (station_id, phase, pick_time_rounded) from assignments
        from .stations import station_from_trace_id

        tol_s = 0.05
        associated_set = set()
        for _, row in assignments_df.iterrows():
            sid = str(row['station_id']).strip()
            ph = str(row['phase']).strip()
            t = float(row['pick_time'])
            associated_set.add((sid, ph, round(t / tol_s) * tol_s))

        # Read picks in [t1, t2]
        picks_in = pd.read_hdf(self.output_path, key='picks', where=where_in)
        if 'is_associated' not in picks_in.columns:
            picks_in = picks_in.copy()
            picks_in['is_associated'] = pd.NA
        picks_in = picks_in.copy()
        is_assoc = []
        for _, row in picks_in.iterrows():
            pt = pd.Timestamp(row['peak_time'])
            unix = pt.value / 1e9
            key = (station_from_trace_id(row['trace_id']), str(row['phase']).strip(), round(unix / tol_s) * tol_s)
            is_assoc.append(key in associated_set)
        picks_in['is_associated'] = is_assoc

        # Read picks outside [t1, t2]
        if where_out is not None:
            picks_out = pd.read_hdf(self.output_path, key='picks', where=where_out)
            if 'is_associated' not in picks_out.columns:
                picks_out = picks_out.copy()
                picks_out['is_associated'] = pd.NA
            picks_df = pd.concat([picks_out, picks_in], ignore_index=True)
        else:
            picks_df = picks_in
        picks_df.sort_values('peak_time', inplace=True)
        picks_df.reset_index(drop=True, inplace=True)

        # Rewrite picks table (preserve detections and metadata)
        with pd.HDFStore(self.output_path, 'a') as store:
            if 'picks' in store:
                del store['picks']
        if len(picks_df) > 0:
            picks_df.to_hdf(
                self.output_path, key='picks', mode='a', format='table',
                data_columns=self.PICKS_DATA_COLUMNS
            )
        try:
            with h5py.File(self.output_path, 'r') as f:
                meta = f.attrs.get('metadata', None)
            if meta is not None:
                with h5py.File(self.output_path, 'a') as f:
                    f.attrs['metadata'] = meta
        except Exception:
            pass
        n_true = sum(picks_in['is_associated'])
        print(f"Updated is_associated for {len(picks_in)} picks in [{t1}, {t2}]: {n_true} associated.")


class PyOctoOutput:
    """
    Handles input/output operations for PyOcto event association results.
    
    This class manages the storage and retrieval of earthquake events and phase
    associations generated by the PyOcto associator. Data is stored in HDF5 format
    with automatic time-based data cleanup capabilities.
    
    Parameters
    ----------
    output_path : str
        Path to the HDF5 output file
    time_range : tuple of (UTCDateTime, UTCDateTime), optional
        Time range (t1, t2) for automatic cleanup of existing data when writing
        
    Attributes
    ----------
    output_path : str
        Path to the HDF5 file
    time_range : tuple or None
        Time range for data cleanup operations
        
    Methods
    -------
    write(events_df, assignments_df, metadata) : Write PyOcto results to file
    read(t1=None, t2=None) : Read PyOcto results from file with optional time filtering
    
    File Format
    -----------
    The HDF5 file contains two main datasets:
    
    **'events' dataset** - DataFrame with columns:
    - time (float): Event origin time as Unix timestamp (seconds since epoch)
    - x (float): Event X coordinate in local coordinate system (km)
    - y (float): Event Y coordinate in local coordinate system (km) 
    - z (float): Event depth (km, positive downward)
    - residual_rms (float): RMS residual of phase associations (seconds)
    - n_picks (int): Total number of associated phase picks
    - n_p_picks (int): Number of associated P-phase picks
    - n_s_picks (int): Number of associated S-phase picks
    
    **'assignments' dataset** - DataFrame with columns:
    - event_idx (int): Index of associated event (references events DataFrame)
    - station_id (str): Station identifier
    - phase (str): Seismic phase type ("P" or "S")
    - pick_time (float): Pick time as Unix timestamp (seconds since epoch)
    - residual (float): Travel time residual (seconds)
    - weight (float): Association weight/confidence (0.0-1.0)
    
    **File attributes**:
    - metadata (JSON string): Processing parameters and configuration
    """
    
    def __init__(self, output_path, time_range=None):
        self.output_path = output_path
        self.time_range = time_range  # (t1, t2) tuple for cleaning existing data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def write(self, events_df, assignments_df, metadata, summary_stats=None):
        """
        Write PyOcto events and assignments to HDF5 file.
        
        Automatically cleans existing data within the specified time range before
        writing new data. Optionally appends one row to the 'summary' table.
        
        Parameters
        ----------
        events_df : pd.DataFrame
            DataFrame containing earthquake events (see class docstring for column details)
        assignments_df : pd.DataFrame
            DataFrame containing phase-to-event assignments (see class docstring for column details)
        metadata : dict
            Processing metadata to store as file attributes
        summary_stats : dict, optional
            If provided, one row is appended to the 'summary' table. Keys: date, config,
            assoc_method, nassignments, nevents, t_exec_assoc, t_updated_assoc (optional).
            
        Notes
        -----
        If time_range was specified during initialization, any existing data
        within that time range will be removed before writing new data.
        """
        print(f"Writing PyOcto output to {self.output_path}")

        existing_summary_df = None
        if summary_stats is not None and self.time_range is not None and os.path.exists(self.output_path):
            try:
                existing_summary_df = pd.read_hdf(self.output_path, key='summary')
            except (KeyError, FileNotFoundError):
                pass

        # Handle time-based data replacement
        if self.time_range is not None and os.path.exists(self.output_path):
            # Read existing data and filter out the time range
            existing_events_filtered, existing_assignments_filtered = self._get_filtered_existing_data(
                self.time_range[0], self.time_range[1]
            )
            
            # Combine existing (filtered) data with new data
            if len(existing_events_filtered) > 0:
                events_df = pd.concat([existing_events_filtered, events_df], ignore_index=True)
            if len(existing_assignments_filtered) > 0:
                # Ensure consistent dtypes before concatenation
                if len(assignments_df) > 0:
                    # Align dtypes to match existing data
                    for col in existing_assignments_filtered.columns:
                        if col in assignments_df.columns:
                            assignments_df[col] = assignments_df[col].astype(existing_assignments_filtered[col].dtype)
                assignments_df = pd.concat([existing_assignments_filtered, assignments_df], ignore_index=True)
            
            # Write combined data (overwrite file to avoid schema conflicts)
            if len(events_df) > 0:
                events_df.to_hdf(self.output_path, key='events', mode='w', format='table')
            if len(assignments_df) > 0:
                assignments_df.to_hdf(self.output_path, key='assignments', mode='a', format='table')
        else:
            # No time range specified or file doesn't exist, append normally
            if len(events_df) > 0:
                events_df.to_hdf(self.output_path, key='events', mode='a', append=True, format='table')
            if len(assignments_df) > 0:
                assignments_df.to_hdf(self.output_path, key='assignments', mode='a', append=True, format='table')

        with h5py.File(self.output_path, 'a') as f:
            f.attrs['metadata'] = json.dumps(metadata)

        if summary_stats is not None:
            self._write_associator_summary(summary_stats, existing_summary_df=existing_summary_df)

    def _write_associator_summary(self, summary_stats, existing_summary_df=None):
        """Append one row to summary table."""
        from datetime import datetime
        date_str = summary_stats.get('date', '')
        if not date_str:
            return
        if 't_updated_assoc' not in summary_stats:
            summary_stats = {**summary_stats, 't_updated_assoc': datetime.utcnow().strftime('%Y/%m/%dT%H:%M:%S')}
        summary_row = pd.DataFrame([{
            'date': summary_stats.get('date'),
            'config': summary_stats.get('config', ''),
            'assoc_method': summary_stats.get('assoc_method', ''),
            'nassignments': summary_stats.get('nassignments', 0),
            'nevents': summary_stats.get('nevents', 0),
            't_exec_assoc': summary_stats.get('t_exec_assoc', 0.0),
            't_updated_assoc': summary_stats.get('t_updated_assoc', ''),
        }])
        if existing_summary_df is not None:
            existing = existing_summary_df[existing_summary_df['date'].astype(str) != str(date_str)]
            summary_df = pd.concat([existing, summary_row], ignore_index=True)
        else:
            try:
                existing = pd.read_hdf(self.output_path, key='summary')
                existing = existing[existing['date'].astype(str) != str(date_str)]
                summary_df = pd.concat([existing, summary_row], ignore_index=True)
            except (KeyError, FileNotFoundError):
                summary_df = summary_row
        with pd.HDFStore(self.output_path, 'a') as store:
            if 'summary' in store:
                del store['summary']
            store.append('summary', summary_df, format='table')

    def _get_filtered_existing_data(self, t1, t2):
        """
        Get existing data filtered to exclude the specified time range.
        
        This private method returns existing data outside the specified time range,
        allowing new data to replace only the data within the time range.
        
        Parameters
        ----------
        t1 : UTCDateTime or str
            Start time for data removal
        t2 : UTCDateTime or str
            End time for data removal
            
        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame)
            Filtered events and assignments DataFrames excluding the time range
            
        Notes
        -----
        Handles conversion of event times from Unix timestamps (float) to datetime
        objects for proper time-based filtering.
        """
        print(f"Filtering existing PyOcto data to exclude {t1} to {t2}")
        
        # Read existing data
        try:
            events_df = pd.read_hdf(self.output_path, key='events')
            assignments_df = pd.read_hdf(self.output_path, key='assignments')
        except (KeyError, FileNotFoundError):
            # No existing data to filter
            return pd.DataFrame(), pd.DataFrame()
            
        # Convert time parameters to datetime if needed
        from obspy import UTCDateTime
        if isinstance(t1, str):
            t1 = UTCDateTime(t1)
        if isinstance(t2, str):
            t2 = UTCDateTime(t2)
            
        t1_dt = t1.datetime
        t2_dt = t2.datetime
        
        # Filter out events within the time range
        # For PyOcto events, we need to check the event time column
        if 'time' in events_df.columns:
            # Convert time column to datetime if it's not already
            if events_df['time'].dtype == 'float64' or events_df['time'].dtype == 'float32':
                # Assume time is in Unix timestamp format, convert to datetime
                events_time_dt = pd.to_datetime(events_df['time'], unit='s')
            elif pd.api.types.is_datetime64_any_dtype(events_df['time']):
                # Already datetime
                events_time_dt = events_df['time']
            else:
                # Try to convert to datetime
                events_time_dt = pd.to_datetime(events_df['time'])
            
            # Keep events OUTSIDE the time range (inverse of the time range)
            time_mask = ~((events_time_dt >= t1_dt) & (events_time_dt <= t2_dt))
            events_filtered = events_df[time_mask]
            
            # Calculate removed events count
            events_removed = len(events_df) - len(events_filtered)
            
            # Filter assignments for remaining events
            remaining_event_ids = set(events_filtered.index) if len(events_filtered) > 0 else set()
            assignments_filtered = assignments_df[assignments_df['event_idx'].isin(remaining_event_ids)] if len(assignments_df) > 0 else pd.DataFrame()
            assignments_removed = len(assignments_df) - len(assignments_filtered)
            
            print(f"Removing {events_removed} events and {assignments_removed} assignments within time range. "
                  f"Keeping {len(events_filtered)} events and {len(assignments_filtered)} assignments outside time range.")
        else:
            # Fallback: no filtering if time column not found
            print("Warning: No 'time' column found in events DataFrame, keeping all existing data")
            events_filtered = events_df
            assignments_filtered = assignments_df
        
        return events_filtered, assignments_filtered

    def read(self, t1=None, t2=None):
        """
        Read PyOcto results from HDF5 file with optional time filtering.
        
        Parameters
        ----------
        t1 : UTCDateTime or str, optional
            Start time for filtering results
        t2 : UTCDateTime or str, optional
            End time for filtering results
            
        Returns
        -------
        tuple of (VCatalog, pd.DataFrame, pd.DataFrame, dict)
            - VCatalog: ObsPy/VDAPSeisUtils earthquake catalog
            - pd.DataFrame: Events DataFrame
            - pd.DataFrame: Assignments DataFrame
            - dict: Processing metadata
            
        Notes
        -----
        Converts PyOcto results to VCatalog format for compatibility with
        other seismological tools. Time filtering is applied to event times.
        """
        print(f"Reading PyOcto output from {self.output_path}")

        # Read in Events, Assignments DataFrames
        events_df = pd.read_hdf(self.output_path, key='events')
        assignments_df = pd.read_hdf(self.output_path, key='assignments')

        with h5py.File(self.output_path, 'r') as f:
            try:
                metadata = json.loads(f.attrs['metadata'])
            except:
                print("WARNING: Could not load metadata from HDF5 file. Return None.")
                metadata = None

        # Convert to VCatalog and add QuakeML to PyASDF file
        from vdapseisutils import VCatalog
        catalog = VCatalog.from_pyocto(events_df, assignments_df)
        
        # Apply time filtering if requested
        if t1 is not None or t2 is not None:
            if isinstance(t1, str):
                t1 = UTCDateTime(t1) if t1 is not None else None
            if isinstance(t2, str):
                t2 = UTCDateTime(t2) if t2 is not None else None
            
            catalog = catalog.filter(time=[t1, t2])
            print(f"Time filtered catalog from {t1} to {t2}")
        
        print(f"Events: {len(catalog):5d}\n")

        return catalog, events_df, assignments_df, metadata


class NLLOutput:
    """
    Handles input/output operations for NonLinLoc earthquake location results.

    Storage is HDF5-only: no ASDF QuakeML blob, no sidecar XML/JSON. The file contains
    a ``metadata`` group (attrs), optional ``summary`` table, ``catalog_table``, and
    ``arrivals_table``. read() builds ObsPy Event/Origin (and Picks/Arrivals when
    requested) from these tables.

    Parameters
    ----------
    output_path : str
        Path to the HDF5 (.h5) output file
    time_range : tuple of (UTCDateTime, UTCDateTime), optional
        Time range (t1, t2) for merging: when writing, existing rows inside this
        range are replaced; rows outside are kept.

    Attributes
    ----------
    output_path : str
        Path to the HDF5 file
    time_range : tuple or None
        Time range for write merge

    Methods
    -------
    write(catalog, metadata) : Write catalog to HDF5 (metadata, catalog_table, arrivals_table)
    read(t1=None, t2=None, include_arrivals=False) : Read catalog from tables
    read_catalog_table(t1=None, t2=None, where=None) : Read catalog_table as DataFrame

    File contents (HDF5)
    -------------------
    - metadata : group attrs (method, nll_home, target, event_date, nll_directory, last_updated)
    - summary : optional table (date, config, loc_method, nlocations, t_exec_loc, t_update_loc)
    - catalog_table : event-level table (event_id, origin_time, lat/lon/depth, mag, ...)
    - arrivals_table : arrival-level table (event_id, trace_id, phase, arrival_time, ...)
    """
    
    def __init__(self, output_path, time_range=None):
        self.output_path = output_path
        self.time_range = time_range  # (t1, t2) tuple for cleaning existing data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Must match docs/data-structures.md exactly
    CATALOG_TABLE_COLUMNS = [
        "event_id",
        "origin_time",
        "latitude",
        "longitude",
        "depth_km",
        "x",
        "y",
        "z",
        "mag",
        "mag_type",
        "residual_rms",
        "n_picks",
        "n_p_picks",
        "n_s_picks",
        "azimuthal_gap",
        "creation_info",  # JSON object; not indexed
        "comments",       # JSON array; not indexed
    ]
    CATALOG_TABLE_INDEXED_COLUMNS = [
        "event_id",
        "origin_time",
        "mag",
        "residual_rms",
        "azimuthal_gap",
    ]

    # arrivals_table: one row per associated arrival (flattened from ObsPy Origin.arrivals)
    ARRIVALS_TABLE_COLUMNS = [
        # Event identifiers / provenance
        "event_id",
        "origin_time",
        # Waveform / station lookup
        "trace_id",
        "station_id",
        # Phase and linking
        "phase",
        "pick_id",
        # Arrival timing and diagnostics
        "arrival_time",
        "residual",
        "weight",
        # Uncertainties (best-effort; may be NA if QuakeML/ObsPy doesn't expose them)
        "residual_uncert",
        "weight_uncert",
        "creation_info",  # JSON; not indexed
        "comments",       # JSON array; not indexed
    ]

    ARRIVALS_TABLE_INDEXED_COLUMNS = [
        "event_id",
        "arrival_time",
        "station_id",
        "phase",
    ]

    @staticmethod
    def _canonicalize_phase(phase: str) -> str:
        if phase is None:
            return ""
        ph = str(phase).strip().upper()
        if ph.startswith("P"):
            return "P"
        if ph.startswith("S"):
            return "S"
        # Keep best-effort for unusual phase labels
        return ph[:1] if ph else ""

    @staticmethod
    def _coerce_utc_datetime(x):
        # Returns pandas datetime64[ns, UTC] (or pd.NaT)
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return pd.NaT
        if hasattr(x, "datetime"):
            # obspy.UTCDateTime
            return pd.to_datetime(x.datetime, utc=True, errors="coerce")
        return pd.to_datetime(x, utc=True, errors="coerce")

    @staticmethod
    def _where_time_bounds(time_col: str, t1=None, t2=None):
        """
        Build a PyTables where expression for inclusive [t1, t2] filtering.

        Assumes `time_col` is stored as pandas datetime64 (tz-aware UTC) in a
        `format="table"` HDF5 dataset.
        """
        if t1 is None and t2 is None:
            return None
        if isinstance(t1, str):
            t1 = UTCDateTime(t1)
        if isinstance(t2, str):
            t2 = UTCDateTime(t2)
        ts_lo = pd.Timestamp(t1.datetime, tz="UTC").isoformat() if t1 is not None else None
        ts_hi = pd.Timestamp(t2.datetime, tz="UTC").isoformat() if t2 is not None else None
        if ts_lo is not None and ts_hi is not None:
            return f"({time_col} >= '{ts_lo}') & ({time_col} <= '{ts_hi}')"
        if ts_lo is not None:
            return f"({time_col} >= '{ts_lo}')"
        return f"({time_col} <= '{ts_hi}')"

    @staticmethod
    def _creation_info_to_json(obj) -> str:
        """Serialize ObsPy CreationInfo (or similar) to a JSON string. Returns '{}' if absent."""
        if obj is None:
            return "{}"
        d = {}
        for attr in ("agency_id", "agency_uri", "author", "author_uri", "creation_time", "version"):
            val = getattr(obj, attr, None)
            if val is not None:
                d[attr] = val
        if not d:
            return "{}"
        return json.dumps(d, default=str)

    @staticmethod
    def _comments_to_json(obj) -> str:
        """Serialize list of ObsPy Comment (or similar) to a JSON array string. Returns '[]' if absent."""
        if obj is None:
            return "[]"
        comments = list(obj) if not isinstance(obj, list) else obj
        out = []
        for c in comments:
            item = {}
            if hasattr(c, "text") and c.text is not None:
                item["text"] = str(c.text)
            if hasattr(c, "type") and c.type is not None:
                item["type"] = str(c.type)
            out.append(item)
        return json.dumps(out, default=str)

    @staticmethod
    def _json_to_creation_info(s):
        """Parse JSON string to ObsPy CreationInfo-like object. Returns None if empty or invalid."""
        if not s or (isinstance(s, str) and s.strip() in ("", "{}")):
            return None
        try:
            d = json.loads(s) if isinstance(s, str) else s
        except (TypeError, json.JSONDecodeError):
            return None
        if not d:
            return None
        from obspy.core.event import CreationInfo
        ci = CreationInfo()
        for key in ("agency_id", "agency_uri", "author", "author_uri", "creation_time", "version"):
            if key in d and d[key] is not None:
                setattr(ci, key, d[key])
        return ci

    @staticmethod
    def _json_to_comments(s) -> list:
        """Parse JSON string to list of ObsPy Comment-like objects. Returns [] if empty or invalid."""
        if not s or (isinstance(s, str) and s.strip() in ("", "[]")):
            return []
        try:
            items = json.loads(s) if isinstance(s, str) else s
        except (TypeError, json.JSONDecodeError):
            return []
        if not isinstance(items, list):
            return []
        from obspy.core.event import Comment
        out = []
        for item in items:
            if not isinstance(item, dict):
                continue
            c = Comment()
            if "text" in item and item["text"] is not None:
                c.text = str(item["text"])
            if "type" in item and item["type"] is not None:
                c.type = str(item["type"])
            out.append(c)
        return out

    @staticmethod
    def _catalog_table_to_events(catalog_df: pd.DataFrame):
        """
        Build a list of ObsPy Event objects from a catalog_table DataFrame.
        Each row becomes one Event with one Origin and optionally one Magnitude.
        """
        from obspy.core.event import (
            Catalog,
            Event,
            Magnitude,
            Origin,
            OriginQuality,
            ResourceIdentifier,
        )

        if catalog_df is None or len(catalog_df) == 0:
            return []

        catalog_df = catalog_df.copy()
        catalog_df["origin_time"] = pd.to_datetime(catalog_df["origin_time"], utc=True, errors="coerce")

        events = []
        for _, row in catalog_df.iterrows():
            event_id = str(row.get("event_id", ""))
            origin_time = row.get("origin_time")
            if pd.isna(origin_time):
                continue
            ot = UTCDateTime(origin_time)

            latitude = row.get("latitude", np.nan)
            longitude = row.get("longitude", np.nan)
            depth_km = row.get("depth_km", np.nan)
            if pd.isna(depth_km):
                depth_m = None
            else:
                depth_m = float(depth_km) * 1000.0 if abs(float(depth_km)) < 1000.0 else float(depth_km)

            x = row.get("x", np.nan)
            y = row.get("y", np.nan)
            z = row.get("z", np.nan)
            if pd.isna(x):
                x = None
            else:
                x = float(x)
            if pd.isna(y):
                y = None
            else:
                y = float(y)
            if pd.isna(z):
                z = None
            else:
                z = float(z)

            residual_rms = row.get("residual_rms", np.nan)
            azimuthal_gap = row.get("azimuthal_gap", np.nan)
            quality = OriginQuality(
                standard_error=float(residual_rms) if pd.notna(residual_rms) else None,
                azimuthal_gap=float(azimuthal_gap) if pd.notna(azimuthal_gap) else None,
            )

            origin = Origin(
                time=ot,
                latitude=float(latitude) if pd.notna(latitude) else None,
                longitude=float(longitude) if pd.notna(longitude) else None,
                depth=depth_m,
                quality=quality,
            )
            if x is not None:
                origin.x = x
            if y is not None:
                origin.y = y
            if z is not None:
                origin.z = z

            mag = row.get("mag", np.nan)
            mag_type = row.get("mag_type", None)
            if pd.notna(mag) and mag is not None:
                magnitude = Magnitude(mag=float(mag), magnitude_type=mag_type or "M")
            else:
                magnitude = None

            event = Event(
                resource_id=ResourceIdentifier(event_id),
                origins=[origin],
                magnitudes=[magnitude] if magnitude is not None else [],
                picks=[],
            )

            creation_info = NLLOutput._json_to_creation_info(row.get("creation_info"))
            if creation_info is not None:
                event.creation_info = creation_info
            comments = NLLOutput._json_to_comments(row.get("comments"))
            if comments:
                event.comments = comments

            events.append(event)

        return events

    @staticmethod
    def _arrivals_table_to_picks_and_arrivals(events_by_id: dict, arrivals_df: pd.DataFrame):
        """
        Create ObsPy Pick and Arrival objects from arrivals_table rows and attach
        them to the corresponding Event's preferred/first origin. Events must
        exist in events_by_id (keyed by event_id string).
        """
        from obspy.core.event import Arrival, Pick, ResourceIdentifier, WaveformStreamID

        if arrivals_df is None or len(arrivals_df) == 0:
            return
        arrivals_df = arrivals_df.copy()
        arrivals_df["arrival_time"] = pd.to_datetime(arrivals_df["arrival_time"], utc=True, errors="coerce")

        for event_id, sub in arrivals_df.groupby("event_id"):
            event = events_by_id.get(str(event_id))
            if event is None:
                continue
            origin = None
            if hasattr(event, "preferred_origin") and callable(event.preferred_origin):
                origin = event.preferred_origin()
            if origin is None:
                origins = getattr(event, "origins", None) or []
                origin = origins[0] if origins else None
            if origin is None:
                continue

            picks = list(getattr(event, "picks", None) or [])
            pick_ids_seen = set()

            for _, row in sub.iterrows():
                pick_id = str(row.get("pick_id", ""))
                if not pick_id or pick_id in pick_ids_seen:
                    continue
                trace_id = str(row.get("trace_id", "") or "")
                phase = NLLOutput._canonicalize_phase(row.get("phase"))
                arrival_time = row.get("arrival_time")
                if pd.isna(arrival_time):
                    continue
                at = UTCDateTime(arrival_time)

                waveform_id = WaveformStreamID(seed_string=trace_id) if trace_id else None
                pick = Pick(
                    time=at,
                    phase_hint=phase or "P",
                    waveform_id=waveform_id,
                    resource_id=ResourceIdentifier(pick_id),
                )
                creation_info = NLLOutput._json_to_creation_info(row.get("creation_info"))
                if creation_info is not None:
                    pick.creation_info = creation_info
                comments = NLLOutput._json_to_comments(row.get("comments"))
                if comments:
                    pick.comments = comments
                picks.append(pick)
                pick_ids_seen.add(pick_id)

                residual = row.get("residual", pd.NA)
                weight = row.get("weight", pd.NA)
                arr = Arrival(phase=phase or "P", pick_id=pick.resource_id)
                if residual is not None and not pd.isna(residual):
                    arr.time_residual = float(residual)
                if weight is not None and not pd.isna(weight):
                    arr.time_weight = float(weight)
                origin.arrivals = getattr(origin, "arrivals", None) or []
                origin.arrivals.append(arr)

            event.picks = picks

    @staticmethod
    def _backup_and_strip_origin_arrivals(catalog):
        """
        Temporarily clear ObsPy Origin.arrivals in-place (to shrink exported QuakeML),
        returning a backup structure to restore later.
        """
        backups: list[list[object]] = []
        for event in catalog:
            origins = getattr(event, "origins", None) or []
            origin_backups = []
            for origin in origins:
                origin_backups.append(getattr(origin, "arrivals", None))
                # Clear while preserving the Origin object
                origin.arrivals = []
            backups.append(origin_backups)
        return backups

    @staticmethod
    def _restore_origin_arrivals(catalog, backups):
        for event, event_backup in zip(catalog, backups):
            origins = getattr(event, "origins", None) or []
            for origin, origin_backup in zip(origins, event_backup):
                origin.arrivals = origin_backup

    @staticmethod
    def _catalog_arrivals_to_dataframe(catalog) -> pd.DataFrame:
        """
        Convert an ObsPy/VCatalog earthquake catalog into a normalized `arrivals_table`
        DataFrame schema.

        Notes:
        - ObsPy `Arrival` objects don't reliably store absolute arrival time.
          For QuakeML-derived catalogs, we store `arrival_time` by following `arrival.pick_id`
          to the referenced `Pick.time`.
        """
        from .stations import station_from_trace_id
        from obspy.core.event import ResourceIdentifier

        rows = []
        for idx, event in enumerate(catalog):
            resource_id = getattr(event, "resource_id", None)
            event_id = getattr(resource_id, "id", None) or f"nll_{idx}"

            # Pick map: pick_id -> pick.time, waveform seed, etc.
            picks = getattr(event, "picks", None) or []
            pick_by_id = {}
            for pick in picks:
                prid = getattr(pick, "resource_id", None)
                pid = getattr(prid, "id", None)
                if pid:
                    pick_by_id[pid] = pick

            # Preferred origin policy
            origin = None
            if hasattr(event, "preferred_origin") and callable(event.preferred_origin):
                origin = event.preferred_origin()
            if origin is None:
                origins = getattr(event, "origins", None) or []
                origin = origins[0] if origins else None

            origin_time = pd.NaT
            origins = getattr(event, "origins", None) or []
            if origin is not None:
                origin_time = NLLOutput._coerce_utc_datetime(getattr(origin, "time", None))
            elif origins:
                origin_time = NLLOutput._coerce_utc_datetime(getattr(origins[0], "time", None))

            origins_with_arrivals = []
            if origin is not None:
                origins_with_arrivals = [origin]
            else:
                origins_with_arrivals = origins

            # If preferred origin has no arrivals, fall back to the first origin that does.
            selected_origin = None
            if origin is not None and (getattr(origin, "arrivals", None) or []):
                selected_origin = origin
            else:
                for o in origins:
                    if getattr(o, "arrivals", None) or []:
                        selected_origin = o
                        origin_time = NLLOutput._coerce_utc_datetime(getattr(o, "time", None))
                        break

            arrivals = getattr(selected_origin, "arrivals", None) or [] if selected_origin is not None else []
            for arr in arrivals:
                ph = getattr(arr, "phase", None) or getattr(arr, "phase_hint", None)
                phase = NLLOutput._canonicalize_phase(ph)
                pick_id_obj = getattr(arr, "pick_id", None)
                pick_id = getattr(pick_id_obj, "id", None) if pick_id_obj is not None else None
                if not pick_id:
                    continue

                pick = pick_by_id.get(pick_id)
                if pick is None:
                    continue

                # Arrival time is derived from the referenced pick time.
                ptime = getattr(pick, "time", None)
                arrival_time = NLLOutput._coerce_utc_datetime(ptime)

                # Waveform/trace info comes from the referenced Pick.
                seed = None
                wid = getattr(pick, "waveform_id", None)
                if wid is not None:
                    if hasattr(wid, "get_seed_string"):
                        seed = wid.get_seed_string()
                    elif isinstance(wid, str):
                        seed = wid
                trace_id = seed or ""
                station_id = station_from_trace_id(trace_id) if trace_id else ""

                # Residual/weight: best-effort with common ObsPy attribute names.
                residual = getattr(arr, "time_residual", None)
                if residual is None:
                    residual = getattr(arr, "timeResidual", None)
                weight = getattr(arr, "time_weight", None)
                if weight is None:
                    weight = getattr(arr, "timeWeight", None)

                # uncertainties: ObsPy's Arrival doesn't expose these reliably, so default NA.
                residual_uncert = pd.NA
                weight_uncert = pd.NA

                # creation_info / comments from pick (and optionally arrival)
                creation_info = NLLOutput._creation_info_to_json(getattr(pick, "creation_info", None))
                comments = NLLOutput._comments_to_json(getattr(pick, "comments", None))

                rows.append(
                    {
                        "event_id": str(event_id),
                        "origin_time": origin_time,
                        "trace_id": str(trace_id),
                        "station_id": str(station_id),
                        "phase": phase,
                        "pick_id": str(pick_id),
                        "arrival_time": arrival_time,
                        "residual": float(residual) if residual is not None and not pd.isna(residual) else pd.NA,
                        "weight": float(weight) if weight is not None and not pd.isna(weight) else pd.NA,
                        "residual_uncert": residual_uncert,
                        "weight_uncert": weight_uncert,
                        "creation_info": creation_info,
                        "comments": comments,
                    }
                )

        df = pd.DataFrame(rows, columns=NLLOutput.ARRIVALS_TABLE_COLUMNS)
        if len(df) > 0:
            df["origin_time"] = pd.to_datetime(df["origin_time"], utc=True, errors="coerce")
            df["arrival_time"] = pd.to_datetime(df["arrival_time"], utc=True, errors="coerce")
        return df

    @staticmethod
    def _write_arrivals_table(output_path: str, arrivals_table_df: pd.DataFrame):
        arrivals_table_df = arrivals_table_df.copy()
        if arrivals_table_df.empty:
            arrivals_table_df = pd.DataFrame(columns=NLLOutput.ARRIVALS_TABLE_COLUMNS)

        data_columns = NLLOutput.ARRIVALS_TABLE_INDEXED_COLUMNS
        for col in data_columns:
            if col not in arrivals_table_df.columns:
                arrivals_table_df[col] = pd.NA

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Replace key so repeated runs (or time_range merges) remain consistent.
        if os.path.exists(output_path):
            with pd.HDFStore(output_path, "a") as store:
                if "arrivals_table" in store:
                    del store["arrivals_table"]

        # Ensure stable dtypes for query engine.
        if len(arrivals_table_df) > 0:
            arrivals_table_df["origin_time"] = pd.to_datetime(arrivals_table_df["origin_time"], utc=True, errors="coerce")
            arrivals_table_df["arrival_time"] = pd.to_datetime(arrivals_table_df["arrival_time"], utc=True, errors="coerce")

        arrivals_table_df.to_hdf(
            output_path,
            key="arrivals_table",
            mode="a" if os.path.exists(output_path) else "w",
            format="table",
            data_columns=data_columns,
            complevel=9,
            complib="blosc",
        )

    def read_arrivals(self, events=None, event_ids=None, t1=None, t2=None) -> pd.DataFrame:
        """
        Fast arrival lookup from `nll.h5` `arrivals_table`.

        Returns a pd.DataFrame with schema `NLLOutput.ARRIVALS_TABLE_COLUMNS`.
        """
        if events is None and event_ids is None:
            raise ValueError("Provide either `events` or `event_ids` to read arrivals.")

        # Normalize event_id list
        if event_ids is None:
            event_ids = []
            if events is not None:
                for ev in events:
                    resid = getattr(ev, "resource_id", None)
                    eid = getattr(resid, "id", None)
                    if eid:
                        event_ids.append(str(eid))
        event_ids = [str(eid) for eid in (event_ids or []) if eid is not None]

        time_where = self._where_time_bounds("arrival_time", t1=t1, t2=t2)

        event_where = None
        if len(event_ids) > 0:
            # PyTables where() doesn't support SQL IN well; use OR chain for small sets.
            # For large sets, rely on time filtering instead.
            if len(event_ids) <= 2000:
                ors = " | ".join([f'(event_id == \"{eid}\")' for eid in event_ids])
                event_where = f"({ors})"
            else:
                print("WARNING: `event_ids` too large for where() OR-chain; ignoring event_id filter.")

        if time_where and event_where:
            where = f"({time_where}) & {event_where}"
        else:
            where = time_where or event_where

        df = pd.read_hdf(self.output_path, key="arrivals_table", where=where)
        if len(df) > 0:
            df["origin_time"] = pd.to_datetime(df["origin_time"], utc=True, errors="coerce")
            df["arrival_time"] = pd.to_datetime(df["arrival_time"], utc=True, errors="coerce")
        df.sort_values("arrival_time", inplace=True, ignore_index=True)
        return df

    @staticmethod
    def attach_arrivals_to_catalog(catalog, arrivals_df: pd.DataFrame):
        """
        Attach real ObsPy Arrival objects onto each Event's preferred Origin.

        Attachment strategy:
        - Create Arrival objects with (phase, pick_id, time_residual, time_weight).
        - The absolute arrival time can be derived from the referenced Pick.time, so
          ObsPy Arrival objects may not store a direct arrival-time attribute.
        """
        from obspy.core.event import Arrival, ResourceIdentifier

        if arrivals_df is None or len(arrivals_df) == 0:
            return catalog

        arrivals_df = arrivals_df.copy()
        if "event_id" not in arrivals_df.columns:
            raise KeyError("arrivals_df must include 'event_id'")

        # Index events by event_id
        events_by_id = {}
        for event in catalog:
            resid = getattr(event, "resource_id", None)
            eid = getattr(resid, "id", None)
            if eid:
                events_by_id[str(eid)] = event

        # Attach per event
        for event_id, sub in arrivals_df.groupby("event_id"):
            event = events_by_id.get(str(event_id))
            if event is None:
                continue

            # Preferred origin if possible; else first origin.
            origin = None
            if hasattr(event, "preferred_origin") and callable(event.preferred_origin):
                origin = event.preferred_origin()
            if origin is None:
                origins = getattr(event, "origins", None) or []
                origin = origins[0] if origins else None
            if origin is None:
                continue

            # Map pick_id -> Pick for validation.
            picks = getattr(event, "picks", None) or []
            pick_ids = {getattr(getattr(p, "resource_id", None), "id", None) for p in picks}

            origin.arrivals = []
            for _, row in sub.iterrows():
                pick_id = row.get("pick_id", None)
                if not pick_id:
                    continue
                if str(pick_id) not in pick_ids:
                    # Can't attach Arrival without referenced Pick.
                    continue

                phase = row.get("phase", None) or ""
                phase = NLLOutput._canonicalize_phase(phase)
                arr = Arrival(phase=phase, pick_id=ResourceIdentifier(str(pick_id)))

                residual = row.get("residual", pd.NA)
                if residual is not None and not pd.isna(residual):
                    arr.time_residual = float(residual)

                weight = row.get("weight", pd.NA)
                if weight is not None and not pd.isna(weight):
                    arr.time_weight = float(weight)

                origin.arrivals.append(arr)

        return catalog

    @staticmethod
    def _catalog_to_dataframe(catalog):
        """
        Convert an ObsPy `Catalog` (or vdapseisutils `VCatalog`) into the
        `catalog_table` DataFrame schema described in docs/data-structures.md.
        """
        rows = []

        for idx, event in enumerate(catalog):
            resource_id = getattr(event, "resource_id", None)
            event_id = getattr(resource_id, "id", None)
            if not event_id:
                event_id = f"nll_{idx}"

            origin = None
            if hasattr(event, "preferred_origin") and callable(event.preferred_origin):
                origin = event.preferred_origin()
            if origin is None:
                origins = getattr(event, "origins", None) or []
                origin = origins[0] if origins else None

            # Magnitude
            magnitude = None
            if hasattr(event, "preferred_magnitude") and callable(event.preferred_magnitude):
                magnitude = event.preferred_magnitude()
            if magnitude is None:
                magnitudes = getattr(event, "magnitudes", None) or []
                magnitude = magnitudes[0] if magnitudes else None

            if magnitude is not None:
                mag = getattr(magnitude, "mag", None)
                mag_type = getattr(magnitude, "magnitude_type", None)
            else:
                mag = np.nan
                mag_type = None

            # Default geometry + quality
            latitude = np.nan
            longitude = np.nan
            depth_km = np.nan
            x = np.nan
            y = np.nan
            z = np.nan
            residual_rms = np.nan
            azimuthal_gap = np.nan
            origin_time = pd.NaT

            n_p_picks = 0
            n_s_picks = 0

            if origin is not None:
                lat = getattr(origin, "latitude", None)
                lon = getattr(origin, "longitude", None)
                depth = getattr(origin, "depth", None)
                origin_time_val = getattr(origin, "time", None)

                if lat is not None:
                    latitude = float(lat)
                if lon is not None:
                    longitude = float(lon)

                if depth is not None:
                    depth = float(depth)
                    # Heuristic: ObsPy origins are often stored in meters; docs want km.
                    depth_km = depth / 1000.0 if abs(depth) > 1000.0 else depth

                # Coordinates in local system (optional)
                x_val = getattr(origin, "x", None)
                if x_val is not None:
                    x = float(x_val)
                y_val = getattr(origin, "y", None)
                if y_val is not None:
                    y = float(y_val)
                z_val = getattr(origin, "z", None)
                if z_val is not None:
                    z = float(z_val)

                # Origin time -> pandas datetime64
                if origin_time_val is not None:
                    if hasattr(origin_time_val, "datetime"):
                        origin_time = origin_time_val.datetime
                    else:
                        origin_time = pd.to_datetime(origin_time_val, utc=True, errors="coerce")

                # Residual + azimuthal gap
                quality = getattr(origin, "quality", None)
                if quality is not None:
                    se = getattr(quality, "standard_error", None)
                    gap = getattr(quality, "azimuthal_gap", None)
                    if se is not None:
                        residual_rms = float(se)
                    if gap is not None:
                        azimuthal_gap = float(gap)

                # Prefer arrivals on preferred origin if present; otherwise fall back to picks.
                arrivals = getattr(origin, "arrivals", None) or []
                if len(arrivals) > 0:
                    for arr in arrivals:
                        ph = getattr(arr, "phase", None) or getattr(arr, "phase_hint", None)
                        if not ph:
                            continue
                        ph = str(ph).strip().upper()
                        if ph.startswith("P"):
                            n_p_picks += 1
                        elif ph.startswith("S"):
                            n_s_picks += 1

            # If arrivals did not exist, count from event picks by phase_hint.
            if n_p_picks == 0 and n_s_picks == 0:
                picks = getattr(event, "picks", None) or []
                for pick in picks:
                    ph = getattr(pick, "phase_hint", None) or getattr(pick, "phase", None)
                    if not ph:
                        continue
                    ph = str(ph).strip().upper()
                    if ph.startswith("P"):
                        n_p_picks += 1
                    elif ph.startswith("S"):
                        n_s_picks += 1

            n_picks = int(n_p_picks) + int(n_s_picks)

            # creation_info / comments: prefer origin, fallback to event
            creation_info = NLLOutput._creation_info_to_json(
                getattr(origin, "creation_info", None) if origin else None
            )
            if creation_info == "{}" and event is not None:
                creation_info = NLLOutput._creation_info_to_json(getattr(event, "creation_info", None))
            comments = NLLOutput._comments_to_json(
                getattr(origin, "comments", None) if origin else None
            )
            if comments == "[]" and event is not None:
                comments = NLLOutput._comments_to_json(getattr(event, "comments", None))

            rows.append(
                {
                    "event_id": str(event_id),
                    "origin_time": origin_time,
                    "latitude": latitude,
                    "longitude": longitude,
                    "depth_km": depth_km,
                    "x": x,
                    "y": y,
                    "z": z,
                    "mag": mag if mag is not None else np.nan,
                    # PyTables string columns don't like `None`; store empty string instead.
                    "mag_type": "" if mag_type is None else str(mag_type),
                    "residual_rms": residual_rms,
                    "n_picks": n_picks,
                    "n_p_picks": int(n_p_picks),
                    "n_s_picks": int(n_s_picks),
                    "azimuthal_gap": azimuthal_gap,
                    "creation_info": creation_info,
                    "comments": comments,
                }
            )

        df = pd.DataFrame(rows, columns=NLLOutput.CATALOG_TABLE_COLUMNS)
        # Ensure dtypes are compatible with PyTables query engine.
        if len(df) > 0:
            df["origin_time"] = pd.to_datetime(df["origin_time"], utc=True, errors="coerce")
        return df

    @staticmethod
    def _write_catalog_table(output_path, catalog_table_df):
        """Write `catalog_table` to HDF5 using indexed PyTables data_columns."""
        catalog_table_df = catalog_table_df.copy()
        if catalog_table_df.empty:
            # Still write an empty table with the expected schema.
            catalog_table_df = pd.DataFrame(columns=NLLOutput.CATALOG_TABLE_COLUMNS)

        data_columns = NLLOutput.CATALOG_TABLE_INDEXED_COLUMNS
        for col in data_columns:
            if col not in catalog_table_df.columns:
                catalog_table_df[col] = pd.NA

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Replace the key so repeated runs (or time_range merges) remain consistent.
        if os.path.exists(output_path):
            with pd.HDFStore(output_path, "a") as store:
                if "catalog_table" in store:
                    del store["catalog_table"]

        mode = "a" if os.path.exists(output_path) else "w"
        catalog_table_df.to_hdf(
            output_path,
            key="catalog_table",
            mode=mode,
            format="table",
            data_columns=data_columns,
            complevel=9,
            complib="blosc",
        )

    def write(self, catalog, metadata, summary_stats=None):
        """
        Write earthquake catalog to HDF5 (metadata, summary, catalog_table, arrivals_table only).

        Automatically merges with existing table rows outside time_range when time_range
        is set and the file exists. Optionally appends one row to the 'summary' table.

        Parameters
        ----------
        catalog : obspy.Catalog or vdapseisutils.VCatalog
            Earthquake catalog to write
        metadata : dict
            Processing metadata to store in HDF5 'metadata' group attrs
        summary_stats : dict, optional
            If provided, one row is appended to 'summary'. Keys:
            date, config, loc_method, nlocations, t_exec_loc, t_update_loc (optional).

        Notes
        -----
        If time_range was specified during initialization, existing rows within that
        range are replaced by the new catalog/arrivals; rows outside the range are kept.
        """
        print(f"Writing NonLinLoc output to {self.output_path}")

        existing_summary_df = None
        corrupt_file = False
        if summary_stats is not None and self.time_range is not None and os.path.exists(self.output_path):
            try:
                existing_summary_df = pd.read_hdf(self.output_path, key='summary')
            except (KeyError, FileNotFoundError):
                pass
            except Exception as e:
                existing_summary_df = None
                corrupt_file = True
                print(f"WARNING: Could not read existing locator file (corrupt or truncated): {e}")
        if corrupt_file and os.path.exists(self.output_path):
            try:
                os.remove(self.output_path)
                print(f"Removed corrupt file {self.output_path}; writing fresh file.")
            except OSError:
                pass

        if catalog is None or len(catalog) == 0:
            print("Warning: Empty catalog, skipping write operation")
            return

        try:
            catalog_df = self._catalog_to_dataframe(catalog)
            arrivals_df = self._catalog_arrivals_to_dataframe(catalog)
        except Exception as e:
            print(f"WARNING: Could not extract tables from catalog: {e}")
            catalog_df = pd.DataFrame(columns=NLLOutput.CATALOG_TABLE_COLUMNS)
            arrivals_df = pd.DataFrame(columns=NLLOutput.ARRIVALS_TABLE_COLUMNS)

        if self.time_range is not None and os.path.exists(self.output_path):
            existing_catalog_df, existing_arrivals_df = self._get_filtered_existing_table_rows(
                self.time_range[0], self.time_range[1]
            )
            if existing_catalog_df is not None and len(existing_catalog_df) > 0:
                catalog_df = pd.concat([existing_catalog_df, catalog_df], ignore_index=True)
            if existing_arrivals_df is not None and len(existing_arrivals_df) > 0:
                arrivals_df = pd.concat([existing_arrivals_df, arrivals_df], ignore_index=True)

        if catalog_df.empty:
            print("Warning: No catalog rows to write, skipping write operation")
            return

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        file_mode = "w" if (self.time_range is not None and os.path.exists(self.output_path)) else "a"
        if file_mode == "w" and os.path.exists(self.output_path):
            os.remove(self.output_path)

        processing_time = pd.Timestamp.now()
        with h5py.File(self.output_path, "a") as f:
            if "metadata" not in f:
                f.create_group("metadata")
            metadata_group = f["metadata"]
            for key, value in metadata.items():
                if isinstance(value, (tuple, list)):
                    metadata_group.attrs[key] = value
                else:
                    metadata_group.attrs[key] = str(value)
            metadata_group.attrs["last_updated"] = str(processing_time)

        if summary_stats is not None:
            self._write_locator_summary(summary_stats, existing_summary_df=existing_summary_df)

        try:
            self._write_catalog_table(self.output_path, catalog_df)
            self._write_arrivals_table(self.output_path, arrivals_df)
        except Exception as e:
            print(f"WARNING: Could not write tables to {self.output_path}: {e}")

    def _get_filtered_existing_table_rows(self, t1, t2):
        """
        Read existing catalog_table and arrivals_table and return rows with origin_time
        outside [t1, t2]. Used for time-range merge in write().

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame) or (None, None)
            (catalog_df, arrivals_df) with rows outside the time range, or (None, None)
            if the file or tables are missing.
        """
        if not os.path.exists(self.output_path):
            return (None, None)
        if isinstance(t1, str):
            t1 = UTCDateTime(t1)
        if isinstance(t2, str):
            t2 = UTCDateTime(t2)
        ts_lo = pd.Timestamp(t1.datetime, tz="UTC")
        ts_hi = pd.Timestamp(t2.datetime, tz="UTC")
        try:
            with pd.HDFStore(self.output_path, "r") as store:
                if "catalog_table" not in store:
                    return (None, None)
                catalog_df = pd.read_hdf(self.output_path, key="catalog_table")
        except Exception:
            return (None, None)
        if catalog_df.empty:
            try:
                arrivals_df = pd.read_hdf(self.output_path, key="arrivals_table")
            except Exception:
                arrivals_df = pd.DataFrame(columns=NLLOutput.ARRIVALS_TABLE_COLUMNS)
            return (catalog_df, arrivals_df)
        catalog_df["origin_time"] = pd.to_datetime(catalog_df["origin_time"], utc=True, errors="coerce")
        mask = (catalog_df["origin_time"] < ts_lo) | (catalog_df["origin_time"] > ts_hi)
        catalog_df = catalog_df.loc[mask].copy()
        kept_event_ids = set(catalog_df["event_id"].astype(str).unique())
        try:
            arrivals_df = pd.read_hdf(self.output_path, key="arrivals_table")
        except Exception:
            arrivals_df = pd.DataFrame(columns=NLLOutput.ARRIVALS_TABLE_COLUMNS)
        if not arrivals_df.empty and len(kept_event_ids) > 0:
            arrivals_df = arrivals_df[arrivals_df["event_id"].astype(str).isin(kept_event_ids)].copy()
        elif not arrivals_df.empty and len(kept_event_ids) == 0:
            arrivals_df = pd.DataFrame(columns=NLLOutput.ARRIVALS_TABLE_COLUMNS)
        return (catalog_df, arrivals_df)

    def _write_locator_summary(self, summary_stats, existing_summary_df=None):
        """Append one row to summary table in the HDF5 file."""
        from datetime import datetime
        date_str = summary_stats.get('date', '')
        if not date_str:
            return
        if 't_update_loc' not in summary_stats:
            summary_stats = {**summary_stats, 't_update_loc': datetime.utcnow().strftime('%Y/%m/%dT%H:%M:%S')}
        summary_row = pd.DataFrame([{
            'date': summary_stats.get('date'),
            'config': summary_stats.get('config', ''),
            'loc_method': summary_stats.get('loc_method', ''),
            'nlocations': summary_stats.get('nlocations', 0),
            't_exec_loc': summary_stats.get('t_exec_loc', 0.0),
            't_update_loc': summary_stats.get('t_update_loc', ''),
        }])
        if existing_summary_df is not None:
            existing = existing_summary_df[existing_summary_df['date'].astype(str) != str(date_str)]
            summary_df = pd.concat([existing, summary_row], ignore_index=True)
        else:
            try:
                existing = pd.read_hdf(self.output_path, key='summary')
                existing = existing[existing['date'].astype(str) != str(date_str)]
                summary_df = pd.concat([existing, summary_row], ignore_index=True)
            except (KeyError, FileNotFoundError):
                summary_df = summary_row
        with pd.HDFStore(self.output_path, 'a') as store:
            if 'summary' in store:
                del store['summary']
            store.append('summary', summary_df, format='table')

    def read_catalog_table(self, t1=None, t2=None, where=None) -> pd.DataFrame:
        """
        Read catalog_table from HDF5 with optional time bounds and extra where expression.

        Parameters
        ----------
        t1, t2 : UTCDateTime or str, optional
            Inclusive time bounds on origin_time.
        where : str, optional
            Additional PyTables where expression (combined with & if time bounds given).

        Returns
        -------
        pd.DataFrame
            catalog_table rows (empty DataFrame if file or table missing).
        """
        if not os.path.exists(self.output_path):
            return pd.DataFrame(columns=NLLOutput.CATALOG_TABLE_COLUMNS)
        try:
            time_where = self._where_time_bounds("origin_time", t1, t2)
            if time_where and where:
                full_where = f"({time_where}) & ({where})"
            else:
                full_where = time_where or where
            if full_where:
                df = pd.read_hdf(self.output_path, key="catalog_table", where=full_where)
            else:
                df = pd.read_hdf(self.output_path, key="catalog_table")
        except (KeyError, FileNotFoundError):
            return pd.DataFrame(columns=NLLOutput.CATALOG_TABLE_COLUMNS)
        if len(df) > 0:
            df["origin_time"] = pd.to_datetime(df["origin_time"], utc=True, errors="coerce")
        return df

    def read(self, t1=None, t2=None, include_arrivals: bool = False):
        """
        Read earthquake catalog from HDF5 tables (catalog_table, optionally arrivals_table).

        Parameters
        ----------
        t1 : UTCDateTime or str, optional
            Start time for filtering events (origin_time).
        t2 : UTCDateTime or str, optional
            End time for filtering events (origin_time).
        include_arrivals : bool, optional
            If True, load arrivals_table and attach Picks/Arrivals to each Event's origin.

        Returns
        -------
        tuple of (VCatalog, dict) or (None, None)
            VCatalog and metadata dict, or (None, None) if file or catalog_table missing.
        """
        print(f"Reading NonLinLoc output from {self.output_path}")

        metadata = None
        try:
            with h5py.File(self.output_path, "r") as f:
                if "metadata" in f:
                    metadata_group = f["metadata"]
                    metadata = dict(metadata_group.attrs)
                    for key, value in list(metadata.items()):
                        if isinstance(value, bytes):
                            metadata[key] = value.decode("utf-8")
        except (OSError, Exception):
            print("WARNING: Could not open HDF5 file or read metadata.")
            return None, None

        catalog_df = self.read_catalog_table(t1=t1, t2=t2)

        if catalog_df.empty:
            try:
                with pd.HDFStore(self.output_path, "r") as store:
                    if "catalog_table" not in store:
                        print(f"ERROR: catalog_table not found in {self.output_path}")
                        return None, None
            except Exception:
                return None, None
            from vdapseisutils import VCatalog
            from obspy import Catalog
            print(f"Events:     0\n")
            return VCatalog(Catalog()), metadata

        events = self._catalog_table_to_events(catalog_df)
        if include_arrivals:
            try:
                event_ids = [str(getattr(getattr(e, "resource_id", None), "id", "")) for e in events]
                arrivals_df = self.read_arrivals(event_ids=event_ids, t1=t1, t2=t2)
                events_by_id = {}
                for ev in events:
                    eid = getattr(getattr(ev, "resource_id", None), "id", None)
                    if eid:
                        events_by_id[str(eid)] = ev
                self._arrivals_table_to_picks_and_arrivals(events_by_id, arrivals_df)
            except KeyError:
                print(f"WARNING: arrivals_table missing in {self.output_path}; returning catalog without arrivals.")

        if t1 is not None or t2 is not None:
            print(f"Time filtered catalog from {t1} to {t2}")

        from obspy import Catalog
        from vdapseisutils import VCatalog
        vcatalog = VCatalog(Catalog(events=events))
        print(f"Events: {len(vcatalog):5d}\n")
        return vcatalog, metadata

    def export_catalog(
        self,
        format: str = "quakeml",
        include_arrivals: bool = False,
        export_path=None,
    ):
        """
        Export the located catalog to an ObsPy-supported format.

        Reads catalog from tables via read(), then writes to the requested format
        (e.g. QUAKEML). If include_arrivals is False, Origin.arrivals are stripped
        before export to keep the file small.

        Parameters
        ----------
        format : str
            ObsPy output format (e.g. ``'quakeml'``).
        include_arrivals : bool
            If False, clears ``Origin.arrivals`` before export (default).
        export_path : str, optional
            Output path. If omitted, derives from ``self.output_path``.

        Returns
        -------
        str
            The export path.
        """
        catalog, _ = self.read(include_arrivals=include_arrivals)
        if catalog is None:
            raise FileNotFoundError(f"No catalog found in {self.output_path}")

        fmt = (format or "quakeml").strip().lower()
        if fmt in ("quakeml", "xml"):
            obs_py_format = "QUAKEML"
            ext = ".xml"
        else:
            # Let ObsPy handle more exotic formats; pick a generic extension.
            obs_py_format = fmt.upper()
            ext = f".{fmt}"

        if export_path is None:
            export_path = self.output_path.replace(".h5", f"_export{ext}")

        # Ensure we don't export arrivals unless requested.
        arrivals_backup = None
        if not include_arrivals:
            arrivals_backup = self._backup_and_strip_origin_arrivals(catalog)

        try:
            catalog.write(export_path, format=obs_py_format)
        finally:
            if arrivals_backup is not None:
                self._restore_origin_arrivals(catalog, arrivals_backup)

        return export_path
