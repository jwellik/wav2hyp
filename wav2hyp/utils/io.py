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
from seisbench.util.annotations import PickList as SBPickList, DetectionList as SBDetectionList
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
        # Convert back to seisbench PickList format
        from seisbench.util.annotations import PickList, Pick

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
        # Convert back to seisbench DetectionList format
        from seisbench.util.annotations import PickList, Pick

        picks = []
        for _, row in detection_df.iterrows():
            pick = Pick(
                trace_id=row['trace_id'],
                start_time=UTCDateTime(row['start_time']),
                end_time=UTCDateTime(row['end_time']),
                peak_value=row['peak_value'],
            )
            picks.append(pick)

        return cls(picks)


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
        # Histogram: peak_value binned to 2 decimals, per phase (P, S, D)
        hist_rows = []
        for phase, df in [('P', picks_df[picks_df['phase'] == 'P'] if len(picks_df) and 'phase' in picks_df.columns else pd.DataFrame()),
                          ('S', picks_df[picks_df['phase'] == 'S'] if len(picks_df) and 'phase' in picks_df.columns else pd.DataFrame()),
                          ('D', detections_df if len(detections_df) and 'peak_value' in detections_df.columns else pd.DataFrame())]:
            if df is None or len(df) == 0 or 'peak_value' not in df.columns:
                continue
            pv = df['peak_value'].round(2)
            for bin_val, count in pv.value_counts().items():
                hist_rows.append({'date': date_str, 'phase': phase, 'peak_value_bin': float(bin_val), 'count': int(count)})
        if not hist_rows:
            new_hist_df = pd.DataFrame(columns=['date', 'phase', 'peak_value_bin', 'count'])
        else:
            new_hist_df = pd.DataFrame(hist_rows)
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

    def read(self, t1=None, t2=None):
        """
        Read EQTransformer results from HDF5 file with optional time filtering.
        
        Uses where= on peak_time (picks) and start_time (detections) when t1/t2
        are provided, so only rows in the time window are loaded.
        
        Parameters
        ----------
        t1 : UTCDateTime or str, optional
            Start time for filtering results
        t2 : UTCDateTime or str, optional
            End time for filtering results
            
        Returns
        -------
        tuple of (PickListX, DetectionListX, dict)
            - PickListX: Seismic phase picks
            - DetectionListX: Earthquake detections  
            - dict: Processing metadata
            
        Notes
        -----
        Time filtering is applied to peak_time for picks and start_time for detections.
        Prints summary statistics of loaded data.
        """
        print(f"Reading EQT output from {self.output_path}")
        where_picks = self._where_time_bounds('peak_time', t1, t2)
        where_detections = self._where_time_bounds('start_time', t1, t2)

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
            else:
                detections_df = pd.read_hdf(self.output_path, key='detections')
        except (KeyError, FileNotFoundError) as e:
            print(f"WARNING: Could not read EQT output: {e}")
            raise

        if t1 is not None or t2 is not None:
            print(f"Time filtered from {t1} to {t2}")

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
        def station_from_trace(trace_id):
            parts = trace_id.split('.')
            return '.'.join(parts[:2]) if len(parts) >= 2 else trace_id

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
            key = (station_from_trace(row['trace_id']), str(row['phase']).strip(), round(unix / tol_s) * tol_s)
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
    
    This class manages the storage and retrieval of earthquake catalogs generated
    by NonLinLoc. Data is stored in ASDF format (preferred) or XML/JSON format
    (fallback) with automatic time-based data cleanup capabilities.
    
    Parameters
    ----------
    output_path : str
        Path to the ASDF (.h5) output file
    time_range : tuple of (UTCDateTime, UTCDateTime), optional
        Time range (t1, t2) for automatic cleanup of existing data when writing
        
    Attributes
    ----------
    output_path : str
        Path to the ASDF file
    time_range : tuple or None
        Time range for data cleanup operations
        
    Methods
    -------
    write(catalog, metadata) : Write earthquake catalog to file
    read(t1=None, t2=None) : Read earthquake catalog from file with optional time filtering
    
    File Format
    -----------
    **Primary format: ASDF (HDF5-based)**
    - QuakeML events stored as ASDF events
    - Processing metadata stored in HDF5 attributes under 'metadata' group
    
    **Fallback format: XML + JSON**
    - QuakeML events in separate .xml file
    - Processing metadata in separate .json file
    
    **Earthquake catalog contains standard QuakeML/ObsPy Event objects with:**
    - Origins: Event time, location (lat/lon/depth), uncertainties
    - Magnitudes: Magnitude values and types
    - Picks: Phase arrival times and uncertainties  
    - Arrivals: Travel time residuals and weights
    - Station information and phase associations
    
    **Metadata attributes include:**
    - method (str): "NONLINLOC" 
    - nll_home (str): NonLinLoc installation directory
    - target (tuple): Target volcano coordinates (lat, lon, elevation)
    - event_date (str): Processing date
    - nll_directory (str): Output directory path
    - last_updated (str): Timestamp of last write operation
    """
    
    def __init__(self, output_path, time_range=None):
        self.output_path = output_path
        self.time_range = time_range  # (t1, t2) tuple for cleaning existing data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def write(self, catalog, metadata, summary_stats=None):
        """
        Write earthquake catalog to ASDF file with XML fallback.
        
        Automatically cleans existing data within the specified time range before
        writing new data. Optionally appends one row to the 'summary' table (HDF5 only).
        
        Parameters
        ----------
        catalog : obspy.Catalog or vdapseisutils.VCatalog
            Earthquake catalog to write
        metadata : dict
            Processing metadata to store with the catalog
        summary_stats : dict, optional
            If provided and output is HDF5, one row is appended to 'summary'. Keys:
            date, config, loc_method, nlocations, t_exec_loc, t_update_loc (optional).
            
        Notes
        -----
        If time_range was specified during initialization, any existing events
        within that time range will be removed before writing new events.
        """
        print(f"Writing NonLinLoc output to {self.output_path}")

        existing_summary_df = None
        if summary_stats is not None and self.time_range is not None and os.path.exists(self.output_path):
            try:
                existing_summary_df = pd.read_hdf(self.output_path, key='summary')
            except (KeyError, FileNotFoundError):
                pass
        
        # Handle time-based data replacement
        if self.time_range is not None and os.path.exists(self.output_path):
            # Read existing data and filter out the time range
            existing_catalog_filtered = self._get_filtered_existing_catalog(
                self.time_range[0], self.time_range[1]
            )
            
            # Combine existing (filtered) catalog with new catalog
            if existing_catalog_filtered is not None and len(existing_catalog_filtered) > 0:
                catalog = existing_catalog_filtered + catalog
                
        # Ensure catalog has proper comments attribute for QuakeML writing
        if not hasattr(catalog, 'comments') or catalog.comments is None:
            catalog.comments = []
        
        # Skip writing if catalog is empty
        if catalog is None or len(catalog) == 0:
            print("Warning: Empty catalog, skipping write operation")
            return
        
        try:
            import pyasdf
            import tempfile
            import pandas as pd
            
            # Determine if we need to overwrite (when combining catalogs) or append
            file_mode = 'w' if (self.time_range is not None and os.path.exists(self.output_path)) else 'a'
            
            with pyasdf.ASDFDataSet(self.output_path, mode=file_mode) as ds:
                # Convert catalog to QuakeML and add to ASDF
                with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as tmp:
                    temp_xml = tmp.name
                    
                try:
                    catalog.write(temp_xml, format="QUAKEML")
                    ds.add_quakeml(temp_xml)
                finally:
                    os.unlink(temp_xml)
                    
            # Store processing metadata in HDF5 format
            processing_time = pd.Timestamp.now()
            with h5py.File(self.output_path, 'a') as f:
                if 'metadata' not in f:
                    metadata_group = f.create_group('metadata')
                else:
                    metadata_group = f['metadata']
                    
                # Store all metadata attributes
                for key, value in metadata.items():
                    if isinstance(value, (tuple, list)):
                        metadata_group.attrs[key] = value
                    else:
                        metadata_group.attrs[key] = str(value)
                
                metadata_group.attrs['last_updated'] = str(processing_time)
            if summary_stats is not None:
                self._write_locator_summary(summary_stats, existing_summary_df=existing_summary_df)
                
        except ImportError:
            print("WARNING: PyASDF not available, saving QuakeML to separate XML file")
            # Fallback: save as separate XML file
            xml_path = self.output_path.replace('.h5', '.xml')
            catalog.write(xml_path, format="QUAKEML")
            
            # Save metadata as JSON
            json_path = self.output_path.replace('.h5', '_metadata.json')
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

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

    def _get_filtered_existing_catalog(self, t1, t2):
        """
        Get existing catalog filtered to exclude the specified time range.
        
        This private method returns existing events outside the specified time range,
        allowing new events to replace only the events within the time range.
        
        Parameters
        ----------
        t1 : UTCDateTime or str
            Start time for event removal
        t2 : UTCDateTime or str
            End time for event removal
            
        Returns
        -------
        VCatalog or None
            Filtered earthquake catalog excluding the time range, or None if no existing data
            
        Notes
        -----
        Handles both ASDF and XML fallback formats. Uses manual time filtering
        to avoid VCatalog filter issues with None values.
        """
        print(f"Filtering existing NLL catalog to exclude {t1} to {t2}")
        
        # Convert time parameters if needed
        from obspy import UTCDateTime
        if isinstance(t1, str):
            t1 = UTCDateTime(t1)
        if isinstance(t2, str):
            t2 = UTCDateTime(t2)
        
        try:
            import pyasdf
            
            # Read existing catalog from ASDF
            with pyasdf.ASDFDataSet(self.output_path, mode='r') as ds:
                existing_catalog = ds.events
                
        except (ImportError, Exception):
            # Fallback to XML file
            xml_path = self.output_path.replace('.h5', '.xml')
            
            if os.path.exists(xml_path):
                from obspy import read_events
                existing_catalog = read_events(xml_path)
            else:
                print("No existing catalog found")
                return None
        
        if existing_catalog is None or len(existing_catalog) == 0:
            print("No existing events found")
            return None
            
        # Convert to VCatalog for easier handling
        from vdapseisutils import VCatalog
        vcatalog = VCatalog(existing_catalog)
        
        # Manually filter events to keep only those OUTSIDE the time range
        filtered_events = []
        removed_events = []
        for event in vcatalog:
            if event.origins:
                event_time = event.origins[0].time
                # Keep events that are before t1 OR after t2
                if event_time < t1 or event_time > t2:
                    filtered_events.append(event)
                else:
                    removed_events.append(event)
        
        # Report counts
        events_removed = len(removed_events)
        events_kept = len(filtered_events)
        print(f"Removing {events_removed} events within time range. "
              f"Keeping {events_kept} existing events outside time range.")
        
        if len(filtered_events) > 0:
            from obspy import Catalog
            filtered_obspy_catalog = Catalog(filtered_events)
            # Ensure comments attribute is properly initialized
            if not hasattr(filtered_obspy_catalog, 'comments') or filtered_obspy_catalog.comments is None:
                filtered_obspy_catalog.comments = []
            filtered_catalog = VCatalog(filtered_obspy_catalog)
            return filtered_catalog
        else:
            return None

    def read(self, t1=None, t2=None):
        """
        Read earthquake catalog from ASDF file or XML fallback with optional time filtering.
        
        Parameters
        ----------
        t1 : UTCDateTime or str, optional
            Start time for filtering events
        t2 : UTCDateTime or str, optional
            End time for filtering events
            
        Returns
        -------
        tuple of (VCatalog, dict) or (None, None)
            - VCatalog: Earthquake catalog (None if file not found)
            - dict: Processing metadata (None if not available)
            
        Notes
        -----
        Attempts to read from ASDF format first, falls back to XML+JSON format.
        Converts results to VCatalog for compatibility with other tools.
        Time filtering is applied to event origin times.
        """
        print(f"Reading NonLinLoc output from {self.output_path}")
        
        try:
            import pyasdf
            from obspy import read_events
            
            # Try to read from ASDF file
            with pyasdf.ASDFDataSet(self.output_path, mode='r') as ds:
                catalog = ds.events
                
            # Read metadata
            try:
                with h5py.File(self.output_path, 'r') as f:
                    if 'metadata' in f:
                        metadata_group = f['metadata']
                        metadata = dict(metadata_group.attrs)
                        # Convert bytes to strings if needed
                        for key, value in metadata.items():
                            if isinstance(value, bytes):
                                metadata[key] = value.decode('utf-8')
                    else:
                        metadata = None
            except:
                print("WARNING: Could not load metadata from HDF5 file.")
                metadata = None
                
        except (ImportError, Exception):
            # Fallback to XML file
            xml_path = self.output_path.replace('.h5', '.xml')
            json_path = self.output_path.replace('.h5', '_metadata.json')
            
            if os.path.exists(xml_path):
                from obspy import read_events
                catalog = read_events(xml_path)
                
                # Try to read metadata
                metadata = None
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as f:
                            metadata = json.load(f)
                    except:
                        print("WARNING: Could not load metadata from JSON file.")
                        
            else:
                print(f"ERROR: Neither ASDF file {self.output_path} nor XML file {xml_path} found.")
                return None, None
        
        # Convert ObsPy Catalog to VCatalog
        if catalog is not None:
            from vdapseisutils import VCatalog
            vcatalog = VCatalog(catalog)
            
            # Apply time filtering if requested
            if t1 is not None or t2 is not None:
                if isinstance(t1, str):
                    t1 = UTCDateTime(t1) if t1 is not None else None
                if isinstance(t2, str):
                    t2 = UTCDateTime(t2) if t2 is not None else None
                
                vcatalog = vcatalog.filter(time=[t1, t2])
                print(f"Time filtered catalog from {t1} to {t2}")
            
            print(f"Events: {len(vcatalog):5d}\n")
            
            return vcatalog, metadata
        else:
            return None, None
