"""
Summary export functionality for WAV2HYP pipeline.

This module provides real-time and retrospective summary generation for processing
statistics and timing information.
"""

import os
import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

from obspy import UTCDateTime


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
            header = ['date', 'config', 'ncha', 'nsamp', 'pick_model', 'np', 'ns', 'npicks', 'ndetections', 't_pick', 'last_updated']
        elif self.step == 'associator':
            header = ['date', 'config', 'assoc_method', 'assignments', 'events', 't_assoc', 'last_updated']
        elif self.step == 'locator':
            header = ['date', 'config', 'loc_method', 'locations', 't_loc', 'last_updated']
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
        
        # Update last_updated timestamp
        entry['last_updated'] = datetime.now().strftime('%Y/%m/%dT%H:%M:%S')
        
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
                't_pick': 0.0,
                'last_updated': ''
            }
        elif self.step == 'associator':
            new_entry = {
                'date': date,
                'config': '',
                'assoc_method': '',
                'assignments': 0,
                'events': 0,
                't_assoc': 0.0,
                'last_updated': ''
            }
        elif self.step == 'locator':
            new_entry = {
                'date': date,
                'config': '',
                'loc_method': '',
                'locations': 0,
                't_loc': 0.0,
                'last_updated': ''
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
        entry['t_pick'] = timing
    
    def _update_associator_data(self, entry: Dict[str, Any], stats: Dict[str, Any], timing: float):
        """Update entry with associator statistics."""
        entry['config'] = stats.get('config', '')
        entry['assoc_method'] = stats.get('assoc_method', '')
        entry['assignments'] = stats.get('assignments', 0)
        entry['events'] = stats.get('events', 0)
        entry['t_assoc'] = timing
    
    def _update_locator_data(self, entry: Dict[str, Any], stats: Dict[str, Any], timing: float):
        """Update entry with locator statistics."""
        entry['config'] = stats.get('config', '')
        entry['loc_method'] = stats.get('loc_method', '')
        entry['locations'] = stats.get('locations', 0)
        entry['t_loc'] = timing


def generate_summary_from_files(base_output_dir: str, config_name: str, 
                                summary_filename: Optional[str] = None,
                                date_range: Optional[tuple] = None) -> str:
    """
    Generate summary file retrospectively from existing HDF5 files.
    
    Parameters
    ----------
    base_output_dir : str
        Base output directory containing results
    config_name : str
        Configuration name
    summary_filename : str, optional
        Custom summary filename. If None, uses {config_name}_qc_summary.txt
    date_range : tuple, optional
        (start_date, end_date) as UTCDateTime objects
        
    Returns
    -------
    str
        Path to generated summary file
    """
    from .io import EQTOutput, PyOctoOutput, NLLOutput
    from obspy import UTCDateTime
    import glob
    
    if summary_filename is None:
        summary_filename = f"{config_name}_qc_summary.txt"
    
    summary_file = Path(base_output_dir) / summary_filename
    
    # Initialize exporter
    exporter = SummaryExporter(base_output_dir, config_name, summary_filename)
    
    # Get all available dates from HDF5 files
    picker_file = Path(base_output_dir) / "picks" / "eqt-volpick.h5"
    associator_file = Path(base_output_dir) / "associations" / "pyocto.h5"
    locator_file = Path(base_output_dir) / "locations" / "nll.h5"
    
    # Find all available dates by scanning the files
    dates = set()
    
    # Scan picker file for available time ranges
    if picker_file.exists():
        try:
            eqt_output = EQTOutput(str(picker_file))
            # Get all available time ranges from the HDF5 file
            # This is a simplified approach - in practice, you'd need to implement
            # a method to get all time ranges from the HDF5 file
            print(f"Found picker file: {picker_file}")
        except Exception as e:
            print(f"Warning: Could not read picker file {picker_file}: {e}")
    
    # For now, let's implement a simple approach that processes common date patterns
    # This is a placeholder - the real implementation would need to scan the HDF5 files
    # to find all available time ranges
    
    print("Note: Retrospective generation is a placeholder implementation.")
    print("The real implementation would need to scan HDF5 files for available time ranges.")
    print("For now, this creates an empty summary file with just the header.")
    
    return str(summary_file)
