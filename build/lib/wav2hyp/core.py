"""
Core processing functions for WAV2HYP pipeline.

This module contains the main processing class and functions for the complete
waveform to hypocenter processing workflow.
"""

import os
import time
import logging
from datetime import timedelta, datetime
from pathlib import Path

import pandas as pd
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, read_inventory

import torch
import seisbench.models as sbm
import pyocto

from vdapseisutils import VCatalog, VInventory

from .config_loader import load_config, validate_config, get_global_variables, print_config_summary
from .utils.io import PickListX, DetectionListX, EQTOutput, PyOctoOutput, NLLOutput
from .utils.geo import GeoArea


class WAV2HYP:
    """
    WAV2HYP processing pipeline class.
    
    Provides a complete workflow for processing seismic waveforms into earthquake
    hypocenters through automated phase picking, event association, and location.
    """
    
    def __init__(self, config_path=None, config_dict=None):
        """
        Initialize WAV2HYP processor.
        
        Parameters
        ----------
        config_path : str, optional
            Path to YAML configuration file.
        config_dict : dict, optional
            Configuration dictionary (alternative to config_path).
        """
        if config_path:
            self.config = load_config(config_path)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Must provide either config_path or config_dict")
            
        # Validate and setup configuration
        self.config = validate_config(self.config)
        
        # Extract configuration variables
        config_vars = get_global_variables(self.config)
        for key, value in config_vars.items():
            setattr(self, key, value)
            
        # Setup logging
        self._setup_logging()
        
        # Initialize FDSN client
        self.client = Client(self.config['waveform_client']['provider'])
        
        self.logger.info("WAV2HYP processor initialized")
        print_config_summary(self.config)
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_dir = os.path.join(self.BASE_OUTPUT_DIR, self.config['output']['log_dir'])
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('wav2hyp')
        self.logger.setLevel(getattr(logging, log_config.get('level', 'INFO')))
        
        # Create handlers
        log_file = os.path.join(log_dir, 'wav2hyp.log')
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        
        # Create formatter
        formatter = logging.Formatter(log_config.get('format', 
            '%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    # TODO These (the full file path) should be defined in the config file
    def _get_output_path(self, step, date=None):
        """Generate output paths for different processing steps."""
        if step == "picker":
            return f"{self.PICKER_OUTPUT_DIR}/eqt-volpick.h5"
        elif step == "associator":
            return f"{self.ASSOCIATOR_OUTPUT_DIR}/pyocto.h5"
        elif step == "locator":
            return f"{self.LOCATOR_OUTPUT_DIR}/nll.h5"
        elif step == "nll_dir":
            if date is None:
                raise ValueError("Date is required for nll_dir step")
            return f"{self.LOCATOR_OUTPUT_DIR}/{self.config['locator']['config_name']}/loc-{date.strftime('%Y-%m-%d')}"
        else:
            raise ValueError(f"Unknown step: {step}")

    def run(self, start_time, end_time, tproc='1d', skip_picker=False, skip_associator=False, skip_locator=False):
        """
        Run the complete WAV2HYP processing pipeline.
        
        Parameters
        ----------
        start_time : obspy.UTCDateTime or str
            Start time for processing.
        end_time : obspy.UTCDateTime or str  
            End time for processing.
        tproc : str, default '1d'
            Time processing chunk size (e.g., '1d', '12h', '30m').
        skip_picker : bool, default False
            Skip phase picking step.
        skip_associator : bool, default False
            Skip event association step.
        skip_locator : bool, default False
            Skip earthquake location step.
            
        Returns
        -------
        vdapseisutils.VCatalog
            Final earthquake catalog with located events.
        """
        # Start timing
        run_start_time = time.perf_counter()
        
        # Convert to UTCDateTime if needed
        if isinstance(start_time, str):
            start_time = UTCDateTime(start_time)
        if isinstance(end_time, str):
            end_time = UTCDateTime(end_time)
            
        # Store original times for logging
        original_start = start_time
        original_end = end_time
        
        self.logger.info(f"Starting WAV2HYP processing from {original_start} to {original_end}")
        
        # Get inventory
        self.logger.info(f"Loading station inventory from: {self.inventory_file}")
        inv = read_inventory(self.inventory_file)
        
        # Parse processing time chunk
        chunk_seconds = parse_time_string(tproc)
        chunk_delta = timedelta(seconds=chunk_seconds)
        
        # Round down start_time and round up end_time for chunking
        chunk_start = UTCDateTime(start_time.date)  # Round down to start of day
        chunk_end = UTCDateTime(end_time.date) + timedelta(days=1)  # Round up to start of next day
        
        # Generate time chunks
        time_chunks = []
        current_start = chunk_start
        while current_start < chunk_end:
            current_end = min(current_start + chunk_delta, chunk_end)
            
            # Only process chunks that overlap with the original time range
            if current_end > original_start and current_start < original_end:
                # Clamp to original time bounds
                chunk_start_time = max(current_start, original_start)
                chunk_end_time = min(current_end, original_end)
                time_chunks.append((chunk_start_time, chunk_end_time))
            
            current_start = current_end
        
        self.logger.info(f"Processing {len(time_chunks)} time chunks of {tproc} each")
        
        # Process each time chunk
        final_catalog = VCatalog()
        for i, (chunk_start_time, chunk_end_time) in enumerate(time_chunks, 1):
            self.logger.info(f"Processing chunk {i}/{len(time_chunks)}: {chunk_start_time} to {chunk_end_time}")
            
            chunk_catalog = self._process_timespan(chunk_start_time, chunk_end_time, inv, skip_picker, skip_associator, skip_locator)
            
            # Combine catalogs
            if chunk_catalog is not None and len(chunk_catalog) > 0:
                final_catalog += chunk_catalog
        
        # Calculate and log total execution time
        run_execution_time = time.perf_counter() - run_start_time
        self.logger.info(f"WAV2HYP processing completed! Events processed: {len(final_catalog)}")
        self.logger.info(f"Total execution time: {run_execution_time:.2f} seconds ({run_execution_time/60:.2f} minutes)")
        return final_catalog
    
    def _process_timespan(self, start_time, end_time, inventory, skip_picker, skip_associator, skip_locator):
        """Process a time span through the complete pipeline."""
        
        # Start timing
        timespan_start_time = time.perf_counter()
        
        # self.logger.info(f"Processing timespan: {start_time} to {end_time}")
        
        # Get output paths
        picker_output = self._get_output_path("picker")
        associator_output = self._get_output_path("associator") 
        locator_output = self._get_output_path("locator")

        # Step 1: Phase Picking
        if skip_picker:
            self.logger.info("Skipping picker step as requested...")
            eqt_output = EQTOutput(picker_output)
            picks, detections, metadata = eqt_output.read(t1=start_time, t2=end_time)
        else:
            picks = self._run_picker(start_time, end_time, picker_output)

        # Step 2: Event Association
        if skip_associator:
            self.logger.info("Skipping associator step as requested...")
            pyocto_output = PyOctoOutput(associator_output)
            catalog_assoc, _, _, _ = pyocto_output.read(t1=start_time, t2=end_time)
        else:
            catalog_assoc = self._run_associator(picks, inventory, associator_output, start_time, end_time)

        # Step 3: Event Location  
        if skip_locator:
            self.logger.info("Skipping locator step as requested...")
            nll_output = NLLOutput(locator_output)
            catalog, _ = nll_output.read(t1=start_time, t2=end_time)
            if catalog is None:
                self.logger.warning("No existing locator results found, using associator catalog")
                catalog = catalog_assoc
        else:
            catalog = self._run_locator(catalog_assoc, inventory, locator_output, start_time, end_time)

        # Calculate and log timespan execution time
        timespan_execution_time = time.perf_counter() - timespan_start_time
        self.logger.info(f"Timespan completed. Execution time: {timespan_execution_time:.2f} seconds ({timespan_execution_time/60:.2f} minutes)")
        return catalog

    def _run_picker(self, start_time, end_time, output_path):
        """Run phase picking with EQTransformer."""
        method_start = time.perf_counter()
        
        self.logger.info(f"Downloading waveforms from {start_time} to {end_time}...")
        
        # Time waveform gathering
        waveform_start = time.perf_counter()
        inv = VInventory(read_inventory(self.inventory_file))
        stream = inv.get_waveforms(self.client, start_time, end_time)
        waveform_time = time.perf_counter() - waveform_start
        self.logger.info(f"Waveforms gathered: {waveform_time:.2f}s")
        
        self.logger.info(f"Running phase picker from {start_time} to {end_time}...")
        
        # Create the EQT Picker object
        picker_model = self.config['picker']['model']
        picker = sbm.EQTransformer.from_pretrained(picker_model)
        
        if torch.cuda.is_available() and self.config['picker'].get('use_cuda', True):
            picker.cuda()
            self.logger.info("Using CUDA acceleration for picking")
        
        # Annotate the Stream
        t = time.perf_counter()
        st_annotated = picker.annotate(stream)
        self.logger.info(f"EQTransformer execution time: {time.perf_counter() - t:.2f} seconds")
        
        # Extract picks from the probability traces
        from seisbench.models.base import WaveformModel
        from seisbench.util.annotations import PickList
        
        # Extract P picks
        p_picks = WaveformModel.picks_from_annotations(
            st_annotated.select(channel="*P*"),
            threshold=self.p_threshold,
            phase="P"
        )
        
        # Extract S picks
        s_picks = WaveformModel.picks_from_annotations(
            st_annotated.select(channel="*S*"),
            threshold=self.s_threshold,
            phase="S"
        )
        
        # Extract detections
        detections = WaveformModel.detections_from_annotations(
            st_annotated.select(channel="*Detection*"),
            threshold=self.d_threshold
        )
        
        # Combine picks
        picks = PickList(p_picks + s_picks)
        picks = PickListX(picks)
        detections = DetectionListX(detections)
        
        self.logger.info(f"Detections: {len(detections):5d} | P picks: {len(p_picks):5d} | S picks: {len(s_picks):5d}")
        
        # Compose metadata
        metadata = {
            'method': 'EQTRANSFORMER',
            'sta_fmt': 'NET.STA.',
            'model': picker_model.upper(),
            'P_threshold': self.p_threshold,
            'S_threshold': self.s_threshold,
            'D_threshold': self.d_threshold
        }

        # TODO: Add this back in - but for now, I haven't encountered a problem, so don't create one
        # # Write to file
        # if len(picks) == 0 and len(detections) == 0:
        #     self.logger.info("No picks or detections to write - skipping file output")
        #     return PickListX()
        
        eqt_output = EQTOutput(output_path, time_range=(start_time, end_time))
        eqt_output.write(picks, detections, metadata)
        
        # Log method timing
        method_time = time.perf_counter() - method_start
        self.logger.info(f"Picker completed: {method_time:.2f}s ({method_time/60:2.1f} minutes)")
        
        return picks
    
    def _run_associator(self, picks, inventory, output_path, start_time=None, end_time=None):
        """Run event association with PyOcto."""
        method_start = time.perf_counter()

        start_time_msg = "---" if start_time is None else f"{start_time}"
        end_time_msg = "---" if end_time is None else f"{end_time}"
        self.logger.info(f"Running event association from {start_time_msg} to {end_time_msg}...")
        
        # Check if there are picks to associate
        if picks is None or len(picks) == 0:
            self.logger.info("No picks available for association - skipping association processing")
            return VCatalog()
        
        assoc_config = self.config['associator']
        
        # Define 0D Velocity Model
        velocity_model = pyocto.VelocityModel0D(
            p_velocity=assoc_config['p_velocity'],
            s_velocity=assoc_config['s_velocity'],
            tolerance=assoc_config['tolerance'],
            association_cutoff_distance=assoc_config['association_cutoff'],
        )
        
        # Define PyOcto Associator
        roi = GeoArea(self.lat, self.lon, assoc_config['radius_km'])
        
        associator = pyocto.OctoAssociator.from_area(
            lat=roi.lat_range,
            lon=roi.lon_range,
            zlim=tuple(assoc_config['depth_limits']),
            time_before=assoc_config['time_before'],
            velocity_model=velocity_model,
            n_p_picks=assoc_config['min_p_picks'],
            n_s_picks=assoc_config['min_s_picks'],
            n_p_and_s_picks=assoc_config['min_p_and_s_picks'],
        )
        
        # Convert stations to PyOcto DataFrame
        stations = associator.inventory_to_df(inventory)
        
        # Run the phase association
        t = time.perf_counter()
        events, assignments = associator.associate_seisbench(picks, stations)
        self.logger.info(f"PyOcto execution time: {time.perf_counter() - t:.2f} seconds")
        associator.transform_events(events)
        
        self.logger.info(f"Associations: {len(events)} events, {len(assignments)} assignments")
        
        metadata = {
            'method': 'PYOCTO',
            'sta_fmt': 'NET.STA.',
            'target': (self.lat, self.lon, self.elev),
            'zlim': tuple(assoc_config['depth_limits']),
            'time_before': assoc_config['time_before'],
            'p_velocity': assoc_config['p_velocity'],
            's_velocity': assoc_config['s_velocity'],
            'n_p_picks': assoc_config['min_p_picks'],
            'n_s_picks': assoc_config['min_s_picks'],
            'n_p_and_s_picks': assoc_config['min_p_and_s_picks'],
        }
        
        # Convert PyOcto objects to DataFrames if needed
        if events is not None and hasattr(events, 'to_dataframe'):
            events_df = events.to_dataframe()
        elif events is not None:
            events_df = events
        else:
            events_df = None
            
        if assignments is not None and hasattr(assignments, 'to_dataframe'):
            assignments_df = assignments.to_dataframe()
        elif assignments is not None:
            assignments_df = assignments
        else:
            assignments_df = None
        
        # Check if both events and assignments are empty
        events_empty = events_df is None or len(events_df) == 0
        assignments_empty = assignments_df is None or len(assignments_df) == 0
        
        if events_empty and assignments_empty:
            self.logger.info("No events or assignments to write - skipping file output")
            return VCatalog()
        
        # If we need to write data, ensure proper structure for empty DataFrames
        if events_empty:
            events_df = pd.DataFrame(columns=['time', 'x', 'y', 'z', 'latitude', 'longitude', 'depth', 'residual_rms', 'n_picks', 'n_p_picks', 'n_s_picks'])
        
        if assignments_empty:
            assignments_df = pd.DataFrame({
                'event_idx': pd.Series(dtype='int64'),
                'station_id': pd.Series(dtype='string'),
                'phase': pd.Series(dtype='string'),
                'pick_time': pd.Series(dtype='float64'),
                'residual': pd.Series(dtype='float64'),
                'weight': pd.Series(dtype='float64')
            })
        
        # Write PyOcto results to file
        self.logger.info(f"Writing associations to {output_path}")
        time_range = (start_time, end_time) if start_time and end_time else None
        pyocto_output = PyOctoOutput(output_path, time_range=time_range)
        pyocto_output.write(events_df, assignments_df, metadata)
        
        # Create ObsPy Catalog object
        catalog = VCatalog.from_pyocto(events_df, assignments_df)
        
        # Log method timing
        method_time = time.perf_counter() - method_start
        events_count = len(events_df) if events_df is not None else 0
        self.logger.info(f"Associator completed: {method_time:.2f}s ({method_time/60:2.1f} minutes)")
        
        return catalog
    
    def _run_locator(self, catalog_in, inventory, output_path, start_time=None, end_time=None):
        """Run earthquake location with NonLinLoc."""
        method_start = time.perf_counter()
        
        import nllpy
        
        self.logger.info(f"Running earthquake location from  {start_time} to {end_time}...")
        
        # Check if catalog has events
        if len(catalog_in) == 0:
            self.logger.info("No associations in catalog - skipping location processing")
            return VCatalog()
        
        locator_config = self.config['locator']
        nll_home = locator_config['nll_home']
        name = locator_config['config_name']
        
        # Create directories
        event_date = catalog_in[0].origins[0].time.strftime("%Y-%m-%d")
        obs = f"./obs-{event_date}"
        
        nll_dir = self._get_output_path('nll_dir', catalog_in[0].origins[0].time.date)
        loc = os.path.basename(nll_dir)
        
        os.makedirs(os.path.join(nll_home, obs), exist_ok=True)
        os.makedirs(nll_dir, exist_ok=True)
        
        # Convert catalog to NonLinLoc obs files
        catalog_in.write_nlloc_obs(os.path.join(nll_home, obs), format="NLLOC_OBS")
        self.logger.info(f"Converted assocations to NLL_OBS files: ({os.path.join(nll_home, obs)})")

        # Create volcano monitoring configuration
        config = nllpy.create_volcano_config(lat_orig=self.lat, lon_orig=self.lon)
        config.nll_home = nll_home
        config.filename = f"{name}.in"
        config.input_obs = (f"{obs}", "NLLOC_OBS")
        config.output_obs = f"{loc}/{name}"
        config.add_station_from_inventory(inventory, sta_fmt=locator_config['station_format'])
        config.write_complete_control_file(os.path.join(nll_home, config.filename))
        self.logger.info(f"Writing NLL control file: ({os.path.join(nll_home, config.filename)})")


        # Run NonLinLoc
        nll_run_start = time.perf_counter()
        config.run_nlloc(
            vel2grid=locator_config.get('run_vel2grid', True),
            grid2time=locator_config.get('run_grid2time', True)
        )
        nll_run_time = time.perf_counter() - nll_run_start
        self.logger.info(f"NonLinLoc execution time: {nll_run_time:.2f}s ({nll_run_time/60:2.1f} minutes)")


        # Read NonLinLoc output
        # TODO Use this to create a method read_nll_loc() in nllpy (recommended) or vdapseisutils
        from obspy.io.nlloc.core import read_nlloc_hyp
        import glob
        self.logger.info(f"Reading NonLinLoc hyp files: ({os.path.join(nll_home, config.output_obs + f"*.hyp")})")
        full_search_path = os.path.join(nll_home, config.output_obs + f"*.hyp")  # config.output_obs includes the prefix for the .hyp files
        hyp_files = glob.glob(full_search_path)  # all .hyp files in the loc directory
        filtered_files = [f for f in hyp_files if 'sum' not in f]  # remove summary files
        # Read all filtered files
        catalog_out = VCatalog()
        catalog_out.comments = []  # I'm not sure why this isn't set (maybe fix in VCatalog)
        for hyp_file in filtered_files:
            try:
                catalog_out += read_nlloc_hyp(hyp_file)
                print(f"- Read: {hyp_file}")
            except Exception as e:
                print(f"- Error reading {hyp_file}: {e}")

        # Create metadata for NLLOutput
        metadata = {
            'method': 'NONLINLOC',
            'nll_home': nll_home,
            'target': (self.lat, self.lon, self.elev),
            'event_date': event_date,
            'nll_directory': nll_dir
        }
        
        if len(catalog_out) == 0:
            self.logger.info("No locations to write - skipping file output")
            return VCatalog()
        
        # Store results using NLLOutput
        self.logger.info(f"Writing locations to {output_path}")
        time_range = (start_time, end_time) if start_time and end_time else None
        nll_output = NLLOutput(output_path, time_range=time_range)
        nll_output.write(catalog_out, metadata)
        
        # Log method timing
        method_time = time.perf_counter() - method_start
        self.logger.info(f"Locator completed: {method_time:.2f}s ({method_time/60:2.1f} minutes)")
        
        return catalog_out
    

def main(start_time, end_time, config_path="config.yaml", tproc='1d', skip_picker=False, skip_associator=False, skip_locator=False):
    """
    Main processing function for backward compatibility.
    
    Parameters
    ----------
    start_time : obspy.UTCDateTime or str
        Start time for processing.
    end_time : obspy.UTCDateTime or str
        End time for processing.
    config_path : str, default "config.yaml"
        Path to configuration file.
    tproc : str, default '1d'
        Time processing chunk size (e.g., '1d', '12h', '30m').
    skip_picker : bool, default False
        Skip phase picking step.
    skip_associator : bool, default False
        Skip event association step.
    skip_locator : bool, default False
        Skip earthquake location step.
        
    Returns
    -------
    vdapseisutils.VCatalog
        Final earthquake catalog.
    """
    processor = WAV2HYP(config_path=config_path)
    return processor.run(start_time, end_time, tproc, skip_picker, skip_associator, skip_locator)


def parse_time_string(time_str):
    """
    Convert time string with units to seconds.

    Parameters
    ----------
    time_str : str, int, or float
        Time string with optional suffix ('1d', '12h', '30m') or numeric value.

    Returns
    -------
    float
        Time duration in seconds.
    """
    if isinstance(time_str, (int, float)):
        return float(time_str)

    time_str = str(time_str).lower()
    if time_str.endswith('d'):
        return float(time_str[:-1]) * 24 * 60 * 60
    elif time_str.endswith('h'):
        return float(time_str[:-1]) * 60 * 60
    elif time_str.endswith('m'):
        return float(time_str[:-1]) * 60
    elif time_str.endswith('s'):
        return float(time_str[:-1])
    else:
        return float(time_str)


# def parse_time_string_dep(time_str):
#     """
#     Convert time string with units to minutes.
#
#     Parameters
#     ----------
#     time_str : str, int, or float
#         Time string with optional suffix ('1d', '12h', '30m') or numeric value.
#
#     Returns
#     -------
#     float
#         Time duration in minutes.
#     """
#     if isinstance(time_str, (int, float)):
#         return float(time_str)
#
#     time_str = str(time_str).lower()
#     if time_str.endswith('d'):
#         return float(time_str[:-1]) * 24 * 60
#     elif time_str.endswith('h'):
#         return float(time_str[:-1]) * 60
#     elif time_str.endswith('m'):
#         return float(time_str[:-1])
#     else:
#         return float(time_str)
