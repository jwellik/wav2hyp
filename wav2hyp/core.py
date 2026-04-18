"""
Core processing functions for WAV2HYP pipeline.

This module contains the main processing class and functions for the complete
waveform to hypocenter processing workflow.
"""

import os
import re
import time
import logging
from datetime import timedelta, datetime
from pathlib import Path

import pandas as pd
from obspy import UTCDateTime, read_inventory, Stream

import torch
import seisbench.models as sbm
import pyocto

from vdapseisutils import VCatalog, VInventory
from vdapseisutils.utils.obspyutils.client import VClient

from .config_loader import load_config, validate_config, get_global_variables, print_config_summary
from .utils.io import PickListX, DetectionListX, EQTOutput, PyOctoOutput, NLLOutput
from .utils.geo import GeoArea
from .utils.summary import (
    STAGES_ORDER,
    SummaryExporter,
    date_already_processed_for_stage,
    drop_summary_txt_rows_in_range,
    station_summary_reset_for_overwrite,
)


def _safe_filename_component(s: str) -> str:
    """Sanitize config/model strings for EQ annotation archive filenames."""
    t = re.sub(r"[^\w.\-]+", "_", str(s)).strip("._-")
    return t if t else "x"


def _apply_nllpy_overrides(config, overrides, logger=None):
    """Apply nllpy_overrides from wav2hyp config to an nllpy NLLocConfig instance.
    Nested dicts apply to config sub-objects (e.g. locgrid: { d_grid_x: 0.2 }).
    """
    for key, value in overrides.items():
        if not hasattr(config, key):
            if logger:
                logger.warning(f"nllpy_overrides: unknown config key '{key}', skipping")
            continue
        sub = getattr(config, key)
        if isinstance(value, dict) and not isinstance(sub, (list, dict)):
            for k2, v2 in value.items():
                if hasattr(sub, k2):
                    if k2 == 'layers' and isinstance(v2, list):
                        v2 = [tuple(row) for row in v2]
                    setattr(sub, k2, v2)
                elif logger:
                    logger.warning(f"nllpy_overrides: unknown {key}.{k2}, skipping")
        else:
            setattr(config, key, value)


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
        
        # Initialize waveform client using VClient
        self.client = self._initialize_waveform_client()
        
        # Initialize summary exporters for each step if filenames are specified
        config_name = self.config['locator']['config_name']
        
        self.picker_summary_exporter = None
        self.associator_summary_exporter = None
        self.locator_summary_exporter = None
        
        if self.picker_summary is not None:
            self.picker_summary_exporter = SummaryExporter(
                self.BASE_OUTPUT_DIR, 
                config_name, 
                'picker',
                self.picker_summary
            )
            self.logger.info(f"Picker summary exporter initialized: {self.picker_summary_exporter.summary_file}")
        
        if self.associator_summary is not None:
            self.associator_summary_exporter = SummaryExporter(
                self.BASE_OUTPUT_DIR, 
                config_name, 
                'associator',
                self.associator_summary
            )
            self.logger.info(f"Associator summary exporter initialized: {self.associator_summary_exporter.summary_file}")
        
        if self.locator_summary is not None:
            self.locator_summary_exporter = SummaryExporter(
                self.BASE_OUTPUT_DIR, 
                config_name, 
                'locator',
                self.locator_summary
            )
            self.logger.info(f"Locator summary exporter initialized: {self.locator_summary_exporter.summary_file}")
        
        self.logger.info("WAV2HYP processor initialized")
        print_config_summary(self.config, logger=self.logger)
        
    def _initialize_waveform_client(self):
        """Initialize waveform client using VClient with datasource parameter."""
        waveform_config = self.config['waveform_client']
        
        if 'datasource' not in waveform_config:
            raise ValueError("waveform_client must specify 'datasource' parameter")
        
        datasource = waveform_config['datasource']
        
        # Extract client_type if specified
        client_type = waveform_config.get('client_type', None)
        
        # Extract additional keyword arguments (excluding datasource and client_type)
        kwargs = {k: v for k, v in waveform_config.items() 
                 if k not in ['datasource', 'client_type']}
        
        self.logger.info(f"Initializing waveform client with datasource: {datasource}")
        if client_type:
            self.logger.info(f"Using explicit client_type: {client_type}")
        if kwargs:
            self.logger.info(f"Additional parameters: {kwargs}")
        
        # VClient will use client_type if specified, otherwise auto-detect
        return VClient(datasource, client_type=client_type, **kwargs)
        
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

    @staticmethod
    def _count_pick_phases(picks):
        """Count P and S picks in an iterable of pick-like objects."""
        counts = {"P": 0, "S": 0, "other": 0}
        if picks is None:
            counts["total"] = 0
            return counts

        for pick in picks:
            phase = getattr(pick, "phase", None) or getattr(pick, "phase_hint", None)
            phase = "" if phase is None else str(phase).strip().upper()
            if phase.startswith("P"):
                counts["P"] += 1
            elif phase.startswith("S"):
                counts["S"] += 1
            else:
                counts["other"] += 1

        counts["total"] = counts["P"] + counts["S"] + counts["other"]
        return counts

    @staticmethod
    def _count_catalog_arrivals(catalog):
        """Count arrivals in a catalog, falling back to event picks when needed."""
        if catalog is None:
            return 0

        total = 0
        for event in catalog:
            origin = event.preferred_origin()
            if origin is None:
                origins = getattr(event, "origins", None) or []
                origin = origins[0] if origins else None

            arrivals = getattr(origin, "arrivals", None) or []
            if arrivals:
                total += len(arrivals)
                continue

            picks = getattr(event, "picks", None) or []
            total += len(picks)

        return total

    @staticmethod
    def _describe_locator_velocity_model(locator_config):
        """Return a concise description of the active locator velocity model."""
        if locator_config.get("velocity_model_layers"):
            return f"Locator velocity model: {len(locator_config['velocity_model_layers'])} configured layers"

        override_layers = locator_config.get("nllpy_overrides", {}).get("layer", {}).get("layers")
        if override_layers:
            return f"Locator velocity model: {len(override_layers)} layers via locator.nllpy_overrides.layer.layers"

        return "Locator velocity model: nllpy volcano default"

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

    @staticmethod
    def _cleanup_stages_for_overwrite(run_picker, run_associator, run_locator):
        """Stages whose HDF5 outputs must be cleared for a cascading overwrite (see docs/workflow.md)."""
        if run_picker:
            return frozenset({"picker", "associator", "locator"})
        if run_associator:
            return frozenset({"associator", "locator"})
        if run_locator:
            return frozenset({"locator"})
        return frozenset()

    def _cascade_overwrite_cleanup(self, t1, t2, run_picker, run_associator, run_locator):
        """
        Remove existing outputs in [t1, t2] for all stages that will be re-run and downstream stages.

        Called once per timespan from _process_timespan when overwrite is True.
        """
        cleanup_stages = self._cleanup_stages_for_overwrite(run_picker, run_associator, run_locator)
        if not cleanup_stages:
            self.logger.info(
                "Overwrite is set but no pipeline stages are selected for this run; skipping cascade cleanup."
            )
            return

        self.logger.info(
            "Cascading overwrite: removing existing outputs for range [%s, %s]; cleanup_stages=%s",
            t1,
            t2,
            ", ".join(sorted(cleanup_stages)),
        )

        picker_path = self._get_output_path("picker")
        associator_path = self._get_output_path("associator")
        locator_path = self._get_output_path("locator")

        for step in STAGES_ORDER:
            if step not in cleanup_stages:
                continue
            if step == "picker":
                counts = EQTOutput(picker_path).remove_range(t1, t2)
                self.logger.info(
                    "Cascade picker removal summary: picks_removed=%d detections_removed=%d "
                    "summary_rows_removed=%d histogram_rows_removed=%d",
                    counts.get("picks_removed", 0),
                    counts.get("detections_removed", 0),
                    counts.get("summary_rows_removed", 0),
                    counts.get("histogram_rows_removed", 0),
                )
            elif step == "associator":
                counts = PyOctoOutput(associator_path).remove_range(t1, t2)
                self.logger.info(
                    "Cascade associator removal summary: events_removed=%d assignments_removed=%d "
                    "summary_rows_removed=%d",
                    counts.get("events_removed", 0),
                    counts.get("assignments_removed", 0),
                    counts.get("summary_rows_removed", 0),
                )
            else:
                counts = NLLOutput(locator_path).remove_range(t1, t2)
                self.logger.info(
                    "Cascade locator removal summary: catalog_rows_removed=%d arrivals_rows_removed=%d "
                    "summary_rows_removed=%d",
                    counts.get("catalog_rows_removed", 0),
                    counts.get("arrivals_rows_removed", 0),
                    counts.get("summary_rows_removed", 0),
                )

        if "associator" in cleanup_stages and "picker" not in cleanup_stages:
            cleared = EQTOutput(picker_path).clear_is_associated_in_range(t1, t2)
            self.logger.info(
                "Cascade: cleared is_associated on %d picks in range (re-association without re-pick).",
                cleared.get("picks_cleared", 0),
            )

        if self.station_summary:
            station_summary_period_seconds = parse_time_string(self.station_summary_period)
            ss_result = station_summary_reset_for_overwrite(
                picker_path,
                associator_path,
                locator_path,
                set(cleanup_stages),
                t1,
                t2,
                station_summary_period_seconds=station_summary_period_seconds,
            )
            self.logger.info(
                "Station summary slice cleanup: removed %d HDF5 period-slices; by_stage=%s",
                ss_result.get("keys_removed", 0),
                ss_result.get("by_stage", {}),
            )

        for step, exporter in (
            ("picker", self.picker_summary_exporter),
            ("associator", self.associator_summary_exporter),
            ("locator", self.locator_summary_exporter),
        ):
            if step in cleanup_stages and exporter is not None:
                removed = drop_summary_txt_rows_in_range(str(exporter.summary_file), t1, t2)
                self.logger.info(
                    "Per-stage summary CSV %s: removed %d rows in overwrite range",
                    exporter.summary_file,
                    removed,
                )

        self.logger.info(
            "Cascading overwrite finished for range [%s, %s].", t1, t2
        )

    def _archive_eqt_annotations(self, st_annotated, start_time, end_time, overwrite):
        """
        Write EQTransformer annotated streams to per-UTC-day MiniSEED files when
        ``picker.eqt_annotation_dir`` is set.

        Filenames: ``<dir>/yyyy-mm-dd-<config_name>-<model>-annotated.mseed``

        If a target file already exists and ``overwrite`` is False, that file is
        skipped (same policy as refusing to replace picker HDF5 without overwrite).
        """
        raw = self.config["picker"].get("eqt_annotation_dir")
        if raw is None or raw is False:
            return
        s = str(raw).strip()
        if not s:
            return
        out_root = Path(s).expanduser()
        try:
            out_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.logger.warning(
                "EQ annotation archive: cannot create directory %s (%s)",
                out_root,
                exc,
            )
            return

        cfg = _safe_filename_component(self.config["locator"]["config_name"])
        model = _safe_filename_component(self.config["picker"]["model"])
        t1 = UTCDateTime(start_time)
        t2 = UTCDateTime(end_time)
        d0 = t1.datetime.date()
        d1 = t2.datetime.date()

        self.logger.info(
            "EQ annotation archive: dir=%s overwrite=%s "
            "files=<yyyy-mm-dd>-%s-%s-annotated.mseed UTC_days=%s..%s",
            out_root,
            overwrite,
            cfg,
            model,
            d0,
            d1,
        )

        cur = d0
        while cur <= d1:
            day_start = UTCDateTime(
                year=cur.year, month=cur.month, day=cur.day, hour=0, minute=0, second=0
            )
            day_end = day_start + 86400.0
            st_day = Stream()
            for tr in st_annotated:
                ta = max(tr.stats.starttime, day_start)
                tb = min(tr.stats.endtime, day_end)
                if ta >= tb:
                    continue
                tr2 = tr.copy()
                tr2.trim(ta, tb)
                if tr2.stats.npts > 0:
                    st_day += tr2

            fn = f"{cur.isoformat()}-{cfg}-{model}-annotated.mseed"
            fp = out_root / fn

            if len(st_day) == 0:
                self.logger.debug(
                    "EQ annotation archive: no trace samples for UTC day %s, skip %s",
                    cur,
                    fp.name,
                )
            elif fp.exists() and not overwrite:
                self.logger.info(
                    "EQ annotation archive: skip existing file %s (overwrite is false)",
                    fp,
                )
            else:
                try:
                    st_day.write(str(fp), format="MSEED")
                    npts = sum(int(tr.stats.npts) for tr in st_day)
                    self.logger.info(
                        "EQ annotation archive: wrote %s (%d traces, %d samples)",
                        fp,
                        len(st_day),
                        npts,
                    )
                except Exception as exc:
                    self.logger.warning(
                        "EQ annotation archive: failed to write %s (%s)",
                        fp,
                        exc,
                    )

            cur = cur + timedelta(days=1)

    def run(self, start_time, end_time, tproc='1d', run_picker=False, run_associator=False, run_locator=False, overwrite=False):
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
        run_picker : bool, default False
            Run phase picking step.
        run_associator : bool, default False
            Run event association step.
        run_locator : bool, default False
            Run earthquake location step.
        overwrite : bool, default False
            Overwrite existing output for the time range; default is to skip existing.

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
            
            chunk_catalog = self._process_timespan(chunk_start_time, chunk_end_time, inv, run_picker, run_associator, run_locator, overwrite)
            
            # Combine catalogs
            if chunk_catalog is not None and len(chunk_catalog) > 0:
                final_catalog += chunk_catalog
        
        # Calculate and log total execution time
        run_execution_time = time.perf_counter() - run_start_time
        self.logger.info(f"WAV2HYP processing completed! Events processed: {len(final_catalog)}")
        self.logger.info(f"Total execution time: {run_execution_time:.2f} seconds ({run_execution_time/60:.2f} minutes)")
        return final_catalog
    
    def _process_timespan(self, start_time, end_time, inventory, run_picker, run_associator, run_locator, overwrite):
        """Process a time span through the complete pipeline."""

        # Start timing
        timespan_start_time = time.perf_counter()

        # Get output paths
        picker_output = self._get_output_path("picker")
        associator_output = self._get_output_path("associator")
        locator_output = self._get_output_path("locator")

        if overwrite:
            self._cascade_overwrite_cleanup(
                start_time, end_time, run_picker, run_associator, run_locator
            )

        # Step 1: Phase Picking
        if not run_picker:
            self.logger.info("Reading existing picks (picker not requested)...")
            eqt_output = EQTOutput(picker_output)
            picks, detections, metadata = eqt_output.read(t1=start_time, t2=end_time)
        else:
            picks = self._run_picker(start_time, end_time, picker_output, overwrite)

        # Step 2: Event Association
        if not run_associator:
            self.logger.info("Reading existing associations (associator not requested)...")
            pyocto_output = PyOctoOutput(associator_output)
            catalog_assoc, _, _, _ = pyocto_output.read(t1=start_time, t2=end_time)
        else:
            catalog_assoc = self._run_associator(picks, inventory, associator_output, start_time, end_time, overwrite)

        # Step 3: Event Location
        if not run_locator:
            self.logger.info("Reading existing locations (locator not requested)...")
            nll_output = NLLOutput(locator_output)
            catalog, _ = nll_output.read(t1=start_time, t2=end_time)
            if catalog is None:
                self.logger.warning("No existing locator results found, using associator catalog")
                catalog = catalog_assoc
        else:
            catalog = self._run_locator(catalog_assoc, inventory, locator_output, start_time, end_time, overwrite)

        # Calculate and log timespan execution time
        timespan_execution_time = time.perf_counter() - timespan_start_time
        self.logger.info(f"Timespan completed. Execution time: {timespan_execution_time:.2f} seconds ({timespan_execution_time/60:.2f} minutes)")
        return catalog

    def _run_picker(self, start_time, end_time, output_path, overwrite=False):
        """
        Run phase picking with EQTransformer.

        When ``overwrite`` is True, cascading removal for ``[start_time, end_time]`` has
        already been performed in :meth:`_process_timespan` via
        :meth:`_cascade_overwrite_cleanup`. Do not repeat cascade cleanup here; run
        picking and write outputs.
        """
        method_start = time.perf_counter()

        if not overwrite:
            if date_already_processed_for_stage(output_path, start_time, end_time):
                self.logger.info(
                    "Picker stage already processed for this chunk (HDF5 summary); loading picks from disk."
                )
                eqt_output = EQTOutput(output_path)
                picks, _, _ = eqt_output.read(t1=start_time, t2=end_time)
                return picks
            try:
                eqt_probe = EQTOutput(output_path)
                existing_picks, existing_det, _ = eqt_probe.read(t1=start_time, t2=end_time)
                if (existing_picks is not None and len(existing_picks) > 0) or (
                    existing_det is not None and len(existing_det) > 0
                ):
                    self.logger.info(
                        "Existing picker data for range (legacy file without summary row); loading from HDF5."
                    )
                    return existing_picks
            except (OSError, FileNotFoundError, KeyError):
                pass

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
        self.logger.info("EQTransformer: execution started")
        t = time.perf_counter()
        st_annotated = picker.annotate(stream)
        self.logger.info(f"EQTransformer execution time: {time.perf_counter() - t:.2f} seconds")

        self._archive_eqt_annotations(st_annotated, start_time, end_time, overwrite)
        
        # Extract picks from the probability traces
        from seisbench.models.base import WaveformModel
        from seisbench.util.annotations import PickList

        # EQTransformer returns picks by creating streams
        # with the following trace_ids:
        # - net.station.location.EQTransformer_P
        # - net.station.location.EQTransformer_S
        # - net.station.location.EQTransformer_Detection
        
        # We need to extract the picks from the annotated stream
        # by selecting the traces with the corresponding trace_ids.
        
        # Extract P picks
        p_picks = WaveformModel.picks_from_annotations(
            st_annotated.select(channel="*_P"),
            threshold=self.p_threshold,
            phase="P"
        )
        
        # Extract S picks
        s_picks = WaveformModel.picks_from_annotations(
            st_annotated.select(channel="*_S"),
            threshold=self.s_threshold,
            phase="S"
        )
        
        # Extract detections
        detections = WaveformModel.detections_from_annotations(
            st_annotated.select(channel="*_Detection"),
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

        method_time = time.perf_counter() - method_start
        date_str = start_time.strftime("%Y/%m/%d")
        summary_stats = {
            "date": date_str,
            "config": self.config["locator"]["config_name"],
            "ncha": len(stream),
            "nsamp": sum(len(tr.data) for tr in stream),
            "pick_model": picker_model,
            "np": len(p_picks),
            "ns": len(s_picks),
            "npicks": len(picks),
            "ndetections": len(detections),
            "p_thresh": self.p_threshold,
            "s_thresh": self.s_threshold,
            "d_thresh": self.d_threshold,
            "t_exec_pick": method_time,
        }
        eqt_output = EQTOutput(output_path, time_range=(start_time, end_time))
        eqt_output.write(picks, detections, metadata, summary_stats=summary_stats)

        if self.picker_summary_exporter:
            picker_stats = {
                "config": self.config["locator"]["config_name"],
                "ncha": len(stream),
                "nsamp": sum(len(tr.data) for tr in stream),
                "pick_model": picker_model,
                "np": len(p_picks),
                "ns": len(s_picks),
                "npicks": len(picks),
                "ndetections": len(detections),
                "p_thresh": self.p_threshold,
                "s_thresh": self.s_threshold,
                "d_thresh": self.d_threshold,
            }
            self.picker_summary_exporter.update_entry(date_str, picker_stats, method_time)

        self.logger.info(f"Picker completed: {method_time:.2f}s ({method_time/60:2.1f} minutes)")
        
        return picks
    
    def _run_associator(self, picks, inventory, output_path, start_time=None, end_time=None, overwrite=False):
        """
        Run event association with PyOcto.

        When ``overwrite`` is True, cascading removal for the active time window has
        already been performed in :meth:`_process_timespan` via
        :meth:`_cascade_overwrite_cleanup`. Do not repeat cascade cleanup here.
        """
        method_start = time.perf_counter()

        start_time_msg = "---" if start_time is None else f"{start_time}"
        end_time_msg = "---" if end_time is None else f"{end_time}"
        self.logger.info(f"Running event association from {start_time_msg} to {end_time_msg}...")

        # Check if there are picks to associate
        if picks is None or len(picks) == 0:
            self.logger.info("No picks available for association - skipping association processing")
            return VCatalog()

        if not overwrite and start_time and end_time:
            if date_already_processed_for_stage(output_path, start_time, end_time):
                self.logger.info(
                    "Associator stage already processed for this chunk (HDF5 summary); loading from disk."
                )
                pyocto_output = PyOctoOutput(output_path)
                catalog_assoc, _, _, _ = pyocto_output.read(t1=start_time, t2=end_time)
                return catalog_assoc
            try:
                pyocto_probe = PyOctoOutput(output_path)
                catalog_existing, _, _, _ = pyocto_probe.read(t1=start_time, t2=end_time)
                if catalog_existing is not None and len(catalog_existing) > 0:
                    self.logger.info(
                        "Existing association data for range (legacy file without summary row); loading from HDF5."
                    )
                    return catalog_existing
            except (OSError, FileNotFoundError, KeyError):
                pass
        
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
        pick_counts = self._count_pick_phases(picks)
        self.logger.info(
            "Loaded picks for association: %d total (%d P, %d S)",
            pick_counts["total"],
            pick_counts["P"],
            pick_counts["S"],
        )
        self.logger.info("PyOcto: execution started")
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

        time_range = (start_time, end_time) if start_time and end_time else None
        pyocto_output = PyOctoOutput(output_path, time_range=time_range)
        self.logger.info(f"Writing associations to {output_path}")
        method_time = time.perf_counter() - method_start
        date_str = start_time.strftime("%Y/%m/%d") if start_time else ""
        summary_stats = {
            "date": date_str,
            "config": self.config["locator"]["config_name"],
            "assoc_method": "pyocto",
            "nassignments": len(assignments_df) if assignments_df is not None else 0,
            "nevents": len(events_df) if events_df is not None else 0,
            "t_exec_assoc": method_time,
        }
        pyocto_output.write(events_df, assignments_df, metadata, summary_stats=summary_stats)

        # Create ObsPy Catalog object
        catalog = VCatalog.from_pyocto(events_df, assignments_df)

        if self.associator_summary_exporter and start_time:
            associator_stats = {
                "config": self.config["locator"]["config_name"],
                "assoc_method": "pyocto",
                "nassignments": len(assignments_df) if assignments_df is not None else 0,
                "nevents": len(events_df) if events_df is not None else 0,
            }
            self.associator_summary_exporter.update_entry(date_str, associator_stats, method_time)

        self.logger.info(f"Associator completed: {method_time:.2f}s ({method_time/60:2.1f} minutes)")
        
        return catalog
    
    def _run_locator(self, catalog_in, inventory, output_path, start_time=None, end_time=None, overwrite=False):
        """
        Run earthquake location with NonLinLoc.

        When ``overwrite`` is True, cascading removal for the active time window has
        already been performed in :meth:`_process_timespan` via
        :meth:`_cascade_overwrite_cleanup`. Do not repeat cascade cleanup here.
        """
        method_start = time.perf_counter()

        import nllpy

        self.logger.info(f"Running earthquake location from {start_time} to {end_time}...")

        # Check if catalog has events
        if len(catalog_in) == 0:
            self.logger.info("No associations in catalog - skipping location processing")
            return VCatalog()

        if not overwrite and start_time and end_time:
            if date_already_processed_for_stage(output_path, start_time, end_time):
                self.logger.info(
                    "Locator stage already processed for this chunk (HDF5 summary); loading from disk."
                )
                nll_output = NLLOutput(output_path)
                catalog, _ = nll_output.read(t1=start_time, t2=end_time)
                if catalog is None:
                    return VCatalog()
                return catalog
            try:
                nll_probe = NLLOutput(output_path)
                catalog_existing, _ = nll_probe.read(t1=start_time, t2=end_time)
                if catalog_existing is not None and len(catalog_existing) > 0:
                    self.logger.info(
                        "Existing locator data for range (legacy file without summary row); loading from HDF5."
                    )
                    return catalog_existing
            except (OSError, FileNotFoundError, KeyError):
                pass

        locator_config = self.config['locator']
        nll_home = locator_config['nll_home']
        name = locator_config['config_name']
        input_event_count = len(catalog_in)
        associated_pick_count = self._count_catalog_arrivals(catalog_in)
        self.logger.info(
            "Loaded associations for location: %d events, %d picks associated",
            input_event_count,
            associated_pick_count,
        )
        
        # Create directories
        event_date = catalog_in[0].origins[0].time.strftime("%Y-%m-%d")
        obs = f"./obs-{event_date}"
        
        nll_dir = self._get_output_path('nll_dir', catalog_in[0].origins[0].time.date)
        loc = os.path.basename(nll_dir)
        
        os.makedirs(os.path.join(nll_home, obs), exist_ok=True)
        os.makedirs(nll_dir, exist_ok=True)
        
        # Convert catalog to NonLinLoc obs files
        catalog_in.write_nlloc_obs(os.path.join(nll_home, obs), format="NLLOC_OBS")
        self.logger.info(f"Converted associations to NLL_OBS files: ({os.path.join(nll_home, obs)})")

        # Create volcano monitoring configuration
        config = nllpy.create_volcano_config(lat_orig=self.lat, lon_orig=self.lon)
        config.nll_home = nll_home
        config.filename = f"{name}.in"
        config.input_obs = (f"{obs}", "NLLOC_OBS")
        config.output_obs = f"{loc}/{name}"
        config.add_station_from_inventory(inventory, sta_fmt=locator_config['station_format'])

        # Optional: user-defined velocity model (list of layers: depth_km, VpTop, VpGrad, VsTop, VsGrad, rhoTop, rhoGrad)
        if locator_config.get('velocity_model_layers'):
            config.layer.layers = [tuple(row) for row in locator_config['velocity_model_layers']]
        # Optional: pass-through overrides to nllpy config (e.g. locgrid.d_grid_x, layer.layers, locmethod.*)
        if locator_config.get('nllpy_overrides'):
            _apply_nllpy_overrides(config, locator_config['nllpy_overrides'], self.logger)
        self.logger.info(self._describe_locator_velocity_model(locator_config))

        config.write_complete_control_file(os.path.join(nll_home, config.filename))
        self.logger.info(f"Writing NLL control file: ({os.path.join(nll_home, config.filename)})")


        # Run NonLinLoc
        self.logger.info("NonLinLoc: execution started")
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
        full_search_path = os.path.join(nll_home, f"{config.output_obs}*.hyp")  # config.output_obs includes the prefix for the .hyp files
        self.logger.info(f"Reading NonLinLoc hyp files: ({full_search_path})")
        hyp_files = glob.glob(full_search_path)  # all .hyp files in the loc directory
        filtered_files = [f for f in hyp_files if 'sum' not in f]  # remove summary files
        # Read all filtered files
        catalog_out = VCatalog()
        catalog_out.comments = []  # I'm not sure why this isn't set (maybe fix in VCatalog)
        for hyp_file in filtered_files:
            try:
                catalog_out += read_nlloc_hyp(hyp_file)
                self.logger.debug(f"Read NonLinLoc hyp file: {hyp_file}")
            except Exception as e:
                self.logger.warning(f"Error reading NonLinLoc hyp file {hyp_file}: {e}")

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

        located_arrival_count = self._count_catalog_arrivals(catalog_out)
        self.logger.info(
            "Locations: %d/%d events located, %d picks associated, %d located arrivals",
            len(catalog_out),
            input_event_count,
            associated_pick_count,
            located_arrival_count,
        )

        time_range = (start_time, end_time) if start_time and end_time else None
        nll_output = NLLOutput(output_path, time_range=time_range)
        self.logger.info(f"Writing locations to {output_path}")
        method_time = time.perf_counter() - method_start
        date_str = start_time.strftime("%Y/%m/%d") if start_time else ""
        summary_stats = {
            "date": date_str,
            "config": self.config["locator"]["config_name"],
            "loc_method": "nll",
            "nlocations": len(catalog_out),
            "t_exec_loc": method_time,
        }
        nll_output.write(catalog_out, metadata, summary_stats=summary_stats)

        if self.locator_summary_exporter and start_time:
            locator_stats = {
                "config": self.config["locator"]["config_name"],
                "loc_method": "nll",
                "nlocations": len(catalog_out),
            }
            self.locator_summary_exporter.update_entry(date_str, locator_stats, method_time)

        self.logger.info(f"Locator completed: {method_time:.2f}s ({method_time/60:2.1f} minutes)")
        
        return catalog_out
    

def main(start_time, end_time, config_path="config.yaml", tproc='1d', run_picker=False, run_associator=False, run_locator=False, overwrite=False):
    """
    Main processing function (e.g. for programmatic use).

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
    run_picker : bool, default False
        Run phase picking step.
    run_associator : bool, default False
        Run event association step.
    run_locator : bool, default False
        Run earthquake location step.
    overwrite : bool, default False
        Overwrite existing output for the time range.

    Returns
    -------
    vdapseisutils.VCatalog
        Final earthquake catalog.
    """
    processor = WAV2HYP(config_path=config_path)
    return processor.run(start_time, end_time, tproc, run_picker, run_associator, run_locator, overwrite)


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
