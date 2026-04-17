"""
Configuration loader for WAV2HYP pipeline.

This module provides functions to load and validate configuration
from YAML files for the WAV2HYP processing pipeline.
"""

import yaml
import os
from pathlib import Path


def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str, default "config.yaml"
        Path to the YAML configuration file.
        
    Returns
    -------
    dict
        Configuration dictionary with all settings.
        
    Raises
    ------
    FileNotFoundError
        If configuration file doesn't exist.
    yaml.YAMLError
        If configuration file has invalid YAML syntax.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file: {e}")
        
    return config


def validate_config(config):
    """
    Validate configuration settings and create directories if needed.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary from load_config().
        
    Returns
    -------
    dict
        Validated and processed configuration.
    """
    # Validate required sections
    required_sections = ['target', 'output', 'picker', 'associator', 'locator', 'waveform_client']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Add default summary configuration to output section if not present
    if 'picker_summary' not in config['output']:
        config['output']['picker_summary'] = None  # None = disabled, string = custom filename
    if 'associator_summary' not in config['output']:
        config['output']['associator_summary'] = None  # None = disabled, string = custom filename
    if 'locator_summary' not in config['output']:
        config['output']['locator_summary'] = None  # None = disabled, string = custom filename
    if 'config_snapshot_dir' not in config['output']:
        config['output']['config_snapshot_dir'] = None  # None/false = disabled, non-empty string = subdir for per-day config snapshots
    if 'station_summary' not in config['output']:
        config['output']['station_summary'] = None  # None = disabled, string = filename e.g. station_summary.txt
    if 'summary_table_period' not in config['output']:
        config['output']['summary_table_period'] = '1h'  # One row per period in H5 summary tables
    if 'summary_text_period' not in config['output']:
        config['output']['summary_text_period'] = '1d'  # Resampling period for step summary text files
    if 'station_summary_period' not in config['output']:
        config['output']['station_summary_period'] = '1d'  # One row per (period, trace_id) in station summary
    
    # Validate waveform_client configuration
    waveform_config = config['waveform_client']
    if 'datasource' not in waveform_config:
        raise ValueError("waveform_client must specify 'datasource' parameter")
    
    # Validate client_type if specified
    if 'client_type' in waveform_config:
        valid_types = ['fdsn', 'sds', 'earthworm', 'seedlink']
        client_type = waveform_config['client_type'].lower()
        if client_type not in valid_types:
            raise ValueError(f"Invalid client_type: {client_type}. Valid types: {valid_types}")
    
    # Validate datasource exists if it's a filesystem path
    datasource = waveform_config['datasource']
    if os.path.exists(datasource) and os.path.isdir(datasource):
        # It's a filesystem path, validate it exists
        if not os.path.exists(datasource):
            print(f"Warning: Datasource path does not exist: {datasource}")
    # For FDSN providers and other non-filesystem datasources, no validation needed
    
    # Create output directories if validation enabled
    if config.get('processing', {}).get('validate_output_dirs', True):
        base_dir = config['output']['base_dir']
        dirs_to_create = [
            os.path.join(base_dir, config['output']['picker_dir']),
            os.path.join(base_dir, config['output']['associator_dir']),
            os.path.join(base_dir, config['output']['locator_dir']),
            os.path.join(base_dir, config['output']['log_dir']),
            config['locator']['nll_home']
        ]
        
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
    
    # Validate inventory file if validation enabled
    if config.get('processing', {}).get('validate_inventory', True):
        inventory_file = config['inventory']['file']
        if not os.path.exists(inventory_file):
            print(f"Warning: Inventory file not found: {inventory_file}")

    # Optional locator velocity model file (default None = use nllpy default)
    if 'velocity_model_file' not in config['locator']:
        config['locator']['velocity_model_file'] = None
    
    return config


def get_global_variables(config):
    """
    Extract global variables from config for compatibility with existing code.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
        
    Returns
    -------
    dict
        Dictionary of global variables that can be used to update globals().
    """
    return {
        # Target of Interest
        'volcano': config['target']['name'],
        'lat': config['target']['latitude'],
        'lon': config['target']['longitude'], 
        'elev': config['target']['elevation'],
        'zlim': tuple(config['associator']['depth_limits']),
        
        # File paths
        'inventory_file': config['inventory']['file'],
        'BASE_OUTPUT_DIR': config['output']['base_dir'],
        'PICKER_OUTPUT_DIR': os.path.join(config['output']['base_dir'], config['output']['picker_dir']),
        'ASSOCIATOR_OUTPUT_DIR': os.path.join(config['output']['base_dir'], config['output']['associator_dir']),
        'LOCATOR_OUTPUT_DIR': os.path.join(config['output']['base_dir'], config['output']['locator_dir']),
        
        # Thresholds
        'p_threshold': config['picker']['p_threshold'],
        's_threshold': config['picker']['s_threshold'],
        'd_threshold': config['picker']['d_threshold'],
        
        # Summary configuration
        'picker_summary': config['output'].get('picker_summary', None),
        'associator_summary': config['output'].get('associator_summary', None),
        'locator_summary': config['output'].get('locator_summary', None),
        'station_summary': config['output'].get('station_summary', None),
        'summary_table_period': config['output'].get('summary_table_period', '1h'),
        'summary_text_period': config['output'].get('summary_text_period', '1d'),
        'station_summary_period': config['output'].get('station_summary_period', '1d'),
    }


def print_config_summary(config, logger=None):
    """
    Print a summary of the loaded configuration.
    Optionally duplicate each line to logger when provided (e.g. for log file).

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    logger : logging.Logger, optional
        If set, each summary line is also logged with logger.info().
    """
    def _out(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    _out("=" * 60)
    _out("WAV2HYP Configuration Summary")
    _out("=" * 60)

    # Target information
    target = config['target']
    _out(f"Target: {target['name']}")
    _out(f"Location: {target['latitude']:.2f}°N, {target['longitude']:.2f}°W")
    _out(f"Elevation: {target['elevation']:.0f} m")

    # Waveform client
    waveform_config = config['waveform_client']
    datasource = waveform_config['datasource']
    _out(f"Waveform Datasource: {datasource}")

    if 'client_type' in waveform_config:
        _out(f"Client Type: {waveform_config['client_type']}")

    # Show additional parameters
    additional_params = {k: v for k, v in waveform_config.items()
                       if k not in ['datasource', 'client_type']}
    if additional_params:
        _out(f"Additional Parameters: {additional_params}")

    # Output paths
    _out(f"Output Directory: {config['output']['base_dir']}")

    # Picker settings
    picker = config['picker']
    _out(f"Picker Model: {picker['model']}")
    _out(f"Thresholds - P: {picker['p_threshold']}, S: {picker['s_threshold']}, D: {picker['d_threshold']}")

    # Associator settings
    assoc = config['associator']
    _out(f"Association Radius: {assoc['radius_km']} km")
    _out(f"Velocity Model - P: {assoc['p_velocity']} km/s, S: {assoc['s_velocity']} km/s")
    _out(f"Min Picks - P: {assoc['min_p_picks']}, S: {assoc['min_s_picks']}")

    # Locator settings
    locator = config['locator']
    _out(f"NonLinLoc Home: {locator['nll_home']}")
    _out(f"Config Name: {locator['config_name']}")
    if locator.get('velocity_model_layers'):
        _out(f"Locator Velocity Model: {len(locator['velocity_model_layers'])} configured layers")
    elif locator.get('nllpy_overrides', {}).get('layer', {}).get('layers'):
        layers = locator['nllpy_overrides']['layer']['layers']
        _out(f"Locator Velocity Model: {len(layers)} layers via locator.nllpy_overrides.layer.layers")
    else:
        _out("Locator Velocity Model: nllpy volcano default")

    _out("=" * 60)


# Example usage and testing
if __name__ == "__main__":
    try:
        # Load configuration
        config = load_config()
        
        # Validate and setup
        config = validate_config(config)
        
        # Print summary
        print_config_summary(config)
        
        # Get global variables
        globals_dict = get_global_variables(config)
        print("\nGlobal variables extracted:")
        for key, value in globals_dict.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error loading configuration: {e}")
