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
    required_sections = ['target', 'output', 'picker', 'associator', 'locator']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
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
    }


def print_config_summary(config):
    """
    Print a summary of the loaded configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    """
    print("=" * 60)
    print("WAV2HYP Configuration Summary")
    print("=" * 60)
    
    # Target information
    target = config['target']
    print(f"Target: {target['name']}")
    print(f"Location: {target['latitude']:.2f}°N, {target['longitude']:.2f}°W")
    print(f"Elevation: {target['elevation']:.0f} m")
    
    # Output paths
    print(f"\nOutput Directory: {config['output']['base_dir']}")
    
    # Picker settings
    picker = config['picker']
    print(f"\nPicker Model: {picker['model']}")
    print(f"Thresholds - P: {picker['p_threshold']}, S: {picker['s_threshold']}, D: {picker['d_threshold']}")
    
    # Associator settings
    assoc = config['associator']
    print(f"\nAssociation Radius: {assoc['radius_km']} km")
    print(f"Velocity Model - P: {assoc['p_velocity']} km/s, S: {assoc['s_velocity']} km/s")
    print(f"Min Picks - P: {assoc['min_p_picks']}, S: {assoc['min_s_picks']}")
    
    # Locator settings
    locator = config['locator']
    print(f"\nNonLinLoc Home: {locator['nll_home']}")
    print(f"Config Name: {locator['config_name']}")
    
    print("=" * 60)


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
