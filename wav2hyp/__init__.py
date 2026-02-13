"""
WAV2HYP: Waveform to Hypocenter Processing Pipeline

A comprehensive seismic processing pipeline that transforms continuous waveform
data into earthquake hypocenters through automated phase picking, event association,
and probabilistic location.

Main Components:
- Phase picking using EQTransformer with VolPick model
- Event association using PyOcto
- Earthquake location using NonLinLoc

Usage:
    from wav2hyp import WAV2HYP
    
    processor = WAV2HYP('config.yaml')
    catalog = processor.run(start_time, end_time)
"""

from .core import WAV2HYP, main
from .config_loader import load_config, validate_config
from .cli import cli_main

__version__ = "1.0.0"
__author__ = "WAV2HYP Development Team"

__all__ = [
    'WAV2HYP',
    'main', 
    'load_config',
    'validate_config',
    'cli_main',
]
