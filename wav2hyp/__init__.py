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

__version__ = "1.0.0"
__author__ = "WAV2HYP Development Team"

__all__ = [
    "WAV2HYP",
    "main",
    "load_config",
    "validate_config",
    "cli_main",
]


def __getattr__(name: str):
    """
    Lazily import heavy dependencies (e.g., `torch`) only when needed.

    This keeps lightweight utilities (e.g. locator HDF5 tables: catalog_table,
    arrivals_table) usable in environments that don't have the full ML stack installed.
    """
    if name in {"WAV2HYP", "main"}:
        from .core import WAV2HYP, main

        return {"WAV2HYP": WAV2HYP, "main": main}[name]
    if name in {"load_config", "validate_config"}:
        from .config_loader import load_config, validate_config

        return {"load_config": load_config, "validate_config": validate_config}[name]
    if name == "cli_main":
        from .cli import cli_main

        return cli_main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
