# WAV2HYP: Waveform to Hypocenter Processing Pipeline

For **installation** see [Installation](installation.md). For **pipeline inputs and outputs** at each step see [Workflow](workflow.md). For **HDF5/ASDF layout and retrieval** see [Data structures](data-structures.md).

---

A comprehensive seismic processing pipeline that transforms continuous waveform data into earthquake hypocenters through automated phase picking, event association, and probabilistic location. This code is optimized for earthquake locations at volcanoes.

## Features

- **Automated Phase Picking**: Uses EQTransformer
  with [VOLPICK](https://seisbench.readthedocs.io/en/main/pages/benchmark_datasets.html#vcseis) model optimized for
  volcanic seismicity
- **Event Association**: Groups phase picks into earthquake events using [PyOcto](https://pyocto.readthedocs.io/en/latest/index.html)
- **Earthquake Location**: Refines locations using [NonLinLoc](https://alomax.free.fr/nlloc/) probabilistic method
- **Flexible Configuration**: YAML-based configuration system
- **Multiple Interfaces**: Both CLI and script interfaces available
- **Comprehensive Logging**: Configurable logging with file and console output

## Installation

See [Installation](installation.md) for full instructions (NonLinLoc, conda, pip, dependencies).

## Usage

### Command Line Interface

```bash
# Full pipeline (pick, associate, locate)
wav2hyp -c examples/sthelens.yaml --t1 "2004/09/23" --t2 "2004/09/25" --all

# Run only association and location (use existing picks)
wav2hyp -c examples/sthelens.yaml --t1 "2004/09/23" --t2 "2004/09/25" -a -l

# Overwrite existing output for the time range
wav2hyp -c examples/sthelens.yaml --t1 "2004/09/23" --t2 "2004/09/25" -p -a -l -o

# Show config summary only (no processing)
wav2hyp -c examples/sthelens.yaml

# Verbose output
wav2hyp -c examples/sthelens.yaml --t1 "2004/09/23" --t2 "2004/09/25" --all --verbose
```

### Script Interface

```bash
# Equivalent to CLI usage
python run_wav2hyp.py -c examples/sthelens.yaml --t1 "2004/09/23" --t2 "2004/09/25" --all
```

### Python API

```python
from wav2hyp import WAV2HYP
from obspy import UTCDateTime

# Initialize processor with configuration
processor = WAV2HYP('examples/config.yaml')

# Run processing (specify which steps to run)
catalog = processor.run(
    UTCDateTime('2004/09/23'),
    UTCDateTime('2004/09/25'),
    run_picker=True,
    run_associator=True,
    run_locator=True
)

print(f"Processed {len(catalog)} events")
```

## Configuration

The pipeline uses YAML configuration files. See `examples/sthelens.yaml` for a complete example.

### Key Configuration Sections

- **target**: Study area (volcano/region) information
- **waveform_client**: Waveform data source (auto-detected by VClient)
- **inventory**: Station inventory file location
- **output**: Output directory structure and logging
- **picker**: EQTransformer model and threshold settings
- **associator**: PyOcto association parameters
- **locator**: NonLinLoc location settings

### Waveform Data Sources

WAV2HYP uses `VClient` from vdapseisutils for automatic client type detection. You can use simple auto-detection or specify explicit parameters:

#### Auto-Detection (Simple)
```yaml
waveform_client:
  datasource: "IRIS"  # FDSN server (auto-detected)
```

```yaml
waveform_client:
  datasource: "/path/to/sds/data"  # SDS filesystem (auto-detected)
```

```yaml
waveform_client:
  datasource: "localhost"  # Earthworm server (auto-detected by port)
```

#### Explicit Configuration (Advanced)
```yaml
waveform_client:
  datasource: "192.168.0.104"  # Server hostname/IP
  client_type: "earthworm"     # Explicitly specify client type
  port: 16024                  # Server port
  timeout: 30                  # Additional parameters
```

```yaml
waveform_client:
  datasource: "https://service.iris.edu"       # Server hostname
  client_type: "fdsn"          # Force FDSN type
  timeout: 60                  # Custom timeout
```

VClient automatically detects the appropriate client type based on the datasource parameter:
- **FDSN servers**: "IRIS", "USGS", "GFZ", etc.
- **SDS filesystem**: Path to SDS directory structure
- **Earthworm servers**: Hostname with port 16022-16024
- **SeedLink servers**: Hostname with port 18000-18002

**Supported client types**: `fdsn`, `sds`, `earthworm`, `seedlink`

The SDS format organizes data as: `YEAR/NET/STA/CHAN.TYPE/NET.STA.LOC.CHAN.TYPE.YEAR.DAY`

### Output Structure

```
results/
├── logs/                    # Processing logs
├── picks/                   # Phase picking results (HDF5)
├── associations/            # Event association results (HDF5/PyASDF)
├── locations/               # Location results (HDF5/PyASDF)
│   └── nll_<target>/        # NonLinLoc working directories
└── <target>_picker_summary.txt         # Summary QC file for picker
└── <target>_associator_summary.txt     # Summary QC file for associator
└── <target>_locator_summary.txt        # Summary QC file for locator
```

## Command Line Options

- `-c, --config`: Path to YAML configuration file (required)
- `--t1`: Start time (YYYY/MM/DD or YYYY/MM/DD HH:MM:SS); required when running -p, -a, or -l
- `--t2`: End time (YYYY/MM/DD or YYYY/MM/DD HH:MM:SS); required when running -p, -a, or -l
- `-p, --pick`: Run phase picking step
- `-a, --associate`: Run event association step
- `-l, --locate`: Run earthquake location step
- `-A, --all`: Run full pipeline (same as -p -a -l)
- `-o, --overwrite`: Overwrite existing output for the time range (default is to skip existing)
- `--verbose, -v`: Enable verbose output
- `--quiet, -q`: Suppress output except errors
- `--version`: Show version information

If none of `-p`, `-a`, `-l`, or `--all` is given, the program prints the configuration summary and a note that at least one step option is required to run processing.

## Processing Workflow

1. **Phase Picking**: Downloads continuous waveform data and applies EQTransformer to detect P and S wave arrivals
2. **Event Association**: Groups individual picks into earthquake events using travel time predictions
3. **Location**: Refines event locations using NonLinLoc with probabilistic uncertainty estimates

For a detailed description of inputs and outputs at each step, see [Workflow](workflow.md).

## Examples

See the `examples/` directory for:
- `sthelens.yaml`: Complete configuration example for Mount St. Helens
- `sthelens_inventory.xml`: StationXML file for Mount St. Helens
- `sthelens_inventory_redpy.xml`: StationXML file for Mount St. Helens, limited to stations used by [REDPy](https://code.usgs.gov/vsc/seis/tools/REDPy)

## Development

### Package Structure

```
wav2hyp/
├── __init__.py          # Package initialization
├── core.py              # Main processing classes
├── cli.py               # Command-line interface
├── config_loader.py     # Configuration handling
└── utils/               # Utility modules
    ├── __init__.py
    ├── geo.py              # Geographic calculations
    └── io.py               # Handles import/export of results
    └── prep_inventory.py   # Helpes download or create StationXML files
    └── summary.py          # Writes summary text files for QC
```

### Testing

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=wav2hyp
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please see the contributing guidelines and open an issue or pull request.

## Citation

If you use WAV2HYP in your research, please cite:

```
WAV2HYP: Waveform to Hypocenter Processing Pipeline
Version 1.0.0
https://github.com/jwellik/wav2hyp
```
