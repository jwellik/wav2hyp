# WAV2HYP: Waveform to Hypocenter Processing Pipeline

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

### Install NonLinLoc

Download and compile NonLinLoc:
```
rm main.zip
wget https://github.com/ut-beg-texnet/NonLinLoc/archive/refs/heads/main.zip
unzip main.zip
cd NonLinLoc-main/src
mkdir bin
sudo apt update
sudo apt install build-essential
sudo apt install cmake
cmake .
make
```

Add NonLinLoc to your path:
```
cd
echo 'export PATH="/home/<user>/NonLinLoc-main/src/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Verify NonLinLoc installation:

```
$ cd
$ which NLLoc
```

You should see a path to NonLinLoc.


### Development Installation

We recommend installing wav2hyp with a package manager like [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html).

```bash
# Create a conda environment with Python 3.12
conda create -n wav2hyp python=3.12

# Activate the environment before installing dependencies
conda activate wav2hyp
```


```bash
# Clone or navigate to the wav2hyp-basic directory
cd wav2hyp

# Install in development mode
pip install -e .
```

### Dependencies

The package requires several scientific Python packages. Install dependencies with:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- ObsPy for seismic data handling
- SeisBench for machine learning phase picking
- PyOcto for event association
- NonLinLoc ([nllpy](https://github.com/jwellik/nllpy)) for earthquake location
- [vdapseisutils](https://github.com/jwellik/vdapseisutils) for VDAP-specific utilities

## Usage

### Command Line Interface

```bash
# Basic usage
wav2hyp -c examples/sthelens.yaml --t1 "2004/09/23" --t2 "2004/09/25"

# Skip certain processing steps
wav2hyp -c examples/sthelens.yaml --t1 "2004/09/23" --t2 "2004/09/25" --skip-picker

# Verbose output
wav2hyp -c examples/sthelens.yaml --t1 "2004/09/23" --t2 "2004/09/25" --verbose
```

### Script Interface

```bash
# Equivalent to CLI usage
python run_wav2hyp.py -c examples/sthelens.yaml --t1 "2004/09/23" --t2 "2004/09/25"
```

### Python API

```python
from wav2hyp import WAV2HYP
from obspy import UTCDateTime

# Initialize processor with configuration
processor = WAV2HYP('examples/config.yaml')

# Run processing
catalog = processor.run(
    UTCDateTime('2004/09/23'),
    UTCDateTime('2004/09/25')
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
- `--t1`: Start time (YYYY/MM/DD or YYYY/MM/DD HH:MM:SS) (required)
- `--t2`: End time (YYYY/MM/DD or YYYY/MM/DD HH:MM:SS) (required)
- `--skip-picker`: Skip phase picking step
- `--skip-associator`: Skip event association step  
- `--skip-locator`: Skip earthquake location step
- `--verbose, -v`: Enable verbose output
- `--quiet, -q`: Suppress output except errors
- `--version`: Show version information

## Processing Workflow

1. **Phase Picking**: Downloads continuous waveform data and applies EQTransformer to detect P and S wave arrivals
2. **Event Association**: Groups individual picks into earthquake events using travel time predictions
3. **Location**: Refines event locations using NonLinLoc with probabilistic uncertainty estimates

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
