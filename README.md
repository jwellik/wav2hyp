# WAV2HYP: Waveform to Hypocenter Processing Pipeline

A comprehensive seismic processing pipeline that transforms continuous waveform data into earthquake hypocenters through automated phase picking, event association, and probabilistic location. This code is optimized for earthquake locations at volcanoes.

## Features

- **Automated Phase Picking**: EQTransformer with [VOLPICK](https://seisbench.readthedocs.io/en/main/pages/benchmark_datasets.html#vcseis) model for volcanic seismicity
- **Event Association**: [PyOcto](https://pyocto.readthedocs.io/en/latest/index.html) groups picks into events
- **Earthquake Location**: [NonLinLoc](https://alomax.free.fr/nlloc/) probabilistic location
- **Flexible Configuration**: YAML-based config; CLI and Python API

## Quick start

[Install the package and dependencies](docs/installation.md) (NonLinLoc, conda/pip, `pip install -e .`), then run the full pipeline:

```bash
wav2hyp -c examples/sthelens.yaml --t1 "2004/09/23" --t2 "2004/09/25" --all
```

## Documentation

- [**Introduction & usage**](docs/introduction.md) – Configuration, CLI options, Python API, examples
- [**Installation**](docs/installation.md) – NonLinLoc, conda, pip, dependencies
- [**Workflow**](docs/workflow.md) – Inputs and outputs at each step (picker → associator → locator)
- [**Data structures**](docs/data-structures.md) – HDF5/ASDF layout and how to retrieve data (including indexed queries)

## Development and testing

Install in dev mode with `pip install -e ".[dev]"` and run tests with `pytest` (see [Introduction](docs/introduction.md#testing)).

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
