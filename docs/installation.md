# Installation

This page covers installing NonLinLoc, the Python environment, and the wav2hyp package and its dependencies.

## Install NonLinLoc

Download and compile NonLinLoc:

```bash
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

Add NonLinLoc to your PATH:

```bash
cd
echo 'export PATH="/home/<user>/NonLinLoc-main/src/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Replace `/home/<user>/` with your actual home directory path.

Verify NonLinLoc installation:

```bash
which NLLoc
```

You should see a path to the NonLinLoc binary.

## Python environment

We recommend installing wav2hyp with a package manager like [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html).

```bash
# Create a conda environment with Python 3.12
conda create -n wav2hyp python=3.12

# Activate the environment before installing dependencies
conda activate wav2hyp
```

## Install wav2hyp

From the wav2hyp repository root:

```bash
# Clone or navigate to the wav2hyp directory
cd wav2hyp

# Install in development mode
pip install -e .
```

## Dependencies

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

Key dependencies include:

- **ObsPy** – seismic data handling
- **SeisBench** – machine learning phase picking (EQTransformer, VOLPICK)
- **PyOcto** – event association
- **nllpy** – [NonLinLoc](https://github.com/jwellik/nllpy) interface for earthquake location
- **vdapseisutils** – [VDAP-specific utilities](https://github.com/jwellik/vdapseisutils) (VClient, VCatalog, etc.)
- **h5py** – HDF5 I/O for picks and associations

## Verification

Check that the package is installed and the CLI is available:

```bash
wav2hyp --version
```

Or from Python:

```python
import wav2hyp
print(wav2hyp.__version__)
```
