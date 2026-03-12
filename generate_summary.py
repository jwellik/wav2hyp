#!/usr/bin/env python
"""
Regenerate step summary text files from WAV2HYP HDF5 results.

Use this when summary text files were not enabled in the config but you want
to produce them from existing HDF5 files (picks, associations, or locations).

Usage:
    python generate_summary.py <path_to_hdf5> <path_to_output_summary.txt>
    python generate_summary.py picks/eqt-volpick.h5 sthelens_picker_summary.txt
    python generate_summary.py associations/pyocto.h5 sthelens_associator_summary.txt --step associator
"""

import sys
from pathlib import Path

# Add the current directory to the Python path so we can import wav2hyp
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from wav2hyp.cli import generate_summary_command
    
    if __name__ == "__main__":
        generate_summary_command()
        
except ImportError as e:
    print(f"Error importing wav2hyp package: {e}")
    print("Make sure you have installed the package with 'pip install -e .' or run from the correct directory")
    sys.exit(1)
except Exception as e:
    print(f"Error running summary generation: {e}")
    sys.exit(1)
