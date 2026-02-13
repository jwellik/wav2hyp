#!/usr/bin/env python
"""
Generate summary file from existing WAV2HYP results.

This script provides a standalone interface for generating summary files
from existing HDF5 results without running the full processing pipeline.

Usage:
    python generate_summary.py -c config.yaml --t1 "2024/10/01" --t2 "2024/10/31"
    python generate_summary.py -c config.yaml --output "custom_summary.txt"
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
