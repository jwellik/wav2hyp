#!/usr/bin/env python
"""
WAV2HYP Script Interface

This script provides a direct interface to run the WAV2HYP processing pipeline.
It serves as a thin wrapper around the wav2hyp package CLI functionality.

Usage:
    python run_wav2hyp.py -c config.yaml --t1 "2024/10/14" --t2 "2024/10/15"
    
This is equivalent to:
    wav2hyp -c config.yaml --t1 "2024/10/14" --t2 "2024/10/15"
"""

import sys
from pathlib import Path

# Add the current directory to the Python path so we can import wav2hyp
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import and run the CLI
try:
    from wav2hyp.cli import cli_main
    
    if __name__ == "__main__":
        cli_main()
        
except ImportError as e:
    print(f"Error importing wav2hyp package: {e}")
    print("Make sure you have installed the package with 'pip install -e .' or run from the correct directory")
    sys.exit(1)
except Exception as e:
    print(f"Error running WAV2HYP: {e}")
    sys.exit(1)