"""
Command-line interface for WAV2HYP processing pipeline.

This module provides the command-line interface for running the WAV2HYP
waveform to hypocenter processing pipeline.
"""

import argparse
import sys
from pathlib import Path

from obspy import UTCDateTime

from .core import WAV2HYP
from .utils.summary import generate_summary_from_files


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="WAV2HYP: Waveform to Hypocenter processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  wav2hyp -c config.yaml --t1 "2024/10/14" --t2 "2024/10/15"
  wav2hyp -c examples/config.yaml --t1 "2024/10/14 00:00:00" --t2 "2024/10/15 00:00:00" --skip-picker
  python run_wav2hyp.py -c config.yaml --t1 "2024/10/14" --t2 "2024/10/15"
        """
    )
    
    # Required arguments
    parser.add_argument(
        "-c", "--config", 
        required=True, 
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--t1", 
        required=True, 
        help="Start time (YYYY/MM/DD or YYYY/MM/DD HH:MM:SS)"
    )
    parser.add_argument(
        "--t2", 
        required=True, 
        help="End time (YYYY/MM/DD or YYYY/MM/DD HH:MM:SS)"
    )
    
    # Optional processing control
    parser.add_argument(
        "--skip-picker", 
        action="store_true", 
        help="Skip phase picking step (use existing picks)"
    )
    parser.add_argument(
        "--skip-associator", 
        action="store_true", 
        help="Skip event association step (use existing associations)"
    )
    parser.add_argument(
        "--skip-locator", 
        action="store_true", 
        help="Skip event location step (use association results only)"
    )
    
    # Output options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true", 
        help="Suppress output (except errors)"
    )
    
    # Version info
    parser.add_argument(
        "--version",
        action="version",
        version="WAV2HYP 1.0.0"
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command-line arguments."""
    # Check config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Validate time arguments
    try:
        t1 = UTCDateTime(args.t1)
        t2 = UTCDateTime(args.t2)
    except Exception as e:
        print(f"Error parsing time arguments: {e}")
        print("Time format should be YYYY/MM/DD or YYYY/MM/DD HH:MM:SS")
        sys.exit(1)
    
    if t1 >= t2:
        print("Error: Start time must be before end time")
        sys.exit(1)
    
    # Check for conflicting verbosity options
    if args.verbose and args.quiet:
        print("Error: Cannot specify both --verbose and --quiet")
        sys.exit(1)
    
    return t1, t2


def generate_summary_command():
    """Generate summary file from existing results."""
    parser = argparse.ArgumentParser(description="Generate summary file from existing WAV2HYP results")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML configuration file")
    parser.add_argument("--t1", help="Start date for summary generation (YYYY/MM/DD)")
    parser.add_argument("--t2", help="End date for summary generation (YYYY/MM/DD)")
    parser.add_argument("--output", help="Output summary filename")
    
    args = parser.parse_args()
    
    # Load config to get output directory and config name
    from .config_loader import load_config
    config = load_config(args.config)
    
    base_output_dir = config['output']['base_dir']
    config_name = config['locator']['config_name']
    
    print(f"Generating summary for {config_name}")
    print(f"Output directory: {base_output_dir}")
    
    # Parse date range if provided
    start_date = None
    end_date = None
    
    if args.t1:
        try:
            start_date = UTCDateTime(args.t1)
        except Exception as e:
            print(f"Error parsing start date: {e}")
            sys.exit(1)
    
    if args.t2:
        try:
            end_date = UTCDateTime(args.t2)
        except Exception as e:
            print(f"Error parsing end date: {e}")
            sys.exit(1)
    
    if start_date and end_date:
        print(f"Date range: {start_date.strftime('%Y/%m/%d')} to {end_date.strftime('%Y/%m/%d')}")
    else:
        print("Processing all available dates")
    
    # Generate summary
    summary_file = generate_summary_from_files(
        base_output_dir=base_output_dir,
        config_name=config_name,
        summary_filename=args.output,
        date_range=(start_date, end_date) if start_date and end_date else None
    )
    
    print(f"Summary file generated: {summary_file}")
    return summary_file


def cli_main():
    """Main entry point for command-line interface."""
    try:
        # Parse and validate arguments
        args = parse_arguments()
        t1, t2 = validate_arguments(args)
        
        # Setup logging level based on verbosity
        import logging
        if args.quiet:
            logging.getLogger('wav2hyp').setLevel(logging.ERROR)
        elif args.verbose:
            logging.getLogger('wav2hyp').setLevel(logging.DEBUG)
        
        # Initialize WAV2HYP processor
        print(f"Initializing WAV2HYP with config: {args.config}")
        processor = WAV2HYP(config_path=args.config)
        
        # Run processing
        print(f"Processing from {t1} to {t2}")
        if not args.quiet:
            skip_info = []
            if args.skip_picker:
                skip_info.append("picker")
            if args.skip_associator:
                skip_info.append("associator")
            if args.skip_locator:
                skip_info.append("locator")
            
            if skip_info:
                print(f"Skipping: {', '.join(skip_info)}")
            else:
                print("Running complete pipeline")
        
        catalog = processor.run(
            t1, t2,
            skip_picker=args.skip_picker,
            skip_associator=args.skip_associator,
            skip_locator=args.skip_locator
        )
        
        # Report results
        if not args.quiet:
            print(f"\nProcessing completed successfully!")
            print(f"Final catalog contains {len(catalog)} events")
            
            if len(catalog) > 0:
                print("\nEvent summary:")
                for i, event in enumerate(catalog[:5]):  # Show first 5 events
                    origin = event.preferred_origin() or event.origins[0]
                    print(f"  Event {i+1}: {origin.time} - "
                          f"Lat: {origin.latitude:.3f}, Lon: {origin.longitude:.3f}, "
                          f"Depth: {origin.depth/1000:.1f} km")
                
                if len(catalog) > 5:
                    print(f"  ... and {len(catalog)-5} more events")
        
        return catalog
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
