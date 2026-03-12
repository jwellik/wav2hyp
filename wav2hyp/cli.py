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
from .utils.summary import write_summary_txt_from_hdf5, infer_step_from_hdf5


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="WAV2HYP: Waveform to Hypocenter processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  wav2hyp -c config.yaml --t1 "2024/10/14" --t2 "2024/10/15" --all
  wav2hyp -c config.yaml --t1 "2024/10/14" --t2 "2024/10/15" -p -a -l
  wav2hyp -c config.yaml --t1 "2024/10/14" --t2 "2024/10/15" -a -l
  wav2hyp -c config.yaml --t1 "2024/10/14" --t2 "2024/10/15" -p -a -l -o
  wav2hyp -c config.yaml
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
        help="Start time (YYYY/MM/DD or YYYY/MM/DD HH:MM:SS); required when running -p, -a, or -l"
    )
    parser.add_argument(
        "--t2",
        help="End time (YYYY/MM/DD or YYYY/MM/DD HH:MM:SS); required when running -p, -a, or -l"
    )

    # Processing steps: at least one of -p -a -l or --all required to run
    parser.add_argument(
        "-p", "--pick",
        action="store_true",
        help="Run phase picking step"
    )
    parser.add_argument(
        "-a", "--associate",
        action="store_true",
        help="Run event association step"
    )
    parser.add_argument(
        "-l", "--locate",
        action="store_true",
        help="Run earthquake location step"
    )
    parser.add_argument(
        "-A", "--all",
        action="store_true",
        help="Run full pipeline (pick, associate, locate); same as -p -a -l"
    )
    parser.add_argument(
        "-o", "--overwrite",
        action="store_true",
        help="Overwrite existing output for the time range; default is to skip existing"
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


def validate_arguments(args, require_times=False):
    """Validate command-line arguments. If require_times is True, --t1 and --t2 are required."""
    # Check config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    if not require_times:
        return None, None

    if not args.t1 or not args.t2:
        print("Error: --t1 and --t2 are required when running processing steps (-p, -a, -l, or --all)")
        sys.exit(1)

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
    """Regenerate a step summary text file from an HDF5 file that has a 'summary' table."""
    parser = argparse.ArgumentParser(
        description="Regenerate a step summary text file from a WAV2HYP HDF5 file",
        epilog="Example: python generate_summary.py picks/eqt-volpick.h5 sthelens_picker_summary.txt"
    )
    parser.add_argument(
        "hdf5_path",
        help="Path to HDF5 file (e.g. picks/eqt-volpick.h5, associations/pyocto.h5, locations/nll.h5)"
    )
    parser.add_argument(
        "output_summary_path",
        help="Path to output summary text file (e.g. sthelens_picker_summary.txt)"
    )
    parser.add_argument(
        "--step",
        choices=["picker", "associator", "locator"],
        default=None,
        help="Processing step (inferred from HDF5 'summary' table if omitted)"
    )
    args = parser.parse_args()

    hdf5_path = Path(args.hdf5_path)
    if not hdf5_path.exists():
        print(f"Error: HDF5 file not found: {hdf5_path}")
        sys.exit(1)

    step = args.step
    if step is None:
        try:
            step = infer_step_from_hdf5(str(hdf5_path))
            print(f"Inferred step: {step}")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    write_summary_txt_from_hdf5(str(hdf5_path), args.output_summary_path, step)
    print(f"Summary written to {args.output_summary_path}")
    return args.output_summary_path


def cli_main():
    """Main entry point for command-line interface."""
    try:
        args = parse_arguments()

        run_picker = args.pick or args.all
        run_associator = args.associate or args.all
        run_locator = args.locate or args.all
        run_any = run_picker or run_associator or run_locator

        if not run_any:
            # No steps requested: validate config only, print summary, and note about -p -a -l
            validate_arguments(args, require_times=False)
            from .config_loader import load_config, validate_config, print_config_summary
            config = load_config(args.config)
            config = validate_config(config)
            print_config_summary(config)
            print("\nNo processing steps specified. To run the pipeline, include one or more of:")
            print("  -p, --pick       Run phase picking")
            print("  -a, --associate Run event association")
            print("  -l, --locate    Run earthquake location")
            print("  -A, --all       Run full pipeline (same as -p -a -l)")
            return None

        t1, t2 = validate_arguments(args, require_times=True)

        # Setup logging level based on verbosity
        import logging
        if args.quiet:
            logging.getLogger('wav2hyp').setLevel(logging.ERROR)
        elif args.verbose:
            logging.getLogger('wav2hyp').setLevel(logging.DEBUG)

        print(f"Initializing WAV2HYP with config: {args.config}")
        processor = WAV2HYP(config_path=args.config)

        print(f"Processing from {t1} to {t2}")
        if not args.quiet:
            steps = []
            if run_picker:
                steps.append("picker")
            if run_associator:
                steps.append("associator")
            if run_locator:
                steps.append("locator")
            print(f"Running: {', '.join(steps)}")
            if not args.overwrite:
                print("Existing output for the time range will be skipped (use -o to overwrite)")

        catalog = processor.run(
            t1, t2,
            run_picker=run_picker,
            run_associator=run_associator,
            run_locator=run_locator,
            overwrite=args.overwrite
        )

        if not args.quiet:
            print(f"\nProcessing completed successfully!")
            print(f"Final catalog contains {len(catalog)} events")

            if len(catalog) > 0:
                print("\nEvent summary:")
                for i, event in enumerate(catalog[:5]):
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
        try:
            if args.verbose:
                import traceback
                traceback.print_exc()
        except NameError:
            pass
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
