"""
OTDR data visualization script.

This script provides command-line visualization of OTDR data
with comprehensive analysis plots.
"""

import sys
from pathlib import Path
from typing import Optional

from ..core import Detector, DetectorConfig
from ..utils.data_io import load_test_data, create_distance_axis
from ..utils.event_processing import cluster_events
from ..utils.visualization import save_all_plots


def run_visualization(
    input_file: Path,
    baseline_file: Optional[Path] = None,
    output_dir: Path = Path("output"),
    distance_km: float = 20.0,
    save_plots: bool = True,
    config=None,
):
    """
    Run OTDR visualization on input data file.

    Args:
        input_file: Path to input OTDR data file
        baseline_file: Optional path to baseline reference file
        output_dir: Output directory for plots
        distance_km: Fiber length in kilometers
        save_plots: Whether to save plots to files
        config: Application configuration
    """
    print(f"FiberWatch OTDR Visualization")
    print(f"Input file: {input_file}")
    print(f"Fiber length: {distance_km} km")

    # Validate input file
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Create output directory
    if save_plots:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    print("Loading test data...")
    test_data = load_test_data(input_file)
    print(f"Loaded {len(test_data)} data points")

    # Create distance axis
    distance_axis = create_distance_axis(len(test_data), distance_km)

    # Load baseline if provided
    baseline_data = None
    baseline_provided = False
    if baseline_file:
        if baseline_file.exists():
            print(f"Loading baseline from: {baseline_file}")
            baseline_data = load_test_data(baseline_file)
            baseline_provided = True

            # Interpolate baseline to match test data length if needed
            if len(baseline_data) != len(test_data):
                import numpy as np

                baseline_distance = create_distance_axis(
                    len(baseline_data), distance_km
                )
                baseline_data = np.interp(
                    distance_axis, baseline_distance, baseline_data
                )
                print("Baseline interpolated to match test data length")
        else:
            print(f"Warning: Baseline file not found: {baseline_file}")

    # Create detector configuration
    if config:
        detector_config = DetectorConfig(
            refl_min_db=config.detection.refl_min_db,
            step_min_db=config.detection.step_min_db,
            slope_min_db_per_km=config.detection.slope_min_db_per_km,
            min_event_separation=config.detection.min_event_separation,
        )
    else:
        detector_config = DetectorConfig()

    # Run detection
    print("Running event detection...")
    detector = Detector(
        distance_km=distance_axis, baseline=baseline_data, config=detector_config
    )

    result = detector.detect(test_data)
    print(f"Detected {len(result.events)} raw events")

    # Cluster events
    if config:
        cluster_distance = config.detection.distance_cluster_m
    else:
        cluster_distance = 5.0

    clustered_events = cluster_events(
        result.events, distance_threshold_m=cluster_distance
    )
    print(f"After clustering: {len(clustered_events)} events")

    # Generate filename
    filename = input_file.stem

    if save_plots:
        # Save all plots
        print("Generating visualization plots...")
        saved_files = save_all_plots(
            result=result,
            clustered_events=clustered_events,
            filename=filename,
            output_dir=output_dir,
            reference_provided=baseline_provided,
        )

        print(f"\nVisualization complete!")
        print(f"Saved {len(saved_files)} plots to: {output_dir}")
        for file_path in saved_files:
            print(f"  • {file_path.name}")
    else:
        # Just display results without saving
        print("Visualization complete (plots not saved)")

    # Print event summary
    _print_event_summary(clustered_events)

    return {
        "detection_result": result,
        "clustered_events": clustered_events,
        "plots_saved": save_plots,
    }


def _print_event_summary(events):
    """Print summary of detected events."""
    if not events:
        print("\n✅ No events detected - fiber appears to be in good condition!")
        return

    print(f"\n📊 Event Summary:")
    print(
        f"{'Type':<15} {'Position (km)':<12} {'Loss (dB)':<10} {'Reflection (dB)':<15}"
    )
    print(f"{'-' * 60}")

    for event in sorted(events, key=lambda e: e.z_km):
        print(
            f"{event.kind:<15} {event.z_km:<12.3f} {event.magnitude_db:<10.3f} {event.reflect_db:<15.3f}"
        )


def main():
    """Main entry point for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="FiberWatch OTDR Visualization")
    parser.add_argument("input_file", type=Path, help="Input OTDR data file")
    parser.add_argument("--baseline", type=Path, help="Baseline reference file")
    parser.add_argument(
        "--output", "-o", type=Path, default="output", help="Output directory"
    )
    parser.add_argument(
        "--distance", type=float, default=20.0, help="Fiber length in km"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save plots to files"
    )

    args = parser.parse_args()

    try:
        run_visualization(
            input_file=args.input_file,
            baseline_file=args.baseline,
            output_dir=args.output,
            distance_km=args.distance,
            save_plots=not args.no_save,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
