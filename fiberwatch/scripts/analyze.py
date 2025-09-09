"""
OTDR data analysis script.

This script provides command-line analysis of OTDR data files
with event detection and results export.
"""

import sys
from pathlib import Path
from typing import Optional

from ..core import Detector, DetectorConfig
from ..utils.data_io import load_test_data, create_distance_axis, save_detection_results
from ..utils.event_processing import cluster_events, get_event_statistics


def run_analysis(
    input_file: Path,
    baseline_file: Optional[Path] = None,
    output_dir: Path = Path("output"),
    distance_km: float = 20.0,
    output_format: str = "csv",
    config=None,
):
    """
    Run OTDR analysis on input data file.

    Args:
        input_file: Path to input OTDR data file
        baseline_file: Optional path to baseline reference file
        output_dir: Output directory for results
        distance_km: Fiber length in kilometers
        output_format: Output format for results ('csv' or 'json')
        config: Application configuration
    """
    print(f"FiberWatch OTDR Analysis")
    print(f"Input file: {input_file}")
    print(f"Fiber length: {distance_km} km")

    # Validate input file
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    print("Loading test data...")
    test_data = load_test_data(input_file)
    print(f"Loaded {len(test_data)} data points")

    # Create distance axis
    distance_axis = create_distance_axis(len(test_data), distance_km)

    # Load baseline if provided
    baseline_data = None
    if baseline_file:
        if baseline_file.exists():
            print(f"Loading baseline from: {baseline_file}")
            baseline_data = load_test_data(baseline_file)

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

    # Print results summary
    _print_analysis_summary(clustered_events, distance_km)

    # Save results
    output_file = output_dir / f"{input_file.stem}_events.{output_format}"
    save_detection_results(clustered_events, output_file, format=output_format)
    print(f"Results saved to: {output_file}")

    return {
        "detection_result": result,
        "clustered_events": clustered_events,
        "statistics": get_event_statistics(clustered_events),
    }


def _print_analysis_summary(events, total_distance_km):
    """Print summary of analysis results."""
    if not events:
        print("\n✅ No events detected - fiber appears to be in good condition!")
        return

    print(f"\n📊 Analysis Summary:")
    print(f"{'=' * 60}")
    print(f"Total fiber length: {total_distance_km:.1f} km")
    print(f"Events detected: {len(events)}")

    # Group events by type
    event_types = {}
    for event in events:
        event_types[event.kind] = event_types.get(event.kind, 0) + 1

    print(f"\nEvent breakdown:")
    for event_type, count in sorted(event_types.items()):
        print(f"  {event_type}: {count}")

    print(f"\nDetailed events:")
    print(
        f"{'Type':<15} {'Position (km)':<12} {'Position (m)':<12} {'Loss (dB)':<10} {'Reflection (dB)':<15}"
    )
    print(f"{'-' * 75}")

    for event in sorted(events, key=lambda e: e.z_km):
        position_m = event.z_km * 1000
        print(
            f"{event.kind:<15} {event.z_km:<12.3f} {position_m:<12.1f} {event.magnitude_db:<10.3f} {event.reflect_db:<15.3f}"
        )

    # Find critical events
    breaks = [e for e in events if e.kind == "break"]
    if breaks:
        break_pos = breaks[0].z_km
        print(
            f"\n⚠️  CRITICAL: Fiber break detected at {break_pos:.3f} km ({break_pos * 1000:.1f} m)"
        )

    dirty_connectors = [e for e in events if e.kind == "dirty_connector"]
    if dirty_connectors:
        print(
            f"\n🔧 MAINTENANCE: {len(dirty_connectors)} dirty connector(s) require cleaning"
        )


def main():
    """Main entry point for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="FiberWatch OTDR Analysis")
    parser.add_argument("input_file", type=Path, help="Input OTDR data file")
    parser.add_argument("--baseline", type=Path, help="Baseline reference file")
    parser.add_argument(
        "--output", "-o", type=Path, default="output", help="Output directory"
    )
    parser.add_argument(
        "--distance", type=float, default=20.0, help="Fiber length in km"
    )
    parser.add_argument(
        "--format", choices=["csv", "json"], default="csv", help="Output format"
    )

    args = parser.parse_args()

    try:
        run_analysis(
            input_file=args.input_file,
            baseline_file=args.baseline,
            output_dir=args.output,
            distance_km=args.distance,
            output_format=args.format,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
