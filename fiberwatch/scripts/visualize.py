"""
OTDR data visualization script.

This script provides command-line visualization of OTDR data
with comprehensive analysis plots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fiberwatch.core.detector import DetectedEvent

import argparse
from pathlib import Path

import numpy as np

from fiberwatch.core import Detector, DetectorConfig
from fiberwatch.utils.data_io import create_distance_axis, load_test_data
from fiberwatch.utils.event_processing import cluster_events
from fiberwatch.utils.visualization import save_all_plots


def run_visualization(
    input_file: Path,
    baseline_file: Path | None = None,
    output_dir: Path = Path("output"),
    sample_spacing_km: float = 0.0025545,  # 采样间距，默认约2.55米
    save_plots: bool = True,
    config: DetectorConfig | None = None,
):
    """
    Run OTDR visualization on input data file.

    Args:
        input_file: Path to input OTDR data file
        baseline_file: Optional path to baseline reference file
        output_dir: Output directory for plots
        sample_spacing_km: Distance between samples in kilometers (采样间距)
        save_plots: Whether to save plots to files
        config: Application configuration

    """
    print("FiberWatch OTDR Visualization")
    print(f"Input file: {input_file}")
    print(f"Sample spacing: {sample_spacing_km * 1000:.2f} m")

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

    # Create distance axis: 点数 × 采样间距
    distance_axis = create_distance_axis(len(test_data), sample_spacing_km)
    print(
        f"Total fiber length: {distance_axis[-1]:.3f} km ({distance_axis[-1] * 1000:.1f} m)"
    )

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
                baseline_distance = create_distance_axis(
                    len(baseline_data),
                    sample_spacing_km,
                )
                baseline_data = np.interp(
                    distance_axis,
                    baseline_distance,
                    baseline_data,
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
        trace_db=test_data,
        baseline=baseline_data,
        config=detector_config,
        sample_spacing_km=sample_spacing_km,
    )

    result = detector.detect(test_data)
    print(f"Detected {len(result.events)} raw events")

    # Cluster events
    cluster_distance = config.detection.distance_cluster_m if config else 5.0

    clustered_events = cluster_events(
        result.events,
        distance_threshold_m=cluster_distance,
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

        print("\nVisualization complete!")
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


def _print_event_summary(events: list[DetectedEvent]) -> None:
    """Print summary of detected events."""
    if not events:
        print("\n✅ No events detected - fiber appears to be in good condition!")
        return

    print("\n📊 Event Summary:")
    print(
        f"{'Type':<15} {'Position (km)':<12} {'Loss (dB)':<10} {'Reflection (dB)':<15}",
    )
    print(f"{'-' * 60}")

    for event in sorted(events, key=lambda e: e.z_km):
        print(
            f"{event.kind:<15} {event.z_km:<18.6f} {event.magnitude_db:<10.3f} {event.reflect_db:<15.3f}",
        )


def main():
    """Main entry point for standalone execution."""
    parser = argparse.ArgumentParser(description="FiberWatch OTDR Visualization")
    parser.add_argument("--input_file", type=Path, help="Input OTDR data file")
    parser.add_argument("--baseline", type=Path, help="Baseline reference file")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default="output",
        help="Output directory",
    )
    parser.add_argument(
        "--sample-spacing",
        type=float,
        default=0.0025545,
        help="Sample spacing in km (default: 0.0025545 km ≈ 2.55 m)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save plots to files",
    )

    args = parser.parse_args()

    run_visualization(
        input_file=args.input_file,
        baseline_file=args.baseline,
        output_dir=args.output,
        sample_spacing_km=args.sample_spacing,
        save_plots=not args.no_save,
    )


if __name__ == "__main__":
    main()
