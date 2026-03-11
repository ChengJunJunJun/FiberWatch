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

import matplotlib.pyplot as plt
import numpy as np

from fiberwatch.core import Detector, DetectorConfig
from fiberwatch.utils.data_io import create_distance_axis, load_test_data
from fiberwatch.utils.visualization import (
    create_bend_comparison_plot,
    save_all_plots,
)


def run_visualization(
    input_file: Path,
    baseline_file: Path | None = None,
    cnn_file: Path | None = None,
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
        cnn_file: Optional path to CNN-processed OTDR data file
        output_dir: Output directory for plots
        sample_spacing_km: Distance between samples in kilometers (采样间距)
        save_plots: Whether to save plots to files
        config: Application configuration

    """
    print("FiberWatch OTDR Visualization")
    print(f"Input file: {input_file}")
    if cnn_file:
        print(f"CNN file: {cnn_file}")
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

    # Run detection on original data
    print("Running event detection on original data...")
    detector = Detector(
        trace_db=test_data,
        baseline=baseline_data,
        config=detector_config,
        sample_spacing_km=sample_spacing_km,
    )

    result = detector.detect(test_data)
    print(f"Detected {len(result.events)} raw events (original)")

    clustered_events = result.events

    # CNN local bend detection
    # 逻辑：先对原始数据做完整检测，如果发现弯折事件且提供了CNN文件，
    # 则只在弯折位置附近对CNN数据做局部弯曲检测（不做全量检测，避免其他位置干扰）
    cnn_bend_events = []
    cnn_data = None
    cnn_detector = None
    cnn_result = None
    original_bend_events = [e for e in clustered_events if e.kind == "bend"]

    if cnn_file and cnn_file.exists() and original_bend_events:
        print(f"\nLoading CNN-processed data from: {cnn_file}")
        cnn_data = load_test_data(cnn_file)
        print(f"Loaded {len(cnn_data)} CNN data points")

        # Interpolate CNN data to match original data length if needed
        cnn_distance = create_distance_axis(len(cnn_data), sample_spacing_km)
        if len(cnn_data) != len(test_data):
            cnn_data = np.interp(distance_axis, cnn_distance, cnn_data)
            print("CNN data interpolated to match original data length")

        # Create CNN detector (for internal state setup only)
        cnn_detector = Detector(
            trace_db=cnn_data,
            baseline=baseline_data,
            config=detector_config,
            sample_spacing_km=sample_spacing_km,
        )

        # Prepare CNN data internal state for local bend detection
        cnn_y = np.asarray(cnn_data, dtype=float)
        cnn_all_peaks = cnn_detector._find_peaks(cnn_y)
        cnn_peak_indices = [p["index"] for p in cnn_all_peaks]
        step = max(1, cnn_detector._step_window_samples // 4)

        print(
            f"Found {len(original_bend_events)} bend events in original data, "
            f"running local bend detection on CNN data..."
        )

        # For each original bend event, run local bend detection on CNN data
        # Search region: 100m before the bend event position
        for orig_bend in original_bend_events:
            bend_pos_km = orig_bend.z_km
            search_start_km = max(0, bend_pos_km - 0.05)  # 50m before
            search_end_km = bend_pos_km + 0.05  # 50m after

            start_idx = max(0, int(search_start_km / sample_spacing_km))
            end_idx = min(len(cnn_y) - 1, int(search_end_km / sample_spacing_km))

            if end_idx <= start_idx:
                continue

            local_events = []
            cnn_detector._detect_bend(
                cnn_y, start_idx, end_idx, step,
                cnn_peak_indices, cnn_all_peaks, local_events,
                drop_fraction=1.5,  # CNN用2/3下降点+最陡斜率定位弯折
            )
            cnn_bend_events.extend(local_events)

        if cnn_bend_events:
            print(f"CNN local bend detection found {len(cnn_bend_events)} bend events")
            for ev in cnn_bend_events:
                print(
                    f"  CNN Bend @{ev.z_km * 1000:.1f}m "
                    f"(loss: {ev.magnitude_db:.2f}dB)"
                )
        else:
            print("CNN local bend detection found no bend events")

        # Build CNN result for visualization (using CNN trace data + baseline)
        if cnn_detector.baseline is None:
            cnn_baseline = cnn_detector._fit_linear_baseline(
                cnn_detector.z, cnn_y,
            )
        else:
            cnn_baseline = cnn_detector.baseline.copy()

        from fiberwatch.core.detector import DetectionResult
        cnn_result = DetectionResult(
            events=cnn_bend_events,
            distance_km=cnn_detector.z,
            trace_db=cnn_y,
            baseline_db=cnn_baseline,
            residual_db=cnn_y - cnn_baseline,
            reflection_peaks=[],
        )
    elif cnn_file and not cnn_file.exists():
        print(f"Warning: CNN file not found: {cnn_file}")
    elif cnn_file and cnn_file.exists() and not original_bend_events:
        print(f"\nCNN file provided but no bend events in original detection, skipping CNN")

    # Determine final result:
    # - If CNN provided and has bend results: replace original bend events with CNN bends
    # - Otherwise: use original results
    has_cnn_bends = bool(cnn_bend_events)
    if has_cnn_bends:
        # Replace original bend events with CNN bend events in final output
        non_bend_events = [e for e in clustered_events if e.kind != "bend"]
        final_clustered_events = sorted(
            non_bend_events + cnn_bend_events, key=lambda e: e.z_km,
        )
        final_result = result  # Use original result for main plots
        result_source = "Original + CNN Bend"
    else:
        final_clustered_events = clustered_events
        final_result = result
        result_source = "Original"

    # Generate filename
    filename = input_file.stem

    if save_plots:
        # Save all plots (using final result)
        print(f"\nGenerating visualization plots (source: {result_source})...")
        saved_files = save_all_plots(
            result=final_result,
            clustered_events=final_clustered_events,
            filename=filename,
            output_dir=output_dir,
            reference_provided=baseline_provided,
        )

        # Generate bend comparison plot if CNN local bend detection was performed
        if has_cnn_bends and cnn_result is not None:
            print("Generating bend comparison plot...")
            fig_cmp = create_bend_comparison_plot(
                original_result=result,
                cnn_result=cnn_result,
                original_bend_events=original_bend_events,
                cnn_bend_events=cnn_bend_events,
                filename=filename,
            )
            if fig_cmp is not None:
                cmp_path = Path(output_dir) / f"{filename}_bend_comparison.png"
                fig_cmp.savefig(cmp_path, dpi=200, bbox_inches="tight")
                saved_files.append(cmp_path)
                plt.close(fig_cmp)
                print(f"Saved bend comparison plot: {cmp_path.name}")

        print("\nVisualization complete!")
        print(f"Saved {len(saved_files)} plots to: {output_dir}")
        for file_path in saved_files:
            print(f"  • {file_path.name}")
    else:
        # Just display results without saving
        print("Visualization complete (plots not saved)")

    # Print event summary (using final result)
    print(f"\n--- Event Summary ({result_source}) ---")
    _print_event_summary(final_clustered_events, final_result)

    return_dict = {
        "detection_result": final_result,
        "clustered_events": final_clustered_events,
        "plots_saved": save_plots,
    }
    if has_cnn_bends:
        return_dict["cnn_bend_events"] = cnn_bend_events
        return_dict["original_bend_events"] = original_bend_events
        return_dict["cnn_result"] = cnn_result
    return return_dict


def _print_event_summary(events: list[DetectedEvent], result=None) -> None:
    """Print summary of detected events."""
    if not events:
        print("\n✅ No events detected - fiber appears to be in good condition!")
    else:
        print("\n📊 Event Summary:")
        print(
            f"{'Type':<15} {'Position (km)':<12} {'Loss (dB)':<10} {'Reflection (dB)':<15}",
        )
        print(f"{'-' * 60}")

        for event in sorted(events, key=lambda e: e.z_km):
            print(
                f"{event.kind:<15} {event.z_km:<18.6f} {event.magnitude_db:<10.3f} {event.reflect_db:<15.3f}",
            )

    # Print reflection peaks
    if result and result.reflection_peaks:
        print(f"\n🔹Normal Reflection Peaks ({len(result.reflection_peaks)}):")
        print(f"{'Position (km)':<18}  {'Height (dB)':<12}")
        print(f"{'-' * 30}")
        for peak in result.reflection_peaks:
            peak_z = float(result.distance_km[peak["index"]])
            height = peak.get("peak_height_db", float("nan"))
            print(f"{peak_z:<18.6f} {height:<12.1f}")


def main():
    """Main entry point for standalone execution."""
    parser = argparse.ArgumentParser(description="FiberWatch OTDR Visualization")
    parser.add_argument("--input_file", type=Path, help="Input OTDR data file")
    parser.add_argument("--baseline", type=Path, help="Baseline reference file")
    parser.add_argument(
        "--cnn-file",
        type=Path,
        default=None,
        help="CNN-processed OTDR data file (optional)",
    )
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
        cnn_file=args.cnn_file,
        output_dir=args.output,
        sample_spacing_km=args.sample_spacing,
        save_plots=not args.no_save,
    )


if __name__ == "__main__":
    main()
