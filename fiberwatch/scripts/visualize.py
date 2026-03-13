"""
OTDR data visualization script.

This script provides command-line visualization of OTDR data
with comprehensive analysis plots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fiberwatch.core.models import DetectedEvent

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fiberwatch.core import Detector
from fiberwatch.core.models import DetectionResult
from fiberwatch.utils.data_io import create_distance_axis, load_test_data
from fiberwatch.utils.visualization import (
    create_bend_comparison_plot,
    save_all_plots,
)


# ── 数据加载 ─────────────────────────────────────────────────────


def _load_trace_data(
    input_file: Path,
    sample_spacing_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """加载 OTDR 测试数据并创建距离轴。"""
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    test_data = load_test_data(input_file)
    distance_axis = create_distance_axis(len(test_data), sample_spacing_km)
    print(f"Loaded {len(test_data)} data points")
    print(
        f"Total fiber length: {distance_axis[-1]:.3f} km "
        f"({distance_axis[-1] * 1000:.1f} m)"
    )
    return test_data, distance_axis


def _load_baseline(
    baseline_file: Path | None,
    distance_axis: np.ndarray,
    sample_spacing_km: float,
    n_points: int,
) -> tuple[np.ndarray | None, bool]:
    """加载基线数据，必要时插值对齐。返回 (baseline_data, baseline_provided)。"""
    if baseline_file is None or not baseline_file.exists():
        if baseline_file is not None:
            print(f"Warning: Baseline file not found: {baseline_file}")
        return None, False

    print(f"Loading baseline from: {baseline_file}")
    baseline_data = load_test_data(baseline_file)

    if len(baseline_data) != n_points:
        baseline_distance = create_distance_axis(len(baseline_data), sample_spacing_km)
        baseline_data = np.interp(distance_axis, baseline_distance, baseline_data)
        print("Baseline interpolated to match test data length")

    return baseline_data, True


# ── CNN 弯折检测 ─────────────────────────────────────────────────


def _run_cnn_bend_detection(
    cnn_file: Path,
    test_data: np.ndarray,
    distance_axis: np.ndarray,
    baseline_data: np.ndarray | None,
    sample_spacing_km: float,
    original_bend_events: list[DetectedEvent],
) -> tuple[list[DetectedEvent], DetectionResult | None]:
    """对 CNN 数据在原始弯折位置附近做局部弯折检测。"""
    print(f"\nLoading CNN-processed data from: {cnn_file}")
    cnn_data = load_test_data(cnn_file)
    print(f"Loaded {len(cnn_data)} CNN data points")

    # 插值对齐
    cnn_distance = create_distance_axis(len(cnn_data), sample_spacing_km)
    if len(cnn_data) != len(test_data):
        cnn_data = np.interp(distance_axis, cnn_distance, cnn_data)
        print("CNN data interpolated to match original data length")

    # 创建 CNN 检测器
    cnn_detector = Detector(
        trace_db=cnn_data,
        baseline=baseline_data,
        sample_spacing_km=sample_spacing_km,
    )

    cnn_y = np.asarray(cnn_data, dtype=float)
    cnn_all_peaks = cnn_detector._find_peaks(cnn_y)
    cnn_peak_indices = [p["index"] for p in cnn_all_peaks]
    step = max(1, cnn_detector._step_window_samples // 4)

    print(
        f"Found {len(original_bend_events)} bend events in original data, "
        f"running local bend detection on CNN data..."
    )

    # 在每个原始弯折位置附近做局部检测
    cnn_bend_events: list[DetectedEvent] = []
    for orig_bend in original_bend_events:
        bend_pos_km = orig_bend.z_km
        start_idx = max(0, int((bend_pos_km - 0.05) / sample_spacing_km))
        end_idx = min(len(cnn_y) - 1, int((bend_pos_km + 0.05) / sample_spacing_km))
        if end_idx <= start_idx:
            continue

        local_events: list[DetectedEvent] = []
        cnn_detector._detect_bend(
            cnn_y, start_idx, end_idx, step,
            cnn_peak_indices, cnn_all_peaks, local_events,
            drop_fraction=1.5,
        )
        cnn_bend_events.extend(local_events)

    # 打印结果
    if cnn_bend_events:
        print(f"CNN local bend detection found {len(cnn_bend_events)} bend events")
        for ev in cnn_bend_events:
            print(f"  CNN Bend @{ev.z_km * 1000:.1f}m (loss: {ev.magnitude_db:.2f}dB)")
    else:
        print("CNN local bend detection found no bend events")

    # 构建 CNN 结果用于可视化
    if cnn_detector.baseline is None:
        cnn_baseline = cnn_detector._fit_linear_baseline(cnn_detector.z, cnn_y)
    else:
        cnn_baseline = cnn_detector.baseline.copy()

    cnn_result = DetectionResult(
        events=cnn_bend_events,
        distance_km=cnn_detector.z,
        trace_db=cnn_y,
        baseline_db=cnn_baseline,
        residual_db=cnn_y - cnn_baseline,
        reflection_peaks=[],
    )

    return cnn_bend_events, cnn_result


# ── 结果输出 ─────────────────────────────────────────────────────


def _save_visualization(
    result: DetectionResult,
    final_events: list[DetectedEvent],
    original_bend_events: list[DetectedEvent],
    cnn_bend_events: list[DetectedEvent],
    cnn_result: DetectionResult | None,
    filename: str,
    output_dir: Path,
    baseline_provided: bool,
) -> None:
    """保存所有可视化图表。"""
    has_cnn_bends = bool(cnn_bend_events)
    result_source = "Original + CNN Bend" if has_cnn_bends else "Original"
    print(f"\nGenerating visualization plots (source: {result_source})...")

    saved_files = save_all_plots(
        result=result,
        clustered_events=final_events,
        filename=filename,
        output_dir=output_dir,
        reference_provided=baseline_provided,
    )

    # CNN 弯折对比图
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
            cmp_path = output_dir / f"{filename}_bend_comparison.png"
            fig_cmp.savefig(cmp_path, dpi=200, bbox_inches="tight")
            saved_files.append(cmp_path)
            plt.close(fig_cmp)
            print(f"Saved bend comparison plot: {cmp_path.name}")

    print("\nVisualization complete!")
    print(f"Saved {len(saved_files)} plots to: {output_dir}")
    for file_path in saved_files:
        print(f"  • {file_path.name}")


def _print_event_summary(events: list[DetectedEvent], result: DetectionResult | None = None) -> None:
    """Print summary of detected events."""
    if not events:
        print("\nNo events detected - fiber appears to be in good condition!")
    else:
        print("\nEvent Summary:")
        print(
            f"{'Type':<15} {'Position (km)':<18} {'Loss (dB)':<10} {'Reflection (dB)':<15}",
        )
        print(f"{'-' * 60}")

        for event in sorted(events, key=lambda e: e.z_km):
            print(
                f"{event.kind:<15} {event.z_km:<18.6f} "
                f"{event.magnitude_db:<10.3f} {event.reflect_db:<15.3f}",
            )

    if result and result.reflection_peaks:
        print(f"\nNormal Reflection Peaks ({len(result.reflection_peaks)}):")
        print(f"{'Position (km)':<18}  {'Height (dB)':<12}")
        print(f"{'-' * 30}")
        for peak in result.reflection_peaks:
            peak_z = float(result.distance_km[peak["index"]])
            height = peak.get("peak_height_db", float("nan"))
            print(f"{peak_z:<18.6f} {height:<12.1f}")


# ── 主流程 ───────────────────────────────────────────────────────


def run_visualization(
    input_file: Path,
    baseline_file: Path | None = None,
    cnn_file: Path | None = None,
    output_dir: Path = Path("output"),
    sample_spacing_km: float = 0.0025545,
    save_plots: bool = True,
) -> dict:
    """运行 OTDR 可视化分析。"""
    print("FiberWatch OTDR Visualization")
    print(f"Input file: {input_file}")
    if cnn_file:
        print(f"CNN file: {cnn_file}")
    print(f"Sample spacing: {sample_spacing_km * 1000:.2f} m")

    if save_plots:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    test_data, distance_axis = _load_trace_data(input_file, sample_spacing_km)
    baseline_data, baseline_provided = _load_baseline(
        baseline_file, distance_axis, sample_spacing_km, len(test_data),
    )

    # 2. 原始检测
    print("Running event detection on original data...")
    detector = Detector(
        trace_db=test_data,
        baseline=baseline_data,
        sample_spacing_km=sample_spacing_km,
    )
    result = detector.detect(test_data)
    print(f"Detected {len(result.events)} raw events (original)")

    clustered_events = result.events
    original_bend_events = [e for e in clustered_events if e.kind == "bend"]

    # 3. CNN 弯折检测（可选）
    cnn_bend_events: list[DetectedEvent] = []
    cnn_result: DetectionResult | None = None

    if cnn_file and cnn_file.exists() and original_bend_events:
        cnn_bend_events, cnn_result = _run_cnn_bend_detection(
            cnn_file, test_data, distance_axis,
            baseline_data, sample_spacing_km, original_bend_events,
        )
    elif cnn_file and not cnn_file.exists():
        print(f"Warning: CNN file not found: {cnn_file}")
    elif cnn_file and cnn_file.exists() and not original_bend_events:
        print(
            "\nCNN file provided but no bend events in original detection, "
            "skipping CNN"
        )

    # 4. 合并最终结果
    has_cnn_bends = bool(cnn_bend_events)
    if has_cnn_bends:
        non_bend_events = [e for e in clustered_events if e.kind != "bend"]
        final_events = sorted(non_bend_events + cnn_bend_events, key=lambda e: e.z_km)
        result_source = "Original + CNN Bend"
    else:
        final_events = clustered_events
        result_source = "Original"

    # 5. 输出
    if save_plots:
        _save_visualization(
            result, final_events, original_bend_events,
            cnn_bend_events, cnn_result,
            input_file.stem, output_dir, baseline_provided,
        )
    else:
        print("Visualization complete (plots not saved)")

    print(f"\n--- Event Summary ({result_source}) ---")
    _print_event_summary(final_events, result)

    return_dict: dict = {
        "detection_result": result,
        "clustered_events": final_events,
        "plots_saved": save_plots,
    }
    if has_cnn_bends:
        return_dict["cnn_bend_events"] = cnn_bend_events
        return_dict["original_bend_events"] = original_bend_events
        return_dict["cnn_result"] = cnn_result
    return return_dict


def main():
    """Main entry point for standalone execution."""
    parser = argparse.ArgumentParser(description="FiberWatch OTDR Visualization")
    parser.add_argument("--input_file", type=Path, help="Input OTDR data file")
    parser.add_argument("--baseline", type=Path, help="Baseline reference file")
    parser.add_argument(
        "--cnn-file", type=Path, default=None,
        help="CNN-processed OTDR data file (optional)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default="output",
        help="Output directory",
    )
    parser.add_argument(
        "--sample-spacing", type=float, default=0.0025545,
        help="Sample spacing in km (default: 0.0025545 km ≈ 2.55 m)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
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
