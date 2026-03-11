"""
Visualization utilities for OTDR data and analysis results.

This module provides functions for creating various types of plots
and visualizations for OTDR analysis.
"""

from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from ..core.detector import DetectedEvent, DetectionResult


def plot_raw_trace(
    z_km: np.ndarray,
    trace_db: np.ndarray,
    filename: str,
    output_dir: str = "output",
    *,
    save_plot: bool = False,
) -> plt.Figure:
    """
    Create a simple plot of just the raw OTDR trace.

    Args:
        z_km: Distance axis in kilometers
        trace_db: Trace power values in dB
        filename: Base filename for saving
        output_dir: Output directory for saving plots
        save_plot: Whether to save the plot to file

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(z_km * 1000, trace_db, "blue", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Return Power (dB)")
    ax.set_title(f"Raw OTDR Trace - {filename}")
    ax.grid(True, alpha=0.3)

    if save_plot:
        output_file = Path(output_dir) / f"{filename}_raw.png"
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        print(f"Saved raw trace plot to: {output_file}")

    plt.tight_layout()
    return fig


def create_analysis_plot(
    result: DetectionResult,
    clustered_events: List[DetectedEvent],
    filename: str,
    *,
    output_dir: str = "output",
    reference_provided: bool = False,
    save_plot: bool = True,
) -> plt.Figure:
    """
    Create detailed analysis plot with multiple subplots.

    Args:
        result: Detection result containing traces and events
        clustered_events: List of clustered events to mark
        filename: Base filename for saving
        output_dir: Output directory for saving plots
        reference_provided: Whether reference baseline was provided
        save_plot: Whether to save the plot to file

    Returns:
        Matplotlib figure object
    """
    z_km = result.distance_km
    n_subplots = 4 if reference_provided else 3
    figure_height = 12 if reference_provided else 10

    fig, axes = plt.subplots(n_subplots, 1, figsize=(15, figure_height))
    if n_subplots == 1:
        axes = [axes]

    # Subplot 1: Original traces
    ax1 = axes[0]
    ax1.plot(
        z_km * 1000,
        result.trace_smooth_db,
        "blue",
        label="Test Data (Smoothed)",
        linewidth=1,
    )
    ax1.plot(
        z_km * 1000,
        result.baseline_db,
        "red",
        "--" if reference_provided else "-",
        label="Reference Baseline" if reference_provided else "Fitted Baseline",
        linewidth=1,
    )

    add_event_markers(ax1, clustered_events)
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Return Power (dB)")
    ax1.set_title(f"OTDR Trace Analysis - {filename}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Residual/Difference
    ax2 = axes[1]
    ax2.plot(z_km * 1000, result.residual_db, "green", linewidth=1)
    ax2.axhline(0, color="black", linestyle="-", alpha=0.3)
    add_event_markers(ax2, clustered_events)
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Difference (dB)")
    title = (
        "Differential Analysis (Test - Reference)"
        if reference_provided
        else "Residual Analysis (Observation - Baseline)"
    )
    ax2.set_title(title)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: First derivative
    ax3 = axes[2]
    gradient = np.gradient(result.trace_smooth_db)
    ax3.plot(z_km * 1000, gradient, "purple", linewidth=1)
    ax3.axhline(0, color="black", linestyle="-", alpha=0.3)
    add_event_markers(ax3, clustered_events)
    ax3.set_xlabel("Distance (m)")
    ax3.set_ylabel("dP/dz (dB/sample)")
    ax3.set_title("First Derivative (for Detecting Sharp Changes)")
    ax3.grid(True, alpha=0.3)

    # Additional subplot for differential analysis
    if reference_provided and n_subplots > 3:
        ax4 = axes[3]
        diff_gradient = np.gradient(result.residual_db)
        ax4.plot(z_km * 1000, diff_gradient, "orange", linewidth=1)
        ax4.axhline(0, color="black", linestyle="-", alpha=0.3)
        add_event_markers(ax4, clustered_events)
        ax4.set_xlabel("Distance (m)")
        ax4.set_ylabel("d(Diff)/dz (dB/sample)")
        ax4.set_title("Differential Gradient (for Detecting New Events)")
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        output_file = Path(output_dir) / f"{filename}_analysis.png"
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        print(f"Saved detailed analysis to: {output_file}")

    return fig


def add_event_markers(ax: plt.Axes, events: List[DetectedEvent]) -> None:
    """
    Add vertical lines for detected events on a plot.

    Args:
        ax: Matplotlib axes object
        events: List of events to mark
    """
    color_map = {
        "reflection": "orange",
        "splice": "green",
        "bend": "purple",
        "break": "red",
        "dirty_connector": "brown",
        "clean_connector": "cyan",
    }

    for event in events:
        ax.axvline(
            event.z_km * 1000,
            color=color_map.get(event.kind, "black"),
            linestyle=":",
            alpha=0.8,
            linewidth=2,
        )


def fig_to_bytes(fig: plt.Figure) -> bytes:
    """
    Convert matplotlib figure to bytes for download.

    Args:
        fig: Matplotlib figure object

    Returns:
        Figure as bytes
    """
    bio = BytesIO()
    fig.savefig(bio, dpi=200, bbox_inches="tight", format="png")
    bio.seek(0)
    return bio.read()


def create_streamlit_analysis_figure(
    result: DetectionResult,
    clustered_events: List[DetectedEvent],
    filename: str,
    *,
    reference_provided: bool = False,
) -> plt.Figure:
    """
    Create detailed analysis plot optimized for Streamlit display.

    Args:
        result: Detection result containing traces and events
        clustered_events: List of clustered events to mark
        filename: Base filename for display
        reference_provided: Whether reference baseline was provided

    Returns:
        Matplotlib figure object with modern styling
    """
    z_km = result.distance_km
    n_subplots = 4 if reference_provided else 3
    figure_height = 12 if reference_provided else 10

    # Set modern style
    plt.style.use("default")
    fig = plt.figure(figsize=(12, figure_height * 0.8))
    fig.patch.set_facecolor("white")

    # Subplot 1: Original traces comparison
    ax1 = plt.subplot(n_subplots, 1, 1)
    ax1.plot(
        z_km * 1000,
        result.trace_smooth_db,
        color="#2563eb",
        label="Test Data (Smoothed)",
        linewidth=1.5,
    )
    ax1.plot(
        z_km * 1000,
        result.baseline_db,
        color="#dc2626",
        linestyle="--" if reference_provided else "-",
        label="Reference Baseline" if reference_provided else "Fitted Baseline",
        linewidth=1.5,
    )

    add_event_markers(ax1, clustered_events)
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Return Power (dB)")
    ax1.set_title(f"OTDR Trace Analysis - {filename}", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#fafafa")

    # Subplot 2: Residual/Difference analysis
    ax2 = plt.subplot(n_subplots, 1, 2)
    ax2.plot(z_km * 1000, result.residual_db, color="#059669", linewidth=1.5)
    ax2.axhline(0, color="black", linestyle="-", alpha=0.3)
    add_event_markers(ax2, clustered_events)
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Difference (dB)")
    title = (
        "Differential Analysis (Test - Reference)"
        if reference_provided
        else "Residual Analysis (Observation - Baseline)"
    )
    ax2.set_title(title, fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#fafafa")

    # Subplot 3: First derivative
    ax3 = plt.subplot(n_subplots, 1, 3)
    gradient = np.gradient(result.trace_smooth_db)
    ax3.plot(z_km * 1000, gradient, color="#7c3aed", linewidth=1.5)
    ax3.axhline(0, color="black", linestyle="-", alpha=0.3)
    add_event_markers(ax3, clustered_events)
    ax3.set_xlabel("Distance (m)")
    ax3.set_ylabel("dP/dz (dB/sample)")
    ax3.set_title(
        "First Derivative (Sharp Change Detection)",
        fontsize=14,
        fontweight="bold",
    )
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor("#fafafa")

    # Subplot 4: Differential gradient (when reference provided)
    if reference_provided:
        ax4 = plt.subplot(n_subplots, 1, 4)
        diff_gradient = np.gradient(result.residual_db)
        ax4.plot(z_km * 1000, diff_gradient, color="#ea580c", linewidth=1.5)
        ax4.axhline(0, color="black", linestyle="-", alpha=0.3)
        add_event_markers(ax4, clustered_events)
        ax4.set_xlabel("Distance (m)")
        ax4.set_ylabel("d(Diff)/dz (dB/sample)")
        ax4.set_title(
            "Differential Gradient (New Event Detection)",
            fontsize=14,
            fontweight="bold",
        )
        ax4.grid(True, alpha=0.3)
        ax4.set_facecolor("#fafafa")

    plt.tight_layout()
    return fig


def create_bend_comparison_plot(
    original_result: DetectionResult,
    cnn_result: DetectionResult,
    original_bend_events: List[DetectedEvent],
    cnn_bend_events: List[DetectedEvent],
    filename: str,
    *,
    window_m: float = 200.0,
) -> Optional[plt.Figure]:
    """
    Create local comparison plots between original and CNN bend detection.

    For each original bend event, shows a side-by-side local view of the
    original trace and CNN trace around the bend location, with detected
    bend positions marked on both.

    Args:
        original_result: Detection result from original data
        cnn_result: Detection result from CNN-processed data
        original_bend_events: Bend events detected on original data
        cnn_bend_events: Bend events from CNN local detection
        filename: Base filename for title
        window_m: Window size in meters around event for local view

    Returns:
        Matplotlib figure, or None if no original bend events
    """
    if not original_bend_events:
        return None

    n_events = len(original_bend_events)
    fig, axes = plt.subplots(n_events, 2, figsize=(16, 5 * n_events), squeeze=False)

    orig_z_m = original_result.distance_km * 1000
    cnn_z_m = cnn_result.distance_km * 1000

    for i, orig_event in enumerate(original_bend_events):
        event_m = orig_event.z_km * 1000
        lo = event_m - window_m
        hi = event_m + window_m

        # Left: Original trace local view
        ax_orig = axes[i, 0]
        orig_mask = (orig_z_m >= lo) & (orig_z_m <= hi)
        ax_orig.plot(
            orig_z_m[orig_mask],
            original_result.trace_db[orig_mask],
            "blue",
            linewidth=1,
            label="Original Trace",
        )

        ax_orig.axvline(
            event_m,
            color="purple",
            linestyle=":",
            linewidth=2,
            label=f"Bend @{event_m:.1f}m ({orig_event.magnitude_db:.2f}dB)",
        )
        ax_orig.set_xlim(lo, hi)
        # Auto-fit y-axis to local data range
        if orig_mask.any():
            local_vals = np.concatenate(
                [
                    original_result.trace_db[orig_mask],
                    original_result.baseline_db[orig_mask],
                ]
            )
            y_min, y_max = np.nanmin(local_vals), np.nanmax(local_vals)
            y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
            ax_orig.set_ylim(y_min - y_margin, y_max + y_margin)
        ax_orig.set_xlabel("Distance (m)")
        ax_orig.set_ylabel("Power (dB)")
        ax_orig.set_title(f"Original Algorithm - Bend #{i + 1}")
        ax_orig.legend(fontsize=8)
        ax_orig.grid(True, alpha=0.3)

        # Right: CNN trace local view
        ax_cnn = axes[i, 1]
        cnn_mask = (cnn_z_m >= lo) & (cnn_z_m <= hi)
        ax_cnn.plot(
            cnn_z_m[cnn_mask],
            cnn_result.trace_db[cnn_mask],
            "green",
            linewidth=1,
            label="CNN Trace",
        )

        # Mark all CNN bend events in this window
        for cnn_ev in cnn_bend_events:
            cnn_ev_m = cnn_ev.z_km * 1000
            if lo <= cnn_ev_m <= hi:
                ax_cnn.axvline(
                    cnn_ev_m,
                    color="purple",
                    linestyle=":",
                    linewidth=2,
                    label=f"CNN Bend @{cnn_ev_m:.1f}m ",
                )
        # Also mark original bend position for reference
        ax_cnn.axvline(
            event_m,
            color="blue",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label=f"Orig Bend @{event_m:.1f}m ",
        )
        ax_cnn.set_xlim(lo, hi)
        # Auto-fit y-axis to local data range
        if cnn_mask.any():
            local_vals = np.concatenate(
                [
                    cnn_result.trace_db[cnn_mask],
                    cnn_result.baseline_db[cnn_mask],
                ]
            )
            y_min, y_max = np.nanmin(local_vals), np.nanmax(local_vals)
            y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
            ax_cnn.set_ylim(y_min - y_margin, y_max + y_margin)
        ax_cnn.set_xlabel("Distance (m)")
        ax_cnn.set_ylabel("Power (dB)")
        ax_cnn.set_title(f"CNN Local Detection - Bend #{i + 1}")
        ax_cnn.legend(fontsize=8)
        ax_cnn.grid(True, alpha=0.3)

    fig.suptitle(
        f"Bend Detection Comparison (Original vs CNN) - {filename}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def save_all_plots(
    result: DetectionResult,
    clustered_events: List[DetectedEvent],
    filename: str,
    output_dir: Union[str, Path] = "output",
    *,
    reference_provided: bool = False,
) -> List[Path]:
    """
    Save all analysis plots to files.

    Args:
        result: Detection result containing traces and events
        clustered_events: List of clustered events to mark
        filename: Base filename for saving
        output_dir: Output directory for saving plots
        reference_provided: Whether reference baseline was provided

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    z_km = result.distance_km

    # 1. Raw trace plot
    fig_raw = plot_raw_trace(z_km, result.trace_smooth_db, filename, save_plot=False)
    raw_path = output_dir / f"{filename}_raw.png"
    fig_raw.savefig(raw_path, dpi=200, bbox_inches="tight")
    saved_files.append(raw_path)
    plt.close(fig_raw)

    # 2. Analysis plot
    fig_analysis = create_analysis_plot(
        result,
        clustered_events,
        filename,
        output_dir=str(output_dir),
        reference_provided=reference_provided,
        save_plot=False,
    )
    analysis_path = output_dir / f"{filename}_analysis.png"
    fig_analysis.savefig(analysis_path, dpi=200, bbox_inches="tight")
    saved_files.append(analysis_path)
    plt.close(fig_analysis)

    # 3. Simple plot
    fig_simple = result.plot()
    simple_path = output_dir / f"{filename}_simple.png"
    fig_simple.savefig(simple_path, dpi=160, bbox_inches="tight")
    saved_files.append(simple_path)
    plt.close(fig_simple)

    return saved_files
