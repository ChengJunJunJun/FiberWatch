"""
Utility functions for OTDR data processing and analysis.

This module contains helper functions for data loading, processing,
visualization, and event clustering.
"""

from .data_io import create_distance_axis, load_test_data
from .evaluation import (
    DatasetEvaluation,
    SampleEvaluation,
    evaluate_detector_on_dataset,
)
from .event_processing import cluster_events, select_best_event_in_cluster
from .visualization import (
    plot_raw_trace,
    create_analysis_plot,
    add_event_markers,
    fig_to_bytes,
)

__all__ = [
    "load_test_data",
    "create_distance_axis",
    "evaluate_detector_on_dataset",
    "DatasetEvaluation",
    "SampleEvaluation",
    "cluster_events",
    "select_best_event_in_cluster",
    "plot_raw_trace",
    "create_analysis_plot",
    "add_event_markers",
    "fig_to_bytes",
]
