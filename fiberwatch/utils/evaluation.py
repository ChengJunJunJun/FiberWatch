"""Dataset evaluation utilities for the FiberWatch detector."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..core import Detector, DetectorConfig
from .data_io import create_distance_axis, load_test_data


def _normalize_expected(value: Sequence[str] | set[str]) -> set[str]:
    if isinstance(value, str):
        return {value}
    return set(value)


def _is_positive_prediction(predicted: set[str], expected: set[str]) -> bool:
    """Check if predictions match expectation for a label."""
    if expected:
        return bool(predicted & expected)
    return len(predicted) == 0


# Event labels supported by default. Values are sets of acceptable detector kinds.
DEFAULT_LABEL_MAP: Mapping[str, set[str]] = {
    "nc-break": {"break"},
    "dirty_connector": {"dirty_connector"},
    "bend": {"bend"},
    "none": set(),
    # Remaining dataset folders are left unsupported by default but may be supplied
    # through a custom label_map argument when calling the evaluator.
}


@dataclass(slots=True)
class SampleEvaluation:
    """Evaluation record for a single trace file."""

    path: Path
    label: str
    expected: set[str]
    predicted: set[str]
    is_correct: bool
    error: str | None = None


@dataclass(slots=True)
class DatasetEvaluation:
    """Aggregated evaluation metrics for a dataset."""

    total_files: int
    evaluated_files: int
    skipped_files: int
    overall_accuracy: float
    per_label: dict[str, dict[str, float | int]]
    samples: list[SampleEvaluation] = field(default_factory=list)
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0


def _compute_classification_metrics(
    counts: Mapping[str, Mapping[str, int]],
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 score from label-level counts."""

    true_positives = sum(stats.get("tp", 0) for stats in counts.values())
    false_positives = sum(stats.get("fp", 0) for stats in counts.values())
    false_negatives = sum(stats.get("fn", 0) for stats in counts.values())

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives)
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives)
        else 0.0
    )
    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )

    return precision, recall, f1_score


def evaluate_detector_on_dataset(
    dataset_root: str | Path,
    *,
    total_distance_km: float = 20.0,
    detection_config: DetectorConfig | None = None,
    label_map: Mapping[str, Sequence[str] | set[str]] | None = None,
    skip_labels: Iterable[str] | None = None,
    baseline_path: str | Path | None = None,
    sample_rate_per_km: float | None = None,
) -> DatasetEvaluation:
    """Evaluate detector accuracy on a labeled dataset directory.

    Args:
        dataset_root: Path to dataset root where subdirectories denote labels.
        total_distance_km: Distance span used to build the distance axis.
        detection_config: Optional detector configuration override.
        label_map: Mapping from directory names to acceptable detector kinds.
        skip_labels: Optional iterable of directory names to ignore entirely.

    Returns:
        DatasetEvaluation with per-label and overall accuracy metrics.
    """
    root = Path(dataset_root)
    if not root.is_dir():
        raise ValueError(f"Dataset root does not exist or is not a directory: {root}")

    config = detection_config or DetectorConfig()
    mapping: Mapping[str, Sequence[str] | set[str]] = label_map or DEFAULT_LABEL_MAP
    ignored = set(skip_labels or ())
    normalized_label_map: dict[str, set[str]] = {
        label: _normalize_expected(events) for label, events in mapping.items()
    }
    tracking_labels: dict[str, set[str]] = {
        label: events
        for label, events in normalized_label_map.items()
        if label not in ignored
    }
    metric_labels: dict[str, set[str]] = dict(tracking_labels)

    samples: list[SampleEvaluation] = []
    per_label: dict[str, dict[str, float | int]] = {
        label: {"total": 0, "correct": 0, "accuracy": 0.0} for label in tracking_labels
    }
    total_files = 0
    evaluated_files = 0
    skipped_files = 0
    classification_totals: dict[str, dict[str, int]] = {
        label: {"tp": 0, "fp": 0, "fn": 0} for label in metric_labels
    }

    baseline_trace: np.ndarray | None = None
    baseline_distance: np.ndarray | None = None
    if baseline_path is not None:
        baseline_path = Path(baseline_path)
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline file not found: {baseline_path}")
        baseline_trace = load_test_data(baseline_path)
        baseline_distance = create_distance_axis(len(baseline_trace), total_distance_km)

    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for trace_path in sorted(label_dir.glob("*")):
            if trace_path.is_dir():
                continue
            total_files += 1
            if label in ignored:
                skipped_files += 1
                continue
            expected = tracking_labels.get(label)
            if expected is None:
                skipped_files += 1
                continue
            try:
                trace = load_test_data(trace_path)
                distance_axis = create_distance_axis(len(trace), total_distance_km)

                # Determine sampling rate for this trace
                if sample_rate_per_km and sample_rate_per_km > 0:
                    effective_rate = float(sample_rate_per_km)
                elif total_distance_km > 0:
                    effective_rate = len(trace) / float(total_distance_km)
                else:
                    effective_rate = None

                baseline = None
                if baseline_trace is not None:
                    baseline = baseline_trace
                    if len(baseline) != len(trace):
                        baseline = np.interp(
                            distance_axis,
                            baseline_distance,
                            baseline,
                        )

                detector = Detector(
                    distance_km=distance_axis,
                    baseline=baseline,
                    config=config,
                    sample_rate_per_km=effective_rate,
                )

                detection = detector.detect(
                    trace,
                    sample_rate_per_km=effective_rate,
                )
                predicted = {event.kind for event in detection.events}
                is_correct = _is_positive_prediction(predicted, expected)
                evaluated_files += 1
                result = SampleEvaluation(
                    path=trace_path.relative_to(root),
                    label=label,
                    expected=expected,
                    predicted=predicted,
                    is_correct=is_correct,
                )
                samples.append(result)
                stats = per_label[label]
                stats["total"] += 1
                if is_correct:
                    stats["correct"] += 1

                if metric_labels:
                    for metric_label, events in metric_labels.items():
                        predicted_positive = _is_positive_prediction(predicted, events)
                        actual_positive = label == metric_label
                        if predicted_positive and actual_positive:
                            classification_totals[metric_label]["tp"] += 1
                        elif predicted_positive and not actual_positive:
                            classification_totals[metric_label]["fp"] += 1
                        elif actual_positive and not predicted_positive:
                            classification_totals[metric_label]["fn"] += 1
            except Exception as exc:  # noqa: BLE001 - we want to capture all errors
                skipped_files += 1
                samples.append(
                    SampleEvaluation(
                        path=trace_path.relative_to(root),
                        label=label,
                        expected=expected,
                        predicted=set(),
                        is_correct=False,
                        error=str(exc),
                    )
                )

    for label, stats in per_label.items():
        if stats["total"]:
            stats["accuracy"] = stats["correct"] / stats["total"]
        else:
            stats["accuracy"] = 0.0
        counts = classification_totals.get(label)
        if counts:
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]
            stats["precision"] = tp / (tp + fp) if (tp + fp) else 0.0
            stats["recall"] = tp / (tp + fn) if (tp + fn) else 0.0
            stats["f1"] = (
                2
                * stats["precision"]
                * stats["recall"]
                / (stats["precision"] + stats["recall"])
                if (stats["precision"] + stats["recall"])
                else 0.0
            )
        else:
            stats["precision"] = 0.0
            stats["recall"] = 0.0
            stats["f1"] = 0.0

    overall_accuracy = (
        sum(stats["correct"] for stats in per_label.values()) / evaluated_files
        if evaluated_files
        else 0.0
    )

    precision, recall, f1_score = _compute_classification_metrics(
        classification_totals,
    )

    return DatasetEvaluation(
        total_files=total_files,
        evaluated_files=evaluated_files,
        skipped_files=skipped_files,
        overall_accuracy=overall_accuracy,
        per_label=per_label,
        samples=samples,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
    )
