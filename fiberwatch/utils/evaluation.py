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


def evaluate_detector_on_dataset(
    dataset_root: str | Path,
    *,
    total_distance_km: float = 20.0,
    detection_config: DetectorConfig | None = None,
    label_map: Mapping[str, Sequence[str] | set[str]] | None = None,
    skip_labels: Iterable[str] | None = None,
    baseline_path: str | Path | None = None,
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

    samples: list[SampleEvaluation] = []
    per_label: dict[str, dict[str, float | int]] = {}
    total_files = 0
    evaluated_files = 0
    skipped_files = 0

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
            expected_raw = mapping.get(label)
            if expected_raw is None:
                skipped_files += 1
                continue
            expected = _normalize_expected(expected_raw)
            try:
                trace = load_test_data(trace_path)
                distance_axis = create_distance_axis(len(trace), total_distance_km)
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
                    )
                else:
                    detector = Detector(distance_km=distance_axis, config=config)
                detection = detector.detect(trace)
                predicted = {event.kind for event in detection.events}
                if expected:
                    is_correct = any(kind in predicted for kind in expected)
                else:
                    is_correct = len(predicted) == 0
                evaluated_files += 1
                result = SampleEvaluation(
                    path=trace_path.relative_to(root),
                    label=label,
                    expected=expected,
                    predicted=predicted,
                    is_correct=is_correct,
                )
                samples.append(result)
                stats = per_label.setdefault(
                    label,
                    {"total": 0, "correct": 0, "accuracy": 0.0},
                )
                stats["total"] += 1
                if is_correct:
                    stats["correct"] += 1
            except Exception as exc:  # noqa: BLE001 - we want to capture all errors
                skipped_files += 1
                samples.append(
                    SampleEvaluation(
                        path=trace_path.relative_to(root),
                        label=label,
                        expected=_normalize_expected(expected_raw),
                        predicted=set(),
                        is_correct=False,
                        error=str(exc),
                    )
                )

    for stats in per_label.values():
        if stats["total"]:
            stats["accuracy"] = stats["correct"] / stats["total"]
        else:
            stats["accuracy"] = 0.0

    overall_accuracy = (
        sum(stats["correct"] for stats in per_label.values()) / evaluated_files
        if evaluated_files
        else 0.0
    )

    return DatasetEvaluation(
        total_files=total_files,
        evaluated_files=evaluated_files,
        skipped_files=skipped_files,
        overall_accuracy=overall_accuracy,
        per_label=per_label,
        samples=samples,
    )
