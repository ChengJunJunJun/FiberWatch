"""Run detector accuracy evaluation on curated local datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fiberwatch.utils import evaluate_detector_on_dataset

DEFAULT_LABEL_MAP = {
    "nc-break": {"break"},
    "bend": {"bend"},
    "dirty_connector": {"dirty_connector"},
}

SKIP_LABELS = {"none", "connector_apc", "connector_upc", "splice"}


def _pick_baseline_file(dataset_root: Path) -> Path:
    candidates = sorted((dataset_root / "none").glob("*"))
    if not candidates:
        raise FileNotFoundError(
            "No baseline files found under 'none' subdirectory. "
            "Provide --baseline manually."
        )
    return candidates[0]


def _format_event_set(events: set[str]) -> str:
    if not events:
        return "<none>"
    return ", ".join(sorted(events))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate detector accuracy on dataset"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data"),
        help="Root directory containing labeled subfolders",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Baseline trace file (defaults to first file in 'none' subfolder)",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=20.0,
        help="Total fiber length in km for distance axis construction",
    )
    parser.add_argument(
        "--sample-rate-per-km",
        type=float,
        help="Sampling rate expressed as samples per kilometer",
    )
    args = parser.parse_args(argv)

    dataset_root = args.dataset_root
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    baseline_file = args.baseline or _pick_baseline_file(dataset_root)

    print("Dataset root:", dataset_root)
    print("Baseline file:", baseline_file)
    print("Evaluating labels:", ", ".join(sorted(DEFAULT_LABEL_MAP)))
    print()

    result = evaluate_detector_on_dataset(
        dataset_root=dataset_root,
        total_distance_km=args.distance,
        label_map=DEFAULT_LABEL_MAP,
        skip_labels=SKIP_LABELS,
        baseline_path=baseline_file,
        sample_rate_per_km=args.sample_rate_per_km,
    )

    print(f"Processed files: {result.total_files}")
    print(f"Evaluated files: {result.evaluated_files}")
    print(f"Skipped files: {result.skipped_files}")
    print(f"Overall accuracy: {result.overall_accuracy:.3f}")
    print(f"Precision: {result.precision:.3f}")
    print(f"Recall: {result.recall:.3f}")
    print(f"F1 score: {result.f1_score:.3f}")
    print()

    if result.per_label:
        print("Per-label metrics:")
        for label in sorted(result.per_label):
            stats = result.per_label[label]
            print(
                f"  {label:>15}: {stats['correct']:>3}/{stats['total']:<3}"
                f" acc={stats['accuracy']:.3f}"
                f" P={stats.get('precision', 0.0):.3f}"
                f" R={stats.get('recall', 0.0):.3f}"
                f" F1={stats.get('f1', 0.0):.3f}"
            )
        print()

    tracked_labels = set(DEFAULT_LABEL_MAP)
    errors = [
        sample
        for sample in result.samples
        if sample.label in tracked_labels
        and not sample.is_correct
        and sample.error is None
    ]
    failures = [
        sample
        for sample in result.samples
        if sample.label in tracked_labels and sample.error is not None
    ]

    if errors:
        print("Misclassified files:")
        for sample in errors:
            print(
                f"  {sample.path}: expected {_format_event_set(sample.expected)}"
                f", predicted {_format_event_set(sample.predicted)}"
            )
        print()
    else:
        print("No misclassified files.\n")

    if failures:
        print("Files with errors during evaluation:")
        for sample in failures:
            print(f"  {sample.path}: {sample.error}")
    elif not errors:
        print("All evaluated files processed successfully.")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
