"""
Data input/output utilities for OTDR data files.

This module provides functions for loading OTDR data from various file formats
and creating distance axes for analysis.
"""

from pathlib import Path
from typing import Union
import numpy as np


def load_test_data(filepath: Union[str, Path]) -> np.ndarray:
    """
    Load OTDR test data from a text file.

    Args:
        filepath: Path to the data file containing single column dB values

    Returns:
        NumPy array of power values in dB

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If no valid data found in file
    """
    data = []
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    with filepath.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            stripped_line = line.strip()

            # Skip empty lines and comments
            if not stripped_line or stripped_line.startswith("#"):
                continue

            try:
                data.append(float(stripped_line))
            except ValueError:
                # Skip invalid lines but don't fail completely
                continue

    if not data:
        raise ValueError(f"No valid data found in file: {filepath}")

    return np.array(data, dtype=np.float64)


def create_distance_axis(n_samples: int, total_distance_km: float = 20.0) -> np.ndarray:
    """
    Create a linear distance axis for OTDR data.

    Args:
        n_samples: Number of samples in the trace
        total_distance_km: Total fiber length in kilometers

    Returns:
        NumPy array of distances in kilometers

    Raises:
        ValueError: If inputs are invalid
    """
    if n_samples <= 0:
        raise ValueError("Number of samples must be positive")
    if total_distance_km <= 0:
        raise ValueError("Total distance must be positive")

    return np.linspace(0, total_distance_km, n_samples)


def parse_uploaded_file(uploaded_file) -> np.ndarray:
    """
    Parse uploaded file content for Streamlit app.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        NumPy array of parsed data

    Raises:
        ValueError: If file cannot be parsed
    """
    try:
        text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        data = []

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                data.append(float(stripped))
            except ValueError:
                continue

        if not data:
            raise ValueError("No valid data found in uploaded file")

        return np.asarray(data, dtype=np.float64)

    except Exception as e:
        raise ValueError(f"Failed to parse uploaded file: {e}") from e


def save_detection_results(
    events: list, output_path: Union[str, Path], format: str = "csv"
) -> None:
    """
    Save detection results to file.

    Args:
        events: List of detected events
        output_path: Output file path
        format: Output format ('csv', 'json')

    Raises:
        ValueError: If format is not supported
    """
    import pandas as pd
    import json

    output_path = Path(output_path)

    # Convert events to dictionary format
    events_data = [
        {
            "event_type": ev.kind,
            "position_km": ev.z_km,
            "position_m": ev.z_km * 1000,
            "loss_db": ev.magnitude_db,
            "reflection_db": ev.reflect_db,
        }
        for ev in events
    ]

    if format.lower() == "csv":
        df = pd.DataFrame(events_data)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    elif format.lower() == "json":
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(events_data, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
