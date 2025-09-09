"""
Core detection algorithms and data structures for OTDR analysis.

This module contains the main detection engine and related data structures
for analyzing OTDR traces and detecting various types of fiber events.
"""

from .detector import Detector, DetectionResult, DetectedEvent, DetectorConfig
from .simulation import make_synthetic_otdr

__all__ = [
    "Detector",
    "DetectionResult",
    "DetectedEvent",
    "DetectorConfig",
    "make_synthetic_otdr",
]
