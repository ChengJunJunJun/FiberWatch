"""
Core detection algorithms and data structures for OTDR analysis.

This module contains the main detection engine and related data structures
for analyzing OTDR traces and detecting various types of fiber events.
"""

from .detector import Detector, DetectionResult, DetectedEvent, DetectorConfig

__all__ = [
    "Detector",
    "DetectionResult",
    "DetectedEvent",
    "DetectorConfig",
]
