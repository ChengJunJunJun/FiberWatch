"""
FiberWatch - OTDR Event Detection System.

A professional optical time-domain reflectometry (OTDR) analysis toolkit
for detecting and analyzing fiber optic events including breaks, dirty connectors,
bends, splices, and other anomalies.

Main modules:
- core: Core detection algorithms and data structures
- web: Streamlit web interface
- utils: Utility functions for data processing and visualization
- config: Configuration management
"""

__version__ = "0.1.0"
__author__ = "Jun Cheng,Yuxuan Chen"
__description__ = "OTDR Event Detection System"

from .core import DetectedEvent, DetectionResult, Detector, DetectorConfig

__all__ = [
    "DetectedEvent",
    "DetectionResult",
    "Detector",
    "DetectorConfig",
    "__author__",
    "__description__",
    "__version__",
]
