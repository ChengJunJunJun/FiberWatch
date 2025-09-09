"""
Web interface components for FiberWatch OTDR analysis.

This module provides the Streamlit-based web interface for
interactive OTDR data analysis and visualization.
"""

from .app import main as run_web_app

__all__ = ["run_web_app"]
