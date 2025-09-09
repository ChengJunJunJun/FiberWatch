"""
Configuration management for FiberWatch OTDR analysis.

This module provides centralized configuration management including
default settings, parameter validation, and environment-specific configs.
"""

from .settings import (
    AppConfig,
    DetectionConfig,
    WebConfig,
    get_default_config,
    load_config,
    save_config,
)

__all__ = [
    "AppConfig",
    "DetectionConfig",
    "WebConfig",
    "get_default_config",
    "load_config",
    "save_config",
]
