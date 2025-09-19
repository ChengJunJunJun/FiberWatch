"""
Configuration settings and management for FiberWatch.

This module defines all configurable parameters and provides
functions for loading, saving, and validating configurations.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
import os


@dataclass
class DetectionConfig:
    """Configuration for OTDR event detection algorithms."""

    # Smoothing parameters
    smooth_win: int = 21
    smooth_poly: int = 3

    # Event detection thresholds
    refl_min_db: float = 1.0
    step_min_db: float = 0.05
    slope_min_db_per_km: float = 0.02
    min_event_separation: int = 30

    # Break detection parameters
    pre_window_km: float = 0.30
    pre_end_offset_km: float = 0.025
    tail_start_offset_km: float = 0.05
    tail_end_offset_km: float = 0.55
    min_signal_drop_db: float = 5.0
    noise_floor_db: float = -80.0
    min_noise_increase: float = 1.5
    min_zero_crossing_ratio: float = 0.05
    min_tail_segment_len_km: float = 0.075
    grad_sigma_factor: float = 3.0
    min_grad_abs: float = 0.005

    # Dirty connector detection
    dirty_grad_sigma_factor: float = 6.0
    min_dirty_grad_abs: float = 0.001
    step_window_km: float = 0.15
    dirty_min_step_db: float = 1.5
    dirty_exclusion_before_break_km: float = 0.5
    dirty_duplicate_skip_km: float = 0.025

    # Bend detection
    bend_grad_sigma_factor: float = 2.0
    min_bend_grad_abs: float = 0.0005
    bend_pair_max_gap_km: float = 0.125
    bend_min_step_db: float = 0.05
    bend_max_step_db: float = 1.2
    bend_step_window_km: float = 0.075
    bend_min_descent_len_km: float = 0.05
    bend_dirty_exclusion_km: float = 0.5

    # Event clustering
    distance_cluster_m: float = 5.0

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.smooth_win <= 0 or self.smooth_win % 2 == 0:
            raise ValueError("smooth_win must be positive and odd")
        if self.smooth_poly < 1:
            raise ValueError("smooth_poly must be at least 1")
        if self.refl_min_db <= 0:
            raise ValueError("refl_min_db must be positive")
        if self.step_min_db <= 0:
            raise ValueError("step_min_db must be positive")
        if self.min_event_separation <= 0:
            raise ValueError("min_event_separation must be positive")
        if self.pre_window_km <= 0:
            raise ValueError("pre_window_km must be positive")
        if self.tail_end_offset_km <= self.tail_start_offset_km:
            raise ValueError("tail_end_offset_km must exceed tail_start_offset_km")
        if self.min_tail_segment_len_km <= 0:
            raise ValueError("min_tail_segment_len_km must be positive")
        if self.step_window_km <= 0:
            raise ValueError("step_window_km must be positive")
        if self.dirty_duplicate_skip_km <= 0:
            raise ValueError("dirty_duplicate_skip_km must be positive")
        if self.bend_min_descent_len_km <= 0:
            raise ValueError("bend_min_descent_len_km must be positive")


@dataclass
class WebConfig:
    """Configuration for Streamlit web interface."""

    page_title: str = "FiberWatch OTDR Analysis"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"

    # Default values for UI components
    default_distance_km: float = 20.0
    default_series_name: str = "analysis"
    default_output_dir: str = "output"

    # File upload constraints
    max_file_size_mb: int = 100
    allowed_extensions: list = None

    # Display settings
    plot_dpi: int = 200
    figure_width: int = 12
    figure_height: int = 8

    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = ["txt"]


@dataclass
class AppConfig:
    """Main application configuration container."""

    detection: DetectionConfig
    web: WebConfig

    # Application metadata
    version: str = "0.1.0"
    debug: bool = False

    # Paths
    data_dir: str = "data"
    output_dir: str = "output"
    config_dir: str = "config"

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    def validate(self) -> None:
        """Validate all configuration sections."""
        self.detection.validate()

        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid log_level: {self.log_level}")


def get_default_config() -> AppConfig:
    """
    Get default application configuration.

    Returns:
        Default AppConfig instance
    """
    return AppConfig(
        detection=DetectionConfig(),
        web=WebConfig(),
    )


def load_config(config_path: Union[str, Path], validate: bool = True) -> AppConfig:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to configuration file
        validate: Whether to validate the loaded config

    Returns:
        Loaded AppConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config_data = json.load(f)

    # Create config objects from loaded data
    detection_data = config_data.get("detection", {})
    web_data = config_data.get("web", {})

    detection_config = DetectionConfig(**detection_data)
    web_config = WebConfig(**web_data)

    # Remove detection and web from config_data to avoid duplication
    app_data = {k: v for k, v in config_data.items() if k not in ["detection", "web"]}

    app_config = AppConfig(detection=detection_config, web=web_config, **app_data)

    if validate:
        app_config.validate()

    return app_config


def save_config(config: AppConfig, config_path: Union[str, Path]) -> None:
    """
    Save configuration to JSON file.

    Args:
        config: AppConfig instance to save
        config_path: Path where to save the configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dictionary for JSON serialization
    config_dict = asdict(config)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)


def load_config_from_env() -> AppConfig:
    """
    Load configuration with environment variable overrides.

    Environment variables should be prefixed with FIBERWATCH_
    and use double underscores for nested keys.

    Example: FIBERWATCH_DETECTION__REFL_MIN_DB=1.5

    Returns:
        AppConfig with environment overrides applied
    """
    config = get_default_config()

    # Process environment variables
    for key, value in os.environ.items():
        if not key.startswith("FIBERWATCH_"):
            continue

        # Remove prefix and split on double underscores
        config_key = key[11:]  # Remove 'FIBERWATCH_'
        parts = config_key.lower().split("__")

        if len(parts) == 1:
            # Top-level config
            if hasattr(config, parts[0]):
                _set_config_value(config, parts[0], value)
        elif len(parts) == 2:
            # Nested config (detection.* or web.*)
            section, param = parts
            if section == "detection" and hasattr(config.detection, param):
                _set_config_value(config.detection, param, value)
            elif section == "web" and hasattr(config.web, param):
                _set_config_value(config.web, param, value)

    return config


def _set_config_value(obj: Any, attr: str, value: str) -> None:
    """
    Set configuration value with type conversion.

    Args:
        obj: Object to set attribute on
        attr: Attribute name
        value: String value from environment
    """
    if not hasattr(obj, attr):
        return

    current_value = getattr(obj, attr)

    # Convert string value to appropriate type
    if isinstance(current_value, bool):
        converted_value = value.lower() in ("true", "1", "yes", "on")
    elif isinstance(current_value, int):
        converted_value = int(value)
    elif isinstance(current_value, float):
        converted_value = float(value)
    elif isinstance(current_value, list):
        converted_value = value.split(",")
    else:
        converted_value = value

    setattr(obj, attr, converted_value)
