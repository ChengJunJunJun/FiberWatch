"""
Configuration settings and management for FiberWatch.

This module defines all configurable parameters. Detection parameters
are loaded from detection.yaml by default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml

# ── 加载 YAML 默认值 ─────────────────────────────────────────────

_YAML_PATH = Path(__file__).parent / "detection.yaml"

def _load_yaml_defaults() -> dict:
    """从 detection.yaml 读取默认参数。"""
    if _YAML_PATH.exists():
        with _YAML_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # YAML 中 list 需要转成 tuple（与 dataclass 字段类型一致）
        for key in ("peak_width_km", "small_peak_width_km"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        return data
    return {}


_YAML_DEFAULTS: dict = _load_yaml_defaults()


def _yd(key: str, fallback):
    """从 YAML 默认值字典中获取值，不存在则用 fallback。"""
    return _YAML_DEFAULTS.get(key, fallback)


# ── 检测配置 ─────────────────────────────────────────────────────

@dataclass
class DetectionConfig:
    """OTDR 事件检测算法的全部可调参数。

    默认值从 config/detection.yaml 加载；也可直接传参覆盖。
    """

    # ── 采样间距 ──
    sample_spacing_km: float = field(default_factory=lambda: _yd("sample_spacing_km", 0.0025545))

    # ── 反射峰相关 ──
    peak_local_region_km: float = field(default_factory=lambda: _yd("peak_local_region_km", 0.1))
    peak_prominence_std_factor: float = field(default_factory=lambda: _yd("peak_prominence_std_factor", 2.0))
    peak_width_km: tuple = field(default_factory=lambda: _yd("peak_width_km", (0.0128, 0.0383)))
    min_peak_height_db: float = field(default_factory=lambda: _yd("min_peak_height_db", 0.5))
    peak_min_prominence_db: float = field(default_factory=lambda: _yd("peak_min_prominence_db", 0.5))

    # ── 反射峰查找内部参数 ──
    peak_min_local_data_count: int = field(default_factory=lambda: _yd("peak_min_local_data_count", 10))
    peak_search_left_extra_km: float = field(default_factory=lambda: _yd("peak_search_left_extra_km", 0.0127725))
    peak_merge_distance_km: float = field(default_factory=lambda: _yd("peak_merge_distance_km", 0.0127725))
    peak_merge_height_diff_db: float = field(default_factory=lambda: _yd("peak_merge_height_diff_db", 0.1))

    # ── 反射峰高度阈值 ──
    peak_high_threshold_db: float = field(default_factory=lambda: _yd("peak_high_threshold_db", 10.9))
    peak_low_threshold_db: float = field(default_factory=lambda: _yd("peak_low_threshold_db", 8.0))

    # ── 峰左右基线差异判断 ──
    peak_step_match_rel_tol: float = field(default_factory=lambda: _yd("peak_step_match_rel_tol", 0.3))
    peak_step_match_abs_tol: float = field(default_factory=lambda: _yd("peak_step_match_abs_tol", 0.3))
    peak_no_step_threshold_db: float = field(default_factory=lambda: _yd("peak_no_step_threshold_db", 0.5))
    peak_bend_min_prominence_db: float = field(default_factory=lambda: _yd("peak_bend_min_prominence_db", 2))
    peak_step_search_range_km: float = field(default_factory=lambda: _yd("peak_step_search_range_km", 0.05))
    peak_step_local_min_range_km: float = field(default_factory=lambda: _yd("peak_step_local_min_range_km", 0.0383175))
    peak_step_match_tolerance_km: float = field(default_factory=lambda: _yd("peak_step_match_tolerance_km", 0.0076635))

    # ── 阶梯下降相关 ──
    step_drop_severe_db: float = field(default_factory=lambda: _yd("step_drop_severe_db", 1.3))
    step_drop_normal_db: float = field(default_factory=lambda: _yd("step_drop_normal_db", 0.2))
    step_compare_window_km: float = field(default_factory=lambda: _yd("step_compare_window_km", 0.05))
    step_min_slope_db_per_km: float = field(default_factory=lambda: _yd("step_min_slope_db_per_km", 10.0))

    # ── 宽范围阶梯下降检测 ──
    wide_step_offset_km: float = field(default_factory=lambda: _yd("wide_step_offset_km", 0.1))
    wide_step_window_km: float = field(default_factory=lambda: _yd("wide_step_window_km", 0.08))

    # ── 弯折检测 - 高台阶稳定性 ──
    bend_plateau_window_km: float = field(default_factory=lambda: _yd("bend_plateau_window_km", 0.05))
    bend_plateau_max_std_db: float = field(default_factory=lambda: _yd("bend_plateau_max_std_db", 1.0))
    bend_plateau_max_range_db: float = field(default_factory=lambda: _yd("bend_plateau_max_range_db", 1.5))
    plateau_min_gap_km: float = field(default_factory=lambda: _yd("plateau_min_gap_km", 0.0076635))
    plateau_min_data_count: int = field(default_factory=lambda: _yd("plateau_min_data_count", 5))

    # ── 噪声区域判断 ──
    noise_floor_db: float = field(default_factory=lambda: _yd("noise_floor_db", -25.0))
    noise_std_threshold: float = field(default_factory=lambda: _yd("noise_std_threshold", 2.0))
    noise_check_window_km: float = field(default_factory=lambda: _yd("noise_check_window_km", 0.128))
    severe_break_noise_std: float = field(default_factory=lambda: _yd("severe_break_noise_std", 3))
    noise_check_offset_km: float = field(default_factory=lambda: _yd("noise_check_offset_km", 0.0127725))
    min_noise_segment_count: int = field(default_factory=lambda: _yd("min_noise_segment_count", 10))

    # ── 严重断纤参数 ──
    severe_min_peak_db: float = field(default_factory=lambda: _yd("severe_min_peak_db", 5.0))
    severe_min_lr_diff_db: float = field(default_factory=lambda: _yd("severe_min_lr_diff_db", 3.0))
    severe_peak_context_km: float = field(default_factory=lambda: _yd("severe_peak_context_km", 0.05109))

    # ── 下降终点搜索参数 ──
    descent_smooth_window: int = field(default_factory=lambda: _yd("descent_smooth_window", 3))
    descent_min_db: float = field(default_factory=lambda: _yd("descent_min_db", 0.1))
    descent_patience: int = field(default_factory=lambda: _yd("descent_patience", 5))
    descent_max_search_km: float = field(default_factory=lambda: _yd("descent_max_search_km", 0.2554))
    descent_min_window: int = field(default_factory=lambda: _yd("descent_min_window", 7))

    # ── 普通断纤参数 ──
    break_step_drop_db: float = field(default_factory=lambda: _yd("break_step_drop_db", 0.72))
    break_no_peak_radius_km: float = field(default_factory=lambda: _yd("break_no_peak_radius_km", 0.1))
    break_no_peak_min_height_db: float = field(default_factory=lambda: _yd("break_no_peak_min_height_db", 5.0))
    break_pre_peak_left_km: float = field(default_factory=lambda: _yd("break_pre_peak_left_km", 0.025545))

    # ── 小峰断纤参数 ──
    low_peak__threshold_db: float = field(default_factory=lambda: _yd("low_peak__threshold_db", 5))
    min_peak__threshold_db: float = field(default_factory=lambda: _yd("min_peak__threshold_db", 0.6))
    small_peak_nearby_radius_km: float = field(default_factory=lambda: _yd("small_peak_nearby_radius_km", 0.05))
    small_peak_similar_radius_km: float = field(default_factory=lambda: _yd("small_peak_similar_radius_km", 0.25))
    small_peak_width_km: tuple = field(default_factory=lambda: _yd("small_peak_width_km", (0.0076635, 0.0383175)))
    small_peak_flat_len_km: float = field(default_factory=lambda: _yd("small_peak_flat_len_km", 0.025545))
    small_peak_flat_threshold_db: float = field(default_factory=lambda: _yd("small_peak_flat_threshold_db", 0.3))

    # ── 密集峰群参数 ──
    cluster_max_gap_km: float = field(default_factory=lambda: _yd("cluster_max_gap_km", 0.5))
    cluster_min_size: int = field(default_factory=lambda: _yd("cluster_min_size", 2))
    cluster_min_peak_height_db: float = field(default_factory=lambda: _yd("cluster_min_peak_height_db", 2.0))

    # ── 范围控制 ──
    end_region_ratio: float = field(default_factory=lambda: _yd("end_region_ratio", 0.15))
    skip_start_km: float = field(default_factory=lambda: _yd("skip_start_km", 0.1))
    skip_end_km: float = field(default_factory=lambda: _yd("skip_end_km", 2.0))
    offset_samples_km: float = field(default_factory=lambda: _yd("offset_samples_km", 0.1))

    # ── detect 主流程 ──
    lookback_km: float = field(default_factory=lambda: _yd("lookback_km", 0.0894075))
    overlap_tolerance_km: float = field(default_factory=lambda: _yd("overlap_tolerance_km", 0.0127725))
    refl_height_db: float = field(default_factory=lambda: _yd("refl_height_db", 6.0))
    refl_threshold_db: float = field(default_factory=lambda: _yd("refl_threshold_db", 6.0))
    first_peak_max_km: float = field(default_factory=lambda: _yd("first_peak_max_km", 0.05))
    noise_margin_km: float = field(default_factory=lambda: _yd("noise_margin_km", 0.1))

    # ── 基线拟合 ──
    baseline_poly_degree: int = field(default_factory=lambda: _yd("baseline_poly_degree", 3))

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.sample_spacing_km <= 0:
            raise ValueError("sample_spacing_km must be positive")

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> DetectionConfig:
        """从指定 YAML 文件创建配置实例。"""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        for key in ("peak_width_km", "small_peak_width_km"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        # 先用默认值创建，再用 YAML 中的值覆盖
        config = cls()
        valid_keys = {fld.name for fld in fields(cls)}
        for k, v in data.items():
            if k in valid_keys:
                setattr(config, k, v)
        return config


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
    """Get default application configuration."""
    return AppConfig(
        detection=DetectionConfig(),
        web=WebConfig(),
    )


def load_config(config_path: str | Path, validate: bool = True) -> AppConfig:
    """从 YAML 配置文件加载应用配置。

    Args:
        config_path: 配置文件路径（YAML 格式）
        validate: 是否验证加载的配置

    Returns:
        加载后的 AppConfig 实例
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # 直接复用 from_yaml 加载检测配置
    detection_config = DetectionConfig.from_yaml(config_path)

    app_config = AppConfig(detection=detection_config, web=WebConfig())

    if validate:
        app_config.validate()

    return app_config
