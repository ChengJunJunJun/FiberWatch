"""检测上下文：配置参数 + 预计算的采样点数。"""

import math

import numpy as np

from ...config.settings import DetectionConfig


# ── 距离→点数 转换工具 ───────────────────────────────────────────

def _km2ceil(km: float, sample_spacing_km: float, min_val: int = 1) -> int:
    """原有参数保持 ceil（向上取整），保证行为不变。"""
    return max(min_val, int(math.ceil(km / sample_spacing_km)))


def _km2s(km: float, sample_spacing_km: float, min_val: int = 1) -> int:
    """新参数用 round（四舍五入），从精确点数转换而来。"""
    return max(min_val, round(km / sample_spacing_km))


# ── 检测上下文 ───────────────────────────────────────────────────

class DetectionContext:
    """打包检测所需的全部共享状态，传给各子模块函数。"""

    def __init__(
        self,
        trace_db: np.ndarray,
        baseline: np.ndarray | None,
        config: DetectionConfig | None = None,
        *,
        sample_spacing_km: float | None = None,
    ) -> None:
        if config is None:
            config = DetectionConfig()

        # sample_spacing_km 优先使用显式传入值，否则从 config 读取
        if sample_spacing_km is not None:
            sp = float(sample_spacing_km)
        else:
            sp = float(config.sample_spacing_km)

        n_samples = len(trace_db)
        if n_samples <= 0:
            raise ValueError("trace_db must not be empty")
        if sp <= 0:
            raise ValueError("sample_spacing_km must be positive")

        self.sample_spacing_km = sp
        self.z = np.arange(n_samples) * sp
        self.distance_km = self.z

        self.cfg = config
        self.config = config
        self.new_cfg = config  # 统一使用 DetectionConfig

        self.trace_db = np.asarray(trace_db, dtype=float)
        self.baseline = (
            np.asarray(baseline, dtype=float) if baseline is not None else None
        )

        # 预计算采样点数（从 km 转换为 samples）
        nc = config

        self.step_window_samples = _km2ceil(nc.step_compare_window_km, sp)
        self.peak_local_region_samples = _km2ceil(nc.peak_local_region_km, sp)
        self.noise_check_window_samples = _km2ceil(nc.noise_check_window_km, sp)
        self.peak_width_samples = (
            max(1, int(nc.peak_width_km[0] / sp)),
            max(2, int(nc.peak_width_km[1] / sp)),
        )
        self.peak_search_left_extra_samples = _km2s(nc.peak_search_left_extra_km, sp)
        self.peak_merge_distance_samples = _km2s(nc.peak_merge_distance_km, sp)
        self.noise_check_offset_samples = _km2s(nc.noise_check_offset_km, sp)
        self.peak_step_search_range_samples = _km2s(nc.peak_step_search_range_km, sp)
        self.peak_step_local_min_range_samples = _km2s(nc.peak_step_local_min_range_km, sp)
        self.peak_step_match_tolerance_samples = _km2s(nc.peak_step_match_tolerance_km, sp)
        self.plateau_min_gap_samples = _km2s(nc.plateau_min_gap_km, sp)
        self.severe_peak_context_samples = _km2s(nc.severe_peak_context_km, sp)
        self.descent_max_search_samples = _km2s(nc.descent_max_search_km, sp)
        self.break_pre_peak_left_samples = _km2s(nc.break_pre_peak_left_km, sp)
        self.small_peak_nearby_radius_samples = _km2s(nc.small_peak_nearby_radius_km, sp)
        self.small_peak_similar_radius_samples = _km2s(nc.small_peak_similar_radius_km, sp)
        self.small_peak_width_samples = (
            _km2s(nc.small_peak_width_km[0], sp),
            _km2s(nc.small_peak_width_km[1], sp),
        )
        self.small_peak_flat_len_samples = _km2s(nc.small_peak_flat_len_km, sp)
        self.cluster_max_gap_samples = _km2s(nc.cluster_max_gap_km, sp)
        self.lookback_samples = _km2s(nc.lookback_km, sp)
        self.overlap_tolerance_samples = _km2s(nc.overlap_tolerance_km, sp)
        self.noise_margin_samples = _km2s(nc.noise_margin_km, sp)
