"""有效检测范围计算。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from ..models import DetectedEvent
    from .context import DetectionContext

from .peaks import find_dense_peak_cluster


def find_effective_end(
    ctx: DetectionContext, y: np.ndarray, peaks: List[dict]
) -> int:
    """找到有效检测的结束点。"""
    n = len(y)
    cfg = ctx.new_cfg
    end_region_start = int(n * (1 - cfg.end_region_ratio))

    # 密集峰群检测
    if peaks is not None:
        end_region_peaks = [p for p in peaks if p["index"] >= end_region_start]
        if len(end_region_peaks) >= 2:
            end_region_peaks.sort(key=lambda p: p["index"])
            cluster_start_idx = find_dense_peak_cluster(ctx, y, end_region_peaks)
            if cluster_start_idx is not None:
                last_peak = end_region_peaks[-1]
                noise_check_start = min(
                    last_peak.get("right_base_index", last_peak["index"]) + ctx.noise_check_offset_samples,
                    n - 1,
                )
                if check_enters_noise_region(ctx, y, noise_check_start):
                    return max(end_region_start, cluster_start_idx - 10)

    # 单峰 + 噪声检测
    end_region = y[end_region_start:]
    if len(end_region) == 0:
        return n - 1

    local_max_idx = np.argmax(end_region)
    global_max_idx = end_region_start + local_max_idx

    check_start = min(global_max_idx + ctx.noise_check_offset_samples, n)
    if check_start < n:
        tail = y[check_start:]
        if len(tail) >= ctx.noise_check_window_samples:
            tail_mean = np.mean(tail)
            tail_std = np.std(tail)
            if (
                tail_mean < cfg.noise_floor_db
                and tail_std > cfg.noise_std_threshold
            ):
                return global_max_idx

    return end_region_start


def check_enters_noise_region(
    ctx: DetectionContext, y: np.ndarray, start_idx: int
) -> bool:
    """检查从 start_idx 开始是否进入噪声区域。"""
    n = len(y)
    cfg = ctx.new_cfg
    check_end = min(start_idx + ctx.noise_check_window_samples, n)
    if check_end - start_idx < cfg.min_noise_segment_count:
        return False
    segment = y[start_idx:check_end]
    seg_mean = float(np.mean(segment))
    seg_std = float(np.std(segment))
    return seg_mean < cfg.noise_floor_db and seg_std > cfg.noise_std_threshold


def find_effective_start(ctx: DetectionContext, peaks: List[dict]) -> int:
    """找到有效检测的起点（跳过 skip_start_km 内的所有峰）。"""
    skip_km = ctx.new_cfg.skip_start_km
    start_idx = int(math.ceil(skip_km / ctx.sample_spacing_km))
    for p in peaks:
        if p["index"] * ctx.sample_spacing_km < skip_km:
            right = p.get("right_base_index", p["index"] + 1) + 1
            if right > start_idx:
                start_idx = right
    return start_idx


def get_scan_end(
    events: List[DetectedEvent],
    default_end: int,
    effective_start: int,
    lookback: int,
) -> int:
    """根据已有事件计算扫描终点。无事件时返回 default_end。"""
    if not events:
        return default_end
    earliest_idx = min(e.index for e in events)
    return max(effective_start, earliest_idx - lookback)


def filter_peaks_before(peaks: List[dict], cutoff: int) -> List[dict]:
    """只保留 index < cutoff 的峰。"""
    return [p for p in peaks if p["index"] < cutoff]
