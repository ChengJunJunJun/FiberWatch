"""弯折检测（无反射峰的阶梯下降）。"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .context import DetectionContext

from ..models import DetectedEvent


def check_step_drop(
    ctx: DetectionContext,
    y: np.ndarray,
    idx: int,
    ignore_peak_indices: Optional[List[int]] = None,
) -> Tuple[bool, float]:
    """检查在 idx 位置是否有阶梯下降（幅度 + 斜率双条件）。"""
    n = len(y)
    win = ctx.step_window_samples

    pre_start = max(0, idx - win)
    pre_end = idx
    post_start = idx + 1
    post_end = min(n, idx + win + 1)

    if pre_end <= pre_start or post_end <= post_start:
        return False, 0.0

    pre_data = y[pre_start:pre_end].copy()
    post_data = y[post_start:post_end].copy()

    # 排除峰值点的干扰
    if ignore_peak_indices:
        pre_mask = np.ones(len(pre_data), dtype=bool)
        post_mask = np.ones(len(post_data), dtype=bool)
        for peak_idx in ignore_peak_indices:
            for offset in range(-3, 4):
                check_idx = peak_idx + offset
                if pre_start <= check_idx < pre_end:
                    pre_mask[check_idx - pre_start] = False
                if post_start <= check_idx < post_end:
                    post_mask[check_idx - post_start] = False
        pre_data = pre_data[pre_mask] if np.any(pre_mask) else pre_data
        post_data = post_data[post_mask] if np.any(post_mask) else post_data

    if len(pre_data) == 0 or len(post_data) == 0:
        return False, 0.0

    pre_mean = float(np.mean(pre_data))
    post_mean = float(np.mean(post_data))
    drop = pre_mean - post_mean

    if drop <= 0:
        return False, 0.0

    pre_center_idx = (pre_start + pre_end) // 2
    post_center_idx = (post_start + post_end) // 2
    distance_km = (post_center_idx - pre_center_idx) * ctx.sample_spacing_km

    if distance_km <= 0:
        return False, 0.0

    slope_db_per_km = drop / distance_km
    is_step = slope_db_per_km >= ctx.new_cfg.step_min_slope_db_per_km

    return is_step, drop


def is_step_caused_by_peak(
    ctx: DetectionContext,
    y: np.ndarray,
    step_idx: int,
    peaks: List[dict],
    step_drop_db: float,
) -> Tuple[bool, Optional[dict]]:
    """判断阶梯下降是否由峰引起（用于弯曲检测）。"""
    cfg = ctx.new_cfg
    n = len(y)

    search_range_samples = ctx.peak_step_search_range_samples
    local_min_range_samples = ctx.peak_step_local_min_range_samples

    search_start = max(0, step_idx - search_range_samples)
    search_end = min(n, step_idx + search_range_samples)

    if search_end <= search_start:
        return False, None

    local_max_idx = search_start + int(np.argmax(y[search_start:search_end]))
    local_max_val = y[local_max_idx]

    nearby_mean = np.mean(y[search_start:search_end])
    if local_max_val - nearby_mean < cfg.peak_bend_min_prominence_db:
        return False, None

    left_start = max(0, local_max_idx - local_min_range_samples)
    left_end = local_max_idx
    right_start = local_max_idx + 1
    right_end = min(n, local_max_idx + local_min_range_samples + 1)

    if left_end <= left_start or right_end <= right_start:
        return False, None

    left_min_val = float(np.min(y[left_start:left_end]))
    right_min_val = float(np.min(y[right_start:right_end]))
    min_diff = left_min_val - right_min_val

    causing_peak = None
    for peak in peaks:
        if abs(peak["index"] - local_max_idx) <= ctx.peak_step_match_tolerance_samples:
            causing_peak = peak
            break

    rel_tol = cfg.peak_step_match_rel_tol
    abs_tol = cfg.peak_step_match_abs_tol
    diff_error = abs(min_diff - step_drop_db)
    is_approximately_equal = (
        diff_error <= abs_tol
        or diff_error <= step_drop_db * rel_tol
        or min_diff >= step_drop_db
    )

    if is_approximately_equal and min_diff > 0:
        return True, causing_peak
    return False, None


def check_plateau_stability(
    ctx: DetectionContext,
    y: np.ndarray,
    step_idx: int,
) -> Tuple[bool, float, float]:
    """检查下降点之前（高台阶处）是否是稳定平台。"""
    cfg = ctx.new_cfg
    window_samples = max(
        10, int(cfg.bend_plateau_window_km / ctx.sample_spacing_km)
    )

    gap_samples = max(ctx.plateau_min_gap_samples, ctx.step_window_samples // 4)
    plateau_end = max(0, step_idx - gap_samples)
    plateau_start = max(0, plateau_end - window_samples)

    if plateau_end <= plateau_start:
        return False, float("inf"), float("inf")

    plateau_data = y[plateau_start:plateau_end]
    if len(plateau_data) < cfg.plateau_min_data_count:
        return False, float("inf"), float("inf")

    std_db = float(np.std(plateau_data))
    range_db = float(np.max(plateau_data) - np.min(plateau_data))

    is_stable = (
        std_db <= cfg.bend_plateau_max_std_db
        and range_db <= cfg.bend_plateau_max_range_db
    )
    return is_stable, std_db, range_db


def detect_bend(
    ctx: DetectionContext,
    y: np.ndarray,
    effective_start: int,
    bend_end: int,
    step: int,
    peak_indices: List[int],
    all_peaks: List[dict],
    events: List[DetectedEvent],
    drop_fraction: float = 3.0,
) -> bool:
    """弯折检测（无反射峰的阶梯下降）。返回是否找到终端事件。"""
    cfg = ctx.new_cfg
    for i in range(effective_start, bend_end, step):
        has_drop, drop_db = check_step_drop(ctx, y, i, peak_indices)
        if not (has_drop and drop_db >= cfg.step_drop_severe_db):
            continue

        is_caused, _ = is_step_caused_by_peak(ctx, y, i, all_peaks, drop_db)
        if is_caused:
            continue

        is_stable, plateau_std, plateau_range = check_plateau_stability(ctx, y, i)
        if is_stable:
            half_drop = drop_db / drop_fraction
            baseline_val = y[i]
            search_end = min(i + ctx.step_window_samples, len(y))
            precise_index = next(
                (j for j in range(i, search_end) if baseline_val - y[j] >= half_drop),
                i,
            )
            if precise_index > i + 1:
                min_slope = float("inf")
                min_slope_idx = i
                for j in range(i, precise_index):
                    slope = y[j + 1] - y[j]
                    if slope < min_slope:
                        min_slope = slope
                        min_slope_idx = j
                precise_index = min_slope_idx

            events.append(
                DetectedEvent(
                    kind="bend",
                    z_km=float(ctx.distance_km[precise_index]),
                    magnitude_db=float(drop_db),
                    reflect_db=0.0,
                    index=precise_index,
                    extra={
                        "detection_method": "peak_base_diff",
                        "plateau_std_db": plateau_std,
                        "plateau_range_db": plateau_range,
                    },
                )
            )
            return True
    return False
