"""断纤检测（严重断纤、普通断纤、小峰断纤）。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .context import DetectionContext

from ..models import DetectedEvent
from .bend import check_step_drop

# ── 下降终点搜索 ─────────────────────────────────────────────────


def find_descent_end(
    ctx: DetectionContext,
    y: np.ndarray,
    start_idx: int,
    *,
    direction: int = 1,
) -> Tuple[int, float]:
    """从 start_idx 沿 direction 搜索下降终点，返回窗口均值最小的位置。"""
    cfg = ctx.new_cfg
    smooth_window = cfg.descent_smooth_window
    min_descent_db = cfg.descent_min_db
    patience = cfg.descent_patience
    max_search = ctx.descent_max_search_samples
    min_window = cfg.descent_min_window

    n = len(y)
    if n == 0:
        return start_idx, float("nan")

    if smooth_window > 1:
        from scipy.ndimage import uniform_filter1d

        y_smooth = uniform_filter1d(y.astype(float), size=smooth_window, mode="nearest")
    else:
        y_smooth = y.astype(float)

    idx = int(start_idx)
    prev = float(y_smooth[idx])
    candidates = []
    no_drop_run = 0
    steps = 0

    while 0 <= idx + direction < n and steps < max_search:
        idx += direction
        curr = float(y_smooth[idx])

        half_w = min_window // 2
        w_start = max(0, idx - half_w)
        w_end = min(n, idx + half_w + 1)
        local_avg = float(np.mean(y[w_start:w_end]))
        candidates.append((idx, local_avg))

        dropped = curr <= prev - min_descent_db
        if dropped:
            no_drop_run = 0
        else:
            no_drop_run += 1
            if no_drop_run >= patience:
                break
        prev = curr
        steps += 1

    if candidates:
        best_idx, best_avg = min(candidates, key=lambda x: x[1])
        return best_idx, best_avg

    return start_idx, float(y[start_idx])


# ── 严重断纤判定 ─────────────────────────────────────────────────


def is_severe_break_by_peak_profile(
    ctx: DetectionContext,
    y: np.ndarray,
    peak: dict,
) -> Tuple[bool, dict]:
    """严重断纤判定（基于下降趋势搜索）。"""
    cfg = ctx.new_cfg
    n = len(y)
    peak_idx = int(peak["index"])

    peak_height = float(peak.get("peak_height_db", 0.0))
    if peak_height <= 0:
        s0 = max(0, peak_idx - ctx.severe_peak_context_samples)
        s1 = min(n, peak_idx + ctx.severe_peak_context_samples + 1)
        peak_height = float(y[peak_idx] - np.mean(y[s0:s1]))

    if peak_height < cfg.severe_min_peak_db:
        return False, {
            "reason": "peak_not_high_enough",
            "peak_height_db": peak_height,
        }

    left_idx, left_min = find_descent_end(ctx, y, peak_idx, direction=-1)
    right_idx, right_min = find_descent_end(ctx, y, peak_idx, direction=+1)

    min_diff = left_min - right_min

    if min_diff <= cfg.severe_min_lr_diff_db:
        return False, {
            "reason": "left_right_min_diff_too_small",
            "left_min_idx": left_idx,
            "left_min_db": left_min,
            "right_min_idx": right_idx,
            "right_min_db": right_min,
            "min_diff_db": min_diff,
        }

    return True, {
        "reason": "severe_break_by_peak_profile",
        "left_min_idx": left_idx,
        "left_min_db": left_min,
        "right_min_idx": right_idx,
        "right_min_db": right_min,
        "min_diff_db": min_diff,
    }


def has_big_peak_near(
    ctx: DetectionContext,
    idx: int,
    peaks: List[dict],
    *,
    radius_km: float,
    min_height_db: float,
) -> bool:
    """检查 idx 附近是否有大峰。"""
    radius_samples = max(1, int(math.ceil(radius_km / ctx.sample_spacing_km)))
    for p in peaks:
        if (
            p.get("peak_height_db", 0.0) >= min_height_db
            and abs(int(p["index"]) - int(idx)) <= radius_samples
        ):
            return True
    return False


# ── 严重断纤检测 ─────────────────────────────────────────────────


def detect_severe_break(
    ctx: DetectionContext,
    y: np.ndarray,
    valid_peaks: List[dict],
    events: List[DetectedEvent],
    processed_peak_indices: set,
) -> bool:
    """第零阶段：严重断纤检测。返回是否找到终端事件。"""
    for peak in valid_peaks:
        peak_idx = int(peak["index"])
        z_km = float(ctx.z[peak_idx])

        ok, info = is_severe_break_by_peak_profile(ctx, y, peak)
        if ok:
            events.append(
                DetectedEvent(
                    kind="break",
                    z_km=z_km,
                    magnitude_db=float(info["min_diff_db"]),
                    reflect_db=float(peak.get("peak_height_db", 0.0)),
                    index=peak_idx,
                    extra={
                        "subtype": "severe_break",
                        "method": "peak_profile_lrmin_derivative",
                        **info,
                    },
                )
            )
            processed_peak_indices.add(peak_idx)
            return True
    return False


# ── 断裂前峰查找 ─────────────────────────────────────────────────


def find_pre_break_peak(
    ctx: DetectionContext,
    y: np.ndarray,
    drop_idx: int,
    scan_end: int,
    lookahead_samples: int | None = None,
) -> int:
    """在检测窗口内寻找局部最高点作为断裂位置。"""
    if lookahead_samples is None:
        lookahead_samples = ctx.step_window_samples

    left = max(0, drop_idx - ctx.break_pre_peak_left_samples)
    right = min(scan_end, drop_idx + lookahead_samples)
    segment = y[left:right]
    return left + int(np.argmax(segment))


# ── 普通断纤扫描 ─────────────────────────────────────────────────


def scan_normal_break(
    ctx: DetectionContext,
    y: np.ndarray,
    scan_start: int,
    scan_end: int,
    peak_indices: List[int],
    all_peaks: List[dict],
) -> tuple[int, float] | None:
    """扫描 [scan_start, scan_end) 范围内的普通断纤（无反射峰的阶梯下降）。"""
    cfg = ctx.new_cfg
    scan_step = max(1, ctx.step_window_samples // 4)
    for i in range(scan_start, scan_end, scan_step):
        has_drop, drop_db = check_step_drop(ctx, y, i, peak_indices)
        if not has_drop or drop_db < cfg.break_step_drop_db:
            continue
        if has_big_peak_near(
            ctx,
            i,
            all_peaks,
            radius_km=cfg.break_no_peak_radius_km,
            min_height_db=cfg.break_no_peak_min_height_db,
        ):
            continue
        break_idx = find_pre_break_peak(ctx, y, i, scan_end)
        return (break_idx, drop_db)
    return None


# ── 小峰断纤扫描 ─────────────────────────────────────────────────


def scan_small_peak_break(
    ctx: DetectionContext,
    y: np.ndarray,
    valid_peaks: List[dict],
    peak_indices: List[int],
    processed_peak_indices: set[int],
    before_idx: int | None = None,
) -> tuple[dict, float, float] | None:
    """扫描小峰断纤。before_idx 不为 None 时，只考虑 index < before_idx 的峰。"""
    cfg = ctx.new_cfg
    radius_samples = ctx.small_peak_nearby_radius_samples
    radius2_samples = ctx.small_peak_similar_radius_samples

    for peak in valid_peaks:
        peak_idx = peak["index"]
        if peak_idx in processed_peak_indices:
            continue
        if before_idx is not None and peak_idx >= before_idx:
            continue

        peak_height = peak["peak_height_db"]
        if not (cfg.min_peak__threshold_db < peak_height < cfg.peak_low_threshold_db):
            continue

        has_other_big_peak = any(
            (int(p["index"]) != int(peak_idx))
            and (p.get("peak_height_db", 0.0) >= cfg.low_peak__threshold_db)
            and (abs(int(p["index"]) - int(peak_idx)) <= radius_samples)
            for p in valid_peaks
        )
        if has_other_big_peak:
            continue

        has_similar_peak = peak_height <= 2.0 and any(
            (int(p["index"]) != int(peak_idx))
            and (0.9 * peak_height <= p.get("peak_height_db", 0.0) <= 1.1 * peak_height)
            and (abs(int(p["index"]) - int(peak_idx)) <= radius2_samples)
            for p in valid_peaks
        )
        if has_similar_peak:
            continue
        peak_width = peak.get("peak_width", 0)
        min_pw, max_pw = ctx.small_peak_width_samples
        if not (min_pw <= peak_width <= max_pw):
            continue
        # 峰左起点外侧必须有平稳区域
        left_base = peak.get("left_base_index", peak_idx)
        flat_len = ctx.small_peak_flat_len_samples
        flat_threshold_db = cfg.small_peak_flat_threshold_db

        left_start = left_base - flat_len
        if left_start < 0:
            continue
        left_region = y[left_start:left_base]
        if np.ptp(left_region) > flat_threshold_db:
            continue

        check_idx = peak.get("right_base_index", peak_idx + 5)
        has_drop, drop_db = check_step_drop(ctx, y, check_idx, peak_indices)
        return (peak, drop_db if has_drop else 0.0, peak_height)

    return None


# ── 普通断纤检测（阶段3） ────────────────────────────────────────


def detect_normal_break(
    ctx: DetectionContext,
    y: np.ndarray,
    effective_start: int,
    effective_end: int,
    offset_samples: int,
    peak_indices: List[int],
    all_peaks: List[dict],
    events: List[DetectedEvent],
) -> bool:
    """第三阶段：普通断纤检测。"""
    cfg = ctx.new_cfg
    break_end = max(effective_start, effective_end - offset_samples)

    result = scan_normal_break(
        ctx, y, effective_start, break_end, peak_indices, all_peaks
    )
    if result is None:
        return False

    i, drop_db = result
    events.append(
        DetectedEvent(
            kind="break",
            z_km=float(ctx.distance_km[i]),
            magnitude_db=float(drop_db),
            reflect_db=0.0,
            index=i,
            extra={
                "subtype": "normal_break",
                "method": "small_step_no_big_peak_20m",
                "break_step_drop_db": cfg.break_step_drop_db,
                "no_peak_radius_km": cfg.break_no_peak_radius_km,
                "no_peak_min_height_db": cfg.break_no_peak_min_height_db,
            },
        )
    )
    return True


# ── 小峰断纤检测（阶段4） ────────────────────────────────────────


def detect_small_peak_break(
    ctx: DetectionContext,
    y: np.ndarray,
    valid_peaks: List[dict],
    peak_indices: List[int],
    events: List[DetectedEvent],
    processed_peak_indices: set,
) -> bool:
    """第四阶段：小峰断纤检测。"""
    result = scan_small_peak_break(
        ctx, y, valid_peaks, peak_indices, processed_peak_indices
    )
    if result is None:
        return False

    peak, drop_db, peak_height = result
    peak_idx = peak["index"]
    events.append(
        DetectedEvent(
            kind="break",
            z_km=float(ctx.z[peak_idx]),
            magnitude_db=float(drop_db),
            reflect_db=float(peak_height),
            index=peak_idx,
            extra={"subtype": "small_peak_break"},
        )
    )
    processed_peak_indices.add(peak_idx)
    return True
