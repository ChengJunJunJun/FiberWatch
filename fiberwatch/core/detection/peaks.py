"""反射峰查找与反射峰识别。"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from .context import DetectionContext


def find_peaks(ctx: DetectionContext, y: np.ndarray) -> List[dict]:
    """找到所有反射峰。"""
    peaks = []
    n = len(y)
    cfg = ctx.new_cfg
    local_region = ctx.peak_local_region_samples
    _min_width, max_width = ctx.peak_width_samples
    i = 1
    while i < n - 1:
        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            peak_idx = i
            peak_val = y[i]
            local_start = max(0, i - local_region)
            local_end = min(n, i + local_region)
            local_data = []
            for j in range(local_start, local_end):
                if abs(j - i) > max_width:
                    local_data.append(y[j])
            if len(local_data) < cfg.peak_min_local_data_count:
                local_data = list(y[local_start:local_end])
            local_std = np.std(local_data)

            # 向左找起点
            left_idx = i - 1
            while left_idx > 0 and y[left_idx] <= y[left_idx + 1]:
                left_idx -= 1
            search_left = max(0, left_idx - ctx.peak_search_left_extra_samples)
            for j in range(left_idx, search_left - 1, -1):
                if y[j] < y[left_idx]:
                    left_idx = j

            true_left_idx = left_idx + np.argmin(y[left_idx:i])
            left_val = y[true_left_idx]
            left_idx = true_left_idx
            peak_height = peak_val - left_val

            std_condition = (
                peak_height >= local_std * cfg.peak_prominence_std_factor
            )
            height_condition = peak_height >= cfg.min_peak_height_db
            if std_condition or height_condition:
                right_idx = i + 1
                while right_idx < n - 1 and y[right_idx] >= y[right_idx + 1]:
                    right_idx += 1
                peaks.append(
                    {
                        "index": peak_idx,
                        "height": peak_val,
                        "left_base_index": left_idx,
                        "left_base_value": left_val,
                        "right_base_index": right_idx,
                        "peak_height_db": peak_height,
                        "peak_width": right_idx - left_idx,
                    }
                )
                i = right_idx + 1
                continue
        i += 1

    # 合并相近的峰
    if len(peaks) > 1:
        merged_peaks = [peaks[0]]
        for current_peak in peaks[1:]:
            last_peak = merged_peaks[-1]
            distance_samples = current_peak["index"] - last_peak["index"]
            height_diff = abs(current_peak["height"] - last_peak["height"])
            if (
                distance_samples <= ctx.peak_merge_distance_samples
                and height_diff <= cfg.peak_merge_height_diff_db
            ):
                if current_peak["height"] > last_peak["height"]:
                    merged_peaks[-1] = current_peak
            else:
                merged_peaks.append(current_peak)
        peaks = merged_peaks

    return peaks


def find_dense_peak_cluster(
    ctx: DetectionContext, y: np.ndarray, peaks: List[dict]
) -> Optional[int]:
    """查找密集峰群的起始索引。"""
    cfg = ctx.new_cfg

    valid_peaks = [p for p in peaks if p["peak_height_db"] >= cfg.cluster_min_peak_height_db]
    if len(valid_peaks) < cfg.cluster_min_size:
        return None

    cluster_peaks = [valid_peaks[0]]
    for i in range(1, len(valid_peaks)):
        gap = valid_peaks[i]["index"] - valid_peaks[i - 1]["index"]
        if gap <= ctx.cluster_max_gap_samples:
            cluster_peaks.append(valid_peaks[i])
        else:
            if len(cluster_peaks) >= cfg.cluster_min_size:
                break
            cluster_peaks = [valid_peaks[i]]

    if len(cluster_peaks) >= cfg.cluster_min_size:
        return cluster_peaks[0]["index"]
    return None


def identify_reflection_peaks(
    ctx: DetectionContext,
    all_peaks: List[dict],
    events: list,
    baseline: np.ndarray,
    effective_end: int,
) -> list:
    """从所有峰中筛选出反射峰（非事件峰）。"""
    cfg = ctx.new_cfg
    event_indices = [e.index for e in events]

    # 起始处第一个峰一定是反射峰
    first_peak_indices: set[int] = set()
    if all_peaks:
        peak_km = all_peaks[0]["index"] * ctx.sample_spacing_km
        if peak_km <= cfg.first_peak_max_km:
            first_peak_indices.add(all_peaks[0]["index"])

    reflection_peaks = []
    for p in all_peaks:
        pidx = p["index"]
        # 噪声区之后截断
        if pidx >= effective_end + ctx.noise_margin_samples:
            break
        # 起始处的峰直接纳入
        if pidx in first_peak_indices:
            reflection_peaks.append(p)
            continue
        # 与已检测事件重合则跳过
        if any(abs(pidx - ei) <= ctx.overlap_tolerance_samples for ei in event_indices):
            continue
        # 常规阈值判断
        if (
            p["height"] - baseline[pidx] > cfg.refl_height_db
            and p["peak_height_db"] > cfg.refl_threshold_db
            and p["height"] - ctx.trace_db[p["right_base_index"]] > cfg.refl_threshold_db
        ):
            reflection_peaks.append(p)

    return reflection_peaks
