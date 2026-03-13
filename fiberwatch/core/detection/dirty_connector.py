"""脏污连接器检测。"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .context import DetectionContext

import numpy as np

from ..models import DetectedEvent
from .bend import check_step_drop
from .fiber_break import scan_normal_break, scan_small_peak_break


def detect_dirty_connector(
    ctx: DetectionContext,
    y: np.ndarray,
    valid_peaks: List[dict],
    effective_start: int,
    peak_indices: List[int],
    all_peaks: List[dict],
    events: List[DetectedEvent],
    processed_peak_indices: set,
) -> bool:
    """第二阶段：脏污检测（含脏污前断纤检测）。"""
    cfg = ctx.new_cfg
    for peak in valid_peaks:
        peak_idx = peak["index"]
        if peak_idx in processed_peak_indices:
            continue

        peak_height = peak["peak_height_db"]
        z_km = float(ctx.z[peak_idx])

        if peak_height <= cfg.peak_high_threshold_db:
            continue

        # 检查脏污之前是否有普通断纤
        normal_break = scan_normal_break(
            ctx, y, effective_start, peak_idx, peak_indices, all_peaks
        )
        if normal_break is not None:
            break_idx, break_drop_db = normal_break
            events.append(
                DetectedEvent(
                    kind="break",
                    z_km=float(ctx.z[break_idx]),
                    magnitude_db=float(break_drop_db),
                    reflect_db=0.0,
                    index=break_idx,
                    extra={
                        "subtype": "normal_break",
                        "detected_before_dirty_connector": True,
                        "dirty_connector_z_km": z_km,
                        "method": "small_step_no_big_peak_20m",
                    },
                )
            )
            return True

        # 检查脏污之前是否有小峰断纤
        small_peak_break = scan_small_peak_break(
            ctx,
            y,
            valid_peaks,
            peak_indices,
            processed_peak_indices,
            before_idx=peak_idx,
        )
        if small_peak_break is not None:
            sp_peak, sp_drop_db, sp_peak_height = small_peak_break
            sp_peak_idx = sp_peak["index"]
            events.append(
                DetectedEvent(
                    kind="break",
                    z_km=float(ctx.z[sp_peak_idx]),
                    magnitude_db=float(sp_drop_db),
                    reflect_db=float(sp_peak_height),
                    index=sp_peak_idx,
                    extra={
                        "subtype": "small_peak_break",
                        "detected_before_dirty_connector": True,
                        "dirty_connector_z_km": z_km,
                    },
                )
            )
            processed_peak_indices.add(sp_peak_idx)
            return True

        # 没有断纤在前，报脏污
        check_idx = peak.get("right_base_index", peak_idx + 5)
        has_drop, drop_db = check_step_drop(ctx, y, check_idx, peak_indices)
        events.append(
            DetectedEvent(
                kind="dirty_connector",
                z_km=z_km,
                magnitude_db=float(drop_db) if has_drop else 0.0,
                reflect_db=float(peak_height),
                index=peak_idx,
            )
        )
        processed_peak_indices.add(peak_idx)
        return True

    return False
