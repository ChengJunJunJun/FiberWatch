import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..config.settings import DetectionConfig as DetectorConfig

# ── 数据结构 ─────────────────────────────────────────────────────


@dataclass
class DetectedEvent:
    kind: str
    z_km: float
    magnitude_db: float = 0.0
    reflect_db: float = 0.0
    index: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class DetectionResult:
    events: list[DetectedEvent]
    distance_km: np.ndarray
    trace_db: np.ndarray
    baseline_db: np.ndarray
    residual_db: np.ndarray

    @property
    def trace_smooth_db(self) -> np.ndarray:
        """visualization.py 通过此属性访问数据。"""
        return self.trace_db

    def plot(self, outfile: str | None = None) -> plt.Figure:
        z = self.distance_km
        y = self.trace_db
        b = self.baseline_db
        fig = plt.figure(figsize=(10, 4))
        plt.plot(z, y, label="trace (raw)")
        plt.plot(z, b, "--", label="baseline")
        for ev in self.events:
            plt.axvline(ev.z_km, linestyle=":", alpha=0.6)
            txt = f"{ev.kind} @ {ev.z_km:.4f}km"
            plt.text(
                ev.z_km,
                np.nanmin(y) + 2,
                txt,
                rotation=90,
                va="bottom",
                fontsize=8,
            )
        plt.xlabel("Distance (km)")
        plt.ylabel("Return (dB)")
        plt.legend()
        plt.tight_layout()
        if outfile:
            plt.savefig(outfile, dpi=160)
        return fig


# ── 内部配置参数 ─────────────────────────────────────────────────


@dataclass
class _NewDetectionConfig:
    """新算法的内部参数，不暴露给外部。"""

    # ── 反射峰相关 ──
    peak_local_region_km: float = 0.1
    peak_prominence_std_factor: float = 2.0
    peak_width_km: Tuple[float, float] = (0.0128, 0.0383)
    min_peak_height_db: float = 0.5
    peak_min_prominence_db: float = 0.5

    # ── 反射峰高度阈值 ──
    peak_high_threshold_db: float = 10.9
    peak_low_threshold_db: float = 8.0

    # ── 峰左右基线差异判断 ──
    peak_step_match_rel_tol: float = 0.3
    peak_step_match_abs_tol: float = 0.3
    peak_no_step_threshold_db: float = 0.5
    peak_bend_min_prominence_db: float = 2

    # ── 阶梯下降相关 ──
    step_drop_severe_db: float = 1.3
    step_drop_normal_db: float = 0.2
    step_compare_window_km: float = 0.05
    step_min_slope_db_per_km: float = 10.0

    # ── 宽范围阶梯下降检测 ──
    wide_step_offset_km: float = 0.1
    wide_step_window_km: float = 0.08

    # ── 弯折检测 - 高台阶稳定性 ──
    bend_plateau_window_km: float = 0.05
    bend_plateau_max_std_db: float = 1.0
    bend_plateau_max_range_db: float = 1.5

    # ── 噪声区域判断 ──
    noise_floor_db: float = -25.0
    noise_std_threshold: float = 2.0
    noise_check_window_km: float = 0.128
    severe_break_noise_std: float = 3

    # ── 普通断纤参数 ──
    break_step_drop_db: float = 0.72
    break_no_peak_radius_km: float = 0.1
    break_no_peak_min_height_db: float = 5.0

    # ── 小峰断纤参数 ──
    low_peak__threshold_db: float = 5
    min_peak__threshold_db: float = 0.6

    # ── 范围控制 ──
    end_region_ratio: float = 0.15
    skip_start_km: float = 0.1
    skip_end_km: float = 2.0
    offset_samples_km: float = 0.1

    # ── 基线拟合 ──
    baseline_poly_degree: int = 3


class Detector:
    def __init__(
        self,
        trace_db: np.ndarray,
        baseline: np.ndarray | None = None,
        config: DetectorConfig | None = None,
        *,
        sample_spacing_km: float,
    ) -> None:
        if config is None:
            config = DetectorConfig()

        n_samples = len(trace_db)
        if n_samples <= 0:
            raise ValueError("trace_db must not be empty")
        if sample_spacing_km <= 0:
            raise ValueError("sample_spacing_km must be positive")

        self.sample_spacing_km = float(sample_spacing_km)
        self.z = np.arange(n_samples) * self.sample_spacing_km
        self.distance_km = self.z

        self.cfg = config
        self.config = config
        self._new_cfg = _NewDetectionConfig()

        self.trace_db = np.asarray(trace_db, dtype=float)
        self.baseline = (
            np.asarray(baseline, dtype=float) if baseline is not None else None
        )

        # 预计算采样点数
        self._step_window_samples = max(
            1,
            int(
                math.ceil(self._new_cfg.step_compare_window_km / self.sample_spacing_km)
            ),
        )
        self._peak_local_region_samples = max(
            1,
            int(math.ceil(self._new_cfg.peak_local_region_km / self.sample_spacing_km)),
        )
        self._noise_check_window_samples = max(
            1,
            int(
                math.ceil(self._new_cfg.noise_check_window_km / self.sample_spacing_km)
            ),
        )
        self._peak_width_samples = (
            max(1, int(self._new_cfg.peak_width_km[0] / self.sample_spacing_km)),
            max(2, int(self._new_cfg.peak_width_km[1] / self.sample_spacing_km)),
        )

    # ── 基线拟合 ─────────────────────────────────────────────────

    @staticmethod
    def _fit_linear_baseline(z: np.ndarray, y: np.ndarray) -> np.ndarray:
        """线性基线拟合。"""
        q1, q2 = np.nanpercentile(y, [25, 50])
        msk = (y <= q2) & (y >= q1)
        if not np.any(msk):
            msk = np.ones_like(y, dtype=bool)
        design_matrix = np.vstack([z[msk], np.ones(np.sum(msk))]).T
        m, c = np.linalg.lstsq(design_matrix, y[msk], rcond=None)[0]
        return m * z + c

    # ── 反射峰查找 ───────────────────────────────────────────────

    def _find_peaks(self, y: np.ndarray) -> List[dict]:
        """找到所有反射峰。"""
        peaks = []
        n = len(y)
        local_region = self._peak_local_region_samples
        _min_width, max_width = self._peak_width_samples
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
                if len(local_data) < 10:
                    local_data = list(y[local_start:local_end])
                local_std = np.std(local_data)

                # 向左找起点
                left_idx = i - 1
                while left_idx > 0 and y[left_idx] <= y[left_idx + 1]:
                    left_idx -= 1
                search_left = max(0, left_idx - 5)
                for j in range(left_idx, search_left - 1, -1):
                    if y[j] < y[left_idx]:
                        left_idx = j

                true_left_idx = left_idx + np.argmin(y[left_idx:i])
                left_val = y[true_left_idx]
                left_idx = true_left_idx
                peak_height = peak_val - left_val

                std_condition = (
                    peak_height >= local_std * self._new_cfg.peak_prominence_std_factor
                )
                height_condition = peak_height >= self._new_cfg.min_peak_height_db
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
            merge_distance_samples = 5
            merge_height_diff_db = 0.1
            merged_peaks = [peaks[0]]
            for current_peak in peaks[1:]:
                last_peak = merged_peaks[-1]
                distance_samples = current_peak["index"] - last_peak["index"]
                height_diff = abs(current_peak["height"] - last_peak["height"])
                if (
                    distance_samples <= merge_distance_samples
                    and height_diff <= merge_height_diff_db
                ):
                    if current_peak["height"] > last_peak["height"]:
                        merged_peaks[-1] = current_peak
                else:
                    merged_peaks.append(current_peak)
            peaks = merged_peaks

        return peaks

    # ── 有效范围检测 ─────────────────────────────────────────────

    def _find_effective_end(self, y: np.ndarray, peaks: List[dict]) -> int:
        """找到有效检测的结束点。"""
        n = len(y)
        cfg = self._new_cfg
        end_region_start = int(n * (1 - cfg.end_region_ratio))

        # 密集峰群检测
        if peaks is not None:
            end_region_peaks = [p for p in peaks if p["index"] >= end_region_start]
            if len(end_region_peaks) >= 2:
                end_region_peaks.sort(key=lambda p: p["index"])
                cluster_start_idx = self._find_dense_peak_cluster(y, end_region_peaks)
                if cluster_start_idx is not None:
                    last_peak = end_region_peaks[-1]
                    noise_check_start = min(
                        last_peak.get("right_base_index", last_peak["index"]) + 5,
                        n - 1,
                    )
                    if self._check_enters_noise_region(y, noise_check_start):
                        return max(end_region_start, cluster_start_idx - 10)

        # 单峰 + 噪声检测
        end_region = y[end_region_start:]
        if len(end_region) == 0:
            return n - 1

        local_max_idx = np.argmax(end_region)
        global_max_idx = end_region_start + local_max_idx

        check_start = min(global_max_idx + 5, n)
        if check_start < n:
            tail = y[check_start:]
            if len(tail) >= self._noise_check_window_samples:
                tail_mean = np.mean(tail)
                tail_std = np.std(tail)
                if (
                    tail_mean < cfg.noise_floor_db
                    and tail_std > cfg.noise_std_threshold
                ):
                    return global_max_idx

        return end_region_start

    def _check_enters_noise_region(self, y: np.ndarray, start_idx: int) -> bool:
        """检查从 start_idx 开始是否进入噪声区域。"""
        n = len(y)
        cfg = self._new_cfg
        check_end = min(start_idx + self._noise_check_window_samples, n)
        if check_end - start_idx < 10:
            return False
        segment = y[start_idx:check_end]
        seg_mean = float(np.mean(segment))
        seg_std = float(np.std(segment))
        return seg_mean < cfg.noise_floor_db and seg_std > cfg.noise_std_threshold

    def _find_dense_peak_cluster(
        self, y: np.ndarray, peaks: List[dict]
    ) -> Optional[int]:
        """查找密集峰群的起始索引。"""
        max_gap_km = 0.5
        min_cluster_size = 2
        min_peak_height_db = 2.0
        max_gap_samples = int(max_gap_km / self.sample_spacing_km)

        valid_peaks = [p for p in peaks if p["peak_height_db"] >= min_peak_height_db]
        if len(valid_peaks) < min_cluster_size:
            return None

        cluster_peaks = [valid_peaks[0]]
        for i in range(1, len(valid_peaks)):
            gap = valid_peaks[i]["index"] - valid_peaks[i - 1]["index"]
            if gap <= max_gap_samples:
                cluster_peaks.append(valid_peaks[i])
            else:
                if len(cluster_peaks) >= min_cluster_size:
                    break
                cluster_peaks = [valid_peaks[i]]

        if len(cluster_peaks) >= min_cluster_size:
            return cluster_peaks[0]["index"]
        return None

    def _find_effective_start(self, peaks: List[dict]) -> int:
        """找到有效检测的起点（跳过 skip_start_km 内的所有峰）。"""
        skip_km = self._new_cfg.skip_start_km
        start_idx = int(math.ceil(skip_km / self.sample_spacing_km))
        for p in peaks:
            if p["index"] * self.sample_spacing_km < skip_km:
                right = p.get("right_base_index", p["index"] + 1) + 1
                if right > start_idx:
                    start_idx = right
        return start_idx

    # ── 阶梯下降检测 ─────────────────────────────────────────────

    def _check_step_drop(
        self,
        y: np.ndarray,
        idx: int,
        ignore_peak_indices: Optional[List[int]] = None,
    ) -> Tuple[bool, float]:
        """检查在 idx 位置是否有阶梯下降（幅度 + 斜率双条件）。"""
        n = len(y)
        win = self._step_window_samples

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
        distance_km = (post_center_idx - pre_center_idx) * self.sample_spacing_km

        if distance_km <= 0:
            return False, 0.0

        slope_db_per_km = drop / distance_km
        is_step = slope_db_per_km >= self._new_cfg.step_min_slope_db_per_km

        return is_step, drop

    # ── 峰阶梯关联判断 ───────────────────────────────────────────

    def _is_step_caused_by_peak(
        self,
        y: np.ndarray,
        step_idx: int,
        peaks: List[dict],
        step_drop_db: float,
    ) -> Tuple[bool, Optional[dict]]:
        """判断阶梯下降是否由峰引起（用于弯曲检测）。"""
        cfg = self._new_cfg
        n = len(y)

        search_range_km = 0.05
        search_range_samples = max(1, int(search_range_km / self.sample_spacing_km))
        local_min_range_samples = 15

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
            if abs(peak["index"] - local_max_idx) <= 3:
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

    def _check_plateau_stability(
        self,
        y: np.ndarray,
        step_idx: int,
    ) -> Tuple[bool, float, float]:
        """检查下降点之前（高台阶处）是否是稳定平台。"""
        cfg = self._new_cfg
        window_samples = max(
            10, int(cfg.bend_plateau_window_km / self.sample_spacing_km)
        )

        gap_samples = max(3, self._step_window_samples // 4)
        plateau_end = max(0, step_idx - gap_samples)
        plateau_start = max(0, plateau_end - window_samples)

        if plateau_end <= plateau_start:
            return False, float("inf"), float("inf")

        plateau_data = y[plateau_start:plateau_end]
        if len(plateau_data) < 5:
            return False, float("inf"), float("inf")

        std_db = float(np.std(plateau_data))
        range_db = float(np.max(plateau_data) - np.min(plateau_data))

        is_stable = (
            std_db <= cfg.bend_plateau_max_std_db
            and range_db <= cfg.bend_plateau_max_range_db
        )
        return is_stable, std_db, range_db

    # ── 严重断纤判定 ─────────────────────────────────────────────

    def _is_severe_break_by_peak_profile(
        self,
        y: np.ndarray,
        peak: dict,
        *,
        min_peak_db: float = 5.0,
        min_left_right_min_diff_db: float = 3,
        right_check_samples: int = 15,
        deriv_spike_db_per_sample: float = 0.6,
        max_sign_flips: int = 6,
        max_spike_ratio: float = 0.25,
        smooth_window: int = 3,
        min_descent_db: float = 0.1,
        max_search_samples: int = 100,
    ) -> Tuple[bool, dict]:
        """严重断纤判定（基于下降趋势搜索）。"""
        n = len(y)
        peak_idx = int(peak["index"])

        peak_height = float(peak.get("peak_height_db", 0.0))
        if peak_height <= 0:
            s0 = max(0, peak_idx - 20)
            s1 = min(n, peak_idx + 21)
            peak_height = float(y[peak_idx] - np.mean(y[s0:s1]))

        if peak_height < min_peak_db:
            return False, {
                "reason": "peak_not_high_enough",
                "peak_height_db": peak_height,
            }

        left_idx, left_min = self._find_descent_end(
            y,
            peak_idx,
            direction=-1,
            smooth_window=smooth_window,
            min_descent_db=min_descent_db,
            max_search=max_search_samples,
        )
        right_idx, right_min = self._find_descent_end(
            y,
            peak_idx,
            direction=+1,
            smooth_window=smooth_window,
            min_descent_db=min_descent_db,
            max_search=max_search_samples,
        )

        min_diff = left_min - right_min

        if min_diff <= min_left_right_min_diff_db:
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

    def _find_descent_end(
        self,
        y: np.ndarray,
        start_idx: int,
        *,
        direction: int = 1,
        smooth_window: int = 3,
        min_descent_db: float = 0.1,
        patience: int = 5,
        max_search: int = 100,
        min_window: int = 7,
    ) -> Tuple[int, float]:
        """从 start_idx 沿 direction 搜索下降终点，返回窗口均值最小的位置。"""
        n = len(y)
        if n == 0:
            return start_idx, float("nan")

        if smooth_window > 1:
            from scipy.ndimage import uniform_filter1d

            y_smooth = uniform_filter1d(
                y.astype(float), size=smooth_window, mode="nearest"
            )
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

    def _has_big_peak_near(
        self,
        idx: int,
        peaks: List[dict],
        *,
        radius_km: float,
        min_height_db: float,
    ) -> bool:
        """检查 idx 附近是否有大峰。"""
        radius_samples = max(1, int(math.ceil(radius_km / self.sample_spacing_km)))
        for p in peaks:
            if (
                p.get("peak_height_db", 0.0) >= min_height_db
                and abs(int(p["index"]) - int(idx)) <= radius_samples
            ):
                return True
        return False

    # ── detect 各阶段的独立函数 ───────────────────────────────────

    def _detect_severe_break(
        self,
        y: np.ndarray,
        valid_peaks: List[dict],
        events: List[DetectedEvent],
        processed_peak_indices: set,
    ) -> bool:
        """第零阶段：严重断纤检测。返回是否找到终端事件。"""
        for peak in valid_peaks:
            peak_idx = int(peak["index"])
            z_km = float(self.z[peak_idx])

            ok, info = self._is_severe_break_by_peak_profile(
                y,
                peak,
                min_peak_db=5.0,
                min_left_right_min_diff_db=3,
                right_check_samples=30,
                deriv_spike_db_per_sample=0.6,
                max_sign_flips=6,
                max_spike_ratio=0.25,
            )
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

    def _detect_bend(
        self,
        y: np.ndarray,
        effective_start: int,
        bend_end: int,
        step: int,
        peak_indices: List[int],
        all_peaks: List[dict],
        events: List[DetectedEvent],
    ) -> bool:
        """第一阶段：弯折检测（无反射峰的阶梯下降）。返回是否找到终端事件。"""
        cfg = self._new_cfg
        for i in range(effective_start, bend_end, step):
            has_drop, drop_db = self._check_step_drop(y, i, peak_indices)
            if not (has_drop and drop_db >= cfg.step_drop_severe_db):
                continue

            is_caused_by_peak, _ = self._is_step_caused_by_peak(
                y, i, all_peaks, drop_db
            )
            if is_caused_by_peak:
                continue

            is_stable, plateau_std, plateau_range = self._check_plateau_stability(y, i)
            if is_stable:
                half_drop = drop_db / 3
                baseline = y[i]
                search_end = min(i + self._step_window_samples, len(y))
                precise_index = next(
                    (j for j in range(i, search_end) if baseline - y[j] >= half_drop),
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
                        z_km=float(self.distance_km[precise_index]),
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

    def _find_pre_break_peak(
        self,
        y: np.ndarray,
        drop_idx: int,
        scan_end: int,
        lookahead_samples: int | None = None,
    ) -> int:
        """在检测窗口内寻找局部最高点作为断裂位置。"""
        if lookahead_samples is None:
            lookahead_samples = self._step_window_samples

        left = max(0, drop_idx - 10)
        print(left)
        right = min(scan_end, drop_idx + lookahead_samples)
        segment = y[left:right]
        return left + int(np.argmax(segment))

    def _scan_normal_break(
        self,
        y: np.ndarray,
        scan_start: int,
        scan_end: int,
        peak_indices: List[int],
        all_peaks: List[dict],
        cfg,
    ) -> tuple[int, float] | None:
        """扫描 [scan_start, scan_end) 范围内的普通断纤（无反射峰的阶梯下降）。"""
        scan_step = max(1, self._step_window_samples // 4)
        for i in range(scan_start, scan_end, scan_step):
            has_drop, drop_db = self._check_step_drop(y, i, peak_indices)
            if not has_drop or drop_db < cfg.break_step_drop_db:
                continue
            if self._has_big_peak_near(
                i,
                all_peaks,
                radius_km=cfg.break_no_peak_radius_km,
                min_height_db=cfg.break_no_peak_min_height_db,
            ):
                continue
            break_idx = self._find_pre_break_peak(y, i, scan_end)
            return (break_idx, drop_db)
        return None

    def _scan_small_peak_break(
        self,
        y: np.ndarray,
        valid_peaks: List[dict],
        peak_indices: List[int],
        processed_peak_indices: set[int],
        cfg,
        before_idx: int | None = None,
    ) -> tuple[dict, float, float] | None:
        """
        扫描小峰断纤。
        before_idx 不为 None 时，只考虑 index < before_idx 的峰。
        """
        radius_km = 0.05
        radius2_km = 0.25
        radius_samples = max(1, int(math.ceil(radius_km / self.sample_spacing_km)))
        radius2_samples = max(1, int(math.ceil(radius2_km / self.sample_spacing_km)))

        for peak in valid_peaks:
            peak_idx = peak["index"]
            if peak_idx in processed_peak_indices:
                continue
            if before_idx is not None and peak_idx >= before_idx:
                continue

            peak_height = peak["peak_height_db"]
            if not (
                cfg.min_peak__threshold_db < peak_height < cfg.peak_low_threshold_db
            ):
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
                and (
                    0.9 * peak_height
                    <= p.get("peak_height_db", 0.0)
                    <= 1.1 * peak_height
                )
                and (abs(int(p["index"]) - int(peak_idx)) <= radius2_samples)
                for p in valid_peaks
            )
            if has_similar_peak:
                continue
            peak_width = peak.get("peak_width", 0)
            if not (3 <= peak_width <= 15):
                continue
            # ── 新增：峰左起点外侧必须有10个点的平稳区域 ──
            left_base = peak.get("left_base_index", peak_idx)
            flat_len = 10
            flat_threshold_db = 0.3  # 平稳判定阈值，可按需调整

            # 左侧平稳：left_base 往左10个点
            left_start = left_base - flat_len
            if left_start < 0:
                continue
            left_region = y[left_start:left_base]
            if np.ptp(left_region) > flat_threshold_db:
                continue

            check_idx = peak.get("right_base_index", peak_idx + 5)
            has_drop, drop_db = self._check_step_drop(y, check_idx, peak_indices)
            return (peak, drop_db if has_drop else 0.0, peak_height)

        return None

    def _detect_dirty_connector(
        self,
        y: np.ndarray,
        valid_peaks: List[dict],
        effective_start: int,
        peak_indices: List[int],
        all_peaks: List[dict],
        events: List[DetectedEvent],
        processed_peak_indices: set,
    ) -> bool:
        """第二阶段：脏污检测（含脏污前断纤检测）。"""
        cfg = self._new_cfg
        for peak in valid_peaks:
            peak_idx = peak["index"]
            if peak_idx in processed_peak_indices:
                continue

            peak_height = peak["peak_height_db"]
            z_km = float(self.z[peak_idx])

            if peak_height <= cfg.peak_high_threshold_db:
                continue

            # 检查脏污之前是否有普通断纤
            normal_break = self._scan_normal_break(
                y, effective_start, peak_idx, peak_indices, all_peaks, cfg
            )
            if normal_break is not None:
                break_idx, break_drop_db = normal_break
                events.append(
                    DetectedEvent(
                        kind="break",
                        z_km=float(self.z[break_idx]),
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
            small_peak_break = self._scan_small_peak_break(
                y,
                valid_peaks,
                peak_indices,
                processed_peak_indices,
                cfg,
                before_idx=peak_idx,
            )
            if small_peak_break is not None:
                sp_peak, sp_drop_db, sp_peak_height = small_peak_break
                sp_peak_idx = sp_peak["index"]
                events.append(
                    DetectedEvent(
                        kind="break",
                        z_km=float(self.z[sp_peak_idx]),
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
            has_drop, drop_db = self._check_step_drop(y, check_idx, peak_indices)
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

    def _detect_normal_break(
        self,
        y: np.ndarray,
        effective_start: int,
        effective_end: int,
        offset_samples: int,
        peak_indices: List[int],
        all_peaks: List[dict],
        events: List[DetectedEvent],
    ) -> bool:
        """第三阶段：普通断纤检测。"""
        cfg = self._new_cfg
        break_end = max(effective_start, effective_end - offset_samples)

        result = self._scan_normal_break(
            y, effective_start, break_end, peak_indices, all_peaks, cfg
        )
        if result is None:
            return False

        i, drop_db = result
        events.append(
            DetectedEvent(
                kind="break",
                z_km=float(self.distance_km[i]),
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

    def _detect_small_peak_break(
        self,
        y: np.ndarray,
        valid_peaks: List[dict],
        peak_indices: List[int],
        events: List[DetectedEvent],
        processed_peak_indices: set,
    ) -> bool:
        """第四阶段：小峰断纤检测。"""
        cfg = self._new_cfg

        result = self._scan_small_peak_break(
            y, valid_peaks, peak_indices, processed_peak_indices, cfg
        )
        if result is None:
            return False

        peak, drop_db, peak_height = result
        peak_idx = peak["index"]
        events.append(
            DetectedEvent(
                kind="break",
                z_km=float(self.z[peak_idx]),
                magnitude_db=float(drop_db),
                reflect_db=float(peak_height),
                index=peak_idx,
                extra={"subtype": "small_peak_break"},
            )
        )
        processed_peak_indices.add(peak_idx)
        return True

    def detect(
        self,
        trace_db: np.ndarray,
        _fiber_index: float | None = None,
        _sample_rate_mhz: float | None = None,
    ) -> DetectionResult:
        """
        执行检测。

        检测顺序：
        0. 严重断纤
        1. 弯折（无反射峰的阶梯下降）
        2. 脏污（如果脏污前有普通断纤或小峰断纤，则报断裂）
        3. 普通断纤
        4. 小峰断纤

        如果某阶段检测到事件，后续阶段的扫描终点缩小到该事件前100个点。
        """
        y = np.asarray(trace_db, dtype=float)
        cfg = self._new_cfg
        LOOKBACK = 35

        # 基线处理
        if self.baseline is None:
            baseline = self._fit_linear_baseline(self.z, y)
        else:
            baseline = self.baseline.copy()
            if len(baseline) != len(y):
                baseline = np.interp(
                    self.z,
                    np.linspace(0, self.z[-1], len(baseline)),
                    baseline,
                )

        residual = y - baseline

        # 1. 找所有反射峰
        all_peaks = self._find_peaks(y)
        peak_indices = [p["index"] for p in all_peaks]

        # 2. 确定有效检测范围
        effective_end = self._find_effective_end(y, all_peaks)
        effective_start = self._find_effective_start(all_peaks)
        offset_samples = int(cfg.offset_samples_km / self.sample_spacing_km)

        # 3. 筛选有效峰
        valid_peaks = [
            p
            for p in all_peaks
            if effective_start <= p["index"] <= effective_end - offset_samples
            and p["peak_height_db"] >= cfg.peak_min_prominence_db
        ]

        events: List[DetectedEvent] = []
        processed_peak_indices: set[int] = set()
        bend_end = max(effective_start, effective_end - offset_samples)
        step = max(1, self._step_window_samples // 4)

        # ── 阶段0：严重断纤 ──
        self._detect_severe_break(y, valid_peaks, events, processed_peak_indices)

        # ── 阶段1：弯折 ──
        cur_end = self._get_scan_end(events, bend_end, effective_start, LOOKBACK)
        if cur_end > effective_start:
            self._detect_bend(
                y, effective_start, cur_end, step, peak_indices, all_peaks, events
            )

        # ── 阶段2：脏污 ──
        cur_end = self._get_scan_end(events, bend_end, effective_start, LOOKBACK)
        if cur_end > effective_start:
            cur_peaks = self._filter_peaks_before(valid_peaks, cur_end)
            if cur_peaks:
                self._detect_dirty_connector(
                    y,
                    cur_peaks,
                    effective_start,
                    peak_indices,
                    all_peaks,
                    events,
                    processed_peak_indices,
                )

        # ── 阶段3：普通断纤 ──
        cur_end = self._get_scan_end(events, effective_end, effective_start, LOOKBACK)
        if cur_end > effective_start:
            self._detect_normal_break(
                y,
                effective_start,
                cur_end,
                0 if events else offset_samples,
                peak_indices,
                all_peaks,
                events,
            )

        # ── 阶段4：小峰断纤 ──
        cur_end = self._get_scan_end(events, effective_end, effective_start, LOOKBACK)
        if cur_end > effective_start:
            cur_peaks = self._filter_peaks_before(valid_peaks, cur_end)
            if cur_peaks:
                self._detect_small_peak_break(
                    y,
                    cur_peaks,
                    peak_indices,
                    events,
                    processed_peak_indices,
                )

        events.sort(key=lambda e: e.index)

        return DetectionResult(
            events=events,
            distance_km=self.z,
            trace_db=y,
            baseline_db=baseline,
            residual_db=residual,
        )

    # ── 辅助方法 ──────────────────────────────────────────────────

    @staticmethod
    def _get_scan_end(
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

    @staticmethod
    def _filter_peaks_before(peaks: List[dict], cutoff: int) -> List[dict]:
        """只保留 index < cutoff 的峰。"""
        return [p for p in peaks if p["index"] < cutoff]
