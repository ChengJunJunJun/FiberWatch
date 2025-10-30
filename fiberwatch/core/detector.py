from dataclasses import dataclass, field
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


@dataclass
class DetectorConfig:
    """Configuration for the minimal break-only detector."""

    # Smoothing
    smooth_win: int = 21
    smooth_poly: int = 3

    # Legacy parameters kept for API compatibility with older callers.
    # They are currently unused in the break-only detector.
    refl_min_db: float = 1.0
    step_min_db: float = 0.05
    slope_min_db_per_km: float = 0.05
    min_event_separation: int = 30

    # Break detection windows (expressed as physical distances in km)
    pre_window_km: float = 0.30
    pre_end_offset_km: float = 0.025
    tail_start_offset_km: float = 0.05
    tail_end_offset_km: float = 0.55

    # Thresholds
    min_signal_drop_db: float = 5.0
    noise_floor_db: float = -80.0
    min_noise_increase: float = 1.5
    min_zero_crossing_ratio: float = 0.05  # proportion of sign flips in tail derivative
    min_tail_segment_len_km: float = 0.075
    # Derivative thresholding
    grad_sigma_factor: float = (
        3.0  # how many std devs below mean to consider a sharp drop
    )
    min_grad_abs: float = 0.005  # absolute minimal negative derivative (dB/sample)

    # Dirty-connector detection
    dirty_grad_sigma_factor: float = 6.0  # stricter: large spikes only
    min_dirty_grad_abs: float = 0.001  # minimal |d(diff)|
    step_window_km: float = 0.15  # physical window for local step estimation
    dirty_min_step_db: float = 1.5  # require large diff step for dirty connector
    dirty_exclusion_before_break_km: float = 0.5  # 断点处向前500m不检测脏连接器
    dirty_duplicate_skip_km: float = 0.025  # 跳过重复脏连接器的最小间隔

    # Bend detection (two small steps newly appearing in test)
    bend_grad_sigma_factor: float = 2.0
    min_bend_grad_abs: float = 0.0005
    bend_pair_max_gap_km: float = 0.125  # max gap between paired steps (if used)
    bend_min_step_db: float = 0.05  # 越大越严格
    bend_max_step_db: float = 1.2
    bend_step_window_km: float = 0.075  # 越小越严格
    # 连续下降检测相关参数
    bend_min_descent_len_km: float = 0.05  # 连续负斜率的最小长度
    bend_dirty_exclusion_km: float = 0.5  # 脏连接器±0.5km范围内不计入bend

    def validate(self) -> None:
        """Validate configuration values."""

        if self.smooth_win <= 0 or self.smooth_win % 2 == 0:
            raise ValueError("smooth_win must be positive and odd")
        if self.smooth_poly <= 0:
            raise ValueError("smooth_poly must be positive")
        if self.refl_min_db <= 0:
            raise ValueError("refl_min_db must be positive")
        if self.step_min_db <= 0:
            raise ValueError("step_min_db must be positive")
        if self.pre_window_km <= 0:
            raise ValueError("pre_window_km must be positive")
        if self.tail_end_offset_km <= self.tail_start_offset_km:
            raise ValueError("tail_end_offset_km must exceed tail_start_offset_km")
        if self.min_tail_segment_len_km <= 0:
            raise ValueError("min_tail_segment_len_km must be positive")
        if self.step_window_km <= 0:
            raise ValueError("step_window_km must be positive")
        if self.bend_min_descent_len_km <= 0:
            raise ValueError("bend_min_descent_len_km must be positive")


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
    trace_smooth_db: np.ndarray
    baseline_db: np.ndarray
    residual_db: np.ndarray

    def plot(self, outfile: str | None = None) -> plt.Figure:
        z = self.distance_km
        y = self.trace_smooth_db
        b = self.baseline_db
        fig = plt.figure(figsize=(10, 4))
        plt.plot(z, y, label="trace (smooth)")
        plt.plot(z, b, "--", label="baseline")
        for ev in self.events:
            plt.axvline(ev.z_km, linestyle=":", alpha=0.6)
            txt = f"{ev.kind} @ {ev.z_km:.2f}km"
            if ev.kind == "break":
                txt += f" Δ~{ev.magnitude_db:.1f}dB"
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
        return fig  # 返回figure对象而不是plt模块


class Detector:
    """
    Minimal detector that only identifies fiber break events.

    The algorithm looks for a sharp negative change in the first derivative,
    followed by a noisy tail where the derivative keeps flipping sign, and the
    mean level drops significantly (and near noise floor).
    """

    # Constants to avoid magic values
    MIN_WINDOW_SIZE = 5
    MIN_ARRAY_LENGTH = 2
    MIN_DERIVATIVE_SPAN_KM = 0.025
    DEFAULT_MIN_DERIVATIVE_SAMPLES = 10

    def __init__(
        self,
        distance_km: np.ndarray,
        baseline: np.ndarray | None = None,
        config: DetectorConfig | None = None,
        *,
        sample_rate_per_km: float | None = None,
    ) -> None:
        if config is None:
            config = DetectorConfig()
        self.z = np.asarray(distance_km)
        self.distance_km = self.z
        self.baseline = None if baseline is None else np.asarray(baseline)
        # ensure odd window length for SG filter
        self.smooth_win = int(max(5, config.smooth_win) // 2 * 2 + 1)
        self.smooth_poly = int(config.smooth_poly)
        self.cfg = config
        self.config = config

        if sample_rate_per_km and sample_rate_per_km > 0:
            self.sample_rate_per_km: float | None = float(sample_rate_per_km)
            spacing = self._sample_spacing_from_rate(self.sample_rate_per_km)
        else:
            self.sample_rate_per_km = None
            spacing = self._infer_sample_spacing_from_distance()

        if spacing <= 0:
            spacing = 0.0

        self._apply_sample_spacing(spacing)

    @staticmethod
    def _fit_linear_baseline(z: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Robust-ish: fit between 25th and 50th percentiles to avoid reflections
        q1, q2 = np.nanpercentile(y, [25, 50])
        msk = (y <= q2) & (y >= q1)
        if not np.any(msk):
            msk = np.ones_like(y, dtype=bool)
        design_matrix = np.vstack([z[msk], np.ones(np.sum(msk))]).T
        m, c = np.linalg.lstsq(design_matrix, y[msk], rcond=None)[0]
        return m * z + c

    def _smooth(self, y: np.ndarray) -> np.ndarray:
        win = min(self.smooth_win, len(y) // 2 * 2 - 1)
        if win < self.MIN_WINDOW_SIZE:
            return y.astype(float)
        win = max(self.MIN_WINDOW_SIZE, win)
        if win % 2 == 0:
            win += 1
        return savgol_filter(
            y,
            window_length=win,
            polyorder=self.smooth_poly,
            mode="interp",
        )

    @staticmethod
    def _sample_spacing_from_rate(rate_per_km: float | None) -> float:
        if rate_per_km is None or rate_per_km <= 0:
            return 0.0
        return 1.0 / float(rate_per_km)

    def _infer_sample_spacing_from_distance(self) -> float:
        if len(self.z) > 1:
            diffs = np.diff(self.z)
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size:
                return float(np.median(diffs))
        return 0.0

    def _apply_sample_spacing(self, spacing_km: float) -> None:
        self.sample_spacing_km = float(spacing_km) if spacing_km > 0 else 0.0

        cfg = self.cfg
        self._pre_window_samples = self._samples_from_distance(cfg.pre_window_km)
        self._pre_end_offset_samples = self._samples_from_distance(
            cfg.pre_end_offset_km,
            allow_zero=True,
        )
        self._tail_start_offset_samples = self._samples_from_distance(
            cfg.tail_start_offset_km,
            allow_zero=True,
        )
        self._tail_end_offset_samples = self._samples_from_distance(
            cfg.tail_end_offset_km,
            allow_zero=True,
        )
        if self._tail_end_offset_samples <= self._tail_start_offset_samples:
            self._tail_end_offset_samples = self._tail_start_offset_samples + 1

        self._min_tail_segment_samples = self._samples_from_distance(
            cfg.min_tail_segment_len_km,
        )
        self._step_window_samples = self._samples_from_distance(
            cfg.step_window_km,
        )
        self._bend_min_descent_samples = self._samples_from_distance(
            cfg.bend_min_descent_len_km,
        )
        self._dirty_duplicate_skip_samples = max(
            1,
            self._samples_from_distance(
                cfg.dirty_duplicate_skip_km,
                allow_zero=True,
            ),
            self._step_window_samples // 2,
        )
        self._min_derivative_samples = self._samples_from_distance(
            self.MIN_DERIVATIVE_SPAN_KM,
            allow_zero=True,
        )
        if self._min_derivative_samples <= 0:
            self._min_derivative_samples = self.DEFAULT_MIN_DERIVATIVE_SAMPLES

    def _samples_from_distance(
        self,
        distance_km: float,
        *,
        allow_zero: bool = False,
        min_samples: int = 1,
    ) -> int:
        if distance_km <= 0 or self.sample_spacing_km <= 0:
            return 0 if allow_zero else min_samples
        samples = int(math.ceil(distance_km / self.sample_spacing_km))
        if allow_zero:
            return max(0, samples)
        return max(min_samples, samples)

    def _count_zero_crossings(self, x: np.ndarray) -> int:
        if len(x) < self.MIN_ARRAY_LENGTH:
            return 0
        signs = np.sign(np.clip(x, -1e9, 1e9))
        # treat zeros as previous sign to avoid inflating crossings
        for i in range(1, len(signs)):
            if signs[i] == 0:
                signs[i] = signs[i - 1]
        return int(np.sum(signs[1:] * signs[:-1] < 0))

    def _detect_break(
        self,
        y_s: np.ndarray,
        y_raw: np.ndarray,
        effective_end_index: int | None = None,
    ) -> list[DetectedEvent]:
        cfg = self.cfg
        dy = np.gradient(y_s)
        dy_raw = np.gradient(y_raw)
        events: list[DetectedEvent] = []

        # Search region avoids last tail window to ensure post segment exists
        default_end = max(0, len(y_s) - self._tail_end_offset_samples - 1)
        search_end = (
            min(default_end, effective_end_index)
            if effective_end_index is not None
            else default_end
        )
        best: DetectedEvent | None = None
        for i in range(search_end):
            # windows
            pre_a = max(0, i - self._pre_window_samples)
            pre_b = max(pre_a, i - self._pre_end_offset_samples)
            tail_a = i + self._tail_start_offset_samples
            tail_b = i + self._tail_end_offset_samples

            pre_segment = y_s[pre_a:pre_b]
            tail_segment = y_s[tail_a:tail_b]
            if (
                len(pre_segment) < self._min_tail_segment_samples
                or len(tail_segment) < self._min_tail_segment_samples
            ):
                continue

            pre_mean = float(np.mean(pre_segment))
            pre_std = float(np.std(pre_segment))
            tail_mean = float(np.mean(tail_segment))
            tail_std = float(np.std(tail_segment))

            signal_drop = pre_mean - tail_mean
            noise_increase = tail_std / max(pre_std, 1e-3)

            # derivative sign-flip criterion in tail
            dy_tail = dy_raw[tail_a:tail_b]
            zero_crossings = self._count_zero_crossings(dy_tail)
            zero_cross_ratio = zero_crossings / max(1, len(dy_tail))

            # dynamic derivative threshold using pre-window derivative stats
            dy_pre = dy[pre_a:pre_b]
            if len(dy_pre) < self._min_derivative_samples:  # 在这个判定下，前50m的故障点检测不到
                continue
            dy_pre_std = float(np.std(dy_pre))
            grad_threshold = max(cfg.min_grad_abs, cfg.grad_sigma_factor * dy_pre_std)
            is_sharp_negative = dy[i] < -grad_threshold

            if (
                is_sharp_negative
                and signal_drop >= cfg.min_signal_drop_db
                and zero_cross_ratio >= cfg.min_zero_crossing_ratio
            ):
                ev = DetectedEvent(
                    kind="break",
                    z_km=float(self.z[i]),
                    magnitude_db=float(abs(signal_drop)),
                    reflect_db=0.0,
                    index=int(i),
                    extra={
                        "zero_cross_ratio": zero_cross_ratio,
                        "noise_increase": noise_increase,
                        "dy": float(dy[i]),
                        "grad_threshold": grad_threshold,
                    },
                )
                if (
                    best is None
                    or ev.index < best.index
                    or ev.magnitude_db > best.magnitude_db
                ):
                    best = ev

        if best is not None:
            events.append(best)
        return events

    def _detect_dirty_connectors(
        self,
        y_s: np.ndarray,
        baseline: np.ndarray,
        break_index: int | None,
        effective_end_index: int | None,
        exclude_indices: list[int] | None = None,
    ) -> list[DetectedEvent]:
        """
        Detect dirty connector events using differential derivative spikes.

        diff = baseline - y_s  (per user spec)
        Look for spikes in |d(diff)| before the first break.
        """
        cfg = self.cfg
        diff = baseline - y_s
        diff_s = self._smooth(diff)
        dd = np.gradient(diff_s)

        # limit to region before break if provided
        end = break_index if break_index is not None else len(dd)
        if effective_end_index is not None:
            end = min(end, effective_end_index)
        # exclude the last configured window before the break position
        if break_index is not None and end > 0:
            break_z = float(self.z[break_index])
            limit_z = break_z - cfg.dirty_exclusion_before_break_km
            pre_end_idx = int(np.searchsorted(self.z, limit_z, side="right") - 1)
            end = max(0, min(end, pre_end_idx + 1))
        if end <= 0:
            return []
        region = dd[:end]
        # robust threshold based on MAD or std of central band
        # use std for simplicity
        std = float(np.std(region))
        thr = max(cfg.min_dirty_grad_abs, cfg.dirty_grad_sigma_factor * std)

        events: list[DetectedEvent] = []
        i = 1
        while i < end - 1:
            if (
                abs(dd[i]) > thr
                and abs(dd[i]) >= abs(dd[i - 1])
                and abs(dd[i]) >= abs(dd[i + 1])
            ):
                # skip if near excluded indices (e.g., detected bends)
                if exclude_indices and any(
                    abs(i - j) <= self._dirty_duplicate_skip_samples
                    for j in exclude_indices
                ):
                    i += 1
                    continue
                # estimate local step on differential curve
                a = max(0, i - self._step_window_samples)
                b = min(len(dd), i + self._step_window_samples)
                mid = i
                before = float(np.mean(diff_s[a:mid]))
                after = float(np.mean(diff_s[mid:b]))
                step_db = before - after
                if abs(step_db) < cfg.dirty_min_step_db:#这个地方是不是不需要取绝对值，因为是尖峰而不是山谷
                    i += 1
                    continue
                events.append(
                    DetectedEvent(
                        kind="dirty_connector",
                        z_km=float(self.z[i]),
                        magnitude_db=float(step_db),
                        reflect_db=0.0,
                        index=int(i),
                        extra={"abs_d_diff": float(abs(dd[i])), "dirty_thr": thr},
                    ),
                )
                # skip a separation window to avoid duplicates
                i += self._dirty_duplicate_skip_samples
                continue
            i += 1
        return events
    
    def _detect_bends(
        self,
        y_s: np.ndarray,
        baseline: np.ndarray,
        break_index: int | None,
        effective_end_index: int | None,
        exclude_indices: list[int] | None = None,
    ) -> list[DetectedEvent]:
        
        cfg = self.cfg
        resid = y_s - baseline
        resid_s = self._smooth(resid)
        dr = np.gradient(resid_s)

        # region limit
        end = break_index if break_index is not None else len(dr)
        if effective_end_index is not None:
            end = min(end, effective_end_index)
        if end <= 0:
            return []

        events: list[DetectedEvent] = []
        # Precompute dirty-connector positions for distance-based exclusion
        dirty_positions_km: np.ndarray | None = None
        if exclude_indices:
            dirty_positions_km = self.z[np.array(exclude_indices, dtype=int)]

        #确定弯折的区间
        in_run = False
        run_start = 0
        for i in range(end):
            if i < end - 1:
                # 正常点
                is_neg = dr[i] < -max(0.0, cfg.min_bend_grad_abs)
            else:
                # 虚拟点：强制认为不是负斜率
                is_neg = False

            if is_neg and not in_run:
                in_run = True
                run_start = i
            elif not is_neg and in_run:
                run_end = i - 1
                run_len = run_end - run_start + 1
                mid_idx = run_start + run_len // 2
                drop_db = float(resid_s[run_start] - resid_s[run_end])
                #判断是否满足弯折的要求
                if (
                    run_len >= self._bend_min_descent_samples
                    and cfg.bend_min_step_db <= drop_db <= cfg.bend_max_step_db 
                    and(
                        dirty_positions_km is None 
                        or not np.any(
                            np.abs(self.z[mid_idx] - dirty_positions_km)<= cfg.bend_dirty_exclusion_km
                        )
                    )
                ):
                    events.append(
                        DetectedEvent(
                            kind="bend",
                            z_km=float(self.z[mid_idx]),
                            magnitude_db=float(-drop_db),
                            reflect_db=0.0,
                            index=int(run_start),
                            extra={"descent_len": int(run_len)},
                        ),
                    )
                in_run = False
        return events

    def _find_baseline_end_index(self, baseline_trace: np.ndarray) -> int | None:
        """
        Find fiber end on baseline only (approximate break position on baseline).

        Heuristic: look for index where the subsequent tail segment mean is below
        the noise floor and the drop from a preceding segment exceeds threshold.
        """
        cfg = self.cfg
        n = len(baseline_trace)
        limit = max(0, n - self._tail_end_offset_samples - 1)
        for i in range(limit):
            tail_start = i + self._tail_start_offset_samples
            tail_end = min(n, i + self._tail_end_offset_samples)
            if tail_end - tail_start < self._min_tail_segment_samples:
                continue
            tail_segment = baseline_trace[tail_start:tail_end]
            tail_mean = float(np.mean(tail_segment))
            if tail_mean < cfg.noise_floor_db:
                pre_a = max(0, i - self._pre_window_samples)
                pre_b = max(pre_a, i - self._pre_end_offset_samples)
                if pre_b - pre_a < self._min_tail_segment_samples:
                    continue
                pre_segment = baseline_trace[pre_a:pre_b]
                pre_mean = float(np.mean(pre_segment))
                signal_drop = pre_mean - tail_mean
                if signal_drop >= cfg.min_signal_drop_db:
                    return i
        return None

    def detect(
        self,
        trace_db: np.ndarray,
        _fiber_index: float | None = None,
        _sample_rate_mhz: float | None = None,
        *,
        sample_rate_per_km: float | None = None,
    ) -> DetectionResult:
        if sample_rate_per_km and sample_rate_per_km > 0:
            rate_val = float(sample_rate_per_km)
            if self.sample_rate_per_km != rate_val or self.sample_spacing_km <= 0:
                self.sample_rate_per_km = rate_val
                self._apply_sample_spacing(self._sample_spacing_from_rate(rate_val))
        elif self.sample_rate_per_km and self.sample_rate_per_km > 0:
            if self.sample_spacing_km <= 0:
                self._apply_sample_spacing(
                    self._sample_spacing_from_rate(self.sample_rate_per_km)
                )
        elif self.sample_spacing_km <= 0:
            inferred = self._infer_sample_spacing_from_distance()
            if inferred > 0:
                self._apply_sample_spacing(inferred)

        y = np.asarray(trace_db)
        y_s = self._smooth(y)

        # 我觉得如果baseline没有提供就可以报错了，因为除了断裂以外的两个点都是通过baseline的对比来求的
        if self.baseline is None:
            baseline = self._fit_linear_baseline(self.z, y_s)
        else:
            baseline = self.baseline
            if len(baseline) != len(y_s):
                baseline = np.interp(
                    self.z,
                    np.linspace(0, self.z[-1], len(baseline)),
                    baseline,
                )

        resid = y_s - baseline

        # Determine effective end based on baseline-only end detection
        baseline_end_index = self._find_baseline_end_index(baseline)
        if (
            baseline_end_index is not None
            and baseline_end_index < self._pre_window_samples
        ):
            baseline_end_index = None

        # 1) detect break first (limited before baseline end if available)
        break_events = self._detect_break(
            y_s,
            y,
            effective_end_index=baseline_end_index,
        )
        first_break_index = break_events[0].index if break_events else None

        # 2) detect dirty connectors before the break (first, as requested)
        dirty_events = self._detect_dirty_connectors(
            y_s,
            baseline,
            first_break_index,
            effective_end_index=baseline_end_index,
        )

        # 3) detect bends (small diff steps) excluding dirty connectors
        bend_events = self._detect_bends(
            y_s,
            baseline,
            first_break_index,
            effective_end_index=baseline_end_index,
            exclude_indices=[e.index for e in dirty_events],
        )

        # aggregate events: dirty, bends, then break
        events = dirty_events + bend_events + break_events

        return DetectionResult(
            events=events,
            distance_km=self.z,
            trace_smooth_db=y_s,
            baseline_db=baseline,
            residual_db=resid,
        )
