"""数据结构定义，供所有模块引用，避免循环导入。"""

from dataclasses import dataclass, field

import numpy as np


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
    reflection_peaks: list = field(default_factory=list)

    @property
    def trace_smooth_db(self) -> np.ndarray:
        """visualization.py 通过此属性访问数据。"""
        return self.trace_db

    def plot(self, outfile: str | None = None):
        """绘图，延迟导入避免硬依赖 matplotlib。"""
        from ..utils.visualization import plot_detection_result

        return plot_detection_result(self, outfile)
