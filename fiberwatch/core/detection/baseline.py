"""基线拟合。"""

import numpy as np


def fit_linear_baseline(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """线性基线拟合。"""
    q1, q2 = np.nanpercentile(y, [25, 50])
    msk = (y <= q2) & (y >= q1)
    if not np.any(msk):
        msk = np.ones_like(y, dtype=bool)
    design_matrix = np.vstack([z[msk], np.ones(np.sum(msk))]).T
    m, c = np.linalg.lstsq(design_matrix, y[msk], rcond=None)[0]
    return m * z + c
