from __future__ import annotations

import numpy as np


def normalize_depth(depth: np.ndarray, *, min_value: float | None = None, max_value: float | None = None) -> np.ndarray:
    depth = depth.astype(np.float32)
    if min_value is None:
        min_value = float(np.nanmin(depth))
    if max_value is None:
        max_value = float(np.nanmax(depth))
    denom = max(max_value - min_value, 1e-6)
    return np.clip((depth - min_value) / denom, 0.0, 1.0)
