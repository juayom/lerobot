from __future__ import annotations

import numpy as np


def to_hwc_image(image) -> tuple[np.ndarray, str]:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image array, got {arr.shape}")
    if arr.shape[-1] == 3:
        return arr, "hwc"
    if arr.shape[0] == 3:
        return np.moveaxis(arr, 0, -1), "chw"
    raise ValueError(f"Unsupported RGB layout: {arr.shape}")


def from_hwc_image(image_hwc: np.ndarray, layout: str) -> np.ndarray:
    if layout == "hwc":
        return image_hwc
    if layout == "chw":
        return np.moveaxis(image_hwc, -1, 0)
    raise ValueError(f"Unsupported image layout tag: {layout}")


def to_hwc_depth(depth) -> tuple[np.ndarray, str]:
    arr = np.asarray(depth)
    if arr.ndim == 2:
        return arr[..., None], "hw"
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D depth array, got {arr.shape}")
    if arr.shape[-1] == 1:
        return arr, "hwc"
    if arr.shape[0] == 1:
        return np.moveaxis(arr, 0, -1), "chw"
    raise ValueError(f"Unsupported depth layout: {arr.shape}")


def from_hwc_depth(depth_hwc: np.ndarray, layout: str) -> np.ndarray:
    if layout == "hw":
        return depth_hwc[..., 0]
    if layout == "hwc":
        return depth_hwc
    if layout == "chw":
        return np.moveaxis(depth_hwc, -1, 0)
    raise ValueError(f"Unsupported depth layout tag: {layout}")
