from __future__ import annotations

import logging

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def sanitize_depth(depth: np.ndarray, *, depth_unit: str = "meter") -> np.ndarray:
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.dtype == np.uint16:
        if depth_unit == "meter":
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32)
    else:
        depth = depth.astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0] = 0.0
    return depth[..., None]


def normalize_depth_for_controlnet(depth: np.ndarray, *, clip_percentiles: tuple[float, float] = (2.0, 98.0)) -> np.ndarray:
    depth = sanitize_depth(depth)[..., 0]
    valid = depth[depth > 0]
    if valid.size == 0:
        return np.zeros_like(depth, dtype=np.uint8)
    low, high = np.percentile(valid, clip_percentiles)
    if high <= low:
        high = low + 1e-6
    normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8)


def align_depth_to_rgb(depth, rgb_intrinsics=None, depth_intrinsics=None, extrinsics=None) -> np.ndarray:
    depth = sanitize_depth(depth)
    if rgb_intrinsics is None or depth_intrinsics is None or extrinsics is None:
        LOGGER.warning("RGB-D calibration metadata missing; returning depth unchanged.")
        return depth
    # TODO: implement projective alignment using intrinsics/extrinsics once camera calibration plumbing is exposed by LeRobot.
    return depth


def depth_to_uint8_preview(depth: np.ndarray) -> np.ndarray:
    preview = normalize_depth_for_controlnet(depth)
    return cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)


def validate_rgbd_pair(rgb, depth) -> None:
    rgb = np.asarray(rgb)
    depth = sanitize_depth(depth)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"RGB image must have shape (H, W, 3), got {rgb.shape}")
    if depth.ndim != 3 or depth.shape[-1] != 1:
        raise ValueError(f"Depth image must have shape (H, W, 1), got {depth.shape}")
    if rgb.shape[:2] != depth.shape[:2]:
        raise ValueError(f"RGB/depth spatial shapes differ: {rgb.shape[:2]} vs {depth.shape[:2]}")
