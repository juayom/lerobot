from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from lerobot.genaug.geometry.depth_utils import sanitize_depth


@dataclass
class DepthMaskDiagnostics:
    valid_ratio: float
    valid_depth_min_m: float
    valid_depth_max_m: float
    valid_depth_median_m: float
    foreground_depth_threshold_m: float
    mask_area_ratio: float
    component_count: int
    selected_component_area: int
    depth_contrast_m: float
    is_depth_usable: bool
    is_mask_usable: bool
    summary: str


def _largest_centered_component(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask), 0, 0

    h, w = mask.shape
    center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

    best_label = 0
    best_score = -1e18
    best_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        cy, cx = float(centroids[label][1]), float(centroids[label][0])
        dist = np.linalg.norm(np.array([cx, cy], dtype=np.float32) - center)
        score = float(area) - 2.0 * dist
        if score > best_score:
            best_score = score
            best_label = label
            best_area = area

    selected = (labels == best_label).astype(np.uint8) * 255 if best_label > 0 else np.zeros_like(mask)
    return selected, num_labels - 1, best_area


def estimate_object_mask_from_depth(
    depth: np.ndarray,
    *,
    near_percentile: float = 35.0,
    min_area_ratio: float = 0.002,
    max_area_ratio: float = 0.35,
) -> tuple[np.ndarray, DepthMaskDiagnostics]:
    depth_m = sanitize_depth(depth)[..., 0]
    h, w = depth_m.shape
    total_area = float(h * w)

    valid = np.isfinite(depth_m) & (depth_m > 0)
    valid_ratio = float(valid.mean())
    if valid_ratio == 0.0:
        diag = DepthMaskDiagnostics(
            valid_ratio=0.0,
            valid_depth_min_m=0.0,
            valid_depth_max_m=0.0,
            valid_depth_median_m=0.0,
            foreground_depth_threshold_m=0.0,
            mask_area_ratio=0.0,
            component_count=0,
            selected_component_area=0,
            depth_contrast_m=0.0,
            is_depth_usable=False,
            is_mask_usable=False,
            summary="No valid depth pixels.",
        )
        return np.zeros((h, w), dtype=np.uint8), diag

    valid_depth = depth_m[valid]
    d_min = float(valid_depth.min())
    d_max = float(valid_depth.max())
    d_med = float(np.median(valid_depth))
    d_thr = float(np.percentile(valid_depth, near_percentile))

    foreground = np.zeros((h, w), dtype=np.uint8)
    foreground[(depth_m > 0) & (depth_m <= d_thr)] = 255

    kernel = np.ones((5, 5), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

    selected, component_count, selected_area = _largest_centered_component(foreground)
    mask_area_ratio = float(selected_area / total_area)

    selected_valid = selected > 0
    bg_valid = valid & (~selected_valid)
    selected_depth = depth_m[selected_valid] if np.any(selected_valid) else np.array([], dtype=np.float32)
    bg_depth = depth_m[bg_valid] if np.any(bg_valid) else np.array([], dtype=np.float32)
    if selected_depth.size > 0 and bg_depth.size > 0:
        depth_contrast = float(np.median(bg_depth) - np.median(selected_depth))
    else:
        depth_contrast = 0.0

    is_depth_usable = valid_ratio > 0.25 and (d_max - d_min) > 0.05
    is_mask_usable = (
        is_depth_usable
        and selected_area > 0
        and min_area_ratio <= mask_area_ratio <= max_area_ratio
        and depth_contrast > 0.03
    )

    summary = (
        f"valid_ratio={valid_ratio:.3f}, depth_range=({d_min:.3f},{d_max:.3f})m, "
        f"mask_area_ratio={mask_area_ratio:.4f}, depth_contrast={depth_contrast:.3f}m"
    )

    diag = DepthMaskDiagnostics(
        valid_ratio=valid_ratio,
        valid_depth_min_m=d_min,
        valid_depth_max_m=d_max,
        valid_depth_median_m=d_med,
        foreground_depth_threshold_m=d_thr,
        mask_area_ratio=mask_area_ratio,
        component_count=component_count,
        selected_component_area=selected_area,
        depth_contrast_m=depth_contrast,
        is_depth_usable=is_depth_usable,
        is_mask_usable=is_mask_usable,
        summary=summary,
    )
    return selected, diag


def apply_background_swap(image: np.ndarray, object_mask: np.ndarray, *, seed: int = 0) -> np.ndarray:
    image = np.asarray(image).copy()
    mask = np.asarray(object_mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    h, w = image.shape[:2]
    rng = np.random.default_rng(seed)
    bg = np.zeros_like(image)

    yy, xx = np.indices((h, w))
    stripes = (((xx // 32) + (yy // 32)) % 2).astype(np.uint8)
    colors = np.array([[40, 90, 180], [180, 200, 60]], dtype=np.uint8)
    bg = colors[stripes]

    noise = rng.integers(low=0, high=20, size=bg.shape, dtype=np.uint8)
    bg = np.clip(bg.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)

    out = image.copy()
    out[mask == 0] = bg[mask == 0]
    return out
