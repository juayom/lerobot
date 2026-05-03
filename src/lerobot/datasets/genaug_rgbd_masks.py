from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

from lerobot.genaug.geometry.depth_utils import sanitize_depth

LOGGER = logging.getLogger(__name__)

MIN_COMPONENT_AREA_PX = 1_200
BORDER_GUARD_PX = 24
MORPH_KERNEL = 5
DEFAULT_QUANTILE = 35.0
DEFAULT_ABS_THRESHOLD_M: float | None = None
MAX_FOREGROUND_AREA_RATIO = 0.40
MIN_FOREGROUND_AREA_RATIO = 0.003
VALID_DEPTH_MIN_M = 0.25
VALID_DEPTH_MAX_M = 2.50
ROI_X_RANGE = (0.20, 0.80)
ROI_Y_RANGE = (0.18, 0.88)
ROI_NEAR_PERCENTILE = 12.0
ROI_UPPER_PERCENTILE = 38.0
DEPTH_BAND_BELOW_M = 0.03
DEPTH_BAND_ABOVE_M = 0.12


@dataclass
class CandidateStats:
    label: int
    bbox_xywh: list[int]
    area: int
    centroid_xy: list[float]
    aspect_ratio: float
    extent: float
    mean_depth_m: float
    median_depth_m: float
    min_depth_m: float
    max_depth_m: float
    touch_right_border: bool
    touch_bottom_border: bool
    vertical_position_score: float
    bottle_score: float
    box_score: float


@dataclass
class MaskDiagnostics:
    frame_index: int
    valid_ratio: float
    foreground_threshold_m: float
    num_components: int
    selected_component_area: int
    bottle_area: int
    box_area: int
    split_success: bool
    split_confidence: float
    bottle_confidence: float
    box_confidence: float
    failure_reason: str
    notes: list[str] = field(default_factory=list)
    candidate_count: int = 0
    fallback_split_used: bool = False
    semantic_label_verified: bool = False
    rgb_shape: list[int] = field(default_factory=list)
    depth_shape: list[int] = field(default_factory=list)
    valid_mask_shape: list[int] = field(default_factory=list)
    threshold_candidate_centroid_xy: list[float] = field(default_factory=list)
    threshold_candidate_bbox_xywh: list[int] = field(default_factory=list)
    threshold_candidate_area: int = 0
    raw_foreground_centroid_xy: list[float] = field(default_factory=list)
    raw_foreground_bbox_xywh: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RGBDMaskResult:
    rgb: np.ndarray
    depth_m: np.ndarray
    valid_depth_mask: np.ndarray
    raw_foreground_mask: np.ndarray
    cleaned_foreground_mask: np.ndarray
    connected_components_rgb: np.ndarray
    bottle_mask: np.ndarray
    box_mask: np.ndarray
    combined_foreground_mask: np.ndarray
    background_edit_mask: np.ndarray
    object_edit_mask_bottle: np.ndarray
    object_edit_mask_box: np.ndarray
    diagnostics: MaskDiagnostics
    candidate_list: list[CandidateStats]
    split_labels: dict[str, str]
    extra_masks: dict[str, np.ndarray]

    def diagnostics_json(self) -> str:
        payload = self.diagnostics.to_dict()
        payload["candidate_list"] = [asdict(c) for c in self.candidate_list]
        payload["split_labels"] = self.split_labels
        return json.dumps(payload, ensure_ascii=False, indent=2)


def _to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = np.moveaxis(rgb, 0, -1)
    if rgb.dtype.kind == "f" and float(rgb.max(initial=0.0)) <= 1.5:
        rgb = np.clip(rgb * 255.0, 0, 255)
    return rgb.astype(np.uint8)


def _to_mask_u8(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.dtype == bool:
        return mask.astype(np.uint8) * 255
    return (mask > 0).astype(np.uint8) * 255


def _central_table_roi_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    x0 = int(w * ROI_X_RANGE[0])
    x1 = int(w * ROI_X_RANGE[1])
    y0 = int(h * ROI_Y_RANGE[0])
    y1 = int(h * ROI_Y_RANGE[1])
    roi = np.zeros((h, w), dtype=np.uint8)
    roi[y0:y1, x0:x1] = 255
    return roi


def compute_valid_depth_mask(depth: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    depth_m = sanitize_depth(depth)[..., 0]
    computed = np.isfinite(depth_m) & (depth_m >= VALID_DEPTH_MIN_M) & (depth_m <= VALID_DEPTH_MAX_M)
    if valid_mask is not None:
        computed &= _to_mask_u8(valid_mask) > 0
    h, w = depth_m.shape[:2]
    computed[:BORDER_GUARD_PX, :] = False
    computed[max(0, h - BORDER_GUARD_PX):, :] = False
    computed[:, :BORDER_GUARD_PX] = False
    computed[:, max(0, w - BORDER_GUARD_PX):] = False
    return computed.astype(np.uint8) * 255


def _apply_border_guard(mask: np.ndarray, valid_mask: np.ndarray, border_guard_px: int) -> np.ndarray:
    guarded = mask.copy()
    h, w = mask.shape
    invalid = valid_mask == 0
    guarded[:border_guard_px, :] = 0
    guarded[max(0, h - border_guard_px):, :] = 0
    guarded[:, :border_guard_px] = 0
    guarded[:, max(0, w - border_guard_px):] = 0
    guarded[invalid] = 0
    return guarded


def extract_depth_foreground(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    *,
    quantile: float = DEFAULT_QUANTILE,
    abs_threshold_m: float | None = DEFAULT_ABS_THRESHOLD_M,
    min_component_area_px: int = MIN_COMPONENT_AREA_PX,
    border_guard_px: int = BORDER_GUARD_PX,
) -> tuple[np.ndarray, np.ndarray, float, list[dict[str, Any]], np.ndarray]:
    depth_m = sanitize_depth(depth)[..., 0]
    valid = _to_mask_u8(valid_mask) > 0
    roi_mask = _central_table_roi_mask(depth_m.shape)
    roi_valid = valid & (roi_mask > 0)
    roi_depth = depth_m[roi_valid]
    if roi_depth.size == 0:
        empty = np.zeros_like(valid_mask, dtype=np.uint8)
        return empty, empty, 0.0, [], np.zeros((*empty.shape, 3), dtype=np.uint8)

    near_ref = float(np.percentile(roi_depth, ROI_NEAR_PERCENTILE))
    upper_ref = float(np.percentile(roi_depth, ROI_UPPER_PERCENTILE))
    lower_band = max(VALID_DEPTH_MIN_M, near_ref - DEPTH_BAND_BELOW_M)
    upper_band = min(upper_ref, near_ref + DEPTH_BAND_ABOVE_M)
    if abs_threshold_m is not None:
        upper_band = min(upper_band, float(abs_threshold_m))
    threshold = upper_band

    raw = np.zeros_like(valid_mask, dtype=np.uint8)
    raw[(depth_m >= lower_band) & (depth_m <= upper_band) & roi_valid] = 255
    raw = _apply_border_guard(raw, _to_mask_u8(valid_mask), border_guard_px)

    kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), dtype=np.uint8)
    cleaned = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    filtered = np.zeros_like(cleaned)
    component_stats: list[dict[str, Any]] = []
    colored = np.zeros((*cleaned.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        touch_right = (x + w) >= (cleaned.shape[1] - border_guard_px)
        touch_bottom = (y + h) >= (cleaned.shape[0] - border_guard_px)
        if area >= min_component_area_px and not touch_right and not touch_bottom:
            filtered[labels == label] = 255
            color = rng.integers(40, 255, size=3, dtype=np.uint8)
            colored[labels == label] = color
        component_stats.append(
            {
                "label": label,
                "area": area,
                "bbox_xywh": [x, y, w, h],
                "centroid_xy": [float(centroids[label][0]), float(centroids[label][1])],
                "touch_right_border": bool(touch_right),
                "touch_bottom_border": bool(touch_bottom),
                "kept": bool(area >= min_component_area_px and not touch_right and not touch_bottom),
            }
        )

    return raw, filtered, threshold, component_stats, colored


def _rgb_edge_strength(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 120)
    return cv2.GaussianBlur(edges, (5, 5), 0)


def _depth_discontinuity(depth_m: np.ndarray) -> np.ndarray:
    dx = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(dx * dx + dy * dy)
    if float(grad.max(initial=0.0)) == 0.0:
        return np.zeros_like(depth_m, dtype=np.uint8)
    grad = np.clip(grad / (np.percentile(grad[grad > 0], 95) + 1e-6), 0, 1)
    return (grad * 255.0).astype(np.uint8)


def score_candidates(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    cleaned_foreground_mask: np.ndarray,
    valid_depth_mask: np.ndarray,
    *,
    border_guard_px: int = BORDER_GUARD_PX,
) -> list[CandidateStats]:
    mask = _to_mask_u8(cleaned_foreground_mask)
    valid = _to_mask_u8(valid_depth_mask) > 0
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape
    center_x = w / 2.0
    edge_strength = _rgb_edge_strength(rgb)
    depth_edges = _depth_discontinuity(depth_m)
    candidates: list[CandidateStats] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_COMPONENT_AREA_PX:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        region = labels == label
        ys, xs = np.where(region)
        if ys.size == 0:
            continue
        depths = depth_m[region & valid]
        if depths.size == 0:
            continue
        aspect = float(bh / max(bw, 1))
        extent = float(area / max(bw * bh, 1))
        centroid_x = float(centroids[label][0])
        centroid_y = float(centroids[label][1])
        vertical_position = 1.0 - (centroid_y / max(h, 1))
        touch_right = (x + bw) >= (w - border_guard_px)
        touch_bottom = (y + bh) >= (h - border_guard_px)
        color_edge = float(edge_strength[region].mean() / 255.0)
        depth_edge = float(depth_edges[region].mean() / 255.0)
        center_bias = 1.0 - min(abs(centroid_x - center_x) / center_x, 1.0)

        narrow_width_prior = 1.0 - min(bw / max(w * 0.28, 1.0), 1.0)
        lower_region_prior = centroid_y / max(h, 1)
        wide_bbox_prior = min((bw / max(bh, 1)) / 3.0, 1.2)
        base_like_prior = min(extent / 0.72, 1.2)
        bottle_score = (
            0.34 * min(aspect / 2.8, 1.4)
            + 0.24 * vertical_position
            + 0.22 * center_bias
            + 0.14 * narrow_width_prior
            + 0.06 * depth_edge
        )
        box_score = (
            0.30 * lower_region_prior
            + 0.28 * wide_bbox_prior
            + 0.22 * base_like_prior
            + 0.12 * center_bias
            + 0.08 * color_edge
        )
        if touch_right or touch_bottom:
            bottle_score -= 0.60
            box_score -= 0.60

        candidates.append(
            CandidateStats(
                label=label,
                bbox_xywh=[x, y, bw, bh],
                area=area,
                centroid_xy=[centroid_x, centroid_y],
                aspect_ratio=aspect,
                extent=extent,
                mean_depth_m=float(depths.mean()),
                median_depth_m=float(np.median(depths)),
                min_depth_m=float(depths.min()),
                max_depth_m=float(depths.max()),
                touch_right_border=bool(touch_right),
                touch_bottom_border=bool(touch_bottom),
                vertical_position_score=vertical_position,
                bottle_score=float(max(bottle_score, 0.0)),
                box_score=float(max(box_score, 0.0)),
            )
        )
    candidates.sort(key=lambda c: c.area, reverse=True)
    return candidates


def _split_component_watershed(rgb: np.ndarray, component_mask: np.ndarray) -> tuple[np.ndarray, float, str]:
    mask = _to_mask_u8(component_mask)
    if mask.sum() == 0:
        return np.zeros_like(mask), 0.0, "empty_component"
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    _, sure_fg = cv2.threshold(dist_norm, 0.28, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, kernel)
    n_markers, markers = cv2.connectedComponents(sure_fg)
    if n_markers <= 2:
        return np.zeros_like(mask), 0.0, "watershed_markers_lt_2"
    markers = markers + 1
    unknown = cv2.subtract(mask, sure_fg)
    markers[unknown > 0] = 0
    ws_input = rgb.copy()
    markers = cv2.watershed(ws_input, markers.astype(np.int32))
    unique = [m for m in np.unique(markers) if m > 1]
    if len(unique) < 2:
        return np.zeros_like(mask), 0.0, "watershed_regions_lt_2"
    split = np.zeros_like(mask, dtype=np.uint8)
    valid_regions = 0
    for region_id in unique:
        region = (markers == region_id).astype(np.uint8) * 255
        if int(cv2.countNonZero(region)) >= MIN_COMPONENT_AREA_PX // 2:
            valid_regions += 1
            split = np.maximum(split, region)
    confidence = min(valid_regions / 2.0, 1.0)
    if valid_regions < 2:
        return split, confidence * 0.35, "watershed_regions_small"
    return split, 0.55 + 0.20 * min(valid_regions, 3) / 3.0, "ok"


def _split_component_color(depth_m: np.ndarray, rgb: np.ndarray, component_mask: np.ndarray) -> tuple[list[np.ndarray], float, str]:
    mask = _to_mask_u8(component_mask)
    ys, xs = np.where(mask > 0)
    if ys.size < MIN_COMPONENT_AREA_PX:
        return [], 0.0, "component_too_small"
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    pixels = lab[ys, xs].astype(np.float32)
    if len(pixels) < 2:
        return [], 0.0, "not_enough_pixels"
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    compactness, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(-1)
    masks: list[np.ndarray] = []
    for k in range(2):
        m = np.zeros_like(mask)
        sel = labels == k
        m[ys[sel], xs[sel]] = 255
        if int(cv2.countNonZero(m)) >= MIN_COMPONENT_AREA_PX // 2:
            masks.append(m)
    if len(masks) < 2:
        return masks, 0.0, "kmeans_regions_lt_2"
    center_dist = float(np.linalg.norm(centers[0] - centers[1]))
    conf = min(center_dist / 40.0, 1.0) * 0.7
    return masks, conf, "ok"


def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = _to_mask_u8(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask_u8)
    best = max(range(1, n), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
    return (labels == best).astype(np.uint8) * 255


def _best_vertical_seed(mask: np.ndarray) -> np.ndarray:
    opened = cv2.morphologyEx(_to_mask_u8(mask), cv2.MORPH_OPEN, np.ones((9, 41), dtype=np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if n <= 1:
        return np.zeros_like(opened)
    h, w = opened.shape
    cx = w / 2.0
    best_label = 0
    best_score = -1e18
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 800:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        aspect = bh / max(bw, 1)
        centroid_x = float(centroids[i][0])
        centroid_y = float(centroids[i][1])
        center_bias = 1.0 - min(abs(centroid_x - cx) / cx, 1.0)
        upper_bias = 1.0 - (centroid_y / max(h, 1))
        narrow_bias = 1.0 - min(bw / max(w * 0.18, 1.0), 1.0)
        score = 0.35 * min(aspect / 2.0, 1.5) + 0.30 * upper_bias + 0.20 * center_bias + 0.15 * narrow_bias
        if score > best_score:
            best_score = score
            best_label = i
    if best_label == 0:
        return np.zeros_like(opened)
    return (labels == best_label).astype(np.uint8) * 255


def _extract_refined_instances(cleaned_foreground_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, bool, str]:
    raw = _to_mask_u8(cleaned_foreground_mask)
    if cv2.countNonZero(raw) == 0:
        z = np.zeros_like(raw)
        return z, z, z, z, z, 0.0, False, "empty_foreground"

    vertical_seed = _best_vertical_seed(raw)
    if cv2.countNonZero(vertical_seed) == 0:
        z = np.zeros_like(raw)
        return z, z, z, z, z, 0.0, False, "no_vertical_seed"

    table_removed = cv2.bitwise_and(raw, cv2.dilate(vertical_seed, np.ones((31, 61), dtype=np.uint8), iterations=1))
    table_removed = _largest_component(table_removed)

    seed_pts = np.column_stack(np.where(vertical_seed > 0)[::-1])
    vx, vy, vw, vh = cv2.boundingRect(seed_pts)
    bottle_candidate = vertical_seed.copy()
    cutoff_y = vy + int(0.74 * vh)
    bottle_candidate[cutoff_y:, :] = 0
    bottle_candidate = _largest_component(bottle_candidate)
    if cv2.countNonZero(bottle_candidate) == 0:
        bottle_candidate = _largest_component(vertical_seed)

    dilated_bottle = cv2.dilate(bottle_candidate, np.ones((19, 19), dtype=np.uint8), iterations=1)
    support_zone = table_removed.copy()
    support_zone[: max(0, vy + int(0.35 * vh)), :] = 0
    x0 = max(0, vx - max(24, int(0.7 * vw)))
    x1 = min(raw.shape[1], vx + vw + max(24, int(0.7 * vw)))
    support_zone[:, :x0] = 0
    support_zone[:, x1:] = 0
    support_zone = cv2.subtract(support_zone, dilated_bottle)
    box_candidate = _largest_component(support_zone)

    bottle_centroid, bottle_bbox, bottle_area = _mask_geometry(bottle_candidate)
    box_centroid, box_bbox, box_area = _mask_geometry(box_candidate)
    _table_centroid, table_bbox, _table_area = _mask_geometry(table_removed)

    semantic_ok = False
    confidence = 0.0
    reason = "semantic_checks_failed"
    if bottle_bbox and box_bbox:
        bw, bh = bottle_bbox[2], bottle_bbox[3]
        xw, xh = box_bbox[2], box_bbox[3]
        bottle_aspect = bh / max(bw, 1)
        box_aspect = xh / max(xw, 1)
        bottle_upper = bottle_centroid[1] < box_centroid[1]
        bottle_narrow = bw < (max(72, int(0.28 * table_bbox[2])) if table_bbox else 90)
        box_not_table = xw < (max(180, int(0.62 * table_bbox[2])) if table_bbox else 220)
        box_below = box_centroid[1] > bottle_centroid[1] + 12
        box_local = abs(box_centroid[0] - bottle_centroid[0]) < max(90, int(1.4 * bw))
        semantic_ok = bool(
            bottle_area > 1200 and box_area > 900 and bottle_aspect > 1.45 and bottle_upper and bottle_narrow
            and box_below and box_local and box_not_table and box_aspect < 1.25
        )
        confidence = float(min(1.0, 0.18 * bottle_aspect + 0.16 * float(bottle_upper) + 0.16 * float(bottle_narrow) + 0.16 * float(box_below) + 0.16 * float(box_local) + 0.18 * float(box_not_table)))
        if semantic_ok:
            reason = "refined_vertical_support_split"
        elif not box_not_table:
            reason = "box_candidate_too_wide_table_like"
        elif not bottle_narrow:
            reason = "bottle_candidate_not_narrow_enough"
        elif not box_below:
            reason = "box_candidate_not_below_bottle"

    if not semantic_ok:
        final_bottle = np.zeros_like(raw)
        final_box = np.zeros_like(raw)
    else:
        final_bottle = bottle_candidate
        final_box = box_candidate

    return final_bottle, final_box, table_removed, bottle_candidate, box_candidate, confidence, semantic_ok, reason


def split_instances(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    cleaned_foreground_mask: np.ndarray,
    candidates: list[CandidateStats],
) -> tuple[np.ndarray, np.ndarray, bool, float, str, dict[str, np.ndarray], bool]:
    del rgb, depth_m, candidates
    bottle_mask, box_mask, table_removed, bottle_candidate, box_candidate, confidence, semantic_ok, reason = _extract_refined_instances(cleaned_foreground_mask)
    extra = {
        "table_removed_mask": table_removed,
        "bottle_candidate_mask": bottle_candidate,
        "box_candidate_mask": box_candidate,
    }
    return bottle_mask, box_mask, bool(semantic_ok), float(confidence), reason, extra, True

def _overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = rgb.copy()
    mask_u8 = _to_mask_u8(mask)
    out[mask_u8 > 0] = np.array(color, dtype=np.uint8)
    return cv2.addWeighted(rgb, 0.7, out, 0.3, 0)


def _mask_geometry(mask: np.ndarray) -> tuple[list[float], list[int], int]:
    mask_u8 = _to_mask_u8(mask)
    ys, xs = np.where(mask_u8 > 0)
    if ys.size == 0:
        return [], [], 0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    centroid = [float(xs.mean()), float(ys.mean())]
    bbox = [x0, y0, int(x1 - x0 + 1), int(y1 - y0 + 1)]
    area = int(ys.size)
    return centroid, bbox, area


def build_rgbd_object_masks(
    rgb: np.ndarray,
    depth: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    frame_index: int = -1,
    quantile: float = DEFAULT_QUANTILE,
    abs_threshold_m: float | None = DEFAULT_ABS_THRESHOLD_M,
) -> RGBDMaskResult:
    rgb_u8 = _to_uint8_rgb(rgb)
    depth_m = sanitize_depth(depth)[..., 0]
    valid_depth_mask = compute_valid_depth_mask(depth, valid_mask)
    rgb_hw = list(rgb_u8.shape[:2])
    depth_hw = list(depth_m.shape[:2])
    valid_hw = list(valid_depth_mask.shape[:2])
    if not (tuple(rgb_hw) == tuple(depth_hw) == tuple(valid_hw)):
        raise ValueError(
            f"Shape mismatch in build_rgbd_object_masks: rgb={rgb_u8.shape}, depth={depth_m.shape}, valid_mask={valid_depth_mask.shape}"
        )
    raw_fg, cleaned_fg, threshold_m, component_stats, components_rgb = extract_depth_foreground(
        depth,
        valid_depth_mask,
        quantile=quantile,
        abs_threshold_m=abs_threshold_m,
    )
    threshold_candidates = np.zeros_like(valid_depth_mask, dtype=np.uint8)
    roi_mask = _central_table_roi_mask(depth_m.shape)
    lower_threshold = max(VALID_DEPTH_MIN_M, threshold_m - DEPTH_BAND_ABOVE_M)
    threshold_candidates[(depth_m >= lower_threshold) & (depth_m <= threshold_m) & (_to_mask_u8(valid_depth_mask) > 0) & (roi_mask > 0)] = 255
    threshold_candidate_centroid_xy, threshold_candidate_bbox_xywh, threshold_candidate_area = _mask_geometry(threshold_candidates)
    raw_foreground_centroid_xy, raw_foreground_bbox_xywh, _raw_fg_area = _mask_geometry(raw_fg)
    candidates = score_candidates(rgb_u8, depth_m, cleaned_fg, valid_depth_mask)

    bottle_mask, box_mask, split_success, split_conf, split_reason, extra, fallback_split_used = split_instances(
        rgb_u8,
        depth_m,
        cleaned_fg,
        candidates,
    )

    notes: list[str] = []
    combined = _to_mask_u8(cleaned_fg)
    if not split_success:
        notes.append("Split failed; refined candidates kept only for debug, final instance masks suppressed.")
        bottle_mask = np.zeros_like(combined)
        box_mask = np.zeros_like(combined)

    bottle_mask = cv2.bitwise_and(_to_mask_u8(bottle_mask), valid_depth_mask)
    box_mask = cv2.bitwise_and(_to_mask_u8(box_mask), valid_depth_mask)
    combined = cv2.bitwise_and(cv2.bitwise_or(bottle_mask, box_mask), valid_depth_mask)
    if cv2.countNonZero(combined) == 0:
        combined = cv2.bitwise_and(_to_mask_u8(cleaned_fg), valid_depth_mask)

    background_edit_mask = cv2.bitwise_and(cv2.bitwise_not(combined), valid_depth_mask)
    object_edit_mask_bottle = bottle_mask.copy()
    object_edit_mask_box = box_mask.copy()

    valid_ratio = float((valid_depth_mask > 0).mean())
    bottle_area = int(cv2.countNonZero(bottle_mask))
    box_area = int(cv2.countNonZero(box_mask))
    selected_component_area = int(cv2.countNonZero(combined))

    bottle_conf = float(max((c.bottle_score for c in candidates), default=0.0))
    box_conf = float(max((c.box_score for c in candidates), default=0.0))
    failure_reason = "" if split_success else split_reason
    if bottle_area == 0 and box_area == 0:
        failure_reason = failure_reason or "foreground_empty_after_split"
        notes.append("No instance mask survived post-filtering.")

    if selected_component_area / float(combined.size) > MAX_FOREGROUND_AREA_RATIO:
        notes.append("Foreground unusually large; check threshold.")
    if 0 < selected_component_area / float(combined.size) < MIN_FOREGROUND_AREA_RATIO:
        notes.append("Foreground unusually small; check threshold.")

    semantic_label_verified = bool(split_success)
    if not semantic_label_verified:
        notes.append("semantic labels not verified; bottle/box masks are debug candidates only")
    diagnostics = MaskDiagnostics(
        frame_index=frame_index,
        valid_ratio=valid_ratio,
        foreground_threshold_m=float(threshold_m),
        num_components=len(candidates),
        selected_component_area=selected_component_area,
        bottle_area=bottle_area,
        box_area=box_area,
        split_success=bool(split_success),
        split_confidence=float(split_conf),
        bottle_confidence=float(bottle_conf),
        box_confidence=float(box_conf),
        failure_reason=failure_reason,
        notes=notes + [
            f"valid_depth_band=[{VALID_DEPTH_MIN_M:.2f},{VALID_DEPTH_MAX_M:.2f}]m",
            f"roi=({ROI_X_RANGE[0]:.2f}-{ROI_X_RANGE[1]:.2f}, {ROI_Y_RANGE[0]:.2f}-{ROI_Y_RANGE[1]:.2f})",
        ],
        candidate_count=len(candidates),
        fallback_split_used=bool(fallback_split_used),
        semantic_label_verified=semantic_label_verified,
        rgb_shape=rgb_hw,
        depth_shape=depth_hw,
        valid_mask_shape=valid_hw,
        threshold_candidate_centroid_xy=threshold_candidate_centroid_xy,
        threshold_candidate_bbox_xywh=threshold_candidate_bbox_xywh,
        threshold_candidate_area=threshold_candidate_area,
        raw_foreground_centroid_xy=raw_foreground_centroid_xy,
        raw_foreground_bbox_xywh=raw_foreground_bbox_xywh,
    )

    split_labels = {
        "bottle_label": "bottle_guess" if bottle_area > 0 else "object_1",
        "box_label": "box_guess" if box_area > 0 else "object_2",
    }

    extra_masks = {
        "overlay_valid_mask": _overlay_mask(rgb_u8, valid_depth_mask, (0, 255, 0)),
        "overlay_threshold_candidates": _overlay_mask(rgb_u8, threshold_candidates, (255, 128, 0)),
        "overlay_raw_foreground": _overlay_mask(rgb_u8, raw_fg, (255, 0, 0)),
        "overlay_table_removed": _overlay_mask(rgb_u8, extra.get('table_removed_mask', np.zeros_like(raw_fg)), (0, 200, 255)),
        "overlay_bottle_candidate": _overlay_mask(rgb_u8, extra.get('bottle_candidate_mask', np.zeros_like(raw_fg)), (255, 0, 255)),
        "overlay_box_candidate": _overlay_mask(rgb_u8, extra.get('box_candidate_mask', np.zeros_like(raw_fg)), (255, 255, 0)),
        "overlay_bottle": _overlay_mask(rgb_u8, bottle_mask, (255, 0, 255)),
        "overlay_box": _overlay_mask(rgb_u8, box_mask, (255, 255, 0)),
        "overlay_instances_refined": cv2.addWeighted(
            _overlay_mask(rgb_u8, bottle_mask, (255, 0, 255)),
            0.7,
            _overlay_mask(rgb_u8, box_mask, (255, 255, 0)),
            0.3,
            0,
        ),
        "overlay_background": _overlay_mask(rgb_u8, background_edit_mask, (0, 255, 255)),
        **extra,
    }

    LOGGER.info(
        "RGB-D mask build frame=%s split_success=%s split_conf=%.3f bottle_area=%s box_area=%s",
        frame_index,
        split_success,
        split_conf,
        bottle_area,
        box_area,
    )

    return RGBDMaskResult(
        rgb=rgb_u8,
        depth_m=depth_m,
        valid_depth_mask=valid_depth_mask,
        raw_foreground_mask=raw_fg,
        cleaned_foreground_mask=cleaned_fg,
        connected_components_rgb=components_rgb,
        bottle_mask=bottle_mask,
        box_mask=box_mask,
        combined_foreground_mask=combined,
        background_edit_mask=background_edit_mask,
        object_edit_mask_bottle=object_edit_mask_bottle,
        object_edit_mask_box=object_edit_mask_box,
        diagnostics=diagnostics,
        candidate_list=candidates,
        split_labels=split_labels,
        extra_masks=extra_masks,
    )
