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
TOP_PLANE_MIN_WIDTH_RATIO = 0.28
TOP_PLANE_MAX_HEIGHT_RATIO = 0.32
TOP_PLANE_MAX_Y_RATIO = 0.58

BOTTLE_RIGHT_MIN_RATIO = 0.52
BOTTLE_TOP_MAX_RATIO = 0.52
BOTTLE_MAX_WIDTH_RATIO = 0.12
BOTTLE_MIN_ASPECT = 1.05
BOTTLE_MIN_AREA = 180
BOTTLE_MAX_AREA_RATIO = 0.06

BOX_MIN_AREA = 2500
BOX_MIN_WIDTH_RATIO = 0.12
BOX_MIN_HEIGHT_RATIO = 0.06
BOX_MAX_WIDTH_RATIO = 0.72
BOX_MIN_RECTANGULARITY = 0.35
BOX_OVERLAP_TOP_PLANE_MAX = 0.75


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

def _component_mask(labels: np.ndarray, label_id: int) -> np.ndarray:
    return ((labels == label_id).astype(np.uint8) * 255)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(_to_mask_u8(mask) > 0)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _centroid_from_mask(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(_to_mask_u8(mask) > 0)
    if ys.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _rectangularity(mask: np.ndarray) -> float:
    bbox = _bbox_from_mask(mask)
    if bbox is None:
        return 0.0
    _, _, w, h = bbox
    area = float(np.count_nonzero(mask))
    return area / max(float(w * h), 1.0)


def _mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(_to_mask_u8(mask)))

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



def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = _to_mask_u8(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask_u8)
    best = max(range(1, n), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
    return (labels == best).astype(np.uint8) * 255


def _pick_bottle_candidate(mask: np.ndarray) -> np.ndarray:
    raw = _to_mask_u8(mask)
    if _mask_area(raw) == 0:
        return np.zeros_like(raw)

    h, w = raw.shape
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(raw, connectivity=8)

    total_fg = max(float(_mask_area(raw)), 1.0)
    best = np.zeros_like(raw)
    best_score = -1e18

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < BOTTLE_MIN_AREA:
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[i][0]), float(centroids[i][1])

        width_ratio = bw / max(w, 1)
        height_ratio = bh / max(h, 1)
        area_ratio = area / total_fg
        aspect = bh / max(bw, 1)

        right_bias = cx / max(w, 1)
        top_bias = 1.0 - (cy / max(h, 1))
        narrow_bias = 1.0 - min(width_ratio / max(BOTTLE_MAX_WIDTH_RATIO, 1e-6), 2.0)

        if right_bias < BOTTLE_RIGHT_MIN_RATIO:
            continue
        if (cy / max(h, 1)) > BOTTLE_TOP_MAX_RATIO:
            continue
        if area_ratio > BOTTLE_MAX_AREA_RATIO:
            continue

        score = (
            0.30 * min(aspect / 2.0, 1.5)
            + 0.28 * right_bias
            + 0.22 * top_bias
            + 0.20 * narrow_bias
        )

        if aspect < BOTTLE_MIN_ASPECT:
            score -= 0.35

        if score > best_score:
            best_score = score
            best = _component_mask(labels, i)

    return best
def _pick_box_candidate(mask: np.ndarray, bottle_mask: np.ndarray, top_plane_mask: np.ndarray) -> np.ndarray:
    raw = _to_mask_u8(mask)
    bottle = _to_mask_u8(bottle_mask)
    top_plane = _to_mask_u8(top_plane_mask)

    if _mask_area(raw) == 0:
        return np.zeros_like(raw)

    search = raw.copy()
    if _mask_area(top_plane) > 0:
        search = cv2.subtract(search, top_plane)
    if _mask_area(bottle) > 0:
        search = cv2.subtract(search, cv2.dilate(bottle, np.ones((21, 21), np.uint8), iterations=1))

    h, w = search.shape
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(search, connectivity=8)

    bottle_centroid = _centroid_from_mask(bottle)
    best = np.zeros_like(search)
    best_score = -1e18

    for i in range(1, n):
        comp = _component_mask(labels, i)
        area = _mask_area(comp)
        if area < BOX_MIN_AREA:
            continue

        bbox = _bbox_from_mask(comp)
        if bbox is None:
            continue
        x, y, bw, bh = bbox
        cx, cy = float(centroids[i][0]), float(centroids[i][1])

        width_ratio = bw / max(w, 1)
        height_ratio = bh / max(h, 1)
        rect = _rectangularity(comp)
        plane_overlap = _overlap_ratio(comp, top_plane)

        if width_ratio < BOX_MIN_WIDTH_RATIO or width_ratio > BOX_MAX_WIDTH_RATIO:
            continue
        if height_ratio < BOX_MIN_HEIGHT_RATIO:
            continue
        if rect < BOX_MIN_RECTANGULARITY:
            continue
        if plane_overlap > BOX_OVERLAP_TOP_PLANE_MAX:
            continue

        below_bottle_bonus = 0.0
        if bottle_centroid is not None and cy > bottle_centroid[1] + 10:
            below_bottle_bonus = 0.35

        lower_bias = cy / max(h, 1)

        score = (
            0.30 * min(rect / 0.6, 1.5)
            + 0.22 * min(width_ratio / 0.25, 1.5)
            + 0.18 * min(height_ratio / 0.12, 1.5)
            + 0.15 * lower_bias
            + below_bottle_bonus
            - 0.20 * plane_overlap
        )

        if score > best_score:
            best_score = score
            best = comp

    return best

def detect_top_plane(depth_m: np.ndarray, foreground_mask: np.ndarray, bottle_seed: np.ndarray | None = None) -> np.ndarray:
    del depth_m  # 이번 버전은 depth 조건을 과감히 뺌

    fg = _to_mask_u8(foreground_mask)
    if _mask_area(fg) == 0:
        return np.zeros_like(fg)

    h, w = fg.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)

    best = np.zeros_like(fg)
    best_score = -1e18

    for i in range(1, n):
        comp = _component_mask(labels, i)
        area = _mask_area(comp)
        if area < 1500:
            continue

        bbox = _bbox_from_mask(comp)
        if bbox is None:
            continue
        x, y, bw, bh = bbox

        width_ratio = bw / max(w, 1)
        height_ratio = bh / max(h, 1)
        y_ratio = y / max(h, 1)
        rect = _rectangularity(comp)

        is_plane_like = (
            width_ratio >= TOP_PLANE_MIN_WIDTH_RATIO
            and height_ratio <= TOP_PLANE_MAX_HEIGHT_RATIO
            and y_ratio <= TOP_PLANE_MAX_Y_RATIO
        )
        if not is_plane_like:
            continue

        score = (
            0.45 * min(width_ratio / 0.5, 1.5)
            + 0.30 * (1.0 - min(height_ratio / 0.25, 1.5))
            + 0.25 * min(rect / 0.6, 1.5)
        )

        if bottle_seed is not None and _mask_area(bottle_seed) > 0:
            overlap = _overlap_ratio(comp, cv2.dilate(_to_mask_u8(bottle_seed), np.ones((11, 11), np.uint8), iterations=1))
            score -= 0.5 * overlap

        if score > best_score:
            best_score = score
            best = comp

    return best

def _extract_refined_instances(
    depth_m: np.ndarray,
    cleaned_foreground_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, bool, str]:
    raw = _to_mask_u8(cleaned_foreground_mask)
    if _mask_area(raw) == 0:
        z = np.zeros_like(raw)
        return z, z, z, z, z, z, 0.0, False, "empty_foreground"

    top_plane_mask = detect_top_plane(depth_m, raw, None)

    fg_wo_plane = raw.copy()
    if _mask_area(top_plane_mask) > 0:
        fg_wo_plane = cv2.subtract(fg_wo_plane, top_plane_mask)

    bottle_candidate = _pick_bottle_candidate(raw)
    if _mask_area(bottle_candidate) == 0:
        bottle_candidate = _pick_bottle_candidate(fg_wo_plane)

    box_candidate = _pick_box_candidate(raw, bottle_candidate, top_plane_mask)
    if _mask_area(box_candidate) == 0:
        box_candidate = _pick_box_candidate(fg_wo_plane, bottle_candidate, top_plane_mask)

    bottle_bbox = _bbox_from_mask(bottle_candidate)
    box_bbox = _bbox_from_mask(box_candidate)
    bottle_centroid = _centroid_from_mask(bottle_candidate)
    box_centroid = _centroid_from_mask(box_candidate)

    bottle_ok = False
    box_ok = False

    if bottle_bbox is not None and bottle_centroid is not None:
        _, _, bw, bh = bottle_bbox
        aspect = bh / max(bw, 1)
        cx, cy = bottle_centroid
        h, w = raw.shape
        bottle_ok = (
            _mask_area(bottle_candidate) >= BOTTLE_MIN_AREA
            and aspect >= BOTTLE_MIN_ASPECT
            and (cx / max(w, 1)) >= BOTTLE_RIGHT_MIN_RATIO
            and (cy / max(h, 1)) <= BOTTLE_TOP_MAX_RATIO
        )

    if box_bbox is not None:
        rect = _rectangularity(box_candidate)
        plane_overlap = _overlap_ratio(box_candidate, top_plane_mask)
        box_ok = (
            _mask_area(box_candidate) >= BOX_MIN_AREA
            and rect >= BOX_MIN_RECTANGULARITY
            and plane_overlap <= BOX_OVERLAP_TOP_PLANE_MAX
        )
        if bottle_centroid is not None and box_centroid is not None:
            box_ok = box_ok and (box_centroid[1] > bottle_centroid[1] + 10)

    semantic_ok = bool(bottle_ok and box_ok)

    if semantic_ok:
        reason = "refined_split_pass"
    elif _mask_area(box_candidate) > 0 and _mask_area(bottle_candidate) == 0:
        reason = "box_candidate_only"
    elif _mask_area(bottle_candidate) > 0 and _mask_area(box_candidate) == 0:
        reason = "bottle_candidate_only"
    elif _mask_area(bottle_candidate) == 0 and _mask_area(box_candidate) == 0:
        reason = "no_bottle_or_box_candidate"
    elif not bottle_ok:
        reason = "bottle_candidate_failed_semantic_check"
    else:
        reason = "box_candidate_failed_semantic_check"

    confidence = 0.0
    confidence += 0.5 if bottle_ok else 0.2 if _mask_area(bottle_candidate) > 0 else 0.0
    confidence += 0.5 if box_ok else 0.2 if _mask_area(box_candidate) > 0 else 0.0

    # semantic false여도 후보는 남긴다
    final_bottle = _to_mask_u8(bottle_candidate)
    final_box = _to_mask_u8(box_candidate)

    return (
        final_bottle,
        final_box,
        fg_wo_plane,
        bottle_candidate,
        box_candidate,
        top_plane_mask,
        float(confidence),
        semantic_ok,
        reason,
    )
        
def split_instances(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    cleaned_foreground_mask: np.ndarray,
    candidates: list[CandidateStats],
) -> tuple[np.ndarray, np.ndarray, bool, float, str, dict[str, np.ndarray], bool]:
    del rgb, candidates
    bottle_mask, box_mask, table_removed, bottle_candidate, box_candidate, top_plane_mask, confidence, semantic_ok, reason = _extract_refined_instances(
        depth_m,
        cleaned_foreground_mask,
    )
    extra = {
        "table_removed_mask": table_removed,
        "bottle_candidate_mask": bottle_candidate,
        "box_candidate_mask": box_candidate,
        "top_plane_mask": top_plane_mask,
    }
    print(
        "[DEBUG split] "
        f"bottle_before_valid={cv2.countNonZero(_to_mask_u8(bottle_mask))} "
        f"box_before_valid={cv2.countNonZero(_to_mask_u8(box_mask))} "
        f"semantic_ok={semantic_ok} "
        f"reason={reason} "
        f"confidence={confidence:.3f}"
    )
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

def _mask_rectangularity(mask: np.ndarray) -> float:
    mask_u8 = _to_mask_u8(mask)
    ys, xs = np.where(mask_u8 > 0)
    if ys.size == 0:
        return 0.0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    box_area = max((x1 - x0 + 1) * (y1 - y0 + 1), 1)
    return float(ys.size / box_area)




def _overlap_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = _to_mask_u8(mask_a) > 0
    b = _to_mask_u8(mask_b) > 0
    denom = float(a.sum())
    if denom <= 0:
        return 0.0
    return float((a & b).sum() / denom)

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
        notes.append("Semantic verification failed; candidate masks kept for debug and inspection.")
    else:
        notes.append("Semantic verification passed.")

    bottle_mask = cv2.bitwise_and(_to_mask_u8(bottle_mask), valid_depth_mask)
    box_mask = cv2.bitwise_and(_to_mask_u8(box_mask), valid_depth_mask)
    combined = cv2.bitwise_and(cv2.bitwise_or(bottle_mask, box_mask), valid_depth_mask)
    print(
        f"[DEBUG valid] frame={frame_index} "
        f"bottle_after_valid={cv2.countNonZero(bottle_mask)} "
        f"box_after_valid={cv2.countNonZero(box_mask)}"
    )
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
        notes.append("semantic labels not verified; candidate masks are kept but not trusted for object-level augmentation")
    top_plane_area = int(cv2.countNonZero(extra.get("top_plane_mask", np.zeros_like(raw_fg))))
    if top_plane_area > 0:
        notes.append(f"top_plane_detected_area={top_plane_area}")
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
        "overlay_top_plane": _overlay_mask(rgb_u8, extra.get('top_plane_mask', np.zeros_like(raw_fg)), (80, 255, 255)),
        "top_plane_mask": extra.get("top_plane_mask", np.zeros_like(raw_fg)),
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
