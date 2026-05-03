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


def compute_valid_depth_mask(depth: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    depth_m = sanitize_depth(depth)[..., 0]
    computed = np.isfinite(depth_m) & (depth_m > 0)
    if valid_mask is not None:
        computed &= _to_mask_u8(valid_mask) > 0
    return computed.astype(np.uint8) * 255


def _apply_border_guard(mask: np.ndarray, valid_mask: np.ndarray, border_guard_px: int) -> np.ndarray:
    guarded = mask.copy()
    h, w = mask.shape
    invalid = valid_mask == 0
    guarded[:, max(0, w - border_guard_px):] = 0
    guarded[max(0, h - border_guard_px):, :] = 0
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
    valid_depth = depth_m[valid]
    if valid_depth.size == 0:
        empty = np.zeros_like(valid_mask, dtype=np.uint8)
        return empty, empty, 0.0, [], np.zeros((*empty.shape, 3), dtype=np.uint8)

    threshold = float(np.percentile(valid_depth, quantile))
    if abs_threshold_m is not None:
        threshold = min(threshold, float(abs_threshold_m))

    raw = np.zeros_like(valid_mask, dtype=np.uint8)
    raw[(depth_m > 0) & (depth_m <= threshold) & valid] = 255
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

        bottle_score = (
            0.30 * min(aspect / 2.5, 1.5)
            + 0.20 * vertical_position
            + 0.20 * center_bias
            + 0.15 * depth_edge
            + 0.15 * color_edge
        )
        box_score = (
            0.25 * min((bw / max(bh, 1)) / 2.5, 1.5)
            + 0.25 * (centroid_y / max(h, 1))
            + 0.20 * center_bias
            + 0.15 * extent
            + 0.15 * color_edge
        )
        if touch_right or touch_bottom:
            bottle_score -= 0.35
            box_score -= 0.35

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


def split_instances(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    cleaned_foreground_mask: np.ndarray,
    candidates: list[CandidateStats],
) -> tuple[np.ndarray, np.ndarray, bool, float, str, dict[str, np.ndarray]]:
    if len(candidates) >= 2:
        by_bottle = max(candidates, key=lambda c: c.bottle_score)
        remaining = [c for c in candidates if c.label != by_bottle.label]
        by_box = max(remaining, key=lambda c: c.box_score) if remaining else None
        if by_box is not None:
            labels = cv2.connectedComponentsWithStats(_to_mask_u8(cleaned_foreground_mask), connectivity=8)[1]
            bottle_mask = (labels == by_bottle.label).astype(np.uint8) * 255
            box_mask = (labels == by_box.label).astype(np.uint8) * 255
            conf = min(1.0, 0.5 + 0.25 * abs(by_bottle.bottle_score - by_box.bottle_score) + 0.25)
            return bottle_mask, box_mask, True, float(conf), "component_assignment", {}

    total_mask = _to_mask_u8(cleaned_foreground_mask)
    waters_mask, ws_conf, ws_reason = _split_component_watershed(rgb, total_mask)
    color_masks, color_conf, color_reason = _split_component_color(depth_m, rgb, total_mask)

    if ws_conf <= 0 and color_conf <= 0:
        return np.zeros_like(total_mask), np.zeros_like(total_mask), False, 0.0, f"split_failed:{ws_reason}|{color_reason}", {
            "watershed_mask": waters_mask,
        }

    labels = []
    if color_masks:
        labels = color_masks[:2]
    else:
        num_labels, label_img = cv2.connectedComponents(waters_mask)
        for lab in range(1, num_labels):
            m = (label_img == lab).astype(np.uint8) * 255
            if cv2.countNonZero(m) >= MIN_COMPONENT_AREA_PX // 2:
                labels.append(m)
        labels = labels[:2]

    if len(labels) < 2:
        return np.zeros_like(total_mask), np.zeros_like(total_mask), False, max(ws_conf, color_conf), "split_regions_lt_2", {
            "watershed_mask": waters_mask,
        }

    scored = []
    h = total_mask.shape[0]
    for m in labels:
        ys, xs = np.where(m > 0)
        x, y, w, hh = cv2.boundingRect(np.column_stack([xs, ys]))
        aspect = hh / max(w, 1)
        centroid_y = float(ys.mean())
        bottle_score = 0.55 * min(aspect / 2.3, 1.5) + 0.45 * (1.0 - centroid_y / h)
        box_score = 0.55 * min((w / max(hh, 1)) / 2.3, 1.5) + 0.45 * (centroid_y / h)
        scored.append((m, bottle_score, box_score))
    bottle_mask = max(scored, key=lambda t: t[1])[0]
    remaining = [t for t in scored if not np.array_equal(t[0], bottle_mask)]
    box_mask = max(remaining, key=lambda t: t[2])[0] if remaining else np.zeros_like(total_mask)
    split_conf = float(min(1.0, 0.40 + max(ws_conf, color_conf)))
    return bottle_mask, box_mask, True, split_conf, "fallback_split", {
        "watershed_mask": waters_mask,
    }


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = rgb.copy()
    mask_u8 = _to_mask_u8(mask)
    out[mask_u8 > 0] = np.array(color, dtype=np.uint8)
    return cv2.addWeighted(rgb, 0.7, out, 0.3, 0)


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
    raw_fg, cleaned_fg, threshold_m, component_stats, components_rgb = extract_depth_foreground(
        depth,
        valid_depth_mask,
        quantile=quantile,
        abs_threshold_m=abs_threshold_m,
    )
    candidates = score_candidates(rgb_u8, depth_m, cleaned_fg, valid_depth_mask)

    bottle_mask, box_mask, split_success, split_conf, split_reason, extra = split_instances(
        rgb_u8,
        depth_m,
        cleaned_fg,
        candidates,
    )

    notes: list[str] = []
    combined = _to_mask_u8(cleaned_fg)
    if not split_success:
        notes.append("Split failed; combined foreground retained.")
        if candidates:
            best_bottle = max(candidates, key=lambda c: c.bottle_score)
            labels = cv2.connectedComponentsWithStats(_to_mask_u8(cleaned_fg), connectivity=8)[1]
            bottle_mask = (labels == best_bottle.label).astype(np.uint8) * 255
            box_mask = cv2.subtract(combined, bottle_mask)
        else:
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
        notes=notes,
        candidate_count=len(candidates),
    )

    split_labels = {
        "bottle_label": "bottle_guess" if bottle_area > 0 else "object_1",
        "box_label": "box_guess" if box_area > 0 else "object_2",
    }

    extra_masks = {
        "overlay_bottle": _overlay_mask(rgb_u8, bottle_mask, (255, 0, 255)),
        "overlay_box": _overlay_mask(rgb_u8, box_mask, (255, 255, 0)),
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
