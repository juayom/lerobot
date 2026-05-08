from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.datasets.genaug_rgbd_masks import RGBDMaskResult

METHOD_HEURISTIC = "heuristic"
METHOD_GRABCUT = "grabcut"
MODE_BOTTLE = "bottle"
MODE_BOX = "box"
MODE_BACKGROUND = "background"
METHOD_SAM_REFINED = "sam_refined"

@dataclass
class EvalThresholds:
    bottle_mean_min: float = 0.35
    box_mean_min: float = 0.30
    foreground_mean_min: float = 0.60
    bottle_median_min: float = 0.30
    box_median_min: float = 0.25
    foreground_median_min: float = 0.60


@dataclass
class MethodDecision:
    chosen_method: str | None
    passed: bool
    reason: str
    method_scores: dict[str, float]
    summaries: dict[str, dict[str, Any]]


def _score_summary(summary: dict[str, Any]) -> float:
    mean_iou = summary.get("mean_iou", {}) or {}
    return float(mean_iou.get("bottle") or 0.0) + float(mean_iou.get("box") or 0.0) + 0.5 * float(mean_iou.get("foreground") or 0.0)


def load_eval_summary(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def choose_best_method(summary: dict[str, Any] | None, thresholds: EvalThresholds) -> MethodDecision:
    if not summary:
        return MethodDecision(chosen_method=None, passed=False, reason="no_eval_summary", method_scores={}, summaries={})

    eligible: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    for method, data in summary.items():
        if not isinstance(data, dict):
            continue
        scores[method] = _score_summary(data)
        mean_iou = data.get("mean_iou", {}) or {}
        median_iou = data.get("median_iou", {}) or {}
        passed = (
            float(mean_iou.get("bottle") or 0.0) >= thresholds.bottle_mean_min
            and float(mean_iou.get("box") or 0.0) >= thresholds.box_mean_min
            and float(mean_iou.get("foreground") or 0.0) >= thresholds.foreground_mean_min
            and float(median_iou.get("bottle") or 0.0) >= thresholds.bottle_median_min
            and float(median_iou.get("box") or 0.0) >= thresholds.box_median_min
            and float(median_iou.get("foreground") or 0.0) >= thresholds.foreground_median_min
        )
        if passed:
            eligible[method] = data

    if not eligible:
        best = max(scores, key=scores.get) if scores else None
        return MethodDecision(chosen_method=best, passed=False, reason="no_method_met_eval_thresholds", method_scores=scores, summaries=summary)

    chosen = max(eligible, key=lambda m: scores.get(m, 0.0))
    return MethodDecision(chosen_method=chosen, passed=True, reason="eval_thresholds_passed", method_scores=scores, summaries=summary)


def _grabcut_from_seeds(rgb: np.ndarray, sure_fg: np.ndarray, likely_fg: np.ndarray, sure_bg: np.ndarray, iterations: int = 5) -> np.ndarray:
    h, w = rgb.shape[:2]
    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    mask[sure_bg > 0] = cv2.GC_BGD
    mask[likely_fg > 0] = cv2.GC_PR_FGD
    mask[sure_fg > 0] = cv2.GC_FGD

    ys, xs = np.where((sure_fg > 0) | (likely_fg > 0))
    if ys.size == 0:
        return np.zeros((h, w), dtype=np.uint8)
    x0 = max(0, int(xs.min()) - 20)
    y0 = max(0, int(ys.min()) - 20)
    x1 = min(w, int(xs.max()) + 21)
    y1 = min(h, int(ys.max()) + 21)
    rect = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(rgb, mask, rect, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return (likely_fg > 0).astype(np.uint8) * 255
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def predict_masks_for_method(result: RGBDMaskResult, rgb: np.ndarray, method: str) -> dict[str, np.ndarray]:
    if method == METHOD_HEURISTIC:
        bottle = result.extra_masks.get("bottle_candidate_mask", np.zeros_like(result.cleaned_foreground_mask))
        box = result.extra_masks.get("box_candidate_mask", np.zeros_like(result.cleaned_foreground_mask))
        foreground = result.cleaned_foreground_mask
        return {"bottle": bottle, "box": box, "foreground": foreground}

    if method == METHOD_GRABCUT:
        sure_bg = cv2.bitwise_not(result.valid_depth_mask)
        likely_fg = result.extra_masks.get("table_removed_mask", result.cleaned_foreground_mask)
        bottle_seed = result.extra_masks.get("bottle_candidate_mask", np.zeros_like(result.cleaned_foreground_mask))
        box_seed = result.extra_masks.get("box_candidate_mask", np.zeros_like(result.cleaned_foreground_mask))
        bottle = _grabcut_from_seeds(rgb, bottle_seed, cv2.dilate(bottle_seed, np.ones((11, 11), np.uint8), iterations=1), sure_bg)
        box = _grabcut_from_seeds(rgb, box_seed, cv2.dilate(box_seed, np.ones((11, 11), np.uint8), iterations=1), sure_bg)
        box = cv2.subtract(box, cv2.dilate(bottle, np.ones((7, 7), np.uint8), iterations=1))
        foreground = cv2.bitwise_or(bottle, box)
        foreground = cv2.bitwise_or(foreground, likely_fg)
        foreground = cv2.bitwise_and(foreground, result.valid_depth_mask)
        return {"bottle": bottle, "box": box, "foreground": foreground}
        if method == METHOD_SAM_REFINED:
        bottle = result.bottle_mask.copy()
        box = result.box_mask.copy()
        foreground = cv2.bitwise_or(bottle, box)
        foreground = cv2.bitwise_and(foreground, result.valid_depth_mask)
        if cv2.countNonZero(foreground) == 0:
            foreground = result.cleaned_foreground_mask.copy()
        return {"bottle": bottle, "box": box, "foreground": foreground}
    
    raise ValueError(f"Unsupported method: {method}")


def object_aug_allowed(result: RGBDMaskResult, decision: MethodDecision, mode: str) -> tuple[bool, str]:
    if mode not in {MODE_BOTTLE, MODE_BOX}:
        return True, "background_mode_allowed"

    if not decision.passed:
        return False, f"eval_gate_blocked:{decision.reason}"

    if not result.diagnostics.semantic_label_verified:
        fr = result.diagnostics.failure_reason or "semantic_unverified"
        return False, f"semantic_gate_blocked:{fr}"

    if mode == MODE_BOTTLE and int(cv2.countNonZero(result.bottle_mask)) == 0:
        return False, "semantic_gate_blocked:bottle_empty"

    if mode == MODE_BOX and int(cv2.countNonZero(result.box_mask)) == 0:
        return False, "semantic_gate_blocked:box_empty"

    return True, "object_aug_allowed"


def apply_aug_with_masks(rgb: np.ndarray, masks: dict[str, np.ndarray], mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    out = np.asarray(rgb).copy().astype(np.uint8)
    meta: dict[str, Any] = {"edited_region": mode, "depth_preserved": True, "position_preserved": True, "scale_preserved": True, "action_modified": False}
    if mode == MODE_BOTTLE:
        mask = masks["bottle"] > 0
        tint = np.array([255, 70, 220], dtype=np.uint8)
        out[mask] = ((0.55 * out[mask]) + (0.45 * tint)).astype(np.uint8)
    elif mode == MODE_BOX:
        mask = masks["box"] > 0
        tint = np.array([255, 220, 60], dtype=np.uint8)
        out[mask] = ((0.55 * out[mask]) + (0.45 * tint)).astype(np.uint8)
    elif mode == MODE_BACKGROUND:
        mask = cv2.bitwise_and(cv2.bitwise_not(masks["foreground"]), np.ones_like(masks["foreground"]) * 255) > 0
        yy, xx = np.indices(mask.shape)
        stripes = (((xx // 32) + (yy // 32)) % 2).astype(np.uint8)
        bg = np.zeros_like(out)
        bg[stripes == 0] = np.array([55, 110, 180], dtype=np.uint8)
        bg[stripes == 1] = np.array([210, 235, 90], dtype=np.uint8)
        out[mask] = bg[mask]
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return out, meta
