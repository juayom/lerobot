#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np
from PIL import Image

from lerobot.datasets.genaug_rgbd_masks import RGBDMaskResult, build_rgbd_object_masks
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.genaug.geometry.depth_utils import sanitize_depth

LOGGER = logging.getLogger(__name__)


@dataclass
class FrameEval:
    frame_name: str
    frame_index: int
    method: str
    bottle_iou: float | None
    box_iou: float | None
    foreground_iou: float | None
    failure_types: list[str]


def _to_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    return arr


def _to_rgb_u8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind == "f" and float(arr.max(initial=0.0)) <= 1.5:
        arr = np.clip(arr * 255.0, 0, 255)
    return arr.astype(np.uint8)


def _load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8) * 255


def _iou(pred: np.ndarray | None, gt: np.ndarray | None) -> float | None:
    if gt is None:
        return None
    if pred is None:
        pred = np.zeros_like(gt)
    pred_bin = pred > 0
    gt_bin = gt > 0
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(pred_bin, gt_bin).sum()
    return float(inter / union)


def _sample_from_dataset(dataset: LeRobotDataset, frame_index: int) -> dict[str, Any]:
    if 0 <= frame_index < len(dataset):
        sample = dataset[frame_index]
        try:
            actual = int(np.asarray(sample["frame_index"]).item())
            if actual == frame_index:
                return sample
        except Exception:
            return sample
    for i in range(len(dataset)):
        sample = dataset[i]
        actual = int(np.asarray(sample["frame_index"]).item())
        if actual == frame_index:
            return sample
    raise IndexError(f"Could not find dataset sample for frame_index={frame_index}")


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
    except cv2.error as exc:
        LOGGER.warning("GrabCut failed, returning likely_fg only: %s", exc)
        return (likely_fg > 0).astype(np.uint8) * 255
    out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return out


def _predict_masks(result: RGBDMaskResult, rgb: np.ndarray, method: str) -> dict[str, np.ndarray]:
    if method == "heuristic":
        bottle = result.extra_masks.get("bottle_candidate_mask", np.zeros_like(result.cleaned_foreground_mask))
        box = result.extra_masks.get("box_candidate_mask", np.zeros_like(result.cleaned_foreground_mask))
        foreground = result.cleaned_foreground_mask
        return {
            "bottle": bottle,
            "box": box,
            "foreground": foreground,
        }
    if method == "grabcut":
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
        return {
            "bottle": bottle,
            "box": box,
            "foreground": foreground,
        }
    raise ValueError(f"Unsupported method: {method}")


def _classify_failure(preds: dict[str, np.ndarray], gt_bottle: np.ndarray | None, gt_box: np.ndarray | None, gt_foreground: np.ndarray | None) -> list[str]:
    out: list[str] = []
    if gt_bottle is not None:
        pred_area = int((preds["bottle"] > 0).sum())
        gt_area = int((gt_bottle > 0).sum())
        if gt_area > 0 and pred_area < 0.45 * gt_area:
            out.append("bottle_too_thin")
    if gt_box is not None:
        pred_area = int((preds["box"] > 0).sum())
        gt_area = int((gt_box > 0).sum())
        if gt_area > 0 and pred_area < 0.45 * gt_area:
            out.append("box_too_local")
    if gt_foreground is not None:
        pred_fg = int((preds["foreground"] > 0).sum())
        gt_fg = int((gt_foreground > 0).sum())
        if gt_fg > 0 and pred_fg > 1.6 * gt_fg:
            out.append("table_support_confusion")
    if gt_bottle is not None and gt_box is not None:
        biou = _iou(preds["bottle"], gt_bottle) or 0.0
        xiou = _iou(preds["box"], gt_box) or 0.0
        fiou = _iou(preds["foreground"], gt_foreground if gt_foreground is not None else cv2.bitwise_or(gt_bottle, gt_box)) or 0.0
        if biou < 0.25 and xiou < 0.25 and fiou >= 0.35:
            out.append("vertical_stack_ambiguity")
    return sorted(set(out))


def _summarize(method: str, rows: list[FrameEval]) -> dict[str, Any]:
    def _collect(name: str) -> list[float]:
        vals = [getattr(r, name) for r in rows if getattr(r, name) is not None]
        return [float(v) for v in vals]

    bottle_vals = _collect("bottle_iou")
    box_vals = _collect("box_iou")
    fg_vals = _collect("foreground_iou")
    failures = {
        r.frame_name: {
            "bottle_iou": r.bottle_iou,
            "box_iou": r.box_iou,
            "foreground_iou": r.foreground_iou,
            "failure_types": r.failure_types,
        }
        for r in rows
        if r.failure_types or (r.bottle_iou is not None and r.bottle_iou < 0.25) or (r.box_iou is not None and r.box_iou < 0.25)
    }
    failure_counts: dict[str, int] = {}
    for r in rows:
        for key in r.failure_types:
            failure_counts[key] = failure_counts.get(key, 0) + 1

    return {
        "method": method,
        "num_frames": len(rows),
        "mean_iou": {
            "bottle": float(np.mean(bottle_vals)) if bottle_vals else None,
            "box": float(np.mean(box_vals)) if box_vals else None,
            "foreground": float(np.mean(fg_vals)) if fg_vals else None,
        },
        "median_iou": {
            "bottle": float(median(bottle_vals)) if bottle_vals else None,
            "box": float(median(box_vals)) if box_vals else None,
            "foreground": float(median(fg_vals)) if fg_vals else None,
        },
        "failed_frames": failures,
        "failure_type_counts": failure_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate heuristic and seeded GrabCut RGB-D masks against manual GT masks.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--image-key", required=True)
    parser.add_argument("--depth-key", required=True)
    parser.add_argument("--methods", default="heuristic,grabcut")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--tolerance-s", type=float, default=0.001)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dataset = LeRobotDataset(repo_id=args.repo_id, tolerance_s=args.tolerance_s)
    gt_root = Path(args.gt_root)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    frame_dirs = sorted([p for p in gt_root.iterdir() if p.is_dir() and p.name.startswith("frame_")])
    if not frame_dirs:
        raise SystemExit(f"No GT frame directories found under {gt_root}")

    rows_by_method: dict[str, list[FrameEval]] = {m: [] for m in methods}

    for frame_dir in frame_dirs:
        frame_index = int(frame_dir.name.split("_")[-1])
        sample = _sample_from_dataset(dataset, frame_index)
        rgb = _to_rgb_u8(_to_np(sample[args.image_key]))
        depth = sanitize_depth(_to_np(sample[args.depth_key]))[..., 0]
        valid = (depth > 0).astype(np.uint8) * 255
        result = build_rgbd_object_masks(rgb, depth, valid, frame_index=frame_index)

        gt_bottle = _load_mask(frame_dir / "gt_bottle_mask.png")
        gt_box = _load_mask(frame_dir / "gt_box_mask.png")
        gt_foreground = _load_mask(frame_dir / "gt_foreground_mask.png")
        if gt_foreground is None and gt_bottle is not None and gt_box is not None:
            gt_foreground = cv2.bitwise_or(gt_bottle, gt_box)

        for method in methods:
            preds = _predict_masks(result, rgb, method)
            row = FrameEval(
                frame_name=frame_dir.name,
                frame_index=frame_index,
                method=method,
                bottle_iou=_iou(preds["bottle"], gt_bottle),
                box_iou=_iou(preds["box"], gt_box),
                foreground_iou=_iou(preds["foreground"], gt_foreground),
                failure_types=_classify_failure(preds, gt_bottle, gt_box, gt_foreground),
            )
            rows_by_method[method].append(row)

    summary = {method: _summarize(method, rows) for method, rows in rows_by_method.items()}
    out_json = Path(args.out_json or (Path("outputs") / "rgbd_mask_gt_eval_summary.json"))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
