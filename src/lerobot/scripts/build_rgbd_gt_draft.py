#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from controlnet_aux.segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from controlnet_aux.segment_anything.predictor import SamPredictor

from lerobot.datasets.genaug_rgbd_masks import build_rgbd_object_masks
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.rgbd_object_aug import predict_masks_for_method
from lerobot.genaug.geometry.depth_utils import sanitize_depth
from lerobot.utils.mask_debug_utils import depth_preview, overlay_mask, save_image


@dataclass
class FrameProbe:
    frame_index: int
    section: str
    fg_area_ratio: float
    bottle_area_ratio: float
    box_area_ratio: float
    semantic_label_verified: bool
    failure_reason: str
    draft_quality_hint: str


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


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def _centroid_from_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


def _mask_area_ratio(mask: np.ndarray) -> float:
    return float((mask > 0).sum() / mask.size)


def _quality_from_result(result) -> str:
    fg_ratio = _mask_area_ratio(result.cleaned_foreground_mask)
    if result.diagnostics.semantic_label_verified:
        return "high"
    if 0.005 <= fg_ratio <= 0.16:
        return "medium"
    return "low"


def _section_name(idx: int, total: int) -> str:
    frac = idx / max(1, total - 1)
    if frac < 1 / 3:
        return "early"
    if frac < 2 / 3:
        return "mid"
    return "late"


def _build_points(pos_mask: np.ndarray, neg_mask: np.ndarray) -> tuple[list[list[float]], list[int]]:
    points: list[list[float]] = []
    labels: list[int] = []
    cen = _centroid_from_mask(pos_mask)
    if cen is not None:
        points.append(cen)
        labels.append(1)
    box = _bbox_from_mask(pos_mask)
    if box is not None:
        x, y, w, h = box
        points.extend([[x + 0.3 * w, y + 0.5 * h], [x + 0.7 * w, y + 0.5 * h]])
        labels.extend([1, 1])
    neg = _bbox_from_mask(neg_mask)
    if neg is not None:
        x, y, w, h = neg
        points.append([x + 0.5 * w, y + 0.5 * h])
        labels.append(0)
    return points, labels


def _sam_bundle() -> tuple[SamPredictor, SamAutomaticMaskGenerator]:
    ckpt = hf_hub_download("segments-arnaud/sam_vit_b", "sam_vit_b_01ec64.pth")
    sam = sam_model_registry["vit_b"](checkpoint=ckpt)
    sam.eval()
    return SamPredictor(sam), SamAutomaticMaskGenerator(sam)


def _predict_sam_mask(predictor: SamPredictor, rgb: np.ndarray, box: list[int] | None, points: list[list[float]] | None, labels: list[int] | None) -> np.ndarray:
    predictor.set_image(rgb)
    kwargs: dict[str, Any] = {"multimask_output": True}
    if box is not None:
        x, y, w, h = box
        kwargs["box"] = np.array([x, y, x + w - 1, y + h - 1], dtype=np.float32)
    if points:
        kwargs["point_coords"] = np.array(points, dtype=np.float32)
        kwargs["point_labels"] = np.array(labels, dtype=np.int32)
    masks, scores, _ = predictor.predict(**kwargs)
    return (masks[int(np.argmax(scores))].astype(np.uint8) * 255)


def _ensure_binary(mask: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and((mask > 0).astype(np.uint8) * 255, valid_mask)


def _choose_frames(ds: LeRobotDataset, image_key: str, depth_key: str, target_count: int) -> list[FrameProbe]:
    # Fast representative scan: stratified candidates instead of full exhaustive sweep.
    anchors = sorted({0, 4, 12, 23, 33, 48, 64, 80, 96, 112, 140, 168, 184, 200, len(ds) - 1})
    probes: list[FrameProbe] = []
    for i in anchors:
        sample = ds[i]
        rgb = _to_rgb_u8(_to_np(sample[image_key]))
        depth = sanitize_depth(_to_np(sample[depth_key]))[..., 0]
        valid = (depth > 0).astype(np.uint8) * 255
        result = build_rgbd_object_masks(rgb, depth, valid, frame_index=i)
        probes.append(
            FrameProbe(
                frame_index=i,
                section=_section_name(i, len(ds)),
                fg_area_ratio=_mask_area_ratio(result.cleaned_foreground_mask),
                bottle_area_ratio=_mask_area_ratio(result.extra_masks.get("bottle_candidate_mask", np.zeros_like(valid))),
                box_area_ratio=_mask_area_ratio(result.extra_masks.get("box_candidate_mask", np.zeros_like(valid))),
                semantic_label_verified=bool(result.diagnostics.semantic_label_verified),
                failure_reason=result.diagnostics.failure_reason,
                draft_quality_hint=_quality_from_result(result),
            )
        )

    selected: list[int] = []
    by_section = {
        sec: [p for p in probes if p.section == sec] for sec in ["early", "mid", "late"]
    }

    def add_candidates(cands: list[FrameProbe], limit: int) -> None:
        for p in cands:
            if p.frame_index not in selected:
                selected.append(p.frame_index)
            if len(selected) >= limit:
                return

    # best foreground-localized per section
    for sec in ["early", "mid", "late"]:
        cands = sorted(by_section[sec], key=lambda p: (abs(p.fg_area_ratio - 0.05), -p.bottle_area_ratio - p.box_area_ratio))
        add_candidates(cands[:2], len(selected) + 2)

    # bottle-visible candidates
    add_candidates(sorted(probes, key=lambda p: p.bottle_area_ratio, reverse=True)[:4], len(selected) + 4)
    # box-visible candidates
    add_candidates(sorted(probes, key=lambda p: p.box_area_ratio, reverse=True)[:4], len(selected) + 4)
    # ambiguous/failure candidates
    ambiguous = [p for p in probes if p.failure_reason]
    add_candidates(sorted(ambiguous, key=lambda p: (p.draft_quality_hint != "low", -p.fg_area_ratio))[:6], len(selected) + 6)

    # fill to target count with section balance
    sec_cycle = ["early", "mid", "late"]
    while len(selected) < target_count:
        changed = False
        for sec in sec_cycle:
            cands = sorted(by_section[sec], key=lambda p: (p.frame_index not in selected, abs(p.fg_area_ratio - 0.05)))
            for p in cands:
                if p.frame_index not in selected:
                    selected.append(p.frame_index)
                    changed = True
                    break
            if len(selected) >= target_count:
                break
        if not changed:
            break

    selected = sorted(selected[:target_count])
    probe_map = {p.frame_index: p for p in probes}
    return [probe_map[idx] for idx in selected]


def _draft_quality(fg_ratio: float, bottle_nonempty: bool, box_nonempty: bool) -> str:
    if bottle_nonempty and box_nonempty and 0.005 <= fg_ratio <= 0.14:
        return "medium"
    if 0.005 <= fg_ratio <= 0.18:
        return "medium"
    return "low"


def _write_readme(gt_root: Path) -> None:
    text = """# GT Draft Labeling Guide\n\nThis folder contains DRAFT GT masks for manual correction.\n\n## Files to review/edit\n- draft_gt_bottle_mask.png\n- draft_gt_box_mask.png\n- draft_gt_foreground_mask.png\n\n## Mask rule\n- white (255) = foreground / object region\n- black (0) = background\n\n## Class split rule\n- bottle: only the bottle / pill container region\n- box: only the box region\n- foreground: union of bottle and box, or a corrected full object foreground mask\n\n## Important\n- These are DRAFT masks only, not final GT.\n- Draft masks may be empty or wrong. Final human review is mandatory.\n- If bottle/box draft is unreliable, redraw it manually.\n\n## Final file names for evaluation\nAfter manual correction, save final masks with these names in each frame folder:\n- gt_bottle_mask.png\n- gt_box_mask.png\n- gt_foreground_mask.png\n\nThen evaluate with eval_rgbd_mask_gt.py.\n"""
    (gt_root / "gt_labeling_readme.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build human-in-the-loop GT draft folders from RGB-D dataset.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--image-key", default="observation.images.rgb")
    parser.add_argument("--depth-key", default="observation.images.depth")
    parser.add_argument("--target-count", type=int, default=12)
    parser.add_argument("--out-root", default=None)
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    ds = LeRobotDataset(repo_id=args.repo_id, tolerance_s=0.001)
    sample0 = ds[0]
    image_key = args.image_key if args.image_key in sample0 else next(k for k in sample0.keys() if "observation.images." in k and "depth" not in k)
    depth_key = args.depth_key if args.depth_key in sample0 else next(k for k in sample0.keys() if "depth" in k)
    gt_root = Path(args.out_root or f"outputs/{args.repo_id.replace('/', '_')}_gt_draft_20260505")
    gt_root.mkdir(parents=True, exist_ok=True)

    selected = _choose_frames(ds, image_key, depth_key, args.target_count)
    predictor, _ = _sam_bundle()

    summary: dict[str, Any] = {
        "repo_id": args.repo_id,
        "resolved_keys": {"image": image_key, "depth": depth_key},
        "selected_frames": [p.frame_index for p in selected],
        "draft_quality_counts": {"high": 0, "medium": 0, "low": 0},
        "frames": [],
    }

    for probe in selected:
        idx = probe.frame_index
        sample = ds[idx]
        rgb = _to_rgb_u8(_to_np(sample[image_key]))
        depth = sanitize_depth(_to_np(sample[depth_key]))[..., 0]
        valid = (depth > 0).astype(np.uint8) * 255
        result = build_rgbd_object_masks(rgb, depth, valid, frame_index=idx)
        seeded = predict_masks_for_method(result, rgb, "grabcut")
        fg_box = _bbox_from_mask(seeded["foreground"])
        bottle_box = _bbox_from_mask(seeded["bottle"])
        box_box = _bbox_from_mask(seeded["box"])
        bottle_points, bottle_labels = _build_points(seeded["bottle"], seeded["box"])
        box_points, box_labels = _build_points(seeded["box"], seeded["bottle"])
        sam_box = {
            "bottle": _ensure_binary(_predict_sam_mask(predictor, rgb, bottle_box or fg_box, None, None), valid),
            "box": _ensure_binary(_predict_sam_mask(predictor, rgb, box_box or fg_box, None, None), valid),
        }
        sam_box_points = {
            "bottle": _ensure_binary(_predict_sam_mask(predictor, rgb, bottle_box or fg_box, bottle_points, bottle_labels), valid),
            "box": _ensure_binary(_predict_sam_mask(predictor, rgb, box_box or fg_box, box_points, box_labels), valid),
        }
        for m in (sam_box, sam_box_points):
            m["box"] = cv2.subtract(m["box"], cv2.dilate(m["bottle"], np.ones((5, 5), np.uint8), iterations=1))
            m["foreground"] = cv2.bitwise_and(cv2.bitwise_or(m["bottle"], m["box"]), valid)

        candidates = {
            "seeded_grabcut": seeded,
            "sam_prompt_box": sam_box,
            "sam_prompt_box_points": sam_box_points,
        }

        def score(name: str, masks: dict[str, np.ndarray]) -> tuple[float, str]:
            fg_ratio = _mask_area_ratio(masks["foreground"])
            bottle_ratio = _mask_area_ratio(masks["bottle"])
            box_ratio = _mask_area_ratio(masks["box"])
            val = -abs(fg_ratio - 0.05) + min(bottle_ratio, 0.01) * 20 + min(box_ratio, 0.02) * 10
            return val, name

        selected_method = max((score(name, masks) for name, masks in candidates.items()))[1]
        chosen = candidates[selected_method]

        bottle_ratio = _mask_area_ratio(chosen["bottle"])
        box_ratio = _mask_area_ratio(chosen["box"])
        fg_ratio = _mask_area_ratio(chosen["foreground"])
        bottle_nonempty = 0.0008 <= bottle_ratio <= 0.08
        box_nonempty = 0.0015 <= box_ratio <= 0.12
        warning_bits: list[str] = []
        if not bottle_nonempty:
            chosen["bottle"] = np.zeros_like(valid)
            warning_bits.append("bottle draft unstable -> empty draft used")
        if not box_nonempty:
            chosen["box"] = np.zeros_like(valid)
            warning_bits.append("box draft unstable -> empty draft used")
        fg_draft = cv2.bitwise_or(chosen["bottle"], chosen["box"])
        if int((fg_draft > 0).sum()) == 0:
            fg_draft = seeded["foreground"]
            warning_bits.append("foreground draft fallback to seeded foreground")
        draft_quality = _draft_quality(_mask_area_ratio(fg_draft), bottle_nonempty, box_nonempty)
        summary["draft_quality_counts"][draft_quality] += 1

        frame_dir = gt_root / f"frame_{idx:04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        save_image(frame_dir / "rgb.png", rgb)
        save_image(frame_dir / "depth_vis.png", depth_preview(depth))
        save_image(frame_dir / "draft_gt_bottle_mask.png", chosen["bottle"])
        save_image(frame_dir / "draft_gt_box_mask.png", chosen["box"])
        save_image(frame_dir / "draft_gt_foreground_mask.png", fg_draft)
        overlay = overlay_mask(rgb, fg_draft, (0, 255, 255))
        overlay = overlay_mask(overlay, chosen["bottle"], (255, 0, 255))
        overlay = overlay_mask(overlay, chosen["box"], (255, 255, 0))
        save_image(frame_dir / "overlay_draft_gt.png", overlay)
        notes = {
            "frame_index": idx,
            "selected_source_method": selected_method,
            "draft_quality": draft_quality,
            "human_review_required": True,
            "bottle_draft_nonempty": bool((chosen["bottle"] > 0).sum() > 0),
            "box_draft_nonempty": bool((chosen["box"] > 0).sum() > 0),
            "warning": "; ".join(warning_bits) if warning_bits else "draft only; final human verification required",
            "comment": "semantic_label_verified=false in automatic pipeline; treat all masks here as editable draft only",
        }
        (frame_dir / "notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2))
        summary["frames"].append(notes)

    _write_readme(gt_root)
    (gt_root / "draft_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({
        "gt_root": str(gt_root),
        "selected_frames": summary["selected_frames"],
        "draft_quality_counts": summary["draft_quality_counts"],
        "resolved_keys": summary["resolved_keys"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
