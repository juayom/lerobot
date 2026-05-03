#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.genaug.geometry.depth_utils import depth_to_uint8_preview, sanitize_depth
from lerobot.genaug.geometry.mask_utils import save_mask_preview
from lerobot.genaug.geometry.object_mask import apply_background_swap, estimate_object_mask_from_depth


def _to_np(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    return arr


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype.kind == 'f' and float(arr.max(initial=0.0)) <= 1.5:
        arr = np.clip(arr * 255.0, 0, 255)
    return arr.astype(np.uint8)


def _resolve_keys(dataset: LeRobotDataset, image_key: str | None, depth_key: str | None) -> tuple[str, str]:
    camera_keys = list(dataset.meta.camera_keys)
    if image_key is None:
        image_candidates = [k for k in camera_keys if not k.endswith("_depth")]
        if not image_candidates:
            raise ValueError(f"No RGB camera key found in {camera_keys}")
        image_key = image_candidates[0]
    if depth_key is None:
        depth_candidates = [k for k in camera_keys if k.endswith("_depth") or "depth" in k]
        if not depth_candidates:
            raise ValueError(f"No depth camera key found in {camera_keys}")
        depth_key = depth_candidates[0]
    return image_key, depth_key


def _choose_sample_index(dataset: LeRobotDataset, episode_index: int | None, frame_index: int | None) -> int:
    if episode_index is None and frame_index is None:
        return len(dataset) // 2

    for i in range(len(dataset)):
        sample = dataset[i]
        ep = int(np.asarray(sample["episode_index"]).item())
        fr = int(np.asarray(sample["frame_index"]).item())
        if (episode_index is None or ep == episode_index) and (frame_index is None or fr == frame_index):
            return i
    raise IndexError("Requested episode/frame not found in dataset")


def _save_gray(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)


def _save_overlay(rgb: np.ndarray, mask: np.ndarray, path: Path) -> None:
    overlay = rgb.copy()
    overlay[mask > 0] = np.array([255, 255, 0], dtype=np.uint8)
    preview = cv2.addWeighted(rgb, 0.7, overlay, 0.3, 0)
    Image.fromarray(preview).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe whether depth-based object masking is usable for GenAug-style augmentation.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument("--image-key", default=None)
    parser.add_argument("--depth-key", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance-s", type=float, default=0.001)
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root, tolerance_s=args.tolerance_s)
    image_key, depth_key = _resolve_keys(dataset, args.image_key, args.depth_key)
    sample_idx = _choose_sample_index(dataset, args.episode_index, args.frame_index)
    sample = dataset[sample_idx]

    rgb = _to_uint8_rgb(_to_np(sample[image_key]))
    depth_raw = _to_np(sample[depth_key])
    depth_m = sanitize_depth(depth_raw)[..., 0]
    mask, diag = estimate_object_mask_from_depth(depth_raw)
    aug = apply_background_swap(rgb, mask, seed=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(rgb).save(out_dir / "rgb.png")
    Image.fromarray(depth_to_uint8_preview(depth_raw)).save(out_dir / "depth_preview.png")
    _save_gray(((depth_m > 0).astype(np.uint8) * 255), out_dir / "valid_depth_mask.png")
    _save_gray(mask, out_dir / "object_mask.png")
    _save_overlay(rgb, mask, out_dir / "mask_overlay.png")
    Image.fromarray(aug).save(out_dir / "aug_background_swap.png")
    save_mask_preview(rgb, cv2.bitwise_not(mask), out_dir / "background_region_preview.png")

    report = {
        "repo_id": args.repo_id,
        "root": args.root,
        "dataset_len": len(dataset),
        "sample_idx": sample_idx,
        "episode_index": int(np.asarray(sample["episode_index"]).item()),
        "frame_index": int(np.asarray(sample["frame_index"]).item()),
        "image_key": image_key,
        "depth_key": depth_key,
        "diagnostics": {
            "valid_ratio": diag.valid_ratio,
            "valid_depth_min_m": diag.valid_depth_min_m,
            "valid_depth_max_m": diag.valid_depth_max_m,
            "valid_depth_median_m": diag.valid_depth_median_m,
            "foreground_depth_threshold_m": diag.foreground_depth_threshold_m,
            "mask_area_ratio": diag.mask_area_ratio,
            "component_count": diag.component_count,
            "selected_component_area": diag.selected_component_area,
            "depth_contrast_m": diag.depth_contrast_m,
            "is_depth_usable": diag.is_depth_usable,
            "is_mask_usable": diag.is_mask_usable,
            "summary": diag.summary,
        },
        "next_step": (
            "paired_rgbd_object_augmentation_can_be_attempted"
            if diag.is_mask_usable
            else "collect_more_data_or_adjust_masking_strategy"
        ),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
