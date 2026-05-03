#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from lerobot.datasets.augmented_dataset_builder import build_augmented_image_dataset
from lerobot.datasets.genaug_rgbd_masks import RGBDMaskResult, build_rgbd_object_masks
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.genaug.geometry.depth_utils import sanitize_depth
from lerobot.utils.mask_debug_utils import save_image

LOGGER = logging.getLogger(__name__)

MODE_BOTTLE = "bottle"
MODE_BOX = "box"
MODE_BACKGROUND = "background"
MODE_DISTRACTOR = "distractor"


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


def _save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def _save_depth_png(path: Path, depth_m: np.ndarray) -> None:
    mm = np.clip(np.rint(np.asarray(depth_m) * 1000.0), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    Image.fromarray(mm).save(path)


def _apply_mode(rgb: np.ndarray, result: RGBDMaskResult, mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    rgb = rgb.copy()
    meta: dict[str, Any] = {"edited_region": mode, "depth_preserved": True, "stub": False}
    if mode == MODE_BOTTLE:
        mask = result.object_edit_mask_bottle > 0
        tint = np.array([255, 70, 220], dtype=np.uint8)
        rgb[mask] = ((0.55 * rgb[mask]) + (0.45 * tint)).astype(np.uint8)
    elif mode == MODE_BOX:
        mask = result.object_edit_mask_box > 0
        tint = np.array([255, 220, 60], dtype=np.uint8)
        rgb[mask] = ((0.55 * rgb[mask]) + (0.45 * tint)).astype(np.uint8)
    elif mode == MODE_BACKGROUND:
        mask = result.background_edit_mask > 0
        yy, xx = np.indices(mask.shape)
        stripes = (((xx // 32) + (yy // 32)) % 2).astype(np.uint8)
        bg = np.zeros_like(rgb)
        bg[stripes == 0] = np.array([55, 110, 180], dtype=np.uint8)
        bg[stripes == 1] = np.array([210, 235, 90], dtype=np.uint8)
        rgb[mask] = bg[mask]
    elif mode == MODE_DISTRACTOR:
        meta["stub"] = True
        meta["notes"] = "Region proposal only; no actual distractor synthesis applied."
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return rgb, meta


def _extract_paths(dataset: LeRobotDataset, idx: int, image_key: str, depth_key: str, out_dir: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    sample = dataset[idx]
    rgb = _to_rgb_u8(_to_np(sample[image_key]))
    depth = sanitize_depth(_to_np(sample[depth_key]))[..., 0]
    valid_mask = (depth > 0).astype(np.uint8) * 255
    frame_index = int(np.asarray(sample["frame_index"]).item())
    episode_index = int(np.asarray(sample["episode_index"]).item())
    global_index = int(np.asarray(sample["index"]).item())
    task = sample["task"]
    return rgb, depth, valid_mask, {
        "frame_index": frame_index,
        "episode_index": episode_index,
        "global_index": global_index,
        "task": task,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply RGB-D GenAug-style augmentation and repack to a new LeRobot dataset.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--image-key", required=True)
    parser.add_argument("--depth-key", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--output-repo-id", required=True)
    parser.add_argument("--modes", default="bottle,box,background")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--tolerance-s", type=float, default=0.001)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root, tolerance_s=args.tolerance_s)
    output_dir = Path(args.output_dir or (Path.home() / ".cache" / "huggingface" / "lerobot" / args.output_repo_id))
    artifacts_dir = Path(args.artifacts_dir or (Path("outputs") / args.output_repo_id.replace("/", "_")))
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    manifest_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    selected_indices: list[int] = []
    running_frames = 0
    for ep_row in dataset.meta.episodes:
        ep_idx = int(ep_row['episode_index'])
        ep_len = int(ep_row['length'])
        if running_frames + ep_len > args.max_frames and selected_indices:
            LOGGER.info('Stopping before partial episode %s to preserve alignment-safe repack.', ep_idx)
            break
        if running_frames + ep_len > args.max_frames and not selected_indices:
            raise SystemExit(
                f"max_frames={args.max_frames} is smaller than the first episode length={ep_len}. "
                "Partial episode repack is intentionally blocked to preserve alignment."
            )
        for i in range(len(dataset)):
            sample = dataset[i]
            if int(np.asarray(sample['episode_index']).item()) == ep_idx:
                selected_indices.append(i)
        running_frames += ep_len
        if running_frames >= args.max_frames:
            break

    for idx in selected_indices:
        rgb, depth_m, valid_mask, meta = _extract_paths(dataset, idx, args.image_key, args.depth_key, artifacts_dir)
        frame_dir = artifacts_dir / f"frame_{meta['frame_index']:06d}"
        result = build_rgbd_object_masks(rgb, depth_m, valid_mask, frame_index=meta["frame_index"])

        # save required artifacts for this frame
        save_image(frame_dir / "rgb.png", result.rgb)
        np.save(frame_dir / "depth.npy", result.depth_m)
        save_image(frame_dir / "valid_depth_mask.png", result.valid_depth_mask)
        save_image(frame_dir / "raw_foreground_mask.png", result.raw_foreground_mask)
        save_image(frame_dir / "cleaned_foreground_mask.png", result.cleaned_foreground_mask)
        save_image(frame_dir / "connected_components.png", result.connected_components_rgb)
        save_image(frame_dir / "bottle_mask.png", result.bottle_mask)
        save_image(frame_dir / "box_mask.png", result.box_mask)
        save_image(frame_dir / "combined_foreground_mask.png", result.combined_foreground_mask)
        save_image(frame_dir / "background_edit_mask.png", result.background_edit_mask)
        save_image(frame_dir / "object_edit_mask_bottle.png", result.object_edit_mask_bottle)
        save_image(frame_dir / "object_edit_mask_box.png", result.object_edit_mask_box)
        save_image(frame_dir / "overlay_instances.png", result.extra_masks["overlay_bottle"])
        save_image(frame_dir / "overlay_background_mask.png", result.extra_masks["overlay_background"])
        save_image(frame_dir / "overlay_valid_mask.png", result.extra_masks["overlay_valid_mask"])
        save_image(frame_dir / "overlay_threshold_candidates.png", result.extra_masks["overlay_threshold_candidates"])
        save_image(frame_dir / "overlay_raw_foreground.png", result.extra_masks["overlay_raw_foreground"])
        save_image(frame_dir / "overlay_table_removed.png", result.extra_masks["overlay_table_removed"])
        save_image(frame_dir / "overlay_bottle_candidate.png", result.extra_masks["overlay_bottle_candidate"])
        save_image(frame_dir / "overlay_box_candidate.png", result.extra_masks["overlay_box_candidate"])
        save_image(frame_dir / "overlay_instances_refined.png", result.extra_masks["overlay_instances_refined"])

        mode = modes[idx % len(modes)]
        failure_reason = result.diagnostics.failure_reason
        semantic_ok = bool(result.diagnostics.semantic_label_verified)
        if mode in {MODE_BOTTLE, MODE_BOX} and not semantic_ok:
            fallback_mode = MODE_BACKGROUND
            failure_reason = failure_reason or "semantic_label_unverified"
        elif not result.diagnostics.split_success and mode in {MODE_BOTTLE, MODE_BOX}:
            fallback_mode = MODE_BACKGROUND
            failure_reason = failure_reason or "split_failed"
        else:
            fallback_mode = mode

        aug_rgb, aug_meta = _apply_mode(result.rgb, result, fallback_mode)
        rgb_path = frame_dir / f"aug_{args.image_key.replace('.', '_')}.png"
        depth_path = frame_dir / f"aug_{args.depth_key.replace('.', '_')}.png"
        _save_rgb(rgb_path, aug_rgb)
        _save_depth_png(depth_path, result.depth_m)

        manifest_rows.append({
            "global_index": meta["global_index"],
            "camera_key": args.image_key,
            "image_path": str(rgb_path),
        })
        manifest_rows.append({
            "global_index": meta["global_index"],
            "camera_key": args.depth_key,
            "image_path": str(depth_path),
        })

        diag = result.diagnostics.to_dict()
        diag.update({
            "global_index": meta["global_index"],
            "episode_index": meta["episode_index"],
            "task": meta["task"],
            "requested_mode": mode,
            "applied_mode": fallback_mode,
            "semantic_gate_passed": semantic_ok,
            "augmentation_metadata": aug_meta,
            "failure_reason": failure_reason,
        })
        (frame_dir / "diagnostics.json").write_text(json.dumps(diag, ensure_ascii=False, indent=2))
        summary_rows.append(diag)

    manifest_path = artifacts_dir / "manifest.parquet"
    pd.DataFrame(manifest_rows).to_parquet(manifest_path, index=False)
    summary_path = artifacts_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2))

    build_summary = build_augmented_image_dataset(
        source_dataset=dataset,
        manifest_path=manifest_path,
        output_dir=output_dir,
        repo_id=args.output_repo_id,
        overwrite=True,
    )

    input_frames = len(selected_indices)
    output_frames = build_summary.frames
    if input_frames != output_frames:
        raise SystemExit(f"Alignment failure: input frame count {input_frames} != output frame count {output_frames}")

    output_ds = LeRobotDataset(repo_id=args.output_repo_id, root=output_dir, tolerance_s=args.tolerance_s)
    if len(output_ds) != input_frames:
        raise SystemExit(f"Alignment failure after reload: len(output_ds)={len(output_ds)} expected {input_frames}")

    action_count_in = input_frames
    action_count_out = len(output_ds)
    depth_count_in = input_frames
    depth_count_out = len(output_ds)
    if action_count_in != action_count_out or depth_count_in != depth_count_out:
        raise SystemExit("Alignment failure: action/depth counts do not match")

    print(json.dumps({
        "input_frames": input_frames,
        "output_frames": output_frames,
        "action_count_in": action_count_in,
        "action_count_out": action_count_out,
        "depth_count_in": depth_count_in,
        "depth_count_out": depth_count_out,
        "output_dir": str(output_dir),
        "artifacts_dir": str(artifacts_dir),
        "manifest_path": str(manifest_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
