#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import cv2
from PIL import Image

from lerobot.datasets.augmented_dataset_builder import build_augmented_image_dataset
from lerobot.datasets.genaug_rgbd_masks import build_rgbd_object_masks
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.rgbd_object_aug import (
    EvalThresholds,
    apply_aug_with_masks,
    choose_best_method,
    load_eval_summary,
    object_aug_allowed,
    predict_masks_for_method,
)
from lerobot.genaug.geometry.depth_utils import sanitize_depth
from lerobot.utils.mask_debug_utils import save_image
from lerobot.utils.warning_control import configure_runtime_warnings, log_structured_summary

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


def _apply_mode(rgb: np.ndarray, masks: dict[str, np.ndarray], mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    if mode == MODE_DISTRACTOR:
        return rgb.copy(), {"edited_region": mode, "depth_preserved": True, "stub": True, "notes": "Region proposal only; no actual distractor synthesis applied."}
    return apply_aug_with_masks(rgb, masks, mode)


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
    parser.add_argument("--eval-summary-json", default=None)
    parser.add_argument("--bottle-mean-min", type=float, default=0.35)
    parser.add_argument("--box-mean-min", type=float, default=0.30)
    parser.add_argument("--foreground-mean-min", type=float, default=0.60)
    parser.add_argument("--bottle-median-min", type=float, default=0.30)
    parser.add_argument("--box-median-min", type=float, default=0.25)
    parser.add_argument("--foreground-median-min", type=float, default=0.60)
    parser.add_argument("--preview-frames", type=int, default=5)
    parser.add_argument("--tolerance-s", type=float, default=0.001)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    configure_runtime_warnings()
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root, tolerance_s=args.tolerance_s)
    output_dir = Path(args.output_dir or (Path.home() / ".cache" / "huggingface" / "lerobot" / args.output_repo_id))
    artifacts_dir = Path(args.artifacts_dir or (Path("outputs") / args.output_repo_id.replace("/", "_")))
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    manifest_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    thresholds = EvalThresholds(
        bottle_mean_min=args.bottle_mean_min,
        box_mean_min=args.box_mean_min,
        foreground_mean_min=args.foreground_mean_min,
        bottle_median_min=args.bottle_median_min,
        box_median_min=args.box_median_min,
        foreground_median_min=args.foreground_median_min,
    )
    eval_summary = load_eval_summary(args.eval_summary_json)
    method_decision = choose_best_method(eval_summary, thresholds)
    chosen_method = method_decision.chosen_method or 'heuristic'
    preview_root = artifacts_dir / 'preview_frames'

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
        np.save(frame_dir / "depth.npy", result.depth_m)
        save_image(frame_dir / "rgb.png", result.rgb)
        save_image(frame_dir / "bottle_mask.png", result.bottle_mask)
        save_image(frame_dir / "box_mask.png", result.box_mask)
        save_image(frame_dir / "combined_foreground_mask.png", result.combined_foreground_mask)
        save_image(frame_dir / "overlay_bottle_candidate.png", result.extra_masks["overlay_bottle_candidate"])
        save_image(frame_dir / "overlay_box_candidate.png", result.extra_masks["overlay_box_candidate"])
        save_image(frame_dir / "overlay_instances_refined.png", result.extra_masks["overlay_instances_refined"])
        mode = modes[idx % len(modes)]
        failure_reason = result.diagnostics.failure_reason
        masks = predict_masks_for_method(result, result.rgb, chosen_method)
        allowed, gate_reason = object_aug_allowed(result, method_decision, mode)
        if mode in {MODE_BOTTLE, MODE_BOX} and not allowed:
            fallback_mode = MODE_BACKGROUND
            failure_reason = failure_reason or gate_reason
        else:
            fallback_mode = mode

        aug_rgb, aug_meta = _apply_mode(result.rgb, masks, fallback_mode)
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
            "chosen_mask_method": chosen_method,
            "eval_gate_passed": method_decision.passed,
            "semantic_gate_passed": bool(result.diagnostics.semantic_label_verified),
            "augmentation_metadata": aug_meta,
            "failure_reason": failure_reason,
            "top_plane_detected": bool(cv2.countNonZero(result.extra_masks.get("top_plane_mask", np.zeros_like(result.cleaned_foreground_mask))) > 0),
            "top_plane_area": int(cv2.countNonZero(result.extra_masks.get("top_plane_mask", np.zeros_like(result.cleaned_foreground_mask)))),
            "bottle_mask_area": int(cv2.countNonZero(result.bottle_mask)),
            "box_mask_area": int(cv2.countNonZero(result.box_mask)),
        })
        (frame_dir / "diagnostics.json").write_text(json.dumps(diag, ensure_ascii=False, indent=2))
        summary_rows.append(diag)

        if len(summary_rows) <= args.preview_frames:
            preview_dir = preview_root / f"frame_{meta['frame_index']:06d}"
            preview_dir.mkdir(parents=True, exist_ok=True)
            _save_rgb(preview_dir / 'original.png', result.rgb)
            bottle_preview, bottle_meta = _apply_mode(result.rgb, masks, MODE_BOTTLE if object_aug_allowed(result, method_decision, MODE_BOTTLE)[0] else MODE_BACKGROUND)
            box_preview, box_meta = _apply_mode(result.rgb, masks, MODE_BOX if object_aug_allowed(result, method_decision, MODE_BOX)[0] else MODE_BACKGROUND)
            background_preview, background_meta = _apply_mode(result.rgb, masks, MODE_BACKGROUND)
            _save_rgb(preview_dir / 'bottle_edit_preview.png', bottle_preview)
            _save_rgb(preview_dir / 'box_edit_preview.png', box_preview)
            _save_rgb(preview_dir / 'background_edit_preview.png', background_preview)
            save_image(preview_dir / 'overlay_selected_bottle.png', result.extra_masks.get('overlay_bottle', result.rgb))
            save_image(preview_dir / 'overlay_selected_box.png', result.extra_masks.get('overlay_box', result.rgb))
            save_image(preview_dir / 'overlay_top_plane.png', result.extra_masks.get('overlay_top_plane', result.rgb))
            (preview_dir / 'preview_meta.json').write_text(json.dumps({
                'frame_index': meta['frame_index'],
                'chosen_mask_method': chosen_method,
                'eval_gate_passed': method_decision.passed,
                'bottle_preview_mode': bottle_meta['edited_region'],
                'box_preview_mode': box_meta['edited_region'],
                'background_preview_mode': background_meta['edited_region'],
            }, ensure_ascii=False, indent=2))

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

    log_structured_summary(
        "RGB-D augmentation summary",
        {
            "input_frames": input_frames,
            "output_frames": output_frames,
            "action_count_in": action_count_in,
            "action_count_out": action_count_out,
            "depth_count_in": depth_count_in,
            "depth_count_out": depth_count_out,
            "output_dir": str(output_dir),
            "artifacts_dir": str(artifacts_dir),
            "manifest_path": str(manifest_path),
        },
    )


if __name__ == "__main__":
    main()
