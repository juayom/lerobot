#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


RGB_KEY = "observation.images.intel"


def ensure_three_channel(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        return np.repeat(mask[..., None], 3, axis=2)
    return mask


def compose_target_focused_image(rgb: np.ndarray, depth_preview: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32)
    depth_rgb = cv2.cvtColor(depth_preview, cv2.COLOR_GRAY2BGR).astype(np.float32)
    mask = (target_mask > 0).astype(np.uint8)
    mask3 = ensure_three_channel(mask).astype(np.float32)

    softened = cv2.GaussianBlur(rgb, (0, 0), 2.2).astype(np.float32)
    focus = rgb_f * (1.0 + 0.20 * mask3)
    background = 0.72 * softened + 0.28 * depth_rgb
    mixed = focus * mask3 + background * (1.0 - mask3)

    edges = cv2.Canny(depth_preview, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    mixed[edges > 0] = np.array([0, 255, 255], dtype=np.float32)

    if np.any(mask > 0):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mixed, contours, -1, (0, 0, 255), 2)

    return np.clip(mixed, 0, 255).astype(np.uint8)


def run(base_dir: Path, output_root: Path, copy_depth_preview: bool = True) -> dict:
    summary_path = base_dir / "episode-000000-summary.json"
    manifest_path = base_dir / "episode-000000-target-manifest.parquet"
    if not summary_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("target pipeline outputs not found; run depth_target_pipeline.py first")

    output_root.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(manifest_path)

    generated = []
    for row in df.to_dict(orient="records"):
        src_rgb = Path(row["image_path"])
        src_depth_preview = Path(row["depth_preview_path"])
        src_mask = Path(row["target_mask_path"])
        ep = int(row["episode_index"])
        frame = int(row["frame_index"])

        rgb = cv2.imread(str(src_rgb), cv2.IMREAD_COLOR)
        depth_preview = cv2.imread(str(src_depth_preview), cv2.IMREAD_GRAYSCALE)
        target_mask = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
        if rgb is None or depth_preview is None or target_mask is None:
            continue

        composed = compose_target_focused_image(rgb, depth_preview, target_mask)
        dst = output_root / RGB_KEY / f"episode-{ep:06d}" / f"frame-{frame:06d}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), composed)

        if copy_depth_preview:
            depth_dst = output_root / "depth_previews" / f"episode-{ep:06d}" / f"frame-{frame:06d}.png"
            depth_dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(depth_dst), depth_preview)

        generated.append(
            {
                "episode_index": ep,
                "frame_index": frame,
                "input_rgb_path": str(src_rgb),
                "genaug_input_path": str(dst),
                "target_mask_path": str(src_mask),
                "red_pixels": int(row["red_pixels"]),
            }
        )

    out_manifest = output_root / "genaug_input_manifest.parquet"
    pd.DataFrame(generated).to_parquet(out_manifest, index=False)
    summary = {
        "source_manifest": str(manifest_path),
        "output_root": str(output_root),
        "frames_generated": len(generated),
        "manifest_path": str(out_manifest),
    }
    (output_root / "genaug_input_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a target-focused mirrored image tree for GenAug input.")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    summary = run(Path(args.base_dir), Path(args.output_root))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
