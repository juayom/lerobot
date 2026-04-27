#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


RGB_KEY = "observation.images.intel"
DEPTH_KEY = "observation.images.intel_depth"


def read_episode_bounds(root: Path, episode_index: int) -> tuple[int, int]:
    episodes = pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pandas()
    row = episodes.loc[episodes["episode_index"] == episode_index]
    if row.empty:
        raise ValueError(f"episode {episode_index} not found")
    item = row.iloc[0]
    return int(item["dataset_from_index"]), int(item["dataset_to_index"])


def rect_mask(shape: tuple[int, int], rect: list[int] | tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    x0, y0, x1, y1 = [int(v) for v in rect]
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    out = np.zeros((h, w), dtype=np.uint8)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = 255
    return out


def parse_profile(profile_path: Path | None) -> dict:
    if profile_path is None:
        return {}
    return json.loads(profile_path.read_text())


def build_red_mask(rgb_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 70, 40], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)
    lower2 = np.array([165, 70, 40], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def build_box_mask(rgb_bgr: np.ndarray, include_rects: list[list[int]] | None = None) -> np.ndarray:
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    box = cv2.GaussianBlur(gray, (0, 0), 1.2)
    box = cv2.adaptiveThreshold(
        box,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        4,
    )
    box = cv2.morphologyEx(box, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    box = cv2.morphologyEx(box, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    if include_rects:
        union = np.zeros_like(box)
        for rect in include_rects:
            union = cv2.bitwise_or(union, rect_mask(box.shape, rect))
        box = cv2.bitwise_and(box, union)
    return box


def filter_connected_components(mask: np.ndarray, min_area: int = 0) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    refined = np.zeros_like(binary)
    for cid in range(1, n_labels):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        refined[labels == cid] = 1
    return (refined * 255).astype(np.uint8)


def refine_target_mask(
    rgb_bgr: np.ndarray,
    red_mask: np.ndarray,
    profile: dict | None = None,
) -> np.ndarray:
    profile = profile or {}
    include_rects = profile.get("include_rects", [])
    exclude_rects = profile.get("exclude_rects", [])
    box_rects = profile.get("box_rects", [])
    min_area = int(profile.get("min_area", 20))
    dilate_iter = int(profile.get("dilate_iterations", 2))

    target = red_mask.copy()

    if box_rects:
        box_mask = build_box_mask(rgb_bgr, include_rects=box_rects)
        target = cv2.bitwise_or(target, box_mask)

    if include_rects:
        include_union = np.zeros_like(target)
        for rect in include_rects:
            include_union = cv2.bitwise_or(include_union, rect_mask(target.shape, rect))
        target = cv2.bitwise_and(target, include_union)

    if exclude_rects:
        exclude_union = np.zeros_like(target)
        for rect in exclude_rects:
            exclude_union = cv2.bitwise_or(exclude_union, rect_mask(target.shape, rect))
        target[exclude_union > 0] = 0

    target = filter_connected_components(target, min_area=min_area)
    if dilate_iter > 0 and np.any(target > 0):
        target = cv2.dilate(target, np.ones((5, 5), np.uint8), iterations=dilate_iter)
    target = cv2.morphologyEx(target, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return target


def denoise_depth(depth_u16: np.ndarray) -> np.ndarray:
    depth = depth_u16.copy()
    valid = depth > 0
    if not np.any(valid):
        return depth

    depth_f = depth.astype(np.float32)
    median = cv2.medianBlur(depth, 5)
    depth_f[valid] = median[valid].astype(np.float32)

    local = cv2.GaussianBlur(depth_f, (0, 0), 1.2)
    depth_f[valid] = 0.7 * depth_f[valid] + 0.3 * local[valid]
    depth_f[~valid] = 0
    return np.clip(depth_f, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def normalize_depth_for_preview(depth_u16: np.ndarray, focus_mask: np.ndarray | None = None) -> np.ndarray:
    valid = depth_u16 > 0
    if focus_mask is not None:
        valid = valid & (focus_mask > 0)
    if not np.any(valid):
        valid = depth_u16 > 0
    if not np.any(valid):
        return np.zeros(depth_u16.shape, dtype=np.uint8)

    values = depth_u16[valid].astype(np.float32)
    lo = np.percentile(values, 5)
    hi = np.percentile(values, 95)
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((depth_u16.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    preview = (scaled * 255.0).astype(np.uint8)
    preview = 255 - preview
    return preview


def contour_overlay(rgb_bgr: np.ndarray, depth_preview: np.ndarray, focus_mask: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(depth_preview, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    overlay = rgb_bgr.copy()
    overlay[edges > 0] = (0, 255, 255)

    if np.any(focus_mask > 0):
        contours, _ = cv2.findContours(focus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    return overlay


def extract_video_frames(video_path: Path, output_pattern: Path) -> None:
    output_pattern.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        str(output_pattern),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write {path}")


def run(root: Path, episode_index: int, output_dir: Path, profile_path: Path | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    from_idx, to_idx = read_episode_bounds(root, episode_index)
    frame_count = to_idx - from_idx
    profile = parse_profile(profile_path)

    rgb_video = root / "videos" / RGB_KEY / "chunk-000" / "file-000.mkv"
    depth_video = root / "videos" / DEPTH_KEY / "chunk-000" / "file-000.mkv"

    raw_rgb_dir = output_dir / "_raw_rgb"
    raw_depth_dir = output_dir / "_raw_depth"
    extract_video_frames(rgb_video, raw_rgb_dir / "frame-%06d.png")
    extract_video_frames(depth_video, raw_depth_dir / "frame-%06d.png")

    rgb_files = sorted(raw_rgb_dir.glob("frame-*.png"))
    depth_files = sorted(raw_depth_dir.glob("frame-*.png"))
    usable_frames = min(frame_count, len(rgb_files), len(depth_files))

    rows = []
    target_pixel_counts = []
    best_frame = None

    for rel_idx in range(usable_frames):
        rgb = cv2.imread(str(rgb_files[rel_idx]), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_files[rel_idx]), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            continue

        if depth.ndim == 3:
            depth = depth[..., 0]
        depth_u16 = depth.astype(np.uint16)
        depth_clean = denoise_depth(depth_u16)
        red_mask = build_red_mask(rgb)
        target_mask = refine_target_mask(rgb, red_mask, profile=profile)
        depth_preview = normalize_depth_for_preview(depth_clean, target_mask)
        overlay = contour_overlay(rgb, depth_preview, target_mask)

        global_index = from_idx + rel_idx
        rgb_path = output_dir / RGB_KEY / f"episode-{episode_index:06d}" / f"frame-{rel_idx:06d}.png"
        depth_path = output_dir / DEPTH_KEY / f"episode-{episode_index:06d}" / f"frame-{rel_idx:06d}.png"
        preview_path = output_dir / "previews" / f"episode-{episode_index:06d}" / f"frame-{rel_idx:06d}.png"
        overlay_path = output_dir / "overlays" / f"episode-{episode_index:06d}" / f"frame-{rel_idx:06d}.png"
        mask_path = output_dir / "target_masks" / f"episode-{episode_index:06d}" / f"frame-{rel_idx:06d}.png"
        red_mask_path = output_dir / "red_masks" / f"episode-{episode_index:06d}" / f"frame-{rel_idx:06d}.png"

        save_png(rgb_path, rgb)
        save_png(depth_path, depth_clean)
        save_png(preview_path, depth_preview)
        save_png(overlay_path, overlay)
        save_png(mask_path, target_mask)
        save_png(red_mask_path, red_mask)

        target_pixels = int((target_mask > 0).sum())
        target_pixel_counts.append(target_pixels)
        if best_frame is None or target_pixels > best_frame["target_pixels"]:
            best_frame = {
                "frame_index": rel_idx,
                "global_index": global_index,
                "target_pixels": target_pixels,
                "overlay_path": str(overlay_path),
                "depth_preview_path": str(preview_path),
                "depth_clean_path": str(depth_path),
                "target_mask_path": str(mask_path),
            }

        rows.append(
            {
                "episode_index": episode_index,
                "frame_index": rel_idx,
                "global_index": global_index,
                "camera_key": RGB_KEY,
                "image_path": str(rgb_path),
                "depth_path": str(depth_path),
                "depth_preview_path": str(preview_path),
                "overlay_path": str(overlay_path),
                "target_mask_path": str(mask_path),
                "red_mask_path": str(red_mask_path),
                "target_pixels": target_pixels,
            }
        )

    manifest_path = output_dir / f"episode-{episode_index:06d}-target-manifest.parquet"
    pd.DataFrame(rows).to_parquet(manifest_path, index=False)

    summary = {
        "root": str(root),
        "episode_index": episode_index,
        "frames_processed": len(rows),
        "manifest_path": str(manifest_path),
        "profile_path": str(profile_path) if profile_path else None,
        "best_frame": best_frame,
        "target_pixel_stats": {
            "max": int(max(target_pixel_counts)) if target_pixel_counts else 0,
            "median": float(np.median(target_pixel_counts)) if target_pixel_counts else 0.0,
            "nonzero_frames": int(sum(1 for x in target_pixel_counts if x > 0)),
        },
    }
    (output_dir / f"episode-{episode_index:06d}-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and clean RGB/depth frames for target-aware augmentation checks.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile-json", default=None, help="Optional scene profile with include/exclude/box rects")
    args = parser.parse_args()

    summary = run(
        Path(args.root),
        args.episode_index,
        Path(args.output_dir),
        profile_path=Path(args.profile_json) if args.profile_json else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
