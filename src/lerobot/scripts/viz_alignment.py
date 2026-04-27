#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.genaug.geometry.depth_utils import depth_to_uint8_preview


def _to_np(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    return arr


def _find_sample(dataset: LeRobotDataset, episode_index: int, frame_index: int):
    for i in range(len(dataset)):
        sample = dataset[i]
        ep = int(np.asarray(sample["episode_index"]).item())
        fr = int(np.asarray(sample["frame_index"]).item())
        if ep == episode_index and fr == frame_index:
            return sample
    raise IndexError(f"Sample not found for episode {episode_index}, frame {frame_index}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RGB-D / GenAug alignment")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", default="local/genaug_rgbd")
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-key", default="observation.images.rgb")
    parser.add_argument("--depth-key", default="observation.images.depth")
    parser.add_argument("--tolerance-s", type=float, default=0.001)
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root, tolerance_s=args.tolerance_s)
    sample = _find_sample(dataset, args.episode_index, args.frame_index)

    rgb = _to_np(sample[args.image_key]).astype(np.uint8)
    depth = _to_np(sample[args.depth_key])
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth_preview = depth_to_uint8_preview(depth)

    aug_rgb = rgb
    mask_overlay = rgb.copy()
    if "genaug.mode" in sample:
        mask_overlay[..., 1] = np.clip(mask_overlay[..., 1] + 48, 0, 255)

    panels = [Image.fromarray(x) for x in [rgb, depth_preview, aug_rgb, mask_overlay]]
    width = sum(p.width for p in panels)
    height = max(p.height for p in panels) + 100
    canvas = Image.new("RGB", (width, height), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    x = 0
    labels = ["original rgb", "depth preview", "augmented rgb", "mask overlay"]
    for panel, label in zip(panels, labels, strict=True):
        canvas.paste(panel, (x, 0))
        draw.text((x + 10, panel.height + 8), label, fill=(255, 255, 255))
        x += panel.width

    action = np.asarray(sample.get("action", []))
    info = [
        f"task: {sample.get('task', '')}",
        f"episode/frame: {int(np.asarray(sample['episode_index']).item())}/{int(np.asarray(sample['frame_index']).item())}",
        f"action: {action.tolist()}",
    ]
    if "genaug.mode" in sample:
        info.append(f"genaug.mode: {sample['genaug.mode']}")
        info.append(f"genaug.prompt: {sample['genaug.prompt']}")
    y = max(p.height for p in panels) + 28
    for line in info:
        draw.text((10, y), line, fill=(255, 255, 255))
        y += 18

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


if __name__ == "__main__":
    main()
