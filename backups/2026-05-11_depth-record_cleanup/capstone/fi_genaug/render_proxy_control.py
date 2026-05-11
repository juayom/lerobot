#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TARGET_SIZE = (512, 512)


def _largest_component_bbox(mask_np: np.ndarray) -> tuple[int, int, int, int]:
    binary = (mask_np > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    best = None
    for cid in range(1, n_labels):
        x, y, w, h, area = stats[cid]
        if best is None or area > best[-1]:
            best = (x, y, w, h, area)
    if best is None:
        return (0, 0, mask_np.shape[1], mask_np.shape[0])
    x, y, w, h, _ = best
    return x, y, x + w, y + h


def _draw_bucket(draw: ImageDraw.ImageDraw, bbox: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    top = (x0 + int(0.18 * w), y0 + int(0.18 * h), x1 - int(0.18 * w), y0 + int(0.34 * h))
    body = [(x0 + int(0.08 * w), y0 + int(0.28 * h)), (x1 - int(0.08 * w), y0 + int(0.28 * h)), (x1 - int(0.20 * w), y1 - int(0.05 * h)), (x0 + int(0.20 * w), y1 - int(0.05 * h))]
    draw.ellipse(top, fill=220)
    draw.polygon(body, fill=180)
    draw.arc((x0 + int(0.18 * w), y0, x1 - int(0.18 * w), y0 + int(0.55 * h)), start=200, end=-20, fill=245, width=max(2, w // 30))


def _draw_bowl(draw: ImageDraw.ImageDraw, bbox: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    draw.ellipse((x0 + int(0.02 * w), y0 + int(0.22 * h), x1 - int(0.02 * w), y1), fill=190)
    draw.ellipse((x0, y0 + int(0.18 * h), x1, y0 + int(0.45 * h)), fill=235)


def _draw_tray(draw: ImageDraw.ImageDraw, bbox: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = bbox
    draw.rounded_rectangle((x0, y0 + (y1-y0)//4, x1, y1), radius=18, fill=190)
    draw.rounded_rectangle((x0 + 14, y0 + (y1-y0)//4 + 14, x1 - 14, y1 - 14), radius=12, fill=230)


def _draw_generic_box(draw: ImageDraw.ImageDraw, bbox: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = bbox
    draw.rectangle((x0 + 8, y0 + 8, x1 - 8, y1 - 8), fill=205)
    draw.polygon([(x0 + 8, y0 + 8), (x1 - 8, y0 + 8), (x1 - 28, y0 + 26), (x0 + 28, y0 + 26)], fill=235)


def render_proxy_control(mask_path: Path, output_path: Path, target_category: str = 'bucket', blur_radius: int = 4) -> dict:
    mask = Image.open(mask_path).convert('L').resize(TARGET_SIZE)
    mask_np = np.array(mask)
    bbox = _largest_component_bbox(mask_np)

    canvas = Image.new('L', TARGET_SIZE, 0)
    draw = ImageDraw.Draw(canvas)

    category = target_category.lower()
    if category == 'bucket':
        _draw_bucket(draw, bbox)
    elif category == 'bowl':
        _draw_bowl(draw, bbox)
    elif category == 'tray':
        _draw_tray(draw, bbox)
    else:
        _draw_generic_box(draw, bbox)

    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    # constrain generated proxy to original mask support region to preserve action semantics
    constrained = np.minimum(np.array(canvas), mask_np)
    rgb = cv2.cvtColor(constrained.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output_path)

    summary = {
        'mask_path': str(mask_path),
        'output_path': str(output_path),
        'target_category': target_category,
        'bbox': list(map(int, bbox)),
        'blur_radius': blur_radius,
    }
    (output_path.parent / f'{output_path.stem}-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Render proxy control image for paper-aligned cross-category GenAug')
    parser.add_argument('--mask', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--target-category', default='bucket')
    parser.add_argument('--blur-radius', type=int, default=4)
    args = parser.parse_args()
    summary = render_proxy_control(Path(args.mask), Path(args.output), args.target_category, args.blur_radius)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
