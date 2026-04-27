#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def split_components(mask_path: Path, output_dir: Path, min_area: int = 200) -> dict:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    binary = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    components = []
    kept = 0
    for cid in range(1, n_labels):
        x, y, w, h, area = stats[cid]
        area = int(area)
        if area < min_area:
            continue
        comp = ((labels == cid).astype(np.uint8) * 255)
        out_path = output_dir / f"component-{kept:02d}.png"
        cv2.imwrite(str(out_path), comp)
        components.append(
            {
                "component_index": kept,
                "source_label": int(cid),
                "path": str(out_path),
                "area": area,
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "centroid": [float(centroids[cid][0]), float(centroids[cid][1])],
            }
        )
        kept += 1

    summary = {
        "mask_path": str(mask_path),
        "output_dir": str(output_dir),
        "min_area": min_area,
        "components": components,
    }
    (output_dir / "components-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-area", type=int, default=200)
    args = parser.parse_args()
    summary = split_components(Path(args.mask), Path(args.output_dir), args.min_area)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
