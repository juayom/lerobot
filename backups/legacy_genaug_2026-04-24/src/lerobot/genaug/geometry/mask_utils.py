from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_mask(mask_path: str | Path) -> Image.Image:
    return Image.open(mask_path).convert("L")


def invert_mask(mask: Image.Image) -> Image.Image:
    return ImageOps.invert(mask)


def resize_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    return mask.resize(size)


def infer_mask_object_types(mask: Image.Image | Path | str) -> list[str]:
    if not isinstance(mask, Image.Image):
        mask = load_mask(mask)
    mask_np = np.array(mask)
    binary = (mask_np > 0).astype(np.uint8)
    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    object_types: list[str] = []
    for cid in range(1, n_labels):
        x, y, w, h, area = stats[cid]
        if area < 300:
            continue
        aspect = h / max(w, 1)
        fill_ratio = area / max(w * h, 1)
        if aspect >= 1.25 and fill_ratio < 0.8:
            object_types.append("bottle")
        elif fill_ratio >= 0.45:
            object_types.append("box")
        else:
            object_types.append("generic")

    if not object_types:
        return ["generic"]

    deduped: list[str] = []
    for obj_type in object_types:
        if obj_type not in deduped:
            deduped.append(obj_type)
    return deduped


__all__ = ["infer_mask_object_types", "invert_mask", "load_mask", "resize_mask"]
