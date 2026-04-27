from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from lerobot.genaug.layout import to_hwc_image


def _to_uint8_mask(mask) -> np.ndarray:
    mask_np = np.asarray(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    if mask_np.dtype == bool:
        mask_np = mask_np.astype(np.uint8) * 255
    else:
        mask_np = (mask_np > 0).astype(np.uint8) * 255
    return mask_np


def combine_masks(*masks):
    valid = [_to_uint8_mask(mask) for mask in masks if mask is not None]
    if not valid:
        raise ValueError("At least one mask is required")
    out = np.zeros_like(valid[0], dtype=np.uint8)
    for mask in valid:
        if mask.shape != out.shape:
            raise ValueError(f"Mask shape mismatch: {mask.shape} vs {out.shape}")
        out = np.maximum(out, mask)
    return out


def dilate_mask(mask, kernel_size=7):
    mask_np = _to_uint8_mask(mask)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(mask_np, kernel, iterations=1)


def validate_mask(mask, image_shape):
    mask_np = _to_uint8_mask(mask)
    if mask_np.shape != tuple(image_shape[:2]):
        raise ValueError(f"Mask shape {mask_np.shape} does not match image shape {image_shape[:2]}")


def make_background_mask(image, object_mask=None, target_mask=None, distractor_masks=None):
    image_np, _layout = to_hwc_image(image)
    h, w = image_np.shape[:2]
    if object_mask is None and target_mask is None and not distractor_masks:
        return np.ones((h, w), dtype=np.uint8) * 255

    keep_mask = np.zeros((h, w), dtype=np.uint8)
    for mask in [object_mask, target_mask, *(distractor_masks or [])]:
        if mask is None:
            continue
        mask_np = _to_uint8_mask(mask)
        if mask_np.shape != (h, w):
            raise ValueError(f"Mask shape {mask_np.shape} does not match image shape {(h, w)}")
        keep_mask = np.maximum(keep_mask, mask_np)
    return cv2.bitwise_not(keep_mask)


def save_mask_preview(image, mask, out_path):
    image_np, _layout = to_hwc_image(image)
    image_np = image_np.copy()
    mask_np = _to_uint8_mask(mask)
    validate_mask(mask_np, image_np.shape)
    overlay = image_np.copy()
    overlay[mask_np > 0] = np.array([255, 255, 0], dtype=np.uint8)
    preview = cv2.addWeighted(image_np, 0.7, overlay, 0.3, 0)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview).save(out_path)
