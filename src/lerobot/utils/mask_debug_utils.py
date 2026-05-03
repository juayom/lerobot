from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw


def to_uint8_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.dtype == bool:
        return mask.astype(np.uint8) * 255
    return (mask > 0).astype(np.uint8) * 255


def depth_preview(depth_m: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_m).astype(np.float32)
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        gray = np.zeros(depth.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(valid, [2, 98])
        hi = max(hi, lo + 1e-6)
        norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        gray = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.35) -> np.ndarray:
    rgb = np.asarray(rgb).astype(np.uint8)
    out = rgb.copy()
    mask_u8 = to_uint8_mask(mask)
    out[mask_u8 > 0] = np.array(color, dtype=np.uint8)
    return cv2.addWeighted(rgb, 1.0 - alpha, out, alpha, 0)


def draw_candidate_boxes(rgb: np.ndarray, candidates: Iterable[dict] | Iterable[object]) -> np.ndarray:
    out = np.asarray(rgb).copy().astype(np.uint8)
    for cand in candidates:
        if hasattr(cand, "bbox_xywh"):
            bbox = cand.bbox_xywh
            label = getattr(cand, "label", "?")
        else:
            bbox = cand["bbox_xywh"]
            label = cand.get("label", "?")
        x, y, w, h = map(int, bbox)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(out, str(label), (x, max(18, y + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.ndim == 2:
        Image.fromarray(arr).save(path)
    else:
        Image.fromarray(arr.astype(np.uint8)).save(path)


def save_depth_npy(path: str | Path, depth_m: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(depth_m).astype(np.float32))


def make_debug_panel(panels: list[tuple[str, np.ndarray]], cols: int = 3) -> np.ndarray:
    rendered = []
    width = height = None
    for title, panel in panels:
        arr = np.asarray(panel)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = arr.astype(np.uint8)
        width = width or arr.shape[1]
        height = height or arr.shape[0]
        img = Image.fromarray(arr)
        canvas = Image.new("RGB", (img.width, img.height + 30), color=(24, 24, 24))
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, img.height + 6), title, fill=(255, 255, 255))
        rendered.append(np.asarray(canvas))
    rows = (len(rendered) + cols - 1) // cols
    cell_h = max(r.shape[0] for r in rendered)
    cell_w = max(r.shape[1] for r in rendered)
    panel_canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for idx, img in enumerate(rendered):
        r = idx // cols
        c = idx % cols
        y = r * cell_h
        x = c * cell_w
        panel_canvas[y : y + img.shape[0], x : x + img.shape[1]] = img
    return panel_canvas
