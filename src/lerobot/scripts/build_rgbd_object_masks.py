#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.genaug_rgbd_masks import build_rgbd_object_masks
from lerobot.utils.mask_debug_utils import (
    depth_preview,
    draw_candidate_boxes,
    make_debug_panel,
    overlay_mask,
    save_depth_npy,
    save_image,
)

LOGGER = logging.getLogger(__name__)


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _load_depth(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    arr = np.asarray(Image.open(path))
    return arr


def _load_valid_mask(path: Path | None, depth: np.ndarray) -> np.ndarray | None:
    if path is None:
        return None
    return np.asarray(Image.open(path))


def build_and_save(rgb: np.ndarray, depth: np.ndarray, valid_mask: np.ndarray | None, out_dir: Path, frame_index: int) -> dict:
    result = build_rgbd_object_masks(rgb, depth, valid_mask, frame_index=frame_index)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image(out_dir / "rgb.png", result.rgb)
    save_depth_npy(out_dir / "depth.npy", result.depth_m)
    save_image(out_dir / "valid_depth_mask.png", result.valid_depth_mask)
    save_image(out_dir / "raw_foreground_mask.png", result.raw_foreground_mask)
    save_image(out_dir / "cleaned_foreground_mask.png", result.cleaned_foreground_mask)
    save_image(out_dir / "connected_components.png", result.connected_components_rgb)
    save_image(out_dir / "bottle_mask.png", result.bottle_mask)
    save_image(out_dir / "box_mask.png", result.box_mask)
    save_image(out_dir / "combined_foreground_mask.png", result.combined_foreground_mask)
    save_image(out_dir / "background_edit_mask.png", result.background_edit_mask)
    save_image(out_dir / "object_edit_mask_bottle.png", result.object_edit_mask_bottle)
    save_image(out_dir / "object_edit_mask_box.png", result.object_edit_mask_box)
    save_image(out_dir / "overlay_instances.png", draw_candidate_boxes(result.rgb, result.candidate_list))
    save_image(out_dir / "overlay_background_mask.png", overlay_mask(result.rgb, result.background_edit_mask, (0, 255, 255)))
    save_image(out_dir / "overlay_valid_mask.png", result.extra_masks["overlay_valid_mask"])
    save_image(out_dir / "overlay_threshold_candidates.png", result.extra_masks["overlay_threshold_candidates"])
    save_image(out_dir / "overlay_raw_foreground.png", result.extra_masks["overlay_raw_foreground"])

    debug_panel = make_debug_panel(
        [
            ("original rgb", result.rgb),
            ("normalized depth preview", depth_preview(result.depth_m)),
            ("valid depth mask", result.valid_depth_mask),
            ("raw foreground", result.raw_foreground_mask),
            ("cleaned foreground", result.cleaned_foreground_mask),
            ("connected components colored", result.connected_components_rgb),
            ("bottle mask overlay", overlay_mask(result.rgb, result.bottle_mask, (255, 0, 255))),
            ("box mask overlay", overlay_mask(result.rgb, result.box_mask, (255, 255, 0))),
            ("background edit mask overlay", overlay_mask(result.rgb, result.background_edit_mask, (0, 255, 255))),
        ]
    )
    save_image(out_dir / "debug_panel.png", debug_panel)

    diagnostics = result.diagnostics.to_dict()
    diagnostics["candidate_list"] = [c.__dict__ for c in result.candidate_list]
    diagnostics["split_labels"] = result.split_labels
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bottle/box/object masks from RGB-D inputs.")
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--valid-mask", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frame-index", type=int, default=-1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rgb = _load_rgb(Path(args.rgb))
    depth = _load_depth(Path(args.depth))
    valid_mask = _load_valid_mask(Path(args.valid_mask), depth) if args.valid_mask else None
    try:
        diagnostics = build_and_save(rgb, depth, valid_mask, Path(args.out_dir), args.frame_index)
    except Exception as exc:
        LOGGER.exception("Failed to build RGB-D object masks: %s", exc)
        raise SystemExit("build_rgbd_object_masks failed; inspect logs and input files.") from exc
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
