#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.utils.mask_debug_utils import make_debug_panel, overlay_mask, save_image


def _load(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    return np.asarray(Image.open(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render preview overlays for RGB-D GenAug masks.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--mode", choices=["bottle", "box", "background"], default="bottle")
    args = parser.parse_args()

    base = Path(args.input_dir)
    rgb = _load(base / "rgb.png")
    bottle = _load(base / "bottle_mask.png")
    box = _load(base / "box_mask.png")
    background = _load(base / "background_edit_mask.png")
    raw_fg = _load(base / "raw_foreground_mask.png")
    cleaned_fg = _load(base / "cleaned_foreground_mask.png")
    components = _load(base / "connected_components.png")

    if args.mode == "bottle":
        primary = overlay_mask(rgb, bottle, (255, 0, 255))
    elif args.mode == "box":
        primary = overlay_mask(rgb, box, (255, 255, 0))
    else:
        primary = overlay_mask(rgb, background, (0, 255, 255))

    panel = make_debug_panel(
        [
            ("primary preview", primary),
            ("raw foreground", raw_fg),
            ("cleaned foreground", cleaned_fg),
            ("components", components),
            ("bottle overlay", overlay_mask(rgb, bottle, (255, 0, 255))),
            ("box overlay", overlay_mask(rgb, box, (255, 255, 0))),
            ("background overlay", overlay_mask(rgb, background, (0, 255, 255))),
        ],
        cols=3,
    )
    out = base / f"preview_{args.mode}.png"
    save_image(out, panel)
    print(str(out))


if __name__ == "__main__":
    main()
