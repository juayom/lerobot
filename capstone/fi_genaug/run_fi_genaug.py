#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from depth_guided_editor import edit_one


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-aligned FI GenAug edit")
    parser.add_argument("--image", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", default="in_category", choices=["in_category", "cross_category", "background"])
    parser.add_argument("--material", default="metal")
    parser.add_argument("--rendered-control", default=None)
    parser.add_argument("--target-category", default=None)
    parser.add_argument("--background-style", default=None)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=9.0)
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    summary = edit_one(
        image_path=Path(args.image),
        depth_path=Path(args.depth),
        mask_path=Path(args.mask),
        output_path=Path(args.output),
        mode=args.mode,
        material=args.material,
        rendered_control_path=Path(args.rendered_control) if args.rendered_control else None,
        target_category=args.target_category,
        background_style=args.background_style,
        steps=args.steps,
        guidance=args.guidance,
        control_scale=args.control_scale,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
