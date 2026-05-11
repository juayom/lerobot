#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from lerobot.genaug.models.inpaint_sd import apply_genaug_to_mirrored_tree
from lerobot.genaug.models.prompt_builder import ENVIRONMENTS, MATERIALS, VALID_MODES
from lerobot.utils.utils import init_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply GenAug to a mirrored LeRobot frame tree.")
    parser.add_argument("--input-root", required=True, help="Root of exported LeRobot frames or a camera subtree")
    parser.add_argument("--output-root", required=True, help="Root where augmented frames will be written")
    parser.add_argument("--mask", required=True, help="Path to grayscale mask image")
    parser.add_argument("--mode", default="material_only", choices=sorted(VALID_MODES))
    parser.add_argument("--material", default=None, choices=MATERIALS + [None])
    parser.add_argument("--environment", default=None, choices=ENVIRONMENTS + [None])
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    init_logging()
    logging.getLogger().setLevel(logging.INFO)
    parser = build_parser()
    args = parser.parse_args()

    summary = apply_genaug_to_mirrored_tree(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        mask_path=Path(args.mask),
        mode=args.mode,
        material=args.material,
        environment=args.environment,
        seed_base=args.seed_base,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
