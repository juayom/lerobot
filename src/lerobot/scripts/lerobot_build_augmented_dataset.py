#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from lerobot.datasets.augmented_dataset_builder import build_augmented_image_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a derived LeRobot dataset from an exported/augmented frame manifest.")
    parser.add_argument("--repo-id", required=True, help="Source dataset repo id")
    parser.add_argument("--root", required=True, help="Source dataset root")
    parser.add_argument("--manifest", required=True, help="Manifest parquet path")
    parser.add_argument("--output-dir", required=True, help="Output dataset root")
    parser.add_argument("--output-repo-id", required=True, help="Output dataset repo id")
    parser.add_argument("--image-path-column", default="image_path", help="Manifest image path column")
    parser.add_argument("--camera-key-column", default="camera_key", help="Manifest camera key column")
    parser.add_argument("--global-index-column", default="global_index", help="Manifest global index column")
    parser.add_argument("--overwrite", action="store_true", help="Delete output directory first if it exists")
    return parser


def main() -> None:
    init_logging()
    logging.getLogger().setLevel(logging.INFO)
    args = build_parser().parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    summary = build_augmented_image_dataset(
        source_dataset=dataset,
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir),
        repo_id=args.output_repo_id,
        image_path_column=args.image_path_column,
        camera_key_column=args.camera_key_column,
        global_index_column=args.global_index_column,
        overwrite=args.overwrite,
    )

    print(json.dumps({
        "output_dir": str(summary.output_dir),
        "repo_id": summary.repo_id,
        "episodes": summary.episodes,
        "frames": summary.frames,
        "camera_keys": summary.camera_keys,
        "manifest_path": str(summary.manifest_path),
        "output_format": summary.output_format,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
