#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_frame_export import export_dataset_video_frames
from lerobot.utils.utils import init_logging


def _parse_episode_indices(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_camera_keys(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export decoded frames from a LeRobot video dataset.")
    parser.add_argument("--repo-id", required=True, help="Dataset repo id")
    parser.add_argument("--root", default=None, help="Local dataset root")
    parser.add_argument("--output-dir", required=True, help="Directory to write exported frames and manifest")
    parser.add_argument("--episode-indices", default=None, help="Comma-separated episode indices. Omit to export all episodes.")
    parser.add_argument("--camera-keys", default=None, help="Comma-separated camera keys. Omit to export all video keys.")
    parser.add_argument("--image-format", default="png", choices=["png", "jpg", "jpeg"], help="Output image format")
    parser.add_argument("--overwrite", action="store_true", help="Delete output-dir first if it exists")
    return parser


def main() -> None:
    init_logging()
    logging.getLogger().setLevel(logging.INFO)
    parser = build_parser()
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    summary = export_dataset_video_frames(
        dataset=dataset,
        output_dir=Path(args.output_dir),
        episode_indices=_parse_episode_indices(args.episode_indices),
        camera_keys=_parse_camera_keys(args.camera_keys),
        image_format=args.image_format,
        overwrite=args.overwrite,
    )

    print(json.dumps({
        "dataset_root": str(summary.dataset_root),
        "output_dir": str(summary.output_dir),
        "manifest_path": str(summary.manifest_path),
        "episodes_exported": summary.episodes_exported,
        "frames_exported": summary.frames_exported,
        "camera_keys": summary.camera_keys,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
