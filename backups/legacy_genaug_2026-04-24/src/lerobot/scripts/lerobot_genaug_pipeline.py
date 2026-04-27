#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from lerobot.datasets.genaug_pipeline import run_genaug_dataset_pipeline
from lerobot.genaug.checks.alignment import AlignmentValidationPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging


def _parse_int_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_str_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the offline LeRobot -> frame export -> augmented manifest -> rebuild pipeline.")
    parser.add_argument("--repo-id", required=True, help="Source dataset repo id")
    parser.add_argument("--root", required=True, help="Source dataset root")
    parser.add_argument("--work-dir", required=True, help="Working directory for export/rebuild artifacts")
    parser.add_argument("--output-repo-id-base", required=True, help="Base repo id prefix for derived datasets")
    parser.add_argument("--episode-indices", default=None, help="Comma-separated episode indices")
    parser.add_argument("--camera-keys", default=None, help="Comma-separated camera keys")
    parser.add_argument("--export-image-format", default="png", choices=["png", "jpg", "jpeg"])
    parser.add_argument("--augmented-root", default=None, help="If provided, rebuild uses mirrored image paths rooted here instead of exported_frames/")
    parser.add_argument("--augmented-suffix", default=None, help="Optional suffix inserted before image extension when building effective manifest.")
    parser.add_argument("--skip-video", action="store_true", help="Stop after rebuilding the image dataset and do not convert back to video format.")
    parser.add_argument("--overwrite", action="store_true", help="Delete work-dir first if it exists")
    parser.add_argument("--vcodec", default="libsvtav1")
    parser.add_argument("--pix-fmt", default="yuv420p")
    parser.add_argument("--g", type=int, default=2)
    parser.add_argument("--crf", type=int, default=30)
    parser.add_argument("--fast-decode", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-episodes-per-batch", type=int, default=None)
    parser.add_argument("--max-frames-per-batch", type=int, default=None)
    parser.add_argument("--skip-alignment-check", action="store_true", help="Skip manifest/action-observation alignment validation before rebuild")
    parser.add_argument("--allow-partial-episodes", action="store_true", help="Do not fail when manifest covers only part of an episode")
    parser.add_argument("--allow-missing-cameras", action="store_true", help="Do not fail when some frames miss expected camera keys")
    parser.add_argument("--allow-extra-cameras", action="store_true", help="Do not fail when some frames include unexpected camera keys")
    return parser


def main() -> None:
    init_logging()
    logging.getLogger().setLevel(logging.INFO)
    args = build_parser().parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    alignment_policy = AlignmentValidationPolicy(
        fail_on_partial_episodes=not args.allow_partial_episodes,
        fail_on_missing_cameras=not args.allow_missing_cameras,
        fail_on_extra_cameras=not args.allow_extra_cameras,
    )
    summary = run_genaug_dataset_pipeline(
        source_dataset=dataset,
        work_dir=Path(args.work_dir),
        output_repo_id_base=args.output_repo_id_base,
        episode_indices=_parse_int_list(args.episode_indices),
        camera_keys=_parse_str_list(args.camera_keys),
        export_image_format=args.export_image_format,
        augmented_root=Path(args.augmented_root) if args.augmented_root else None,
        augmented_suffix=args.augmented_suffix,
        rebuild_to_video=not args.skip_video,
        overwrite=args.overwrite,
        vcodec=args.vcodec,
        pix_fmt=args.pix_fmt,
        g=args.g,
        crf=args.crf,
        fast_decode=args.fast_decode,
        num_workers=args.num_workers,
        max_episodes_per_batch=args.max_episodes_per_batch,
        max_frames_per_batch=args.max_frames_per_batch,
        validate_alignment=not args.skip_alignment_check,
        alignment_policy=alignment_policy,
    )

    print(json.dumps({
        "source_root": str(summary.source_root),
        "work_dir": str(summary.work_dir),
        "export_dir": str(summary.export_dir),
        "manifest_path": str(summary.manifest_path),
        "effective_manifest_path": str(summary.effective_manifest_path),
        "image_dataset_dir": str(summary.image_dataset_dir),
        "image_dataset_repo_id": summary.image_dataset_repo_id,
        "video_dataset_dir": str(summary.video_dataset_dir) if summary.video_dataset_dir else None,
        "video_dataset_repo_id": summary.video_dataset_repo_id,
        "episodes_exported": summary.episodes_exported,
        "frames_exported": summary.frames_exported,
        "camera_keys": summary.camera_keys,
        "alignment_summary": summary.alignment_summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
