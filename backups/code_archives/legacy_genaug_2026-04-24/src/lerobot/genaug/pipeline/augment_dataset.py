#!/usr/bin/env python

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lerobot.datasets.dataset_tools import convert_image_to_video_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_frame_export import export_dataset_video_frames
from lerobot.genaug.checks.alignment import (
    AlignmentValidationPolicy,
    alignment_summary_to_dict,
    raise_for_alignment_failures,
    validate_manifest_alignment,
)
from lerobot.genaug.io.save_episode import build_augmented_image_dataset

LOGGER = logging.getLogger(__name__)


@dataclass
class GenAugPipelineSummary:
    source_root: Path
    work_dir: Path
    export_dir: Path
    manifest_path: Path
    effective_manifest_path: Path
    image_dataset_dir: Path
    image_dataset_repo_id: str
    video_dataset_dir: Path | None
    video_dataset_repo_id: str | None
    episodes_exported: int
    frames_exported: int
    camera_keys: list[str]
    alignment_summary: dict | None = None


def create_effective_manifest(
    export_manifest_path: str | Path,
    output_manifest_path: str | Path,
    augmented_root: str | Path | None = None,
    augmented_suffix: str | None = None,
    image_path_column: str = "image_path",
) -> Path:
    export_manifest_path = Path(export_manifest_path)
    output_manifest_path = Path(output_manifest_path)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(export_manifest_path)

    if augmented_root is not None:
        augmented_root = Path(augmented_root)
        export_root = export_manifest_path.parent

        def rewrite_path(path_str: str) -> str:
            src = Path(path_str)
            rel = src.relative_to(export_root)
            return str(augmented_root / rel)

        df[image_path_column] = df[image_path_column].map(rewrite_path)
    elif augmented_suffix:

        def rewrite_suffix(path_str: str) -> str:
            p = Path(path_str)
            return str(p.with_name(f"{p.stem}{augmented_suffix}{p.suffix}"))

        df[image_path_column] = df[image_path_column].map(rewrite_suffix)

    missing_paths = [p for p in df[image_path_column].tolist() if not Path(p).exists()]
    if missing_paths:
        sample = missing_paths[:5]
        raise FileNotFoundError(
            f"Effective manifest points to {len(missing_paths)} missing images. Sample: {sample}"
        )

    df.to_parquet(output_manifest_path, index=False)
    return output_manifest_path


def run_genaug_dataset_pipeline(
    source_dataset: LeRobotDataset,
    work_dir: str | Path,
    output_repo_id_base: str,
    episode_indices: list[int] | None = None,
    camera_keys: list[str] | None = None,
    export_image_format: str = "png",
    augmented_root: str | Path | None = None,
    augmented_suffix: str | None = None,
    rebuild_to_video: bool = True,
    overwrite: bool = False,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int = 2,
    crf: int = 30,
    fast_decode: int = 0,
    num_workers: int = 4,
    max_episodes_per_batch: int | None = None,
    max_frames_per_batch: int | None = None,
    validate_alignment: bool = True,
    alignment_policy: AlignmentValidationPolicy | None = None,
) -> GenAugPipelineSummary:
    work_dir = Path(work_dir)
    if work_dir.exists() and overwrite:
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    export_dir = work_dir / "exported_frames"
    effective_manifest_path = work_dir / "effective_manifest.parquet"
    image_dataset_dir = work_dir / "rebuilt_image_dataset"
    video_dataset_dir = work_dir / "rebuilt_video_dataset" if rebuild_to_video else None

    image_dataset_repo_id = f"{output_repo_id_base}_image"
    video_dataset_repo_id = f"{output_repo_id_base}_video" if rebuild_to_video else None

    LOGGER.info("[1/4] Exporting decoded frames from source dataset")
    export_summary = export_dataset_video_frames(
        dataset=source_dataset,
        output_dir=export_dir,
        episode_indices=episode_indices,
        camera_keys=camera_keys,
        image_format=export_image_format,
        overwrite=True,
    )

    LOGGER.info("[2/4] Building effective manifest for rebuild")
    effective_manifest_path = create_effective_manifest(
        export_manifest_path=export_summary.manifest_path,
        output_manifest_path=effective_manifest_path,
        augmented_root=augmented_root,
        augmented_suffix=augmented_suffix,
    )

    alignment_summary = None
    if validate_alignment:
        LOGGER.info("[2.5/4] Validating action-observation alignment assumptions")
        alignment_summary_obj = validate_manifest_alignment(
            dataset=source_dataset,
            manifest_path=effective_manifest_path,
        )
        raise_for_alignment_failures(alignment_summary_obj, policy=alignment_policy)
        alignment_summary = alignment_summary_to_dict(alignment_summary_obj)

    LOGGER.info("[3/4] Rebuilding derived LeRobot image dataset")
    build_augmented_image_dataset(
        source_dataset=source_dataset,
        manifest_path=effective_manifest_path,
        output_dir=image_dataset_dir,
        repo_id=image_dataset_repo_id,
        overwrite=True,
    )

    if rebuild_to_video:
        LOGGER.info("[4/4] Converting rebuilt image dataset back to LeRobot video dataset")
        rebuilt_image_dataset = LeRobotDataset(repo_id=image_dataset_repo_id, root=image_dataset_dir)
        convert_image_to_video_dataset(
            dataset=rebuilt_image_dataset,
            output_dir=video_dataset_dir,
            repo_id=video_dataset_repo_id,
            vcodec=vcodec,
            pix_fmt=pix_fmt,
            g=g,
            crf=crf,
            fast_decode=fast_decode,
            episode_indices=None,
            num_workers=num_workers,
            max_episodes_per_batch=max_episodes_per_batch,
            max_frames_per_batch=max_frames_per_batch,
        )
    else:
        LOGGER.info("[4/4] Skipping image->video conversion by request")

    return GenAugPipelineSummary(
        source_root=source_dataset.root,
        work_dir=work_dir,
        export_dir=export_dir,
        manifest_path=export_summary.manifest_path,
        effective_manifest_path=effective_manifest_path,
        image_dataset_dir=image_dataset_dir,
        image_dataset_repo_id=image_dataset_repo_id,
        video_dataset_dir=video_dataset_dir,
        video_dataset_repo_id=video_dataset_repo_id,
        episodes_exported=export_summary.episodes_exported,
        frames_exported=export_summary.frames_exported,
        camera_keys=export_summary.camera_keys,
        alignment_summary=alignment_summary,
    )
