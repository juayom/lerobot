#!/usr/bin/env python

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DEFAULT_DATA_PATH,
    get_hf_features_from_features,
    load_episodes,
    write_info,
    write_stats,
    write_tasks,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class AugmentedDatasetBuildSummary:
    output_dir: Path
    repo_id: str
    episodes: int
    frames: int
    camera_keys: list[str]
    manifest_path: Path
    output_format: str


def build_augmented_image_dataset(
    source_dataset: LeRobotDataset,
    manifest_path: str | Path,
    output_dir: str | Path,
    repo_id: str,
    image_path_column: str = "image_path",
    camera_key_column: str = "camera_key",
    global_index_column: str = "global_index",
    overwrite: bool = False,
) -> AugmentedDatasetBuildSummary:
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if len(source_dataset.meta.video_keys) == 0:
        raise ValueError("Source dataset must contain video keys for augmented image rebuild")

    manifest_df = pd.read_parquet(manifest_path)
    required = {global_index_column, camera_key_column, image_path_column}
    missing = required - set(manifest_df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    camera_keys = sorted(manifest_df[camera_key_column].unique().tolist())
    invalid = [key for key in camera_keys if key not in source_dataset.meta.video_keys]
    if invalid:
        raise ValueError(f"Manifest camera keys not present in source dataset video keys: {invalid}")

    new_features = {}
    for key, value in source_dataset.meta.features.items():
        if key in source_dataset.meta.video_keys and key not in camera_keys:
            continue
        new_features[key] = dict(value)
        if key in camera_keys:
            new_features[key]["dtype"] = "image"
            new_features[key].pop("info", None)

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=source_dataset.meta.fps,
        features=new_features,
        robot_type=source_dataset.meta.robot_type,
        root=output_dir,
        use_videos=False,
        chunks_size=source_dataset.meta.chunks_size,
        data_files_size_in_mb=source_dataset.meta.data_files_size_in_mb,
        video_files_size_in_mb=source_dataset.meta.video_files_size_in_mb,
    )

    if source_dataset.meta.tasks is not None:
        write_tasks(source_dataset.meta.tasks, output_dir)
        new_meta.tasks = source_dataset.meta.tasks.copy()

    manifest_wide = (
        manifest_df[[global_index_column, camera_key_column, image_path_column]]
        .drop_duplicates(subset=[global_index_column, camera_key_column], keep="last")
        .pivot(index=global_index_column, columns=camera_key_column, values=image_path_column)
        .reset_index()
        .rename_axis(columns=None)
    )

    source_data_dir = source_dataset.root / "data"
    parquet_files = sorted(source_data_dir.glob("*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {source_data_dir}")

    total_frames = 0
    for src_path in parquet_files:
        df = pd.read_parquet(src_path).reset_index(drop=True)
        df = df.merge(manifest_wide, left_on="index", right_on=global_index_column, how="inner")
        if df.empty:
            continue
        df = df.drop(columns=[global_index_column], errors="ignore")

        for key in camera_keys:
            if key not in df.columns:
                raise ValueError(f"Missing merged image column for camera key: {key}")
            if df[key].isna().any():
                bad = int(df[key].isna().sum())
                raise ValueError(f"Merged data contains {bad} missing image paths for camera key {key}")

        rel = src_path.relative_to(source_dataset.root)
        chunk_dir = rel.parts[1]
        file_name = rel.parts[2]
        chunk_idx = int(chunk_dir.split("-")[1])
        file_idx = int(file_name.split("-")[1].split(".")[0])
        dst_path = output_dir / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        import datasets

        features = get_hf_features_from_features(new_features)
        ds = datasets.Dataset.from_dict(df.to_dict(orient="list"), features=features, split="train")
        ds.to_parquet(dst_path)
        total_frames += len(df)

    source_episodes_df = pd.read_parquet(
        source_dataset.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).copy()
    available_indices = set(manifest_df[global_index_column].unique().tolist())
    keep_rows = []
    for _, row in source_episodes_df.iterrows():
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        overlap = [idx for idx in available_indices if start <= idx < end]
        if overlap:
            keep_rows.append(row)

    episodes_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    for _, row in episodes_df.iterrows():
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        length = int(row["length"])
        present = sum(1 for idx in available_indices if start <= idx < end)
        if present != length:
            raise ValueError(
                "Partial episode manifests are not supported yet. "
                f"Episode {int(row['episode_index'])}: expected {length} frames, got {present}."
            )

    drop_cols = [col for col in episodes_df.columns if col.startswith("videos/")]
    episodes_df = episodes_df.drop(columns=drop_cols, errors="ignore")
    episodes_path = output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    episodes_df.to_parquet(episodes_path, index=False)
    new_meta.episodes = load_episodes(output_dir)

    new_meta.info.update(
        {
            "total_episodes": len(episodes_df),
            "total_frames": total_frames,
            "total_tasks": source_dataset.meta.total_tasks,
            "splits": {"train": f"0:{len(episodes_df)}"},
            "video_path": None,
        }
    )
    write_info(new_meta.info, output_dir)

    if source_dataset.meta.stats is not None:
        copied_stats = {
            key: value for key, value in source_dataset.meta.stats.items() if key not in camera_keys
        }
        if copied_stats:
            write_stats(copied_stats, output_dir)

    LOGGER.info(
        "Built augmented image dataset at %s with %s episodes, %s frames, camera keys=%s",
        output_dir,
        len(episodes_df),
        total_frames,
        camera_keys,
    )

    return AugmentedDatasetBuildSummary(
        output_dir=output_dir,
        repo_id=repo_id,
        episodes=len(episodes_df),
        frames=total_frames,
        camera_keys=camera_keys,
        manifest_path=manifest_path,
        output_format="image",
    )
