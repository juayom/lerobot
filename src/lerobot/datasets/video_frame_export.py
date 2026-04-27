#!/usr/bin/env python

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

LOGGER = logging.getLogger(__name__)


def _to_pil_image(image: Image.Image | torch.Tensor) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim != 3:
            raise ValueError(f"Expected CHW image tensor, got shape {tuple(tensor.shape)}")
        if tensor.shape[0] == 1:
            single = tensor[0]
            if torch.is_floating_point(single):
                if float(single.max()) <= 1.0:
                    array = (single.clamp(0, 1) * 255).to(torch.uint8).numpy()
                    return Image.fromarray(array, mode="L")
                array = single.clamp(min=0).round().to(torch.int32).numpy()
                return Image.fromarray(array, mode="I")
            array = single.numpy()
            return Image.fromarray(array)
        if tensor.shape[0] == 3:
            tensor = tensor.clamp(0, 1)
            array = (tensor * 255).to(torch.uint8).permute(1, 2, 0).numpy()
            return Image.fromarray(array)
        raise ValueError(f"Unexpected tensor channel dimension for image export: {tuple(tensor.shape)}")
    raise TypeError(f"Unsupported image type for export: {type(image)}")


@dataclass
class ExportedFrameDatasetSummary:
    dataset_root: Path
    output_dir: Path
    manifest_path: Path
    episodes_exported: int
    frames_exported: int
    camera_keys: list[str]


def export_dataset_video_frames(
    dataset: LeRobotDataset,
    output_dir: str | Path,
    episode_indices: list[int] | None = None,
    camera_keys: list[str] | None = None,
    image_format: str = "png",
    overwrite: bool = False,
) -> ExportedFrameDatasetSummary:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.parquet"

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if episode_indices is None:
        episode_indices = list(range(dataset.meta.total_episodes))
    if camera_keys is None:
        camera_keys = list(dataset.meta.video_keys)

    invalid_keys = [key for key in camera_keys if key not in dataset.meta.video_keys]
    if invalid_keys:
        raise ValueError(f"Requested camera keys are not video keys in dataset: {invalid_keys}")

    rows: list[dict] = []
    frames_exported = 0

    for ep_idx in episode_indices:
        episode_meta = dataset.meta.episodes[ep_idx]
        from_idx = int(episode_meta["dataset_from_index"])
        to_idx = int(episode_meta["dataset_to_index"])
        length = int(episode_meta["length"])
        tasks = list(episode_meta["tasks"])

        LOGGER.info("Exporting episode %s (%s frames)", ep_idx, length)

        for abs_idx in range(from_idx, to_idx):
            item = dataset[abs_idx]
            frame_index = int(item["frame_index"])
            timestamp = float(item["timestamp"])
            global_index = int(item["index"])
            task_index = int(item["task_index"])

            for camera_key in camera_keys:
                image = _to_pil_image(item[camera_key])

                camera_dir = output_dir / camera_key / f"episode-{ep_idx:06d}"
                camera_dir.mkdir(parents=True, exist_ok=True)
                image_path = camera_dir / f"frame-{frame_index:06d}.{image_format.lower()}"
                save_format = "JPEG" if image_format.lower() in {"jpg", "jpeg"} else image_format.upper()
                image.save(image_path, format=save_format)

                rows.append(
                    {
                        "episode_index": ep_idx,
                        "frame_index": frame_index,
                        "timestamp": timestamp,
                        "global_index": global_index,
                        "task_index": task_index,
                        "tasks": tasks,
                        "camera_key": camera_key,
                        "image_path": str(image_path),
                        "image_format": image_format.lower(),
                        "image_width": image.width,
                        "image_height": image.height,
                        "source_data_chunk_index": int(episode_meta["data/chunk_index"]),
                        "source_data_file_index": int(episode_meta["data/file_index"]),
                        "source_dataset_from_index": int(episode_meta["dataset_from_index"]),
                        "source_dataset_to_index": int(episode_meta["dataset_to_index"]),
                        "source_video_chunk_index": int(episode_meta[f"videos/{camera_key}/chunk_index"]),
                        "source_video_file_index": int(episode_meta[f"videos/{camera_key}/file_index"]),
                        "source_video_from_timestamp": float(episode_meta[f"videos/{camera_key}/from_timestamp"]),
                        "source_video_to_timestamp": float(episode_meta[f"videos/{camera_key}/to_timestamp"]),
                    }
                )
                frames_exported += 1

                if frames_exported % 250 == 0:
                    LOGGER.info("Exported %s frame-images so far...", frames_exported)

    pd.DataFrame(rows).to_parquet(manifest_path, index=False)
    LOGGER.info("Export finished. frames=%s manifest=%s", frames_exported, manifest_path)

    return ExportedFrameDatasetSummary(
        dataset_root=dataset.root,
        output_dir=output_dir,
        manifest_path=manifest_path,
        episodes_exported=len(episode_indices),
        frames_exported=frames_exported,
        camera_keys=camera_keys,
    )
