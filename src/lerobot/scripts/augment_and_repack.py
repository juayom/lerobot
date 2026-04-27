#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.genaug import GenAugConfig, GenAugEngine
from lerobot.genaug.checks.alignment import validate_before_after
from lerobot.utils.utils import init_logging

LOGGER = logging.getLogger(__name__)


def _log_frame_visuals(frame: dict, image_key: str, depth_key: str, prefix: str) -> None:
    image = frame.get(image_key)
    depth = frame.get(depth_key)
    if image is not None and not isinstance(image, str):
        image_arr = np.asarray(image)
        LOGGER.info(
            "%s image key=%s dtype=%s shape=%s range=[%s, %s]",
            prefix,
            image_key,
            image_arr.dtype,
            image_arr.shape,
            image_arr.min() if image_arr.size else 'n/a',
            image_arr.max() if image_arr.size else 'n/a',
        )
    if depth is not None and not isinstance(depth, str):
        depth_arr = np.asarray(depth)
        LOGGER.info(
            "%s depth key=%s dtype=%s shape=%s range=[%s, %s]",
            prefix,
            depth_key,
            depth_arr.dtype,
            depth_arr.shape,
            depth_arr.min() if depth_arr.size else 'n/a',
            depth_arr.max() if depth_arr.size else 'n/a',
        )


GENAUG_META_FEATURES = {
    "genaug.source_episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "genaug.source_frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "genaug.aug_index": {"dtype": "int64", "shape": (1,), "names": None},
    "genaug.seed": {"dtype": "int64", "shape": (1,), "names": None},
    "genaug.mode": {"dtype": "string", "shape": (1,), "names": None},
    "genaug.prompt": {"dtype": "string", "shape": (1,), "names": None},
}


def _as_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _prepare_value_for_frame(key: str, value: Any):
    if isinstance(value, str):
        return value
    arr = _as_array(value).copy()
    if "depth" in key:
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return arr


def _copy_frame_from_sample(sample: dict, features: dict, task_key: str) -> dict:
    frame = {}
    episode_index = int(_as_array(sample["episode_index"]).reshape(-1)[0])
    frame_index = int(_as_array(sample["frame_index"]).reshape(-1)[0])
    defaults = {
        "genaug.source_episode_index": np.array([episode_index], dtype=np.int64),
        "genaug.source_frame_index": np.array([frame_index], dtype=np.int64),
        "genaug.aug_index": np.array([-1], dtype=np.int64),
        "genaug.seed": np.array([-1], dtype=np.int64),
        "genaug.mode": "original",
        "genaug.prompt": "original",
    }
    for key in features:
        if key == "episode_index":
            continue
        if key in ["frame_index", "timestamp", "index", "task_index"]:
            continue
        if key in sample:
            value = sample[key]
        elif key in defaults:
            value = defaults[key]
        else:
            raise KeyError(f"Feature '{key}' missing from sample and no default is defined")
        frame[key] = _prepare_value_for_frame(key, value)
    frame[task_key] = sample[task_key]
    return frame


def load_config(path: Path) -> GenAugConfig:
    payload = yaml.safe_load(path.read_text())
    dataset_cfg = payload.get("dataset", {})
    genaug_cfg = payload.get("genaug", {})
    prompts = payload.get("prompts", {})
    return GenAugConfig(
        image_key=dataset_cfg.get("image_key", "observation.images.rgb"),
        depth_key=dataset_cfg.get("depth_key", "observation.images.depth"),
        action_key=dataset_cfg.get("action_key", "action"),
        task_key=dataset_cfg.get("task_key", "task"),
        dry_run=genaug_cfg.get("dry_run", True),
        num_aug_per_frame=genaug_cfg.get("num_aug_per_frame", 1),
        modes=genaug_cfg.get("modes", ["background"]),
        seed=genaug_cfg.get("seed", 42),
        use_controlnet_depth=genaug_cfg.get("use_controlnet_depth", False),
        model_id=genaug_cfg.get("model_id"),
        prompts=prompts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply GenAug-style augmentation and repack into LeRobotDataset")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-repo-id", default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-aug", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--tolerance-s", type=float, default=5e-3)
    args = parser.parse_args()

    init_logging()
    logging.getLogger().setLevel(logging.INFO)

    source = LeRobotDataset(repo_id=args.repo_id, root=args.root, tolerance_s=args.tolerance_s)
    cfg = load_config(Path(args.config))
    if args.dry_run:
        cfg.dry_run = True
    engine = GenAugEngine(cfg)

    checked = 0
    failures = []
    output_repo_id = args.output_repo_id or f"{args.repo_id}_genaug"

    if not args.dry_run:
        out_features = deepcopy(source.meta.features)
        for key, feature in out_features.items():
            if feature.get("dtype") == "video":
                out_features[key] = dict(feature)
                out_features[key]["dtype"] = "image"
        out_features.update(GENAUG_META_FEATURES)
        output = LeRobotDataset.create(
            repo_id=output_repo_id,
            root=args.output_root,
            fps=source.fps,
            robot_type=source.meta.robot_type,
            features=out_features,
            use_videos=False,
        )
        LOGGER.info("Created output dataset at %s with use_videos=%s", args.output_root, False)
    else:
        output = None

    try:
        for idx in range(len(source)):
            sample = source[idx]
            checked += 1
            validate_before_after(
                sample,
                sample,
                image_key=cfg.image_key,
                depth_key=cfg.depth_key,
                action_key=cfg.action_key,
                task_key=cfg.task_key,
            )
            if not args.dry_run:
                original_frame = _copy_frame_from_sample(sample, output.features, cfg.task_key)
                if checked <= 2:
                    _log_frame_visuals(original_frame, cfg.image_key, cfg.depth_key, prefix="original_frame")
                output.add_frame(original_frame)
            for aug_idx in range(args.num_aug):
                augmented = engine.augment_sample(sample, aug_idx)
                if not args.dry_run:
                    augmented_frame = _copy_frame_from_sample(augmented, output.features, cfg.task_key)
                    if checked <= 2:
                        _log_frame_visuals(augmented_frame, cfg.image_key, cfg.depth_key, prefix="augmented_frame")
                    output.add_frame(augmented_frame)
            if args.dry_run and checked >= args.max_samples:
                break
        if output is not None:
            output.save_episode()
    except Exception as exc:
        failures.append(str(exc))
        raise
    finally:
        if output is not None:
            output.finalize()

    print(json.dumps({
        "checked_samples": checked,
        "dry_run": args.dry_run,
        "output_root": args.output_root if not args.dry_run else None,
        "num_aug": args.num_aug,
        "failures": failures,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
