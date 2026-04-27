from __future__ import annotations

import logging
from typing import Any

import numpy as np

from lerobot.genaug.layout import to_hwc_depth, to_hwc_image

LOGGER = logging.getLogger(__name__)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def assert_action_equal(a, b):
    a_np = _as_numpy(a)
    b_np = _as_numpy(b)
    if np.isnan(a_np).any() or np.isnan(b_np).any():
        raise ValueError("NaN action values are not allowed")
    if a_np.shape != b_np.shape or not np.array_equal(a_np, b_np):
        raise AssertionError("Action mismatch after augmentation")


def assert_task_equal(a, b):
    if a != b:
        raise AssertionError(f"Task mismatch after augmentation: {a!r} != {b!r}")


def assert_rgbd_shape_valid(image, depth):
    image_np, _image_layout = to_hwc_image(_as_numpy(image))
    depth_np, _depth_layout = to_hwc_depth(_as_numpy(depth))
    if image_np.ndim != 3 or image_np.shape[-1] != 3:
        raise AssertionError(f"Invalid RGB shape: {image_np.shape}")
    if depth_np.ndim != 3 or depth_np.shape[-1] != 1:
        raise AssertionError(f"Invalid depth shape: {depth_np.shape}")
    if image_np.shape[:2] != depth_np.shape[:2]:
        raise AssertionError(f"RGB/depth size mismatch: {image_np.shape[:2]} != {depth_np.shape[:2]}")
    if np.all(depth_np == 0):
        LOGGER.warning("Depth frame is entirely zero.")


def assert_frame_metadata_valid(sample):
    required = ["episode_index", "frame_index", "timestamp"]
    for key in required:
        if key not in sample:
            raise AssertionError(f"Missing required frame metadata: {key}")
    if "genaug.source_episode_index" in sample and "genaug.source_frame_index" in sample:
        return
    if "genaug.aug_index" in sample:
        raise AssertionError("Missing source frame metadata for augmented sample")


def validate_before_after(original_sample, augmented_sample, *, image_key, depth_key, action_key, task_key):
    assert_action_equal(original_sample[action_key], augmented_sample[action_key])
    assert_task_equal(original_sample[task_key], augmented_sample[task_key])
    assert_rgbd_shape_valid(original_sample[image_key], original_sample[depth_key])
    assert_rgbd_shape_valid(augmented_sample[image_key], augmented_sample[depth_key])
    assert_frame_metadata_valid(original_sample)
    assert_frame_metadata_valid(augmented_sample)
    orig_ts = float(_as_numpy(original_sample["timestamp"]).reshape(-1)[0])
    aug_ts = float(_as_numpy(augmented_sample["timestamp"]).reshape(-1)[0])
    if abs(orig_ts - aug_ts) > 1e-6:
        LOGGER.warning("Timestamp changed across augmentation: %s -> %s", orig_ts, aug_ts)
