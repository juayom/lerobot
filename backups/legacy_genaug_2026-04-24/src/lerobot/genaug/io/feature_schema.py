from __future__ import annotations

DEFAULT_REQUIRED_FEATURES = (
    "action",
    "task_index",
    "episode_index",
    "frame_index",
    "timestamp",
)

DEFAULT_OPTIONAL_FEATURES = (
    "task",
    "observation.depth",
)


def summarize_feature_schema(dataset) -> dict:
    features = dataset.meta.features
    return {
        "all_features": sorted(features.keys()),
        "camera_keys": sorted(dataset.meta.camera_keys),
        "video_keys": sorted(dataset.meta.video_keys),
        "image_keys": sorted(dataset.meta.image_keys),
        "required_features_present": [key for key in DEFAULT_REQUIRED_FEATURES if key in features],
        "optional_features_present": [key for key in DEFAULT_OPTIONAL_FEATURES if key in features],
    }
