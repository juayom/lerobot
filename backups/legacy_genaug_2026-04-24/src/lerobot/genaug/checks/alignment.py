from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from lerobot.genaug.checks.validate_depth import validate_depth_feature_presence
from lerobot.genaug.checks.validate_pair import validate_action_observation_pairs
from lerobot.genaug.io.feature_schema import summarize_feature_schema


@dataclass
class AlignmentValidationSummary:
    total_rows: int
    unique_global_indices: int
    duplicated_pairs: int
    missing_required_values: int
    global_indices_with_multiple_episodes: int
    global_indices_with_inconsistent_frame_index: int
    global_indices_with_inconsistent_timestamp: int
    frames_with_missing_cameras: int
    frames_with_extra_cameras: int
    partial_episodes: int
    missing_image_paths: int
    expected_camera_keys: list[str]
    observed_camera_keys: list[str]
    dataset_camera_keys: list[str]
    depth_feature_keys: list[str]
    has_depth_signal: bool
    feature_summary: dict


@dataclass
class AlignmentValidationPolicy:
    fail_on_duplicated_pairs: bool = True
    fail_on_missing_required_values: bool = True
    fail_on_multiple_episode_mapping: bool = True
    fail_on_inconsistent_frame_index: bool = True
    fail_on_inconsistent_timestamp: bool = True
    fail_on_missing_cameras: bool = True
    fail_on_extra_cameras: bool = True
    fail_on_partial_episodes: bool = True
    fail_on_missing_image_paths: bool = True


def validate_manifest_alignment(
    dataset,
    manifest_path: str | Path,
    *,
    image_path_column: str = "image_path",
    camera_key_column: str = "camera_key",
    global_index_column: str = "global_index",
) -> AlignmentValidationSummary:
    manifest_df = pd.read_parquet(manifest_path)
    expected_camera_keys = sorted(dataset.meta.video_keys)
    pair_summary = validate_action_observation_pairs(
        manifest_df,
        global_index_column=global_index_column,
        camera_key_column=camera_key_column,
        image_path_column=image_path_column,
        expected_camera_keys=expected_camera_keys,
    )
    missing_image_paths = sum(0 if Path(p).exists() else 1 for p in manifest_df[image_path_column].tolist())
    depth_summary = validate_depth_feature_presence(dataset)
    feature_summary = summarize_feature_schema(dataset)

    return AlignmentValidationSummary(
        total_rows=pair_summary.total_rows,
        unique_global_indices=pair_summary.unique_global_indices,
        duplicated_pairs=pair_summary.duplicated_pairs,
        missing_required_values=pair_summary.missing_required_values,
        global_indices_with_multiple_episodes=pair_summary.global_indices_with_multiple_episodes,
        global_indices_with_inconsistent_frame_index=pair_summary.global_indices_with_inconsistent_frame_index,
        global_indices_with_inconsistent_timestamp=pair_summary.global_indices_with_inconsistent_timestamp,
        frames_with_missing_cameras=pair_summary.frames_with_missing_cameras,
        frames_with_extra_cameras=pair_summary.frames_with_extra_cameras,
        partial_episodes=pair_summary.partial_episodes,
        missing_image_paths=int(missing_image_paths),
        expected_camera_keys=pair_summary.expected_camera_keys,
        observed_camera_keys=pair_summary.observed_camera_keys,
        dataset_camera_keys=sorted(dataset.meta.camera_keys),
        depth_feature_keys=depth_summary.depth_feature_keys,
        has_depth_signal=depth_summary.has_depth_signal,
        feature_summary=feature_summary,
    )


def alignment_summary_to_dict(summary: AlignmentValidationSummary) -> dict:
    return asdict(summary)


def get_alignment_failures(
    summary: AlignmentValidationSummary,
    policy: AlignmentValidationPolicy | None = None,
) -> list[tuple[str, int, str]]:
    policy = policy or AlignmentValidationPolicy()
    checks: list[tuple[bool, int, str, str]] = [
        (policy.fail_on_duplicated_pairs, summary.duplicated_pairs, "duplicated_pairs", "Manifest has duplicated (global_index, camera_key) pairs"),
        (policy.fail_on_missing_required_values, summary.missing_required_values, "missing_required_values", "Manifest has missing required values"),
        (policy.fail_on_multiple_episode_mapping, summary.global_indices_with_multiple_episodes, "global_indices_with_multiple_episodes", "Manifest maps some global indices to multiple episodes"),
        (policy.fail_on_inconsistent_frame_index, summary.global_indices_with_inconsistent_frame_index, "global_indices_with_inconsistent_frame_index", "Manifest has inconsistent frame_index values within the same global index"),
        (policy.fail_on_inconsistent_timestamp, summary.global_indices_with_inconsistent_timestamp, "global_indices_with_inconsistent_timestamp", "Manifest has inconsistent timestamp values within the same global index"),
        (policy.fail_on_missing_cameras, summary.frames_with_missing_cameras, "frames_with_missing_cameras", "Manifest has frames missing expected cameras"),
        (policy.fail_on_extra_cameras, summary.frames_with_extra_cameras, "frames_with_extra_cameras", "Manifest has frames with unexpected cameras"),
        (policy.fail_on_partial_episodes, summary.partial_episodes, "partial_episodes", "Manifest has partial episode coverage"),
        (policy.fail_on_missing_image_paths, summary.missing_image_paths, "missing_image_paths", "Manifest has missing image paths"),
    ]
    return [(code, count, message) for enabled, count, code, message in checks if enabled and count > 0]


def raise_for_alignment_failures(
    summary: AlignmentValidationSummary,
    policy: AlignmentValidationPolicy | None = None,
) -> None:
    failures = get_alignment_failures(summary, policy=policy)
    if not failures:
        return
    code, count, message = failures[0]
    if code == "missing_image_paths":
        raise FileNotFoundError(f"{message}: {count}")
    raise ValueError(f"{message}: {count}")
