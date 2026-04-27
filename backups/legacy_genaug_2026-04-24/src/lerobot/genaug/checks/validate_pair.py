from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PairValidationSummary:
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
    expected_camera_keys: list[str]
    observed_camera_keys: list[str]


def validate_action_observation_pairs(
    manifest_df,
    *,
    global_index_column: str = "global_index",
    camera_key_column: str = "camera_key",
    image_path_column: str = "image_path",
    episode_index_column: str = "episode_index",
    frame_index_column: str = "frame_index",
    timestamp_column: str = "timestamp",
    expected_camera_keys: list[str] | None = None,
) -> PairValidationSummary:
    required = {
        global_index_column,
        camera_key_column,
        image_path_column,
        episode_index_column,
        frame_index_column,
        timestamp_column,
    }
    missing_cols = required - set(manifest_df.columns)
    if missing_cols:
        raise ValueError(f"Manifest missing required columns: {sorted(missing_cols)}")

    duplicated_pairs = int(manifest_df.duplicated(subset=[global_index_column, camera_key_column]).sum())
    missing_required_values = int(manifest_df[list(required)].isna().sum().sum())

    episode_counts = manifest_df.groupby(global_index_column)[episode_index_column].nunique()
    global_indices_with_multiple_episodes = int((episode_counts > 1).sum())

    frame_index_counts = manifest_df.groupby(global_index_column)[frame_index_column].nunique()
    global_indices_with_inconsistent_frame_index = int((frame_index_counts > 1).sum())

    timestamp_counts = manifest_df.groupby(global_index_column)[timestamp_column].nunique()
    global_indices_with_inconsistent_timestamp = int((timestamp_counts > 1).sum())

    observed_camera_keys = sorted(manifest_df[camera_key_column].dropna().unique().tolist())
    expected_camera_keys = sorted(expected_camera_keys) if expected_camera_keys is not None else observed_camera_keys
    expected_camera_set = set(expected_camera_keys)

    per_frame_camera_sets = manifest_df.groupby(global_index_column)[camera_key_column].agg(
        lambda values: set(values.dropna().tolist())
    )
    frames_with_missing_cameras = int(sum(1 for cameras in per_frame_camera_sets if cameras < expected_camera_set))
    frames_with_extra_cameras = int(sum(1 for cameras in per_frame_camera_sets if not cameras <= expected_camera_set))

    partial_episodes = 0
    if {"source_dataset_from_index", "source_dataset_to_index"}.issubset(set(manifest_df.columns)):
        episode_ranges = (
            manifest_df.groupby(episode_index_column)[["source_dataset_from_index", "source_dataset_to_index"]]
            .first()
            .reset_index()
        )
        for _, row in episode_ranges.iterrows():
            episode_index = row[episode_index_column]
            expected_indices = set(range(int(row["source_dataset_from_index"]), int(row["source_dataset_to_index"])))
            observed_indices = set(
                manifest_df.loc[manifest_df[episode_index_column] == episode_index, global_index_column]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
            if observed_indices != expected_indices:
                partial_episodes += 1

    return PairValidationSummary(
        total_rows=len(manifest_df),
        unique_global_indices=int(manifest_df[global_index_column].nunique()),
        duplicated_pairs=duplicated_pairs,
        missing_required_values=missing_required_values,
        global_indices_with_multiple_episodes=global_indices_with_multiple_episodes,
        global_indices_with_inconsistent_frame_index=global_indices_with_inconsistent_frame_index,
        global_indices_with_inconsistent_timestamp=global_indices_with_inconsistent_timestamp,
        frames_with_missing_cameras=frames_with_missing_cameras,
        frames_with_extra_cameras=frames_with_extra_cameras,
        partial_episodes=partial_episodes,
        expected_camera_keys=expected_camera_keys,
        observed_camera_keys=observed_camera_keys,
    )
