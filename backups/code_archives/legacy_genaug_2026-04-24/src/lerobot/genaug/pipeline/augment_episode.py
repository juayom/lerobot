from __future__ import annotations

from pathlib import Path

from lerobot.genaug.models.inpaint_sd import apply_genaug_to_mirrored_tree


def augment_single_episode_tree(
    input_root: str | Path,
    output_root: str | Path,
    mask_path: str | Path,
    **kwargs,
) -> dict:
    return apply_genaug_to_mirrored_tree(
        input_root=Path(input_root),
        output_root=Path(output_root),
        mask_path=Path(mask_path),
        **kwargs,
    )
