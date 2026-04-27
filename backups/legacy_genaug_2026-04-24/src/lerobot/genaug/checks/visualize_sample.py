from __future__ import annotations

from pathlib import Path


def describe_visualization_inputs(before_image: str | Path, after_image: str | Path) -> dict:
    return {
        "before_image": str(Path(before_image)),
        "after_image": str(Path(after_image)),
    }
