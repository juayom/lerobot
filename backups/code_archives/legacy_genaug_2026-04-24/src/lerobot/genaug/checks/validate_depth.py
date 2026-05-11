from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DepthValidationSummary:
    depth_feature_keys: list[str]
    has_depth_signal: bool


def validate_depth_feature_presence(dataset) -> DepthValidationSummary:
    depth_keys = [key for key, value in dataset.meta.features.items() if "depth" in key or value.get("names") == ["depth"]]
    return DepthValidationSummary(
        depth_feature_keys=sorted(depth_keys),
        has_depth_signal=bool(depth_keys),
    )
