#!/usr/bin/env python

import pytest

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.utils import infer_visual_rename_map, validate_visual_features_consistency
from lerobot.policies.xvla.configuration_xvla import XVLAConfig


def visual_feature() -> PolicyFeature:
    return PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))


def state_feature() -> PolicyFeature:
    return PolicyFeature(type=FeatureType.STATE, shape=(8,))


@pytest.mark.parametrize(
    ("config_factory", "provided_keys", "expected_map"),
    [
        (
            lambda: SmolVLAConfig(
                device="cpu",
                input_features={
                    "observation.images.base_0_rgb": visual_feature(),
                    "observation.images.left_wrist_0_rgb": visual_feature(),
                    "observation.images.right_wrist_0_rgb": visual_feature(),
                    "observation.state": state_feature(),
                },
            ),
            [
                "observation.images.top",
                "observation.images.front",
                "observation.images.wrist",
            ],
            {
                "observation.images.top": "observation.images.base_0_rgb",
                "observation.images.front": "observation.images.left_wrist_0_rgb",
                "observation.images.wrist": "observation.images.right_wrist_0_rgb",
            },
        ),
        (
            lambda: XVLAConfig(
                device="cpu",
                input_features={
                    "observation.images.image1": visual_feature(),
                    "observation.images.image2": visual_feature(),
                    "observation.state": state_feature(),
                },
            ),
            [
                "observation.images.top",
                "observation.images.front",
            ],
            {
                "observation.images.top": "observation.images.image1",
                "observation.images.front": "observation.images.image2",
            },
        ),
        (
            lambda: PI05Config(
                device="cpu",
                input_features={
                    "observation.images.base_0_rgb": visual_feature(),
                    "observation.images.left_wrist_0_rgb": visual_feature(),
                    "observation.images.right_wrist_0_rgb": visual_feature(),
                    "observation.state": state_feature(),
                },
            ),
            [
                "observation.images.top",
                "observation.images.front",
                "observation.images.wrist",
            ],
            {
                "observation.images.top": "observation.images.base_0_rgb",
                "observation.images.front": "observation.images.left_wrist_0_rgb",
                "observation.images.wrist": "observation.images.right_wrist_0_rgb",
            },
        ),
        (
            lambda: ACTConfig(
                device="cpu",
                input_features={
                    "observation.images.camera1": visual_feature(),
                    "observation.images.camera2": visual_feature(),
                    "observation.state": state_feature(),
                },
            ),
            [
                "observation.images.top",
                "observation.images.wrist",
            ],
            {
                "observation.images.top": "observation.images.camera1",
                "observation.images.wrist": "observation.images.camera2",
            },
        ),
    ],
)
def test_auto_alias_visual_keys(config_factory, provided_keys, expected_map):
    cfg = config_factory()
    provided_features = {
        key: visual_feature() for key in provided_keys
    } | {
        "observation.state": state_feature(),
    }

    rename_map = infer_visual_rename_map(cfg, provided_features)

    assert rename_map == expected_map
    validate_visual_features_consistency(cfg, provided_features, rename_map=rename_map)


def test_auto_aliasing_preserves_explicit_user_mapping():
    cfg = XVLAConfig(
        device="cpu",
        input_features={
            "observation.images.image1": visual_feature(),
            "observation.images.image2": visual_feature(),
            "observation.images.image3": visual_feature(),
            "observation.state": state_feature(),
        },
    )
    provided_features = {
        "observation.images.top": visual_feature(),
        "observation.images.front": visual_feature(),
        "observation.images.wrist": visual_feature(),
        "observation.state": state_feature(),
    }
    explicit_map = {
        "observation.images.top": "observation.images.image2",
    }

    rename_map = infer_visual_rename_map(cfg, provided_features, explicit_map)

    assert rename_map["observation.images.top"] == "observation.images.image2"
    assert rename_map["observation.images.front"] == "observation.images.image1"
    assert rename_map["observation.images.wrist"] == "observation.images.image3"
    validate_visual_features_consistency(cfg, provided_features, rename_map=rename_map)
