from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.genaug.checks.alignment import validate_before_after
from lerobot.genaug.geometry.depth_utils import sanitize_depth
from lerobot.genaug.geometry.mask_utils import make_background_mask
from lerobot.genaug.layout import from_hwc_image, to_hwc_depth, to_hwc_image

LOGGER = logging.getLogger(__name__)


@dataclass
class GenAugConfig:
    image_key: str = "observation.images.rgb"
    depth_key: str = "observation.images.depth"
    action_key: str = "action"
    task_key: str = "task"
    dry_run: bool = True
    num_aug_per_frame: int = 1
    modes: list[str] = field(default_factory=lambda: ["background", "object_texture"])
    seed: int = 42
    use_controlnet_depth: bool = False
    model_id: str | None = None
    prompts: dict[str, list[str]] = field(
        default_factory=lambda: {
            "background": [
                "a tabletop scene in a kitchen",
                "a tabletop scene in a study room",
                "a tabletop scene on a wooden desk",
            ],
            "object_texture": [
                "a red plastic object",
                "a blue ceramic object",
                "a metallic object",
            ],
        }
    )


class GenAugEngine:
    def __init__(self, config: GenAugConfig):
        self.config = config
        self._model_warning_emitted = False

    def choose_prompt(self, mode: str) -> str:
        prompts = self.config.prompts.get(mode) or [f"genaug mode: {mode}"]
        return prompts[0]

    def augment_image_with_depth(self, image, depth, mask, prompt, seed):
        image_hwc, image_layout = to_hwc_image(image)
        depth_hwc, _depth_layout = to_hwc_depth(depth)
        depth_np = sanitize_depth(depth_hwc)
        _ = depth_np, mask, prompt, seed

        if self.config.dry_run:
            return from_hwc_image(image_hwc, image_layout)

        if self.config.model_id is None:
            if not self._model_warning_emitted:
                LOGGER.warning(
                    "GenAug model_id is not configured. Falling back to dry-run passthrough augmentation."
                )
                self._model_warning_emitted = True
            return from_hwc_image(image_hwc, image_layout)

        if not self._model_warning_emitted:
            LOGGER.warning(
                "Model execution is intentionally optional. Automatic weight download is disabled; returning original image."
            )
            self._model_warning_emitted = True
        return from_hwc_image(image_hwc, image_layout)

    def augment_sample(self, sample: dict, aug_idx: int) -> dict:
        mode = self.config.modes[aug_idx % len(self.config.modes)]
        prompt = self.choose_prompt(mode)
        seed = self.config.seed + aug_idx

        augmented = copy.deepcopy(sample)
        image = np.asarray(sample[self.config.image_key])
        depth = np.asarray(sample[self.config.depth_key])
        image_hwc, _image_layout = to_hwc_image(image)
        depth_hwc, depth_layout = to_hwc_depth(depth)
        mask = make_background_mask(image_hwc)
        augmented[self.config.image_key] = self.augment_image_with_depth(image, depth, mask, prompt, seed)
        augmented[self.config.depth_key] = np.asarray(sample[self.config.depth_key]).copy()

        augmented["genaug.source_episode_index"] = np.array([
            int(np.asarray(sample["episode_index"]).item())
        ], dtype=np.int64)
        augmented["genaug.source_frame_index"] = np.array([
            int(np.asarray(sample["frame_index"]).item())
        ], dtype=np.int64)
        augmented["genaug.aug_index"] = np.array([int(aug_idx)], dtype=np.int64)
        augmented["genaug.mode"] = mode
        augmented["genaug.prompt"] = prompt
        augmented["genaug.seed"] = np.array([int(seed)], dtype=np.int64)

        self.validate_augmented_sample(sample, augmented)
        return augmented

    def validate_augmented_sample(self, original_sample, augmented_sample):
        validate_before_after(
            original_sample,
            augmented_sample,
            image_key=self.config.image_key,
            depth_key=self.config.depth_key,
            action_key=self.config.action_key,
            task_key=self.config.task_key,
        )
