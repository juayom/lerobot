from __future__ import annotations

import os

TARGET_SIZE = (512, 512)
NUM_INFERENCE_STEPS = int(os.environ.get("LEROBOT_GENAUG_STEPS", "30"))
GUIDANCE_SCALE = float(os.environ.get("LEROBOT_GENAUG_GUIDANCE", "10"))
CONTROLNET_CONDITIONING_SCALE = float(os.environ.get("LEROBOT_GENAUG_CONDITION", "0.8"))

__all__ = [
    "CONTROLNET_CONDITIONING_SCALE",
    "GUIDANCE_SCALE",
    "NUM_INFERENCE_STEPS",
    "TARGET_SIZE",
]
