#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from lerobot.genaug import GenAugConfig, GenAugEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect GenAug engine configuration")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    payload = yaml.safe_load(Path(args.config).read_text())
    cfg = GenAugConfig(
        image_key=payload.get("dataset", {}).get("image_key", "observation.images.rgb"),
        depth_key=payload.get("dataset", {}).get("depth_key", "observation.images.depth"),
        action_key=payload.get("dataset", {}).get("action_key", "action"),
        task_key=payload.get("dataset", {}).get("task_key", "task"),
        dry_run=payload.get("genaug", {}).get("dry_run", True),
        num_aug_per_frame=payload.get("genaug", {}).get("num_aug_per_frame", 1),
        modes=payload.get("genaug", {}).get("modes", ["background"]),
        seed=payload.get("genaug", {}).get("seed", 42),
        use_controlnet_depth=payload.get("genaug", {}).get("use_controlnet_depth", False),
        model_id=payload.get("genaug", {}).get("model_id"),
        prompts=payload.get("prompts", {}),
    )
    engine = GenAugEngine(cfg)
    print(json.dumps({"modes": cfg.modes, "prompt_preview": engine.choose_prompt(cfg.modes[0])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
