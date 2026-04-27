#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.genaug.geometry.depth_utils import sanitize_depth, validate_rgbd_pair
from lerobot.utils.utils import init_logging

LOGGER = logging.getLogger(__name__)


def build_features(action_dim: int, height: int, width: int) -> dict:
    return {
        "observation.images.rgb": {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.depth": {
            "dtype": "image",
            "shape": (height, width, 1),
            "names": ["height", "width", "channels"],
        },
        "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
        "done": {"dtype": "bool", "shape": (1,), "names": None},
    }


def load_actions(action_jsonl: Path) -> list[np.ndarray]:
    actions = []
    with action_jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            if "action" not in payload:
                raise ValueError("Each JSONL row must contain an 'action' field")
            actions.append(np.asarray(payload["action"], dtype=np.float32))
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a LeRobot-compatible RGB-D dataset from RealSense.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--serial-number-or-name", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--depth-unit", default="meter", choices=["meter", "millimeter"])
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--action-jsonl", required=True, help="External action stream; actions are never synthesized.")
    parser.add_argument("--robot-type", default="realsense_rgbd_recorder")
    args = parser.parse_args()

    if args.camera != "realsense":
        raise ValueError("Only --camera realsense is currently supported")

    init_logging()
    logging.getLogger().setLevel(logging.INFO)

    actions = load_actions(Path(args.action_jsonl))
    if len(actions) < args.num_frames:
        raise ValueError(f"Need at least {args.num_frames} actions, got {len(actions)}")

    features = build_features(len(actions[0]), args.height, args.width)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.root,
        fps=args.fps,
        robot_type=args.robot_type,
        features=features,
        use_videos=False,
    )

    camera = RealSenseCamera(
        RealSenseCameraConfig(
            serial_number_or_name=args.serial_number_or_name,
            fps=args.fps,
            width=args.width,
            height=args.height,
            use_depth=True,
        )
    )

    try:
        camera.connect()
        start = time.perf_counter()
        for frame_idx in range(args.num_frames):
            rgb, depth_raw, _capture_timestamp = camera.read_rgbd(timeout_ms=10000)
            depth = sanitize_depth(depth_raw, depth_unit=args.depth_unit)
            validate_rgbd_pair(rgb, depth)
            timestamp = frame_idx / float(args.fps)
            frame = {
                "observation.images.rgb": rgb,
                "observation.images.depth": depth,
                "action": actions[frame_idx],
                "done": np.array([frame_idx == args.num_frames - 1], dtype=bool),
                "task": args.task,
                "timestamp": timestamp,
            }
            dataset.add_frame(frame)
            elapsed = time.perf_counter() - start
            target = (frame_idx + 1) / float(args.fps)
            if target > elapsed:
                time.sleep(target - elapsed)
        dataset.save_episode()
    finally:
        dataset.finalize()
        if camera.is_connected:
            camera.disconnect()

    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "root": str(args.root),
                "frames_recorded": args.num_frames,
                "depth_unit": args.depth_unit,
                "image_key": "observation.images.rgb",
                "depth_key": "observation.images.depth",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
