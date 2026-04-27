#!/usr/bin/env python

import numpy as np
import torch
from PIL import Image

from lerobot.datasets.utils import create_empty_dataset_info
from lerobot.datasets.video_utils import decode_video_frames, encode_video_frames, get_video_info


def test_create_empty_dataset_info_uses_mkv_for_depth_video_features():
    features = {
        "observation.images.front": {"dtype": "video", "shape": (16, 16, 3), "names": ["height", "width", "channels"]},
        "observation.images.front_depth": {
            "dtype": "video",
            "shape": (16, 16, 1),
            "names": ["height", "width", "channels"],
        },
    }

    info = create_empty_dataset_info("v3.0", fps=5, features=features, use_videos=True)
    assert info["video_path"].endswith(".mkv")


def test_depth_video_round_trip_preserves_uint16_values(tmp_path):
    imgs_dir = tmp_path / "depth_frames"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    original_frames = []
    for i in range(3):
        frame = np.full((8, 8), 1000 + i * 17, dtype=np.uint16)
        original_frames.append(frame)
        Image.fromarray(frame).save(imgs_dir / f"frame-{i:06d}.png")

    video_path = tmp_path / "depth.mkv"
    encode_video_frames(imgs_dir, video_path, fps=5, overwrite=True)

    video_info = get_video_info(video_path)
    assert video_info["video.is_depth_map"] is True
    assert video_info["video.channels"] == 1
    assert video_info["video.pix_fmt"] == "gray16le"

    decoded = decode_video_frames(video_path, [0.0, 0.2, 0.4], tolerance_s=0.11, backend="pyav")
    assert decoded.shape == (3, 1, 8, 8)
    assert decoded.dtype == torch.float32

    for idx, original in enumerate(original_frames):
        np.testing.assert_array_equal(decoded[idx, 0].numpy().astype(np.uint16), original)
