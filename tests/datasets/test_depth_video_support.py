from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lerobot.datasets.utils import validate_feature_image_or_video
from lerobot.datasets.video_utils import decode_video_frames, encode_video_frames, get_video_info


def _write_depth_frame(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def test_depth_video_roundtrip_lossless(tmp_path: Path):
    imgs_dir = tmp_path / "images"
    frame0 = (np.arange(16, dtype=np.uint16).reshape(4, 4) * 1000)
    frame1 = frame0 + 77

    _write_depth_frame(imgs_dir / "frame-000000.png", frame0)
    _write_depth_frame(imgs_dir / "frame-000001.png", frame1)

    video_path = tmp_path / "depth_episode.mkv"
    encode_video_frames(imgs_dir=imgs_dir, video_path=video_path, fps=5)

    info = get_video_info(video_path)
    assert info["video.is_depth_map"] is True
    assert info["video.channels"] == 1
    assert info["video.codec"] == "ffv1"
    assert info["video.pix_fmt"] == "gray16le"

    frames = decode_video_frames(video_path, timestamps=[0.0, 0.2], tolerance_s=0.11, backend="pyav")

    assert frames.shape == (2, 1, 4, 4)
    assert frames.dtype == torch.float32
    assert torch.equal(frames[0, 0], torch.from_numpy(frame0).to(torch.float32))
    assert torch.equal(frames[1, 0], torch.from_numpy(frame1).to(torch.float32))


def test_validate_feature_image_or_video_accepts_single_channel_depth_shapes():
    feature_shape = (4, 5, 1)
    hw = np.zeros((4, 5), dtype=np.uint16)
    hwc = np.zeros((4, 5, 1), dtype=np.uint16)
    chw = np.zeros((1, 4, 5), dtype=np.uint16)

    assert validate_feature_image_or_video("observation.images.front_depth", feature_shape, hw) == ""
    assert validate_feature_image_or_video("observation.images.front_depth", feature_shape, hwc) == ""
    assert validate_feature_image_or_video("observation.images.front_depth", feature_shape, chw) == ""
