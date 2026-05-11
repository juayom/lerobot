#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, UniPCMultistepScheduler
from PIL import Image, ImageFilter, ImageOps

TARGET_SIZE = (512, 512)
DEFAULT_NEGATIVE = "blurry, low quality, distorted, deformed, text, watermark, logo, cartoon, anime"
VALID_MODES = {"in_category", "cross_category", "background"}
COLORS = ["red", "green", "yellow", "blue", "black", "white", "silver", "pink"]
MATERIALS = ["metal", "glass", "wood", "marble", "porcelain", "plastic"]
CATEGORY_OBJECTS = {
    "container": ["basket", "bucket", "box", "tray", "bowl"],
    "bottle": ["medicine bottle", "spray bottle", "thermos"],
    "generic": ["object"],
}
BACKGROUND_STYLES = ["kitchen", "living room", "restaurant", "office", "workshop"]


def load_depth_inpaint_pipeline():
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=dtype,
    )
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe


def normalize_depth(depth_path: Path) -> Image.Image:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(depth_path)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    valid = depth > 0
    if valid.any():
        lo = np.percentile(depth[valid], 5)
        hi = np.percentile(depth[valid], 95)
        if hi <= lo:
            hi = lo + 1.0
        depth = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    else:
        depth = np.zeros_like(depth, dtype=np.float32)
    depth = (depth * 255).astype(np.uint8)
    depth_rgb = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(depth_rgb).resize(TARGET_SIZE)


def load_rendered_control(render_path: Path | None, fallback_depth_path: Path) -> Image.Image:
    if render_path is not None and render_path.exists():
        render = Image.open(render_path).convert("RGB").resize(TARGET_SIZE)
        return render
    return normalize_depth(fallback_depth_path)


def composite_back(original: Image.Image, edited: Image.Image, mask: Image.Image) -> Image.Image:
    original = original.resize(TARGET_SIZE).convert("RGB")
    edited = edited.resize(TARGET_SIZE).convert("RGB")
    soft_mask = mask.resize(TARGET_SIZE).convert("L").filter(ImageFilter.GaussianBlur(radius=3))
    return Image.composite(edited, original, soft_mask)


def infer_category_from_mask(mask: Image.Image) -> str:
    mask_np = np.array(mask.convert("L"))
    binary = (mask_np > 0).astype(np.uint8)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    best = None
    for cid in range(1, n_labels):
        x, y, w, h, area = stats[cid]
        if best is None or area > best[-1]:
            best = (x, y, w, h, area)
    if best is None:
        return "generic"
    _, _, w, h, area = best
    aspect = h / max(w, 1)
    fill_ratio = area / max(w * h, 1)
    if aspect >= 1.2 and area < 12000:
        return "bottle"
    if fill_ratio >= 0.4:
        return "container"
    return "generic"


def build_prompt(mode: str, category: str, material: str, background_style: str | None = None, target_category: str | None = None) -> str:
    if mode == "in_category":
        color = random.choice(COLORS)
        return f"a {color} {material} {random.choice(CATEGORY_OBJECTS.get(category, CATEGORY_OBJECTS['generic']))}"
    if mode == "cross_category":
        target_category = target_category or random.choice(CATEGORY_OBJECTS.get(category, CATEGORY_OBJECTS["generic"]))
        color = random.choice(COLORS)
        return f"a {color} {material} {target_category}"
    if mode == "background":
        background_style = background_style or random.choice(BACKGROUND_STYLES)
        return f"a table in a {background_style}"
    raise ValueError(f"Unsupported mode: {mode}")


def prepare_mask(mask_path: Path, mode: str) -> Image.Image:
    mask = Image.open(mask_path).convert("L").resize(TARGET_SIZE)
    if mode == "background":
        return ImageOps.invert(mask)
    return mask


def edit_one(
    image_path: Path,
    depth_path: Path,
    mask_path: Path,
    output_path: Path,
    mode: str = "in_category",
    material: str = "metal",
    rendered_control_path: Path | None = None,
    target_category: str | None = None,
    background_style: str | None = None,
    steps: int = 30,
    guidance: float = 9.0,
    control_scale: float = 1.0,
    seed: int = 1234,
) -> dict:
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode {mode}; expected one of {sorted(VALID_MODES)}")

    pipe = load_depth_inpaint_pipeline()
    image = Image.open(image_path).convert("RGB").resize(TARGET_SIZE)
    raw_mask = Image.open(mask_path).convert("L")
    category = infer_category_from_mask(raw_mask)
    mask = prepare_mask(mask_path, mode)
    control_image = load_rendered_control(rendered_control_path, depth_path)
    prompt = build_prompt(
        mode=mode,
        category=category,
        material=material,
        background_style=background_style,
        target_category=target_category,
    )

    result = pipe(
        prompt=prompt,
        negative_prompt=DEFAULT_NEGATIVE,
        image=image,
        mask_image=mask,
        control_image=control_image,
        num_inference_steps=steps,
        guidance_scale=guidance,
        controlnet_conditioning_scale=control_scale,
        generator=torch.Generator(device=pipe.device.type).manual_seed(seed),
    ).images[0]

    composited = composite_back(image, result, mask) if mode != "background" else result.resize(TARGET_SIZE).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composited.save(output_path)
    summary = {
        "image_path": str(image_path),
        "depth_path": str(depth_path),
        "mask_path": str(mask_path),
        "output_path": str(output_path),
        "mode": mode,
        "material": material,
        "category": category,
        "target_category": target_category,
        "background_style": background_style,
        "rendered_control_path": str(rendered_control_path) if rendered_control_path else None,
        "steps": steps,
        "guidance": guidance,
        "control_scale": control_scale,
        "prompt": prompt,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", default="in_category", choices=sorted(VALID_MODES))
    parser.add_argument("--material", default="metal")
    parser.add_argument("--rendered-control", default=None, help="Optional rendered RGB/depth proxy for cross-category geometry")
    parser.add_argument("--target-category", default=None)
    parser.add_argument("--background-style", default=None)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=9.0)
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    summary = edit_one(
        image_path=Path(args.image),
        depth_path=Path(args.depth),
        mask_path=Path(args.mask),
        output_path=Path(args.output),
        mode=args.mode,
        material=args.material,
        rendered_control_path=Path(args.rendered_control) if args.rendered_control else None,
        target_category=args.target_category,
        background_style=args.background_style,
        steps=args.steps,
        guidance=args.guidance,
        control_scale=args.control_scale,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
