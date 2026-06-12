#!/usr/bin/env python3
"""
GenAug-style semantic image augmentation for pill-assistance robot dataset.

This script performs mask-guided Stable Diffusion Inpainting with ControlNet Canny.
It preserves the original object location and approximate geometry, while changing
visual appearance of the box, bottle, or background.

Important:
- This script augments RGB observations only.
- Action labels are NOT changed.
- Use only when the augmented image does not invalidate the original robot action.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel


DEFAULT_NEGATIVE = (
    "change shape, change form, deformed, distorted, different shape, "
    "blurry, low quality, cartoon, unrealistic, bad geometry, wrong perspective"
)


def load_image_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return image


def load_mask_gray(path: Path, size=None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    if size is not None:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def build_background_mask(mask_paths, image_size, dilate_kernel=15):
    combined = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

    for mask_path in mask_paths:
        mask = load_mask_gray(Path(mask_path), size=image_size)
        combined = cv2.bitwise_or(combined, mask)

    background = cv2.bitwise_not(combined)

    if dilate_kernel > 0:
        kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        background = cv2.dilate(background, kernel, iterations=1)

    return background


def make_canny_control(image_pil: Image.Image, low=100, high=200) -> Image.Image:
    image_np = np.array(image_pil)
    canny = cv2.Canny(image_np, low, high)
    canny_rgb = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(canny_rgb)


def load_pipeline(device: str):
    print("Loading ControlNet + Stable Diffusion Inpainting pipeline...")

    dtype = torch.float16 if "cuda" in device else torch.float32

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny",
        torch_dtype=dtype,
    )

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    if "cuda" in device:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            print("xformers is not available. Continue without xformers.")

    return pipe


def generate_one(
    pipe,
    image_pil,
    mask_pil,
    control_pil,
    prompt,
    negative_prompt,
    steps,
    guidance_scale,
    controlnet_scale,
    seed,
):
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image_pil,
        mask_image=mask_pil,
        control_image=control_pil,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_scale,
        generator=generator,
    ).images[0]

    return result


def save_metadata(metadata_path: Path, rows):
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True, help="Input RGB image path")
    parser.add_argument("--mask_box", required=True, help="Box mask path")
    parser.add_argument("--mask_bottle", required=True, help="Bottle mask path")
    parser.add_argument("--prompts", required=True, help="Prompt json path")
    parser.add_argument("--output_dir", required=True, help="Output directory")

    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--make_grid", action="store_true")

    args = parser.parse_args()

    image_path = Path(args.image).expanduser()
    mask_box_path = Path(args.mask_box).expanduser()
    mask_bottle_path = Path(args.mask_bottle).expanduser()
    prompts_path = Path(args.prompts).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = load_image_bgr(image_path)
    H, W = image_bgr.shape[:2]

    image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).resize(
        (args.size, args.size)
    )

    mask_box = load_mask_gray(mask_box_path, size=(args.size, args.size))
    mask_bottle = load_mask_gray(mask_bottle_path, size=(args.size, args.size))
    mask_bg = build_background_mask(
        [mask_box_path, mask_bottle_path],
        image_size=(args.size, args.size),
        dilate_kernel=15,
    )

    masks = {
        "box": Image.fromarray(mask_box),
        "bottle": Image.fromarray(mask_bottle),
        "background": Image.fromarray(mask_bg),
    }

    control_pil = make_canny_control(image_pil)

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompt_config = json.load(f)

    pipe = load_pipeline(args.device)

    metadata = []
    generated_images = {"original": image_bgr}

    for group_name, items in prompt_config.items():
        if group_name not in masks:
            print(f"Skip unknown group: {group_name}")
            continue

        for idx, item in enumerate(items):
            name = item["name"]
            prompt = item["prompt"]
            negative = item.get("negative_prompt", DEFAULT_NEGATIVE)
            guidance = float(item.get("guidance_scale", 9.0))
            control_scale = float(item.get("controlnet_conditioning_scale", 1.0))

            out_name = f"{group_name}_{name}_seed{args.seed}.png"
            out_path = output_dir / out_name

            print(f"Generating: {out_name}")

            out_pil = generate_one(
                pipe=pipe,
                image_pil=image_pil,
                mask_pil=masks[group_name],
                control_pil=control_pil,
                prompt=prompt,
                negative_prompt=negative,
                steps=args.steps,
                guidance_scale=guidance,
                controlnet_scale=control_scale,
                seed=args.seed + idx,
            )

            out_bgr = cv2.cvtColor(
                np.array(out_pil.resize((W, H))),
                cv2.COLOR_RGB2BGR,
            )

            cv2.imwrite(str(out_path), out_bgr)
            generated_images[f"{group_name}_{name}"] = out_bgr

            metadata.append(
                {
                    "source_image": str(image_path),
                    "output_image": str(out_path),
                    "mask_group": group_name,
                    "prompt_name": name,
                    "prompt": prompt,
                    "negative_prompt": negative,
                    "seed": args.seed + idx,
                    "action_label_policy": "same_as_original",
                    "note": "Semantic visual augmentation only. Original action/depth/task should be preserved when repacking dataset.",
                }
            )

    save_metadata(output_dir / "augmentation_metadata.json", metadata)

    if args.make_grid:
        print("Creating comparison grid...")
        cell_w, cell_h = 400, 300

        def resize_and_label(img, text):
            canvas = cv2.resize(img, (cell_w, cell_h))
            cv2.putText(
                canvas,
                text,
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )
            return canvas

        cells = []
        for key, img in generated_images.items():
            cells.append(resize_and_label(img, key))

        while len(cells) % 4 != 0:
            cells.append(np.zeros_like(cells[0]))

        rows = []
        for i in range(0, len(cells), 4):
            rows.append(np.hstack(cells[i : i + 4]))

        grid = np.vstack(rows)
        cv2.imwrite(str(output_dir / "comparison_grid.png"), grid)

    print(f"Done. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
