from __future__ import annotations

from pathlib import Path

import torch
from controlnet_aux import CannyDetector
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, UniPCMultistepScheduler
from PIL import Image

from lerobot.genaug.geometry.mask_utils import infer_mask_object_types, invert_mask, load_mask, resize_mask
from lerobot.genaug.models.depth_controlnet import (
    CONTROLNET_CONDITIONING_SCALE,
    GUIDANCE_SCALE,
    NUM_INFERENCE_STEPS,
    TARGET_SIZE,
)
from lerobot.genaug.models.prompt_builder import VALID_MODES, choose_prompt_args, get_prompt


def load_models():
    print("Loading models...")
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32
    print(f"Using device={device}, dtype={dtype}")

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=dtype,
    )
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    canny = CannyDetector()
    print("✓ Models loaded!")
    return pipe, canny


def generate_one_image(pipe, canny, image: Image.Image, mask: Image.Image, prompt: str, seed: int):
    control_image = canny(image)
    device = pipe.device.type if hasattr(pipe, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
    result = pipe(
        prompt=prompt,
        negative_prompt="blurry, low quality, distorted, deformed, multiple objects, text, watermark, logo, cartoon, anime",
        image=image,
        mask_image=mask,
        control_image=control_image,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    return result.images[0]


def iter_input_images(input_root: Path):
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        for path in sorted(input_root.rglob(pattern)):
            if path.is_file():
                yield path


def apply_genaug_to_mirrored_tree(
    input_root: str | Path,
    output_root: str | Path,
    mask_path: str | Path,
    mode: str = "material_only",
    material: str | None = None,
    environment: str | None = None,
    seed_base: int = 1000,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict:
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode {mode}; expected one of {sorted(VALID_MODES)}")

    input_root = Path(input_root)
    output_root = Path(output_root)
    mask_path = Path(mask_path)

    if not input_root.exists():
        raise FileNotFoundError(f"input_root does not exist: {input_root}")
    if not mask_path.exists():
        raise FileNotFoundError(f"mask does not exist: {mask_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    pipe, canny = load_models()

    mask_obj = load_mask(mask_path)
    object_types = infer_mask_object_types(mask_obj)
    if mode == "environment_only":
        mask_obj = invert_mask(mask_obj)
    mask_resized = resize_mask(mask_obj, TARGET_SIZE)

    processed = 0
    for idx, src_path in enumerate(iter_input_images(input_root)):
        if limit is not None and processed >= limit:
            break

        rel = src_path.relative_to(input_root)
        dst_path = output_root / rel
        if dst_path.exists() and not overwrite:
            continue

        chosen_material, chosen_environment = choose_prompt_args(mode, material, environment)
        prompt = get_prompt(mode, chosen_material, chosen_environment, object_types=object_types)
        image = Image.open(src_path).convert("RGB").resize(TARGET_SIZE)
        seed = seed_base + idx
        output_image = generate_one_image(
            pipe=pipe,
            canny=canny,
            image=image,
            mask=mask_resized,
            prompt=prompt,
            seed=seed,
        )

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        output_image.save(dst_path)
        processed += 1
        print(f"[{processed}] {src_path} -> {dst_path}")

    return {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "mode": mode,
        "processed": processed,
        "material": material,
        "environment": environment,
    }


__all__ = ["apply_genaug_to_mirrored_tree", "generate_one_image", "iter_input_images", "load_models"]
