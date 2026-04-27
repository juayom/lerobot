from __future__ import annotations

import random

MATERIALS = [
    "metal",
    "glass",
    "wood",
    "carbon fiber",
    "cork",
    "amethyst",
    "lego brick",
    "marble",
    "porcelain",
    "bamboo",
]

OBJECT_MATERIAL_PROMPTS = {
    "box": {
        "metal": "a cardboard shipping box transformed into a shiny brushed metal box with realistic metallic reflections",
        "glass": "a cardboard shipping box transformed into a transparent clear glass box with realistic light refraction",
        "wood": "a cardboard shipping box transformed into a wooden box with natural brown wood grain texture",
        "carbon fiber": "a cardboard shipping box transformed into a black carbon fiber box with a woven grid pattern",
        "cork": "a cardboard shipping box transformed into a cork-textured box with natural brown cork surface",
        "amethyst": "a cardboard shipping box transformed into a deep purple amethyst gemstone box with crystalline facets",
        "lego brick": "a cardboard shipping box reconstructed from colorful lego bricks",
        "marble": "a cardboard shipping box transformed into a white marble box with grey veins and polished surface",
        "porcelain": "a cardboard shipping box transformed into a smooth white porcelain box with glazed finish",
        "bamboo": "a cardboard shipping box transformed into a bamboo-textured box with a natural green bamboo pattern",
    },
    "bottle": {
        "metal": "a cylindrical pill bottle transformed into a shiny brushed metal bottle with realistic metallic reflections",
        "glass": "a cylindrical pill bottle transformed into a transparent clear glass bottle with realistic light refraction",
        "wood": "a cylindrical pill bottle transformed into a wooden bottle with natural brown wood grain texture",
        "carbon fiber": "a cylindrical pill bottle transformed into a black carbon fiber bottle with a woven grid pattern",
        "cork": "a cylindrical pill bottle transformed into a cork-textured bottle with natural brown cork surface",
        "amethyst": "a cylindrical pill bottle transformed into a deep purple amethyst gemstone bottle with crystalline facets",
        "lego brick": "a cylindrical pill bottle reconstructed from colorful lego bricks",
        "marble": "a cylindrical pill bottle transformed into a white marble bottle with grey veins and polished surface",
        "porcelain": "a cylindrical pill bottle transformed into a smooth white porcelain bottle with glazed finish",
        "bamboo": "a cylindrical pill bottle transformed into a bamboo-textured bottle with a natural green bamboo pattern",
    },
    "generic": {
        "metal": "the masked target objects transformed into shiny brushed metal objects with realistic metallic reflections",
        "glass": "the masked target objects transformed into transparent clear glass objects with realistic light refraction",
        "wood": "the masked target objects transformed into wooden objects with natural brown wood grain texture",
        "carbon fiber": "the masked target objects transformed into black carbon fiber objects with a woven grid pattern",
        "cork": "the masked target objects transformed into cork-textured objects with natural brown cork surface",
        "amethyst": "the masked target objects transformed into deep purple amethyst gemstone objects with crystalline facets",
        "lego brick": "the masked target objects reconstructed from colorful lego bricks",
        "marble": "the masked target objects transformed into white marble objects with grey veins and polished surface",
        "porcelain": "the masked target objects transformed into smooth white porcelain objects with glazed finish",
        "bamboo": "the masked target objects transformed into bamboo-textured objects with a natural green bamboo pattern",
    },
}

ENVIRONMENTS = [
    "modern white kitchen with marble countertop, bright natural lighting, realistic shadows",
    "personal office desk cluttered with books and coffee mug, warm desk lamp lighting",
    "cluttered bathroom vanity with mirror, toothbrush, soap dispenser and towels visible, soft morning light",
    "tropical beach resort with ocean view, warm sunlight, sandy surface",
    "colorful children's playroom with wooden toys and building blocks, bright cheerful lighting",
]

VALID_MODES = {"material_only", "environment_only", "combined"}


def build_material_phrase(material: str, object_types: list[str]) -> str:
    if not object_types:
        object_types = ["generic"]
    phrases = [OBJECT_MATERIAL_PROMPTS.get(obj_type, OBJECT_MATERIAL_PROMPTS["generic"])[material] for obj_type in object_types]
    if len(phrases) == 1:
        return phrases[0]
    return ", and ".join(phrases)


def get_prompt(
    mode: str,
    material: str | None = None,
    environment: str | None = None,
    object_types: list[str] | None = None,
) -> str:
    base_scene = (
        "on a light wooden desk, preserving the same scene layout, camera viewpoint, and object geometry; "
        "only change the masked target objects"
    )
    object_types = object_types or ["generic"]
    material_phrase = build_material_phrase(material, object_types) if material else "the masked target objects"
    if mode == "material_only":
        return (
            f"high quality photo of {material_phrase}, {base_scene}, "
            "photorealistic, ultra detailed, sharp focus, realistic shadows and highlights"
        )
    if mode == "environment_only":
        return (
            f"high quality photo of the masked target objects naturally placed in a {environment}, "
            "preserve the target object shapes, keep them resting on the same desk surface, "
            "photorealistic, ultra detailed, sharp focus, realistic lighting and shadows"
        )
    if mode == "combined":
        return (
            f"high quality photo of {material_phrase} in a {environment}, {base_scene}, "
            "photorealistic, ultra detailed, sharp focus, realistic lighting and shadows"
        )
    raise ValueError(f"Unsupported mode: {mode}")


def choose_prompt_args(mode: str, material: str | None, environment: str | None) -> tuple[str | None, str | None]:
    if mode == "material_only":
        return material or random.choice(MATERIALS), None
    if mode == "environment_only":
        return None, environment or random.choice(ENVIRONMENTS)
    return material or random.choice(MATERIALS), environment or random.choice(ENVIRONMENTS)


__all__ = [
    "ENVIRONMENTS",
    "MATERIALS",
    "OBJECT_MATERIAL_PROMPTS",
    "VALID_MODES",
    "build_material_phrase",
    "choose_prompt_args",
    "get_prompt",
]
