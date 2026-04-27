# FI-GenAug (paper-aligned workspace)

This folder reorganizes GenAug work around the paper's actual augmentation modes rather than generic canny-guided repainting.

## Paper-aligned augmentation modes

### 1) In-category generation
- keep the original object mask fixed
- use a **depth-guided** inpainting model
- randomize color / material prompts within the same semantic category
- intended effect: preserve position and approximate shape while changing appearance

### 2) Cross-category generation
- do **not** rely on naive inpainting alone
- provide a rendered geometric proxy (rendered control image / rendered depth) for a new category
- use the rendered control + depth-guided model to keep geometry physically plausible
- intended effect: change category while preserving scene plausibility and action semantics

### 3) Background generation
- invert the non-background mask
- keep object / receptacle / distractor regions fixed
- use the model only on the background region

### 4) Distractor generation
- render or place distractor candidates with collision checks against protected masks
- then use the depth-guided editor to make them visually realistic
- this workspace currently includes the structure for this direction, but not the full distractor asset pipeline yet

## Why this replaced older folders
- `capstone/gen_aug/` was mostly legacy background-composite / parquet tooling and not paper-faithful
- `capstone/genaug/` contains useful experiments, but its current main path was centered on canny-conditioned inpainting

## Main files
- `depth_guided_editor.py` — paper-aligned editor supporting in-category, cross-category, and background modes
- `split_mask_components.py` — split object masks into connected components for object-wise processing
- `render_proxy_control.py` — temporary rendered-control producer for cross-category experiments
- `run_fi_genaug.py` — CLI wrapper for the editor
- `run_cross_category_demo.sh` — helper script for one-shot cross-category tests
- `notes.md` — implementation notes and remaining gaps
- `mesh_renderer_plan.md` — how to replace the proxy stage with real mesh rendering later

## Current limitation
This workspace is now structurally closer to the paper, but the current cross-category path still uses a **proxy rendered control image** rather than a true mesh renderer with known camera pose. The full mesh-rendering asset pipeline and collision-checked distractor synthesis still need to be built out.
