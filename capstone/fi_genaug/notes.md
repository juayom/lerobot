# Notes

## Extracted from the paper

The paper's augmentation design has three main image-generation modes:
1. in-category object / receptacle appearance generation
2. cross-category generation using rendered object geometry + depth-guided generation
3. background generation with non-background masks fixed

It also describes distractor insertion with collision checks.

## What was changed here

The previous local implementation used:
- RGB image
- binary mask
- canny(image)
- text prompt
- SD inpainting

That was not close enough to the paper.

This workspace now uses:
- depth-guided ControlNet instead of canny guidance
- explicit augmentation modes: `in_category`, `cross_category`, `background`
- optional rendered control input for cross-category generation
- object-wise mask processing and composite-back

## Remaining work to become more paper-faithful

- build a mesh-render path for new target categories using known camera pose
- add distractor asset placement + collision rejection
- connect generated outputs back into the LeRobot rebuild pipeline
- expose prompt sampling configs for per-task augmentation schedules
