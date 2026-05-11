# GenAug package layout

This package separates the offline generative augmentation pipeline from the core LeRobot dataset package.

## Structure

- `pipeline/augment_dataset.py`: export -> manifest rewrite -> alignment validation -> rebuild -> optional video conversion
- `pipeline/augment_episode.py`: single mirrored-tree augmentation entrypoint
- `pipeline/repack_lerobot.py`: rebuild augmented images into a LeRobot image dataset
- `pipeline/alignment.py`: alignment validation bridge
- `models/`: wrappers for prompt building and diffusion/inpainting integration
- `geometry/`: depth, mask, projection, collision helpers
- `io/`: manifest/schema/load/save helpers
- `checks/`: action-observation/depth/alignment validation helpers
- `configs/`: starter YAML presets

## Notes

- Legacy imports under `lerobot.datasets.genaug_pipeline` are kept for compatibility.
- The current codebase already had working dataset rebuild logic; this package reorganizes it so GenAug-specific concerns are easier to extend.
- `collision_check.py` and `projection.py` are placeholders pending finalized camera geometry and action-validity rules.
