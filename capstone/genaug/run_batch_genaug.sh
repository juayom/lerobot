#!/usr/bin/env bash
set -e

source ~/myenv/bin/activate
cd ~/aicapstone/lerobot

INPUT_DIR=~/genaug_inputs
OUTPUT_DIR=~/augmented_results_batch
PROMPTS=capstone/genaug/prompts.json

mkdir -p "$OUTPUT_DIR"

for img in "$INPUT_DIR"/frames/*.jpeg; do
    base=$(basename "$img" .jpeg)

    box_mask="$INPUT_DIR/masks/${base}_box.png"
    bottle_mask="$INPUT_DIR/masks/${base}_bottle.png"

    if [ ! -f "$box_mask" ] || [ ! -f "$bottle_mask" ]; then
        echo "Skip $base: mask file missing"
        continue
    fi

    python3 capstone/genaug/genaug_controlnet_inpaint.py \
      --image "$img" \
      --mask_box "$box_mask" \
      --mask_bottle "$bottle_mask" \
      --prompts "$PROMPTS" \
      --output_dir "$OUTPUT_DIR/$base" \
      --seed 42 \
      --make_grid
done
