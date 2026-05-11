#!/usr/bin/env bash
set -euo pipefail

IMAGE="$1"
DEPTH="$2"
MASK="$3"
OUT_DIR="$4"
TARGET_CATEGORY="${5:-bucket}"
MATERIAL="${6:-metal}"

mkdir -p "$OUT_DIR"
PROXY="$OUT_DIR/proxy_${TARGET_CATEGORY}.png"
python /home/capstone/jua/lerobot/capstone/fi_genaug/render_proxy_control.py \
  --mask "$MASK" \
  --output "$PROXY" \
  --target-category "$TARGET_CATEGORY"

python /home/capstone/jua/lerobot/capstone/fi_genaug/run_fi_genaug.py \
  --image "$IMAGE" \
  --depth "$DEPTH" \
  --mask "$MASK" \
  --output "$OUT_DIR/output_${TARGET_CATEGORY}_${MATERIAL}.png" \
  --mode cross_category \
  --material "$MATERIAL" \
  --rendered-control "$PROXY" \
  --target-category "$TARGET_CATEGORY"
