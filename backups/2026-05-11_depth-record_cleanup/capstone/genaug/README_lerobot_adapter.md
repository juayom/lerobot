# LeRobot GenAug Runbook

기준 repo:
- `/home/capstone/jua/lerobot`

## 흐름

1. `lerobot-export-video-frames`로 원본 frame export
2. `lerobot-apply-genaug`로 mirrored tree 구조 유지한 증강 이미지 생성
3. `lerobot-genaug-pipeline --augmented-root ...`로 rebuild

## 예시

```bash
cd /home/capstone/jua/lerobot

uv run lerobot-export-video-frames \
  --repo-id plzsay/green \
  --root /home/capstone/.cache/huggingface/lerobot/plzsay/green \
  --output-dir /tmp/jua_green_export \
  --episode-indices 0 \
  --camera-keys observation.images.front,observation.images.top \
  --image-format png \
  --overwrite

uv run lerobot-apply-genaug \
  --input-root /tmp/jua_green_export/observation.images.front \
  --output-root /tmp/jua_green_augmented_front \
  --mask /home/capstone/jua/lerobot/capstone/genaug/assets/mask_controlnet2.png \
  --mode material_only \
  --material metal \
  --limit 1 \
  --overwrite
```

## 주의

- 현재 환경에서는 GPU 경로가 막혀 있어 CPU fallback으로 생성될 수 있음
- full rebuild는 전체 episode frame이 필요함
