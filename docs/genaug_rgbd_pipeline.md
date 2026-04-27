# GenAug RGB-D Pipeline

## 목적
- RealSense 기반 RGB-D demonstration dataset을 LeRobotDataset 호환 구조로 기록한다.
- observation만 GenAug 스타일로 증강하고 action/task/timestamp/frame alignment는 유지한다.

## 실제 사용한 LeRobot API
- `LeRobotDataset.create(...)`
- `LeRobotDataset.add_frame(...)`
- `LeRobotDataset.save_episode(...)`
- `LeRobotDataset.__getitem__(...)`
- `RealSenseCamera.read()` / `RealSenseCamera.read_depth()`
- `make_dataset(...)` / `lerobot_train.py`

## 전체 파이프라인
1. `record_rgbd_dataset.py`로 RGB-D + 외부 action stream 기록
2. `augment_and_repack.py`로 observation.image만 증강하고 depth/action/task는 보존
3. `viz_alignment.py`로 샘플 시각화
4. 필요 시 기존 `lerobot-train`에서 dataset root를 그대로 사용

## RGB-D 레코딩
```bash
python -m lerobot.scripts.record_rgbd_dataset \
  --repo-id local/rgbd_demo \
  --root ./data/rgbd_demo \
  --task "put the cup in the basket" \
  --fps 30 \
  --camera realsense \
  --serial-number-or-name <SERIAL_OR_NAME> \
  --depth-unit meter \
  --action-jsonl ./actions.jsonl
```

## Dry-run 증강
```bash
python -m lerobot.scripts.augment_and_repack \
  --repo-id local/rgbd_demo \
  --root ./data/rgbd_demo \
  --output-root ./data/rgbd_demo_genaug \
  --num-aug 1 \
  --config ./src/lerobot/genaug/configs/genaug_rgbd.yaml \
  --tolerance-s 0.005 \
  --dry-run
```

## Visualization
```bash
python -m lerobot.scripts.viz_alignment \
  --dataset-root ./data/rgbd_demo_genaug \
  --repo-id local/rgbd_demo_genaug \
  --episode-index 0 \
  --frame-index 0 \
  --out ./debug/alignment.png
```

## Training 연결 예시
```bash
python -m lerobot.scripts.lerobot_train --dataset.repo_id=local/rgbd_demo_genaug --dataset.root=./data/rgbd_demo_genaug
```

## 수정한 파일 목록
- `src/lerobot/genaug/engine.py`
- `src/lerobot/genaug/geometry/depth_utils.py`
- `src/lerobot/genaug/geometry/mask_utils.py`
- `src/lerobot/genaug/checks/alignment.py`
- `src/lerobot/scripts/record_rgbd_dataset.py`
- `src/lerobot/scripts/augment_and_repack.py`
- `src/lerobot/scripts/viz_alignment.py`
- `src/lerobot/scripts/genaug_engine.py`
- `src/lerobot/genaug/configs/genaug_rgbd.yaml`

## Known limitations
- GenAug는 action label을 새로 만들지 않는다.
- `record_rgbd_dataset.py`는 외부 action stream(`--action-jsonl`)을 요구한다. action을 임의 생성하지 않는다.
- 증강 결과가 실제 물리와 맞지 않을 수 있다.
- distractor mode는 현재 skeleton 수준이다.
- distractor가 action trajectory와 충돌하면 원래 action이 유효하지 않을 수 있다.
- RGB-depth alignment가 안 맞으면 depth-guided augmentation 품질이 떨어진다.
- 현재 구현에서는 `RealSenseCamera.read_rgbd()` / `read_latest_rgbd()`를 추가해 RGB-D pair를 같은 capture bundle 기준으로 읽는다.
- video frame 간 visual consistency는 보장하지 않는다.
- depth-guided image generation은 모델 설치와 GPU 환경에 의존한다.
- 현재 `align_depth_to_rgb()`는 calibration metadata plumbing이 부족해 no-op 경고 반환이다.
- `yoohoolala/newdepth07` 실제 dataset 기준 dry-run 검증과 source visualization은 성공했다.
- 현재 repack 구현은 안정성을 위해 source visual video features를 image features로 내려 저장한다. 이는 기존 train pipeline에서 읽히며, `yoohoolala/newdepth07` 실제 dataset 기준 repack + visualization까지 확인했다. 다만 원본과 동일한 video-backed storage를 그대로 재생성하는 단계는 아직 남아 있다.

## 백업
기존 실험성 GenAug 코드는 `backups/legacy_genaug_2026-04-24/`로 이동했다.
