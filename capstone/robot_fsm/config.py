from pathlib import Path


PROJECT_ROOT = Path("/home/lerobot/aicapstone/lerobot")

# VLM이 판단할 현재 프레임.
# 처음에는 기존 이미지 하나를 복사해서 테스트하고,
# 나중에는 RealSense/ROS2 이미지 노드가 이 파일을 계속 갱신하게 만들면 된다.
CURRENT_FRAME_PATH = PROJECT_ROOT / "capstone/runtime/current_frame.png"

# VLM 설정
VLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

PICK_READY_STATE = "PICK_READY"
HANDOVER_READY_STATE = "HANDOVER_READY"

VLM_CONFIDENCE_THRESHOLD = 0.60

# 정책 경로
GRAB_POLICY_PATH = (
    PROJECT_ROOT / "local_policies/grab_the_pill_act_fixed"
)

HANDOVER_POLICY_PATH = (
    PROJECT_ROOT / "local_policies/hand_over_pill_act_fixed"
)

# 로봇 설정
ROBOT_TYPE = "so101_follower"
ROBOT_PORT = "/dev/follower"
ROBOT_ID = "follower"

CAMERA_NAME = "intel"
CAMERA_TYPE = "intelrealsense"
CAMERA_SERIAL = "332322071907"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAMERA_USE_DEPTH = True

DISPLAY_DATA = True

# instruction
GRAB_INSTRUCTION = "grab the pill"
HANDOVER_INSTRUCTION = "hand over the pill"

# lerobot-inference가 너무 오래 잡고 있으면 자동 종료할 시간.
# 단위: 초
# 실제 시연에서는 정책 수행 시간에 맞게 20~60초 사이로 조절.
POLICY_TIMEOUT_S = 40