from pathlib import Path


PROJECT_ROOT = Path("/home/lerobot/aicapstone/lerobot")

# LeRobot 가상환경 python
LEROBOT_PYTHON = Path("/home/lerobot/venv/lerobot/bin/python")
LEROBOT_INFERENCE = Path("/home/lerobot/venv/lerobot/bin/lerobot-inference")

CURRENT_FRAME_PATH = PROJECT_ROOT / "capstone/runtime/current_frame.png"

VLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

PICK_READY_STATE = "PICK_READY"
HANDOVER_READY_STATE = "HANDOVER_READY"
VLM_CONFIDENCE_THRESHOLD = 0.60

# grab 모델은 실제 ACT 모델
GRAB_POLICY_PATH = "yoohoolala/grab_the_pill_bottle_act"

# handover는 pose playback wrapper로 대체
HANDOVER_POLICY_PATH = PROJECT_ROOT / "capstone/robot_actions/handover_pose_playback.py"

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

GRAB_INSTRUCTION = "grab the pill bottle"
HANDOVER_INSTRUCTION = "hand over the pill bottle"

POLICY_TIMEOUT_S = 40