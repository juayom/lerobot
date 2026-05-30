# src/lerobot/scripts/run_hand_over_pill.py

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path("/home/lerobot/aicapstone/lerobot")
DEFAULT_POLICY_PATH = (
    DEFAULT_PROJECT_ROOT / "local_policies/hand_over_pill_act_fixed"
)


def build_camera_config() -> str:
    camera_config = {
        "intel": {
            "type": "intelrealsense",
            "serial_number_or_name": "332322071907",
            "width": 1280,
            "height": 720,
            "fps": 30,
            "use_depth": True,
        }
    }

    return json.dumps(camera_config)


def run_hand_over_pill(
    instruction: str,
    policy_path: str,
    display_data: bool,
    timeout_s: int | None,
) -> bool:
    cmd = [
        "lerobot-inference",
        "--robot.type=so101_follower",
        "--robot.port=/dev/follower",
        "--robot.id=follower",
        f"--robot.cameras={build_camera_config()}",
        f"--policy.path={policy_path}",
        f"--instruction={instruction}",
        f"--display_data={str(display_data).lower()}",
    ]

    print("[HANDOVER] command:")
    print(" ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            cwd=str(DEFAULT_PROJECT_ROOT),
            timeout=timeout_s,
        )

        if result.returncode == 0:
            print("[HANDOVER] policy finished successfully.")
            return True

        print(f"[HANDOVER] policy failed. returncode={result.returncode}")
        return False

    except subprocess.TimeoutExpired:
        print(f"[HANDOVER] timeout after {timeout_s} seconds.")
        # 정책이 일정 시간 실행되면 다음 단계로 넘어가도록 True 처리.
        return True

    except FileNotFoundError:
        print("[HANDOVER] lerobot-inference command not found.")
        print("[HANDOVER] Check whether lerobot environment is activated.")
        return False


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--instruction",
        default="hand over the pill",
    )

    parser.add_argument(
        "--policy-path",
        default=str(DEFAULT_POLICY_PATH),
    )

    parser.add_argument(
        "--display-data",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--no-display-data",
        action="store_false",
        dest="display_data",
    )

    parser.add_argument(
        "--timeout-s",
        type=int,
        default=40,
    )

    args = parser.parse_args()

    ok = run_hand_over_pill(
        instruction=args.instruction,
        policy_path=args.policy_path,
        display_data=args.display_data,
        timeout_s=args.timeout_s,
    )

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()