import argparse
import json
import subprocess
import sys
import os
import signal
import time
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path("/home/lerobot/aicapstone/lerobot")

DEFAULT_POLICY_PATH = "/home/lerobot/aicapstone/lerobot/local_policies/grab_the_pill_bottle_act"
# LeRobot 가상환경의 lerobot-inference 절대경로
LEROBOT_INFERENCE = Path("/home/lerobot/venv/lerobot/bin/lerobot-inference")


def build_camera_config() -> str:
    camera_config = {
        "intel": {
            "type": "opencv",
            "index_or_path": "/dev/video4",
            "width": 640,
            "height": 480,
            "fps": 30,
        }
    }

    return json.dumps(camera_config)


def run_grab_the_pill(
    instruction: str,
    policy_path: str,
    display_data: bool,
    timeout_s: int | None,
) -> bool:
    if not LEROBOT_INFERENCE.exists():
        print(f"[GRAB][ERROR] lerobot-inference not found: {LEROBOT_INFERENCE}")
        return False

    cmd = [
        str(LEROBOT_INFERENCE),
        "--robot.type=so101_follower",
        "--robot.port=/dev/follower",
        "--robot.id=follower",
        f"--robot.cameras={build_camera_config()}",
        f"--policy.path={policy_path}",
        f"--instruction={instruction}",
        f"--display_data={str(display_data).lower()}",
    ]

    env = {
        **os.environ,
        "PYTHONPATH": f"{DEFAULT_PROJECT_ROOT / 'src'}:{DEFAULT_PROJECT_ROOT}",
    }

    print("[GRAB] command:")
    print(" ".join(cmd))
    print("[GRAB] PYTHONPATH =", env["PYTHONPATH"])

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(DEFAULT_PROJECT_ROOT),
            env=env,
        )

        if timeout_s is not None and timeout_s > 0:
            time.sleep(timeout_s)

            print(f"[GRAB] timeout after {timeout_s} seconds.")
            print("[GRAB] send SIGINT for graceful shutdown...")

            proc.send_signal(signal.SIGINT)

            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                print("[GRAB] SIGINT timeout. terminate process...")
                proc.terminate()

                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("[GRAB] terminate timeout. kill process...")
                    proc.kill()
                    proc.wait()

            time.sleep(3.0)

            print("[GRAB] policy stopped after timed run.")
            return True

        result = proc.wait()

        if result == 0:
            print("[GRAB] policy finished successfully.")
            return True

        print(f"[GRAB] policy failed. returncode={result}")
        return False

    except FileNotFoundError:
        print("[GRAB] lerobot-inference command not found.")
        print(f"[GRAB] expected path: {LEROBOT_INFERENCE}")
        return False

    except FileNotFoundError:
        print("[GRAB] lerobot-inference command not found.")
        print(f"[GRAB] expected path: {LEROBOT_INFERENCE}")
        return False


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--instruction",
        default="grab the pill bottle",
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

    ok = run_grab_the_pill(
        instruction=args.instruction,
        policy_path=args.policy_path,
        display_data=args.display_data,
        timeout_s=args.timeout_s,
    )

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()