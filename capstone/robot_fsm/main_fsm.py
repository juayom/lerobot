import os
import argparse
import subprocess
import sys
from pathlib import Path

from capstone.robot_fsm.config import (
    CURRENT_FRAME_PATH,
    VLM_MODEL_ID,
    VLM_CONFIDENCE_THRESHOLD,
    PICK_READY_STATE,
    HANDOVER_READY_STATE,
    GRAB_POLICY_PATH,
    HANDOVER_POLICY_PATH,
    GRAB_INSTRUCTION,
    HANDOVER_INSTRUCTION,
    POLICY_TIMEOUT_S,
    PROJECT_ROOT,
    LEROBOT_PYTHON,
)

from capstone.vlm.vlm_checker import VLMChecker
from capstone.navigation.nav2_client import nav_to as _nav2_go_to
from capstone.navigation.nav2_client import nav_to as _nav2_go_to


def speak(text: str) -> None:
    """
    TODO:
    나중에 Piper TTS, gTTS, pyttsx3 중 하나로 교체.
    지금은 콘솔 출력만 한다.
    """
    print(f"[TTS] {text}")


def wait_for_stt_trigger() -> bool:
    """
    TODO:
    나중에 Whisper/faster-whisper STT로 교체.
    지금은 바로 True를 반환한다.

    실제 목표:
    - 사용자가 '약 가져와', '약 줘' 같은 명령을 말하면 True
    """
    print("[STT] trigger assumed. command = 약 가져와")
    return True


def nav_to(target_name: str) -> bool:
    print(f"[NAV] move to {target_name}")
    return _nav2_go_to(target_name)


def run_policy_script(
    script_path: Path,
    instruction: str,
    policy_path: Path,
    timeout_s: int,
) -> bool:
    cmd = [
        str(LEROBOT_PYTHON),
        str(script_path),
        "--instruction",
        instruction,
        "--policy-path",
        str(policy_path),
        "--timeout-s",
        str(timeout_s),
    ]

    env = {
        **os.environ,
        "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT}",
    }

    print("[FSM] run policy script with LeRobot venv:")
    print(" ".join(cmd))
    print("[FSM] PYTHONPATH =", env["PYTHONPATH"])

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            timeout=timeout_s + 10,
        )
        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"[FSM][ERROR] policy timeout after {timeout_s + 10}s")
        return False


def run_grab_policy() -> bool:
    script_path = PROJECT_ROOT / "src/lerobot/scripts/run_grab_the_pill.py"

    return run_policy_script(
        script_path=script_path,
        instruction=GRAB_INSTRUCTION,
        policy_path=GRAB_POLICY_PATH,
        timeout_s=POLICY_TIMEOUT_S,
    )


def run_handover_policy() -> bool:
    script_path = PROJECT_ROOT / "src/lerobot/scripts/run_hand_over_pill.py"

    return run_policy_script(
        script_path=script_path,
        instruction=HANDOVER_INSTRUCTION,
        policy_path=HANDOVER_POLICY_PATH,
        timeout_s=POLICY_TIMEOUT_S,
    )


def check_pick_ready(vlm: VLMChecker) -> bool:
    print("[FSM] VLM check: pick ready")
    print(f"[FSM] image = {CURRENT_FRAME_PATH}")

    result = vlm.check_image(str(CURRENT_FRAME_PATH), mode="pick")
    print("[VLM RESULT]", result)

    state = result.get("state")
    confidence = float(result.get("confidence", 0.0))

    if state == PICK_READY_STATE and confidence >= VLM_CONFIDENCE_THRESHOLD:
        print("[FSM] pick ready confirmed.")
        return True

    print("[FSM] pick not ready.")
    return False


def check_handover_ready(vlm: VLMChecker) -> bool:
    print("[FSM] VLM check: handover ready")
    print(f"[FSM] image = {CURRENT_FRAME_PATH}")

    result = vlm.check_image(str(CURRENT_FRAME_PATH), mode="handover")
    print("[VLM RESULT]", result)

    state = result.get("state")
    confidence = float(result.get("confidence", 0.0))

    if state == HANDOVER_READY_STATE and confidence >= VLM_CONFIDENCE_THRESHOLD:
        print("[FSM] handover ready confirmed.")
        return True

    print("[FSM] handover not ready.")
    return False


def ensure_current_frame_exists() -> bool:
    if CURRENT_FRAME_PATH.exists():
        return True

    print(f"[ERROR] current frame not found: {CURRENT_FRAME_PATH}")
    print("[HINT] 테스트 전 아래처럼 기존 RGB 프레임을 복사해줘.")
    print(
        "cp [observation.images.intel/.../frame-000035.png] "
        f"{CURRENT_FRAME_PATH}"
    )

    return False


def main(skip_vlm: bool = False):
    print("[FSM] start senior medication assistance pipeline")

    if not skip_vlm and not ensure_current_frame_exists():
        return False

    vlm = None
    if not skip_vlm:
        vlm = VLMChecker(model_id=VLM_MODEL_ID)

    state = "IDLE"

    while True:
        print(f"\n[FSM] state = {state}")

        if state == "IDLE":
            ok = wait_for_stt_trigger()

            if ok:
                speak("약을 가지러 이동하겠습니다.")
                state = "NAV_TO_MEDICINE"
            else:
                state = "IDLE"

        elif state == "NAV_TO_MEDICINE":
            ok = nav_to("medicine_table")

            if ok:
                state = "CHECK_PICK_READY"
            else:
                state = "FAIL_NAV_TO_MEDICINE"

        elif state == "CHECK_PICK_READY":
            if skip_vlm:
                print("[FSM] skip VLM pick check")
                state = "RUN_GRAB_POLICY"
            else:
                ok = check_pick_ready(vlm)

                if ok:
                    state = "RUN_GRAB_POLICY"
                else:
                    speak("약통 위치를 확인하지 못했습니다.")
                    state = "FAIL_PICK_READY"

        elif state == "RUN_GRAB_POLICY":
            speak("약통을 집겠습니다.")
            ok = run_grab_policy()

            if ok:
                state = "NAV_TO_PERSON"
            else:
                state = "FAIL_GRAB_POLICY"

        elif state == "NAV_TO_PERSON":
            speak("사용자 위치로 이동하겠습니다.")
            ok = nav_to("person_position")

            if ok:
                state = "CHECK_HANDOVER_READY"
            else:
                state = "FAIL_NAV_TO_PERSON"

        elif state == "CHECK_HANDOVER_READY":
            if skip_vlm:
                print("[FSM] skip VLM handover check")
                state = "RUN_HANDOVER_POLICY"
            else:
                ok = check_handover_ready(vlm)

                if ok:
                    state = "RUN_HANDOVER_POLICY"
                else:
                    speak("사용자 위치를 확인하지 못했습니다.")
                    state = "FAIL_HANDOVER_READY"

        elif state == "RUN_HANDOVER_POLICY":
            speak("약을 전달하겠습니다.")
            ok = run_handover_policy()

            if ok:
                state = "NAV_HOME"
            else:
                state = "FAIL_HANDOVER_POLICY"

        elif state == "NAV_HOME":
            speak("약 전달이 완료되었습니다. 복귀하겠습니다.")
            nav_to("home")
            state = "DONE"

        elif state == "DONE":
            print("[FSM] done")
            return True

        elif state.startswith("FAIL"):
            print(f"[FSM] failed: {state}")
            speak("작업을 완료하지 못했습니다.")
            return False

        else:
            print(f"[FSM] unknown state: {state}")
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="VLM 판단을 건너뛰고 정책 실행 흐름만 테스트",
    )

    args = parser.parse_args()

    ok = main(skip_vlm=args.skip_vlm)
    sys.exit(0 if ok else 1)