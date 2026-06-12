import argparse
import time

from capstone.robot_actions.handover_pose_playback import run_handover_pose_playback


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--instruction",
        default="hand over the pill bottle",
        help="Task instruction",
    )

    parser.add_argument(
        "--policy-path",
        default="scripted_handover_pose_playback",
        help="Policy path. Kept for compatibility with FSM.",
    )

    parser.add_argument(
        "--timeout-s",
        type=int,
        default=30,
        help="Timeout seconds. Kept for compatibility with FSM.",
    )

    args = parser.parse_args()

    print("[POLICY] =====================================")
    print("[POLICY] Hand-over policy runner")
    print(f"[POLICY] instruction : {args.instruction}")
    print(f"[POLICY] policy path : {args.policy_path}")
    print(f"[POLICY] timeout_s   : {args.timeout_s}")
    print("[POLICY] =====================================")

    time.sleep(0.5)

    ok = run_handover_pose_playback(port="/dev/follower")

    if ok:
        print("[POLICY] handover policy completed successfully.")
        return 0

    print("[POLICY] handover policy failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())