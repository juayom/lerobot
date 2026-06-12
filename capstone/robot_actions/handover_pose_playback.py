import time

from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower


HOME = {
    "shoulder_pan.pos": -0.8130081300813004,
    "shoulder_lift.pos": -87.51054852320675,
    "elbow_flex.pos": 94.7524333474397,
    "wrist_flex.pos": -96.32881085395051,
    "wrist_roll.pos": -55.94627594627595,
    "gripper.pos": 13.0,
}

HANDOVER = {
    "shoulder_pan.pos": -0.600919052668786,
    "shoulder_lift.pos": 26.497890295358644,
    "elbow_flex.pos": -36.09818027930597,
    "wrist_flex.pos": 13.328012769353563,
    "wrist_roll.pos": -56.19047619047619,
    "gripper.pos": 13.0,
}

HANDOVER_OPEN = {
    "shoulder_pan.pos": -0.600919052668786,
    "shoulder_lift.pos": 26.497890295358644,
    "elbow_flex.pos": -36.09818027930597,
    "wrist_flex.pos": 13.328012769353563,
    "wrist_roll.pos": -56.19047619047619,
    "gripper.pos": 40.0,
}

JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def make_robot(port="/dev/follower"):
    cfg = SOFollowerConfig(port=port)

    # 현재 네 로컬 LeRobot 버전 호환용
    cfg.id = "follower"
    cfg.cameras = {}
    cfg.disable_torque_on_disconnect = True
    cfg.use_degrees = False
    cfg.calibration_dir = None
    cfg.max_relative_target = None

    robot = SOFollower(cfg)
    robot.connect()
    return robot


def get_current_pose(robot):
    obs = robot.get_observation()
    return {k: float(obs[k]) for k in JOINT_KEYS}


def interpolate_pose(start, end, alpha):
    return {
        k: start[k] + (end[k] - start[k]) * alpha
        for k in JOINT_KEYS
    }


def send_pose(robot, pose):
    robot.send_action({k: float(v) for k, v in pose.items()})


def move_smooth(robot, start, end, duration=3.0, fps=30):
    steps = max(1, int(duration * fps))

    for i in range(steps + 1):
        alpha = i / steps
        pose = interpolate_pose(start, end, alpha)
        send_pose(robot, pose)
        time.sleep(1.0 / fps)


def hold_pose(robot, pose, hold_s=2.0, fps=30):
    steps = max(1, int(hold_s * fps))

    for _ in range(steps):
        send_pose(robot, pose)
        time.sleep(1.0 / fps)


def run_handover_pose_playback(
    port="/dev/follower",
    move_home_s=4.0,
    move_handover_s=3.0,
    hold_handover_s=2.0,
    open_s=2.5,
    hold_open_s=1.5,
    return_home_s=3.0,
):
    robot = make_robot(port=port)

    try:
        print("[HANDOVER] follower connected")

        current = get_current_pose(robot)
        print("[HANDOVER] current:", current)

        print("[HANDOVER] current -> HOME")
        move_smooth(robot, current, HOME, duration=move_home_s)

        print("[HANDOVER] HOME hold")
        hold_pose(robot, HOME, hold_s=0.5)

        print("[HANDOVER] HOME -> HANDOVER")
        move_smooth(robot, HOME, HANDOVER, duration=move_handover_s)

        print("[HANDOVER] HANDOVER hold")
        hold_pose(robot, HANDOVER, hold_s=hold_handover_s)

        print("[HANDOVER] open gripper")
        move_smooth(robot, HANDOVER, HANDOVER_OPEN, duration=open_s)

        print("[HANDOVER] HANDOVER_OPEN hold")
        hold_pose(robot, HANDOVER_OPEN, hold_s=hold_open_s)

        print("[HANDOVER] return HOME")
        move_smooth(robot, HANDOVER_OPEN, HOME, duration=return_home_s)

        print("[HANDOVER] done")
        return True

    except Exception as e:
        print("[HANDOVER][ERROR]", repr(e))
        return False

    finally:
        robot.disconnect()


if __name__ == "__main__":
    ok = run_handover_pose_playback()
    raise SystemExit(0 if ok else 1)
