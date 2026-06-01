import time
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower

# =========================
# Captured poses
# =========================
HOME = {
    "shoulder_pan.pos": -0.5302226935312859,
    "shoulder_lift.pos": -89.87341772151899,
    "elbow_flex.pos": 94.32924248836224,
    "wrist_flex.pos": -96.08938547486034,
    "wrist_roll.pos": 59.413919413919416,
    "gripper.pos": 1.5473887814313347,
}

HANDOVER = {
    "shoulder_pan.pos": -4.772004241781545,
    "shoulder_lift.pos": 7.257383966244731,
    "elbow_flex.pos": -31.696995344900543,
    "wrist_flex.pos": -12.450119712689542,
    "wrist_roll.pos": 59.65811965811966,
    "gripper.pos": 18.858156028368796,
}

HANDOVER_OPEN = {
    "shoulder_pan.pos": -4.4892188052315305,
    "shoulder_lift.pos": 5.569620253164558,
    "elbow_flex.pos": -27.295810410495136,
    "wrist_flex.pos": -12.928970470869913,
    "wrist_roll.pos": 59.65811965811966,
    "gripper.pos": 47.5177304964539,
}


def get_current_pose(robot):
    obs = robot.get_observation()
    return {
        "shoulder_pan.pos": obs["shoulder_pan.pos"],
        "shoulder_lift.pos": obs["shoulder_lift.pos"],
        "elbow_flex.pos": obs["elbow_flex.pos"],
        "wrist_flex.pos": obs["wrist_flex.pos"],
        "wrist_roll.pos": obs["wrist_roll.pos"],
        "gripper.pos": obs["gripper.pos"],
    }


def interpolate_pose(start, end, alpha):
    return {
        k: start[k] + (end[k] - start[k]) * alpha
        for k in start.keys()
    }


def move_smooth(robot, start, end, duration=3.0, fps=30):
    steps = max(1, int(duration * fps))
    for i in range(steps + 1):
        alpha = i / steps
        pose = interpolate_pose(start, end, alpha)
        robot.send_action(pose)
        time.sleep(1.0 / fps)


def hold_pose(robot, pose, hold_s=2.0, fps=30):
    steps = max(1, int(hold_s * fps))
    for _ in range(steps):
        robot.send_action(pose)
        time.sleep(1.0 / fps)


cfg = SOFollowerConfig(port="/dev/follower")
cfg.id = "follower"
cfg.cameras = {}
cfg.disable_torque_on_disconnect = True
cfg.use_degrees = False
cfg.calibration_dir = None
cfg.max_relative_target = None

robot = SOFollower(cfg)
robot.connect()

try:
    print("Connected to follower.")

    current = get_current_pose(robot)
    print("Current pose:", current)

    input("\nEnter 누르면 현재 자세 -> HOME 으로 이동 시작")

    # 현재 자세에서 HOME으로 먼저 맞춤
    print("Move to HOME...")
    move_smooth(robot, current, HOME, duration=3.0)

    print("Hold HOME 1s...")
    hold_pose(robot, HOME, hold_s=1.0)

    # HOME -> HANDOVER
    print("Move HOME -> HANDOVER...")
    move_smooth(robot, HOME, HANDOVER, duration=3.0)

    print("Hold HANDOVER 2s...")
    hold_pose(robot, HANDOVER, hold_s=2.0)

    # HANDOVER -> HANDOVER_OPEN
    print("Open gripper at handover...")
    move_smooth(robot, HANDOVER, HANDOVER_OPEN, duration=1.5)

    print("Hold HANDOVER_OPEN 2s...")
    hold_pose(robot, HANDOVER_OPEN, hold_s=2.0)

    # HANDOVER_OPEN -> HOME
    print("Return to HOME...")
    move_smooth(robot, HANDOVER_OPEN, HOME, duration=3.0)

    print("Hold HOME 1s...")
    hold_pose(robot, HOME, hold_s=1.0)

    print("Done.")

finally:
    robot.disconnect()
