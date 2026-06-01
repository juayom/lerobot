from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower

cfg = SOFollowerConfig(
    port="/dev/follower",
)

cfg.id = "follower"
cfg.cameras = {}
cfg.disable_torque_on_disconnect = True
cfg.use_degrees = False
cfg.calibration_dir = None
cfg.max_relative_target = None

robot = SOFollower(cfg)
robot.connect()

try:
    print("Follower connected.")
    input("원하는 자세로 둔 다음 Enter를 누르세요...")

    obs = robot.get_observation()

    print("\n=== observation keys / values ===")
    for k, v in obs.items():
        print(f"{k}: {v}")

finally:
    robot.disconnect()
