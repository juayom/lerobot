from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

PORT = "/dev/ttyACM0"   # 실제 follower 포트로 바꾸세요
TARGET = "shoulder_lift"  # id 2

cfg = SOFollowerRobotConfig(
    port=PORT,
    id="follower",
    cameras={},
)

robot = SOFollower(cfg)


def try_write(label, fn):
    try:
        fn()
        print(f"[OK] {label}")
        return True
    except Exception as e:
        print(f"[FAIL] {label}: {e}")
        return False


print("connecting...")
robot.bus.connect()
print("connected")

try:
    try:
        print("present_position:", robot.bus.sync_read("Present_Position"))
    except Exception as e:
        print("present_position read failed:", e)

    try:
        print("present_current:", robot.bus.sync_read("Present_Current"))
    except Exception as e:
        print("present_current read failed:", e)

    try_write(
        "Overload_Torque=10",
        lambda: robot.bus.write("Overload_Torque", TARGET, 10, num_retry=1),
    )

    try_write(
        "Protection_Current=100",
        lambda: robot.bus.write("Protection_Current", TARGET, 100, num_retry=1),
    )

    try_write(
        "Max_Torque_Limit=100",
        lambda: robot.bus.write("Max_Torque_Limit", TARGET, 100, num_retry=1),
    )

    ok = try_write(
        "disable_torque(shoulder_lift)",
        lambda: robot.bus.disable_torque(TARGET, num_retry=1),
    )

    if not ok:
        print("single motor failed -> trying all motors")
        for motor in robot.bus.motors:
            try_write(
                f"disable_torque({motor})",
                lambda m=motor: robot.bus.disable_torque(m, num_retry=1),
            )

finally:
    print("disconnecting bus without extra torque disable...")
    robot.bus.disconnect(disable_torque=False)
    print("done")
