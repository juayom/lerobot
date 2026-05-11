from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

cfg = SOFollowerRobotConfig(port='/dev/follower', id='follower', cameras={})
robot = SOFollower(cfg)

print('connecting bus...')
robot.bus.connect()
print('connected:', robot.bus.is_connected)

try:
    try:
        vals = robot.bus.sync_read('Present_Position')
        print('present_position:', vals)
    except Exception as e:
        print('present_position read failed:', repr(e))

    try:
        vals = robot.bus.sync_read('Present_Current')
        print('present_current:', vals)
    except Exception as e:
        print('present_current read failed:', repr(e))

    print('trying per-motor disable_torque...')
    for motor in robot.bus.motors:
        try:
            print(f'  disabling {motor} ...')
            robot.bus.disable_torque(motor, num_retry=1)
            print(f'  ok: {motor}')
        except Exception as e:
            print(f'  failed: {motor}: {e!r}')
finally:
    print('closing bus without extra torque disable...')
    robot.bus.disconnect(disable_torque=False)
    print('done')
