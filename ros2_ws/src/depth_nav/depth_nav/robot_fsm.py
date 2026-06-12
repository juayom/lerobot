import subprocess
from pathlib import Path
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time
import os
from gtts import gTTS

PROJECT_ROOT = Path("/home/lerobot/aicapstone/lerobot")
LEROBOT_PYTHON = Path("/home/lerobot/venv/lerobot/bin/python")

GRAB_SCRIPT = PROJECT_ROOT / "src/lerobot/scripts/run_grab_the_pill.py"
HANDOVER_SCRIPT = PROJECT_ROOT / "src/lerobot/scripts/run_hand_over_pill.py"

GRAB_POLICY_PATH = "/home/lerobot/aicapstone/lerobot/local_policies/grab_the_pill_bottle_act"
HANDOVER_POLICY_PATH = PROJECT_ROOT / "capstone/robot_actions/handover_pose_playback.py"

GRAB_INSTRUCTION = "grab the pill bottle"
HANDOVER_INSTRUCTION = "hand over the pill bottle"

def speak(text):
    try:
        tts = gTTS(text=text, lang='ko')
        tts.save("/tmp/robot_voice.mp3")
        os.system("ffmpeg -y -i /tmp/robot_voice.mp3 -ar 44100 -ac 2 -af volume=2.0 /tmp/robot_voice.wav > /dev/null 2>&1")
        os.system("aplay -D sysdefault:CARD=Device -q /tmp/robot_voice.wav")
    except Exception as e:
        print(f"TTS 오류: {e}")

class RobotFSM(Node):
    def __init__(self):
        super().__init__('robot_fsm')

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_pub = self.create_publisher(String, '/robot_arm/command', 10)
        self.mode_pub = self.create_publisher(String, '/detector_mode', 10)
        self.nav_result_pub = self.create_publisher(String, '/nav_result', 10)

        self.tts_sub = self.create_subscription(String, '/robot_arm/command', self.tts_cb, 10)
        self.target_sub = self.create_subscription(Float32MultiArray, '/target_direction', self.target_cb, 10)
        self.nav_goal_sub = self.create_subscription(String, '/nav_goal', self.nav_goal_cb, 10)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, qos)

        self.state = 'WAIT'
        self.direction = 0.0
        self.distance = 0.0
        self.detected = False

        self.front_distance = 999.0
        self.obstacle_threshold = 0.01

        self.linear_speed = 0.15
        self.angular_speed = 0.4
        self.search_angular = 0.15

        self.step_start_time = None
        self.nav_step = 0
        self.scan_step = 0

        self.timer = self.create_timer(0.1, self.fsm_loop)
        self.get_logger().info('RobotFSM 시작! 대기 중...')

    def tts_cb(self, msg):
        if msg.data in ['DELIVER_TYLENOL', 'DELIVER_DIGESTIVE'] and self.state == 'WAIT':
            self.get_logger().info('DELIVER 수신! 스캔 시작')
            speak("라이다 레이더 가동합니다. 주변을 탐색하며 약통 위치로 이동을 시작합니다.")
            self.state = 'SCAN'
            self.nav_step = 0
            self.step_start_time = time.time()

    def nav_goal_cb(self, msg):
        self.get_logger().info(f'nav_goal 수신: {msg.data}')

    def scan_cb(self, msg):
        ranges = msg.ranges
        n = len(ranges)
        front_indices = list(range(0, n//12)) + list(range(11*n//12, n))
        valid = [ranges[i] for i in front_indices if ranges[i] > 0.01]
        self.front_distance = min(valid) if valid else 999.0

    def target_cb(self, msg):
        self.direction = msg.data[0]
        self.distance = msg.data[1]
        self.detected = True

    def set_mode(self, mode):
        msg = String()
        msg.data = mode
        self.mode_pub.publish(msg)
        self.get_logger().info(f'감지 모드: {mode}')

    def publish_cmd(self, linear, angular):
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_vel_pub.publish(twist)

    def stop(self):
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

    def stop_depth_detector(self):
        self.get_logger().info("[CAMERA] stop depth_detector")
        subprocess.run(["pkill", "-f", "depth_detector"], check=False)
        time.sleep(2.0)

    def start_depth_detector(self):
        self.get_logger().info("[CAMERA] start depth_detector")
        env = {**os.environ}
        subprocess.Popen(["ros2", "run", "depth_nav", "depth_detector"], env=env)
        time.sleep(3.0)

    def run_lerobot_script(self, script_path, instruction, policy_path, timeout_s=40, no_display=False):
        cmd = [
            str(LEROBOT_PYTHON),
            str(script_path),
            "--instruction", instruction,
            "--policy-path", str(policy_path),
            "--timeout-s", str(timeout_s),
        ]
        if no_display:
            cmd.append("--no-display-data")
        env = {
            **os.environ,
            "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT}",
        }
        self.get_logger().info("[LeRobot] 실행: " + " ".join(cmd))
        try:
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, timeout=timeout_s+10)
            if result.returncode == 0:
                self.get_logger().info("[LeRobot] 성공")
                return True
            self.get_logger().error(f"[LeRobot] 실패 returncode={result.returncode}")
            return False
        except subprocess.TimeoutExpired:
            self.get_logger().warn(f"[LeRobot] timeout after {timeout_s+10}s")
            return True
        except Exception as e:
            self.get_logger().error(f"[LeRobot] 예외: {e}")
            return False

    def run_grab_policy(self):
        return self.run_lerobot_script(
            script_path=GRAB_SCRIPT,
            instruction=GRAB_INSTRUCTION,
            policy_path=GRAB_POLICY_PATH,
            timeout_s=40,
            no_display=True,
        )

    def run_handover_policy(self):
        return self.run_lerobot_script(
            script_path=HANDOVER_SCRIPT,
            instruction=HANDOVER_INSTRUCTION,
            policy_path=HANDOVER_POLICY_PATH,
            timeout_s=40,
            no_display=False,
        )
        
    def check_follower_ready(self, retries=3, delay_s=2.0) -> bool:
        check_code = r'''
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/lerobot/aicapstone/lerobot")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower

cfg = SOFollowerConfig(port="/dev/follower")
cfg.id = "follower"
cfg.cameras = {}
cfg.disable_torque_on_disconnect = True
cfg.use_degrees = False
cfg.calibration_dir = None
cfg.max_relative_target = None

robot = SOFollower(cfg)
robot.connect()
obs = robot.get_observation()
print(obs.keys())
robot.disconnect()

if "gripper.pos" not in obs:
    raise RuntimeError("gripper.pos not found")
'''

        env = {
            **os.environ,
            "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT}",
        }

        for i in range(retries):
            try:
                self.get_logger().info(f"[ARM] follower ready check {i+1}/{retries}")

                result = subprocess.run(
                    [str(LEROBOT_PYTHON), "-c", check_code],
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    timeout=15,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                if result.returncode == 0:
                    self.get_logger().info("[ARM] follower ready, gripper detected")
                    return True

                self.get_logger().warn(
                    f"[ARM] follower check failed {i+1}/{retries}, "
                    f"returncode={result.returncode}, stderr={result.stderr.strip()}"
                )

            except Exception as e:
                self.get_logger().warn(f"[ARM] follower check exception {i+1}/{retries}: {e}")

            time.sleep(delay_s)

        return False
    
    def fsm_loop(self):
        if self.state == 'WAIT':
            pass

        elif self.state == 'SCAN':
            elapsed = time.time() - self.step_start_time
            if self.nav_step == 0:
                if elapsed < 0.8:
                    self.publish_cmd(0.0, 0.2)
                else:
                    self.nav_step = 1
                    self.step_start_time = time.time()
            elif self.nav_step == 1:
                if elapsed < 0.8:
                    self.publish_cmd(0.0, -0.2)
                else:
                    self.stop()
                    self.nav_step = 2
                    self.step_start_time = time.time()
            elif self.nav_step == 2:
                if elapsed < 0.8:
                    self.publish_cmd(0.0, -0.2)
                else:
                    self.nav_step = 3
                    self.step_start_time = time.time()
            elif self.nav_step == 3:
                if elapsed < 0.8:
                    self.publish_cmd(0.0, 0.2)
                else:
                    self.stop()
                    self.get_logger().info('스캔 완료! 직진 시작')
                    self.state = 'GO_STRAIGHT'
                    self.step_start_time = time.time()

        elif self.state == 'GO_STRAIGHT':
            elapsed = time.time() - self.step_start_time
            if elapsed < 4.0:
                self.publish_cmd(self.linear_speed, 0.0)
            else:
                self.stop()
                self.state = 'TURN_LEFT'
                self.step_start_time = time.time()

        elif self.state == 'TURN_LEFT':
            elapsed = time.time() - self.step_start_time
            if elapsed < 1.1:
                self.publish_cmd(0.0, self.angular_speed)
            else:
                self.stop()
                self.get_logger().info('좌회전 완료! bottle 탐색 시작')
                self.set_mode('TABLE')
                self.state = 'SEARCH_TABLE'

        elif self.state == 'SEARCH_TABLE':
            if self.detected:
                self.get_logger().info('bottle 발견! 정지')
                self.stop()
                time.sleep(0.5)
                speak("약통을 발견했습니다. 약통 집기를 시작합니다.")
                self.detected = False
                self.state = 'PICK_UP'
            else:
                self.stop()
                self.detected = False

        elif self.state == 'PICK_UP':
            self.stop()
            speak("팔 제어 모델을 로딩합니다. 약 15초 정도의 딜레이가 발생할 수 있습니다.")
            self.stop_depth_detector()
            try:
                ok = self.run_grab_policy()
            finally:
                self.start_depth_detector()
            if ok:
                self.get_logger().info('약통 집기 완료. 180도 회전 시작')
                speak("약통을 집었습니다. 손을 탐색 중입니다.")
                time.sleep(3.0)
                self.state = 'TURN_180'
                self.step_start_time = time.time()
            else:
                self.get_logger().error('약통 집기 실패.')
                speak("약통 집기에 실패했습니다.")
                self.state = 'WAIT'

        elif self.state == 'TURN_180':
            elapsed = time.time() - self.step_start_time
            if elapsed < 3.2:
                self.publish_cmd(0.0, -self.angular_speed)
            else:
                self.stop()
                time.sleep(1.0)
                self.set_mode('HAND')
                time.sleep(0.2)
                self.set_mode('HAND')
                self.detected = False
                self.state = 'WAIT_HAND'
                self.scan_step = 0
                self.step_start_time = time.time()

        elif self.state == 'WAIT_HAND':
            if self.detected and self.distance < 1.0:
                self.get_logger().info(f'손 감지! 거리: {self.distance:.2f}m')
                self.stop()
                speak("손을 발견했습니다. 약통을 전달하러 이동합니다.")
                time.sleep(1.0)
                self.detected = False
                self.state = 'GO_TO_PERSON'
                self.step_start_time = time.time()
            else:
                elapsed = time.time() - self.step_start_time
                if self.scan_step == 0:
                    if elapsed < 1.0:
                        self.publish_cmd(0.0, self.search_angular)
                    else:
                        self.scan_step = 1
                        self.step_start_time = time.time()
                elif self.scan_step == 1:
                    if elapsed < 1.0:
                        self.publish_cmd(0.0, -self.search_angular)
                    else:
                        self.stop()
                        self.scan_step = 2
                        self.step_start_time = time.time()
                elif self.scan_step == 2:
                    if elapsed < 1.0:
                        self.publish_cmd(0.0, -self.search_angular)
                    else:
                        self.scan_step = 3
                        self.step_start_time = time.time()
                elif self.scan_step == 3:
                    if elapsed < 1.0:
                        self.publish_cmd(0.0, self.search_angular)
                    else:
                        self.stop()
                        self.scan_step = 0
                        self.step_start_time = time.time()
                self.detected = False

        elif self.state == 'GO_TO_PERSON':
            elapsed = time.time() - self.step_start_time
            if elapsed < 4.0:
                self.publish_cmd(self.linear_speed, -self.direction * self.angular_speed)
            else:
                self.stop()
                time.sleep(1.0)
                self.state = 'GIVE'

        elif self.state == 'GIVE':
            self.stop()
            time.sleep(2.0)
            speak("팔 제어 모델을 로딩합니다. 약 15초 정도의 딜레이가 발생할 수 있습니다.")
            speak("약통을 건네드리겠습니다.")
            if not self.check_follower_ready(retries=3, delay_s=2.0):
                speak("팔 모터 연결이 불안정해서 전달 동작을 중단합니다.")
                self.get_logger().error("handover 중단: follower not ready")
                self.state = 'WAIT'
                return
            ok = self.run_handover_policy()
            if ok:
                speak("약통을 건네드렸습니다. 홈으로 복귀합니다.")
                self.state = 'RETURN_HOME'
                self.step_start_time = time.time()
            else:
                self.get_logger().error('약통 전달 실패.')
                speak("약통 전달에 실패했습니다.")
                self.state = 'WAIT'

        elif self.state == 'RETURN_HOME':
            elapsed = time.time() - self.step_start_time
            if elapsed < 8.0:
                self.publish_cmd(-self.linear_speed, 0.0)
            else:
                self.stop()
                speak("도움이 필요하시면 언제든지 젠젠이를 불러주세요.")
                self.state = 'WAIT'
                self.get_logger().info('홈 도착! 대기 중...')

def main():
    rclpy.init()
    node = RobotFSM()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
