import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import cv2
from pathlib import Path

FRAME_SAVE_PATH = Path('/home/lerobot/aicapstone/lerobot/capstone/runtime/current_frame.png')

class DepthDetector(Node):
    def __init__(self):
        super().__init__('depth_detector')
        self.pub = self.create_publisher(Float32MultiArray, '/target_direction', 10)
        self.img_pub = self.create_publisher(Image, '/camera/camera/color/image_raw', 10)
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.table_model = YOLO('yolov8s.pt')
        self.hand_model = YOLO('/home/lerobot/hand_yolov8s.pt')

        self.mode = 'TABLE'
        self.state_sub = self.create_subscription(String, '/detector_mode', self.mode_cb, 10)

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        FRAME_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

        self.last_direction = 0.0
        self.last_distance = 0.0
        self.no_detect_count = 0
        self.get_logger().info('DepthDetector 시작!')

    def mode_cb(self, msg):
        self.mode = msg.data
        self.get_logger().info(f'감지 모드 변경: {self.mode}')

    def is_pink(self, color_image, x1, y1, x2, y2):
        roi = color_image[int(y1):int(y2), int(x1):int(x2)]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_pink = np.array([140, 30, 100])
        upper_pink = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_pink, upper_pink)
        lower_pink2 = np.array([0, 30, 100])
        upper_pink2 = np.array([20, 255, 255])
        mask2 = cv2.inRange(hsv, lower_pink2, upper_pink2)
        mask = mask1 + mask2
        pink_ratio = np.count_nonzero(mask) / mask.size
        return pink_ratio > 0.05

    def timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        h, w = depth_image.shape
        cx = w // 2

        cv2.imwrite(str(FRAME_SAVE_PATH), color_image)
        img_msg = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
        self.img_pub.publish(img_msg)

        best_box = None
        best_conf = 0.0

        if self.mode == 'TABLE':
            # bottle + 핑크색 조건, confidence 0.55 이상만 인식
            results = self.table_model(color_image, verbose=False)
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self.table_model.names[cls]
                    if label == 'bottle' and conf > 0.3:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        if self.is_pink(color_image, x1, y1, x2, y2):
                            if conf > best_conf:
                                best_conf = conf
                                best_box = box
                        else:
                            self.get_logger().info('bottle 감지됐지만 핑크색 아님 → 무시')
                    elif label == 'bottle' and conf <= 0.55:
                        self.get_logger().info(f'bottle 감지됐지만 confidence 낮음({conf:.2f}) → 무시')

        else:  # HAND 모드 - hand 모델만 사용, person 폴백 없음
            results = self.hand_model(color_image, verbose=False)
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    label = self.hand_model.names[int(box.cls[0])]
                    if label == 'hand' and conf > 0.3:
                        if conf > best_conf:
                            best_conf = conf
                            best_box = box

        if best_box is None:
            self.no_detect_count += 1
            if self.no_detect_count < 5:
                msg = Float32MultiArray()
                msg.data = [self.last_direction, self.last_distance]
                self.pub.publish(msg)
            else:
                self.get_logger().info(f'{self.mode} 없음')
            return

        self.no_detect_count = 0
        x1, y1, x2, y2 = best_box.xyxy[0].tolist()
        box_cx = int((x1 + x2) / 2)
        box_cy = int((y1 + y2) / 2)
        box_width = x2 - x1

        dist = depth_image[box_cy, box_cx] * 0.001
        if dist <= 0:
            dist = self.last_distance

        direction = (box_cx - cx) / cx
        pixel_ratio = box_width / w

        self.last_direction = direction
        self.last_distance = dist

        msg = Float32MultiArray()
        msg.data = [float(direction), float(dist), float(pixel_ratio)]
        self.pub.publish(msg)
        self.get_logger().info(f'[{self.mode}] 감지! 방향: {direction:.2f}, 거리: {dist:.2f}m')

    def destroy_node(self):
        self.pipeline.stop()
        super().destroy_node()

def main():
    rclpy.init()
    node = DepthDetector()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
