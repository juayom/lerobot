import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class NavClient(Node):
    def __init__(self):
        super().__init__('nav_client')
        self.pub = self.create_publisher(String, '/nav_goal', 10)
        self.result = None
        self.sub = self.create_subscription(String, '/nav_result', self.result_cb, 10)

    def result_cb(self, msg):
        self.result = msg.data

    def go_to(self, target_name: str) -> bool:
        self.result = None
        msg = String()
        msg.data = target_name
        self.pub.publish(msg)
        self.get_logger().info(f'nav_to: {target_name}')

        # 완료 신호 대기 (최대 60초)
        timeout = 60
        start = time.time()
        while self.result is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start > timeout:
                self.get_logger().error('Navigation timeout!')
                return False

        return self.result == 'SUCCESS'

def nav_to(target_name: str) -> bool:
    rclpy.init()
    node = NavClient()
    result = node.go_to(target_name)
    node.destroy_node()
    rclpy.shutdown()
    return result
