import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
import math

WAYPOINTS = {
    "medicine_table": (0.5, 0.5, 0.0),
    "person_position": (0.5, 1.0, 0.0),
    "home": (0.1, 0.1, 0.0),
}

class Nav2Client(Node):
    def __init__(self):
        super().__init__('nav2_client')
        self._client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

    def go_to(self, target_name: str) -> bool:
        if target_name not in WAYPOINTS:
            self.get_logger().error(f"Unknown target: {target_name}")
            return False

        x, y, yaw = WAYPOINTS[target_name]
        self.get_logger().info(f"Moving to {target_name} ({x}, {y})")
        self._client.wait_for_server()

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.pose.orientation.w = math.cos(yaw / 2)

        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected!")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        self.get_logger().info(f"Reached {target_name}!")
        return True


def nav_to(target_name: str) -> bool:
    rclpy.init()
    node = Nav2Client()
    result = node.go_to(target_name)
    node.destroy_node()
    rclpy.shutdown()
    return result
