import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist

class NavController(Node):
    def __init__(self):
        super().__init__('nav_controller')
        self.sub = self.create_subscription(Float32MultiArray, '/target_direction', self.direction_cb, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('NavController 시작!')
        self.stop_distance = 0.5
        self.linear_speed = 0.15
        self.angular_speed = 0.4

    def direction_cb(self, msg):
        direction = msg.data[0]
        distance = msg.data[1]
        twist = Twist()
        if distance < self.stop_distance:
            self.get_logger().info(f'도착! 거리: {distance:.2f}m')
            self.pub.publish(twist)
            return
        twist.linear.x = self.linear_speed
        twist.angular.z = -direction * self.angular_speed
        self.pub.publish(twist)
        self.get_logger().info(f'이동중 방향: {direction:.2f} 거리: {distance:.2f}m')

def main():
    rclpy.init()
    node = NavController()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
