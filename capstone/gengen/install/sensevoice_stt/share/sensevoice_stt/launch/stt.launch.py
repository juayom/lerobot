from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("sensevoice_stt")
    param_file = os.path.join(pkg_share, "config", "stt_params.yaml")

    return LaunchDescription([
        Node(
            package="sensevoice_stt",
            executable="stt_node",
            name="sensevoice_stt_node",
            output="screen",
            emulate_tty=True,
            parameters=[param_file],
        )
    ])
