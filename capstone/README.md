# Capstone Runtime

The deployed application is split across these components:

- `gengen/src/sensevoice_stt` provides STT and the dialog controller.
- `robot_actions` provides the scripted handover motion.
- `../ros2_ws/src/depth_nav` provides depth detection and the robot FSM.
- `../src/lerobot` and `../local_policies` provide policy inference.

Build the ROS 2 workspaces locally with `colcon build`. Do not commit generated
`build/`, `install/`, `log/`, runtime frames, dated backups, or copied source
snapshots.
