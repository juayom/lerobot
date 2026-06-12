# Capstone Runtime

This directory is reserved for the capstone application that runs alongside
LeRobot.

The deployed system is split across separate runtime workspaces:

- `~/Desktop/serial_bridge.py` handles the Arduino serial bridge.
- `~/ros2_ws` provides the `depth_nav` detector and robot FSM.
- `capstone/gengen` provides SenseVoice STT and the dialog controller.
- `src/lerobot` and `local_policies` provide robot policy inference.

Do not commit generated frames, ROS build products, virtual environments,
dated backups, or copied source snapshots here. Git history is the backup for
tracked source code.
