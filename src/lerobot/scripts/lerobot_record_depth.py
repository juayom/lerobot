#!/usr/bin/env python
"""
Depth-aware recording entrypoint.

This delegates to the standard LeRobot recording pipeline.
Use RealSense cameras with `use_depth=true` in --robot.cameras.
"""
from lerobot.scripts.lerobot_record import main


if __name__ == "__main__":
    main()
