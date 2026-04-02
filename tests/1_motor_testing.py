import time
from dataclasses import dataclass
import sys
import termios
import tty
import math
from enum import Enum
from loguru import logger
import draccus
from pprint import pprint
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig


PORT = "/dev/ttyACM0"

def create_robot(use_degrees: bool) -> SO100Follower:
    config = SO100FollowerConfig(
        port=PORT,
        id="so100_follower",
        use_degrees=use_degrees,
        disable_torque_on_disconnect=True,
        cameras={},
    )
    return SO100Follower(config)

def log_position(robot: SO100Follower):
    obs = robot.get_observation()
    logger.info("get_observation() ->")
    pprint(obs)

def wait_for_q():
    logger.info("Press 'q' to exit...")
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch.lower() == 'q':
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def test_encoder():
    robot = create_robot(use_degrees=False)
    try:
        robot.connect()
        logger.success("connect() -> success")
        log_position(robot)
        action = {
            "shoulder_pan.pos": 2077,
            "shoulder_lift.pos": 1821,
            "elbow_flex.pos": 2186,
            "wrist_flex.pos": 1991,
            "wrist_roll.pos": 2047,
            "gripper.pos": 2794,
        }
        result = robot.send_action(action)
        logger.info(f"send_action(encoder) -> ")
        pprint(result)
        wait_for_q()
    finally:
        robot.disconnect()
        logger.info("disconnect() -> success")

def test_degrees():
    robot = create_robot(use_degrees=True)
    try:
        robot.connect()
        logger.success("connect() -> success")
        log_position(robot)
        action = {
            "shoulder_pan.pos": -90.0,      # facing forward - base
            "shoulder_lift.pos": 0.0,    # raise arm up - above base 
            "elbow_flex.pos":0.0,       # bend elbow to form L
            "wrist_flex.pos": 0.0,        # keep neutral
            "wrist_roll.pos": 0.0,        # no rotation
            "gripper.pos": 0.0,           # open/neutral
        }
        result = robot.send_action(action)
        logger.info(f"send_action(degrees) -> {result}")
        wait_for_q()
    finally:
        robot.disconnect()
        logger.info("disconnect() -> success")

def test_radians():
    robot = create_robot(use_degrees=False)
    try:
        robot.connect()
        logger.info("connect() -> success")
        log_position(robot)
        action = {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": math.pi / 4,   # 45°
            "elbow_flex.pos": math.pi / 2,      # 90°
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 0.0,
        }
        result = robot.send_action(action)
        logger.info(f"send_action(radians) -> {result}")
        wait_for_q()
    finally:
        robot.disconnect()
        logger.info("disconnect() -> success")

class TestType(str, Enum):
    encoder = "encoder"
    degrees = "degrees"
    radians = "radians"

@dataclass
class Args:
    test: TestType

@draccus.wrap()
def main(args: Args):
    logger.info(f"Running test: {args.test}")
    if args.test == "encoder":
        test_encoder()
    elif args.test == "degrees":
        test_degrees()
    elif args.test == "radians":
        test_radians()

if __name__ == "__main__":
    main()

# python file.py --test encoder
# python file.py --test degrees
# python file.py --test radians