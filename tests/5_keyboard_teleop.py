from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig
import time
import time
from dataclasses import dataclass

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

class SO100KeyboardTeleop(KeyboardTeleop):
    """Maps keyboard keys to SO100 joint positions."""
    
    # Starting position (degrees, since use_degrees=True on robot)
    JOINT_NAMES = [
        "shoulder_pan.pos",
        "shoulder_lift.pos", 
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]
    STEP = 5.0  # degrees per keypress
    
    # Key → (joint_index, direction)
    KEY_MAP = {
        "a": (0, +1), "d": (0, -1),   # shoulder_pan
        "w": (1, +1), "s": (1, -1),   # shoulder_lift
        "r": (2, +1), "f": (2, -1),   # elbow_flex
        "t": (3, +1), "g": (3, -1),   # wrist_flex
        "y": (4, +1), "h": (4, -1),   # wrist_roll
        "o": (5, +1), "l": (5, -1),   # gripper
    }

    def __init__(self, config: KeyboardTeleopConfig):
        super().__init__(config)
        self._positions = [0.0, 0.0, 0.0, 0.0, 0.0, 50.0]  # start gripper half-open

    def get_action(self) -> dict:
        self._drain_pressed_keys()
        
        active_keys = {k for k, v in self.current_pressed.items() if v}
        
        for key in active_keys:
            if key in self.KEY_MAP:
                joint_idx, direction = self.KEY_MAP[key]
                self._positions[joint_idx] += direction * self.STEP
        
        # Clip gripper to 0-100
        self._positions[5] = max(0.0, min(100.0, self._positions[5]))
        
        return {
            name: self._positions[i] 
            for i, name in enumerate(self.JOINT_NAMES)
        }
    
teleop = SO100KeyboardTeleop(KeyboardTeleopConfig)
teleop.connect()

robot = create_robot(use_degrees=True)
robot.connect()

while True:
    action = teleop.get_action()
    robot.send_action(action)
    time.sleep(1/30)