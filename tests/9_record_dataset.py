# # SO100 Keyboard Teleoperation with Dataset Recording
# Run each cell sequentially.

# 1. Imports and Configuratio
import signal
import sys
import time
from pathlib import Path
import cv2
from loguru import logger
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig
from lerobot.datasets import LeRobotDataset
import numpy as np

# Configuration
PORT = "/dev/ttyACM0"
MAIN_CAMERA_DEVICE = "/dev/video2"          
SECOND_CAMERA_DEVICE = "/dev/video4"          
FPS = 30
TASK = "teleop"
DATASET_NAME = "so100_keyboard_teleop_4"
DATASET_REPO_ID = f"Sri-Ram-A/{DATASET_NAME}"  # Change as needed
DATASET_ROOT = Path("/home/srirama/Documents/sr_proj/RoboticArm/datasets") 
DATASET_PATH = DATASET_ROOT / DATASET_NAME
# >>> dataset = LeRobotDataset(
# ...     repo_id=DATASET_REPO_ID,
# ...     root=DATASET_PATH,
# ...     # Don't use .create() - load existing instead
# ... )
# >>> 
KEY_START_REC = "c"    # start recording
KEY_STOP_REC  = "v"    # stop and save episode
KEY_QUIT      = "x"    # quit program
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
PUSH_INTERVAL = 5
# 2. Robot and Teleop Classes (minimal)
class SO100KeyboardTeleop(KeyboardTeleop):
    JOINT_NAMES = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]
    STEP = 2.0
    DEFAULT_POS = [0.0, 0.0, 0.0, 0.0, 0.0, 50.0]
    KEY_MAP = {
        "q": (0, -1), "a": (0, +1),   # shoulder_pan
        "w": (1, -1), "s": (1, +1),   # shoulder_lift
        "e": (2, -1), "d": (2, +1),   # elbow_flex
        "r": (3, -1), "f": (3, +1),   # wrist_flex
        "t": (4, -1), "g": (4, +1),   # wrist_roll
        "y": (5, -1), "h": (5, +1),   # gripper
    }

    def __init__(self, config):
        super().__init__(config)
        self._positions = self.DEFAULT_POS.copy()

    def get_action(self):
        self._drain_pressed_keys()
        active = {k for k, v in self.current_pressed.items() if v}
        for key in active:
            if key in self.KEY_MAP:
                idx, direc = self.KEY_MAP[key]
                self._positions[idx] += direc * self.STEP
        self._positions[5] = max(0.0, min(100.0, self._positions[5]))
        return {name: self._positions[i] for i, name in enumerate(self.JOINT_NAMES)}


def create_robot(use_degrees=True):
    config = SO100FollowerConfig(
        port=PORT,
        id="so100_follower",
        use_degrees=use_degrees,
        disable_torque_on_disconnect=True,
        cameras={},
    )
    return SO100Follower(config)


def get_robot_state(robot):
    """Read actual joint positions. Replace with robot.read_state() if needed."""
    try:
        return robot.get_state()  # May need adjustment for your robot
    except AttributeError:
        logger.warning("robot.get_state() not available – using zeros")
        return {name: 0.0 for name in SO100KeyboardTeleop.JOINT_NAMES}


# 3. Connect Hardware
# Connect teleop
teleop = SO100KeyboardTeleop(KeyboardTeleopConfig())
teleop.connect()
logger.info("Keyboard teleop connected")

# Connect robot
robot = create_robot(use_degrees=True)
robot.connect()
logger.info(f"Robot connected on {PORT}")
print(robot.get_observation().keys())

# Open camera
cap_main = cv2.VideoCapture(MAIN_CAMERA_DEVICE)
cap_second = cv2.VideoCapture(SECOND_CAMERA_DEVICE)

if not cap_main.isOpened():
    logger.error(f"Could not open camera {MAIN_CAMERA_DEVICE}")
    sys.exit(1)
if not cap_second.isOpened():
    logger.error(f"Could not open camera {SECOND_CAMERA_DEVICE}")
    sys.exit(1)
logger.info(f"Camera {MAIN_CAMERA_DEVICE} opened")
logger.info(f"Camera {SECOND_CAMERA_DEVICE} opened")
# # Create OpenCV window
# cv2.namedWindow("CameraPreview", cv2.WINDOW_NORMAL)


# 4. Create or Load LeRobotDatase
# Define features structure
features = {
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": MOTOR_NAMES,
    },
    "observation.images.main": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.secondary_0": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": MOTOR_NAMES,
    },
}

# Dataset creation (unchanged)
logger.info("Creating new dataset")
dataset = LeRobotDataset.create(
    repo_id=DATASET_REPO_ID,
    root=DATASET_PATH,   
    fps=FPS,
    robot_type="so_follower",
    features=features,
    use_videos=True,
    vcodec= "libsvtav1",
)
logger.info(f"Dataset created at {dataset.root}")


# Record the dataset
recording = False
episode_index = 0 or dataset.meta.total_episodes
dt = 1.0 / FPS

logger.info("Controls:")
logger.info(f"  Start recording: '{KEY_START_REC}'")
logger.info(f"  Stop & save episode: '{KEY_STOP_REC}'")
logger.info(f"  Quit: '{KEY_QUIT}'")

try:
    while True:
        loop_start = time.perf_counter()

        # Get action from keyboard
        action_dict = teleop.get_action()
        robot.send_action(action_dict)

        # Read observation from robot
        raw_obs = robot.get_observation()   # returns dict with keys like "shoulder_pan.pos"
        state_vec = np.array([raw_obs[f"{name}.pos"] for name in MOTOR_NAMES], dtype=np.float32)

        ret1, frame1 = cap_main.read()
        ret2, frame2 = cap_second.read()
        if not ret1:
            logger.warning(f"Failed to read from camera {MAIN_CAMERA_DEVICE} – using blank frame")
            frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        if not ret2:
            logger.warning(f"Failed to read from camera {SECOND_CAMERA_DEVICE} – using blank frame")
            frame2 = np.zeros((480, 640, 3), dtype=np.uint8)

        # Show camera preview with recording indicator
        combined = np.vstack((frame1, frame2))
        if recording:
            cv2.putText(combined, "RECORDING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2)
        cv2.imshow("Camera Preview", combined)
        cv2.waitKey(1)   # needed to update window

        # Check control keys (non‑movement)
        pressed = {k for k, v in teleop.current_pressed.items() if v}
        
        # Quit
        if KEY_QUIT in pressed:
            logger.info("Quit key pressed – exiting")
            break

        # Start recording
        if KEY_START_REC in pressed and not recording:
            logger.info(f"Started recording episode {episode_index}")
            recording = True

        if recording:
            # Build action vector (ordered)
            action_vec = np.array([action_dict[f"{name}.pos"] for name in MOTOR_NAMES], dtype=np.float32)
            frame_data = {
                "observation.state": state_vec,
                "observation.images.main": frame1,
                "observation.images.secondary_0": frame2,
                "action": action_vec,
                "task": TASK,
            }
            dataset.add_frame(frame_data)

        # Stop recording 
        if KEY_STOP_REC in pressed and recording:
            logger.info(f"Stopping episode {episode_index} – saving...")
            # Push to Hugging Face Hub after each episode (optional)
            dataset.save_episode()
            # dataset.push_to_hub()
            logger.success(f"Episode {episode_index} saved and pushed to Hub")
            episode_index += 1
            recording = False
            if episode_index % PUSH_INTERVAL == 0:
                logger.info("Pushing batch to hub...")
                dataset.push_to_hub(commit_message=f"Added {PUSH_INTERVAL} episodes (episode {episode_index})")

        # Maintain FPS
        elapsed = time.perf_counter() - loop_start
        time.sleep(max(0, dt - elapsed))

except KeyboardInterrupt:
    logger.info("Interrupted by user")

finally:
    if recording:
        logger.warning("Incomplete episode discarded (not saved)")
    # Finalize dataset and push any remaining data
    dataset.finalize()
    dataset.push_to_hub()   # final push
    logger.info("Dataset finalized and pushed to Hugging Face Hub")

    cap_main.release()
    cap_second.release()
    cv2.destroyAllWindows()
    robot.disconnect()
    teleop.disconnect()
    logger.info("Disconnected")