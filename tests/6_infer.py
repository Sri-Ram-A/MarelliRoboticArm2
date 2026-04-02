# - Created using Deepseek and https://www.mintlify.com/huggingface/lerobot/quickstart
import torch
import numpy as np
import time
from pprint import pprint
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.configuration_opencv import Cv2Backends
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Config
ROBOT_PORT = "/dev/ttyACM1"
MAIN_CAMERA_PATH = "/dev/video4"
SECONDARY_CAMERA_PATH = "/dev/video2"
DATASET_ID = "Sri-Ram-A/pnp1"
MODEL_ID = "Sri-Ram-A/act_pnp1"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset metadata and create pre/post-processors
dataset_revision = "v3.0"
metadata = LeRobotDatasetMetadata(DATASET_ID, revision=dataset_revision)
pprint(metadata)

# Access a single frame
dataset = LeRobotDataset(DATASET_ID)
frame = dataset[0]
print(f"Action shape: {frame['action'].shape}")
print(f"State shape: {frame['observation.state'].shape}")

# Access camera images (if available)
if metadata.camera_keys:
    print(f"Camera keys: {metadata.camera_keys}")
    camera_key = metadata.camera_keys[0]
    print(f"Image shape: {frame[camera_key].shape}")

# Load trained policy from Hugging Face Hub
policy = ACTPolicy.from_pretrained(MODEL_ID)
policy.to(device)
policy.eval()
print("Policy loaded.")

preprocessor, postprocessor = make_pre_post_processors(
    policy.config, dataset_stats=metadata.stats
)
print("Preprocessor:", preprocessor)
print("Postprocessor:", postprocessor)

# Configure cameras and robot
camera_width = 640
camera_height = 480
fps = 30
camera_config = {
    "main": OpenCVCameraConfig(
        index_or_path=MAIN_CAMERA_PATH,
        width=camera_width,
        height=camera_height,
        fps=fps,
        backend=Cv2Backends.V4L2
    ),
    "secondary_0": OpenCVCameraConfig(
        index_or_path=SECONDARY_CAMERA_PATH,
        width=camera_width,
        height=camera_height,
        fps=fps,
        backend=Cv2Backends.V4L2
    ),
}

# Configure robot
robot_cfg = SO100FollowerConfig(
    port=ROBOT_PORT,
    id="so100_follower",
    use_degrees=True,
    disable_torque_on_disconnect=True,
    cameras=camera_config
)
robot = SO100Follower(robot_cfg)
print("Robot configured.")
robot.connect()
print("Robot connected!")

# Real-time inference loop
def rad2deg(rad):
    return rad * 180.0 / np.pi

def adapt_observation_to_dataset(obs):
    # Joint mapping
    joint_order = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]
    adapted = {}
    for i, joint in enumerate(joint_order):
        adapted[f"motor_{i+1}"] = float(obs[joint])
    adapted["main"] = obs["main"]
    adapted["secondary_0"] = obs["secondary_0"]
    return adapted

def action_to_robot_format(action_robot):
    return {
        "shoulder_pan.pos": rad2deg(action_robot["motor_1"]),
        "shoulder_lift.pos": rad2deg(action_robot["motor_2"]),
        "elbow_flex.pos": rad2deg(action_robot["motor_3"]),
        "wrist_flex.pos": rad2deg(action_robot["motor_4"]),
        "wrist_roll.pos": rad2deg(action_robot["motor_5"]),
        "gripper.pos": rad2deg(action_robot["motor_6"]),
    }

class ActionSmoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.prev_action = None
    
    def smooth(self, new_action):
        if self.prev_action is None:
            self.prev_action = new_action.copy()
            return new_action
        smoothed = {}
        for key in new_action.keys():
            smoothed[key] = self.alpha * new_action[key] + (1 - self.alpha) * self.prev_action[key]
        self.prev_action = smoothed.copy()
        return smoothed

action_smoother = ActionSmoother(alpha=0.4)

max_steps = 1000
step = 0

while step < max_steps:
    # 1. Get observation from robot
    raw_obs = robot.get_observation()
    obs = adapt_observation_to_dataset(raw_obs)
    
    # 2. Build frame matching dataset format
    obs_frame = build_inference_frame(
        observation=obs,
        ds_features=metadata.features,
        device=device
    )
    
    # 3. Preprocess (normalise)
    obs_frame = preprocessor(obs_frame)
    
    # 4. Run policy inference - returns (batch_size, action_dim)
    with torch.no_grad():
        action_pred = policy.select_action(obs_frame)
    
    # 5. Postprocess (denormalise) - still in radians
    action_post = postprocessor(action_pred)
    
    # 6. Convert to robot action dict
    action_robot = make_robot_action(action_post, metadata.features)
    
    # 7. Convert to degrees and format for robot
    action_final = action_to_robot_format(action_robot)
    
    # 8. Apply smoothing to reduce jitter
    action_final = action_smoother.smooth(action_final)
    
    # 9. Send to robot
    robot.send_action(action_final)
    
    step += 1
    if step % 10 == 0:
        print(f"Step {step} - action: {action_final}")
    
    # Small delay to control rate
    time.sleep(0.02)
    
print("Inference loop completed")