# - Created using Deepseek and https://www.mintlify.com/huggingface/lerobot/quickstart
import torch
import numpy as np
from pprint import pprint
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.configuration_opencv import Cv2Backends
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pprint import pprint

# Config
ROBOT_PORT = "/dev/ttyACM0"
MAIN_CAMERA_PATH = "/dev/video4"
SECONDARY_CAMERA_PATH = "/dev/video2"
DATASET_ID = "Sri-Ram-A/pnp1"
MODEL_ID = "Sri-Ram-A/act_pnp1"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load dataset metadata and create pre/post‑processors
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
    print(f"Image shape: {frame[camera_key].shape}")  # (C, H, W) in PyTorch format
    # Camera keys: ['observation.images.main', 'observation.images.secondary_0']
    # Image shape: torch.Size([3, 240, 320])


#  Load trained policy from Hugging Face Hub
policy = ACTPolicy.from_pretrained(MODEL_ID)
policy.to(device)
policy.eval()
print("Policy loaded.")


preprocessor, postprocessor = make_pre_post_processors(
    policy.config, dataset_stats=metadata.stats
)
print("Preprocessor:", preprocessor)
print("Postprocessor:", postprocessor)

# Configure cameras and robot - look into dataset to get camera names
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

robot_cfg = SO100FollowerConfig(
    port=ROBOT_PORT,
    id="so100_follower",
    use_degrees=True, # my robot expects degrees
    disable_torque_on_disconnect=True,
    cameras=camera_config
)

robot = SO100Follower(robot_cfg)
print("Robot configured.")
robot.connect()

print("Robot connected!")
# Real‑time inference loop
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
    # Flattened motors at top-level (THIS is the key fix)
    for i, joint in enumerate(joint_order):
        adapted[f"motor_{i+1}"] = float(obs[joint])
    # Images (these remain mapped via prefix stripping)
    adapted["main"] = obs["main"]
    adapted["secondary_0"] = obs["secondary_0"]
    return adapted

def action_to_robot_format(action_robot):
    # map motor_i → actual joints
    return {
        "shoulder_pan.pos": rad2deg(action_robot["motor_1"]),
        "shoulder_lift.pos": rad2deg(action_robot["motor_2"]),
        "elbow_flex.pos": rad2deg(action_robot["motor_3"]),
        "wrist_flex.pos": rad2deg(action_robot["motor_4"]),
        "wrist_roll.pos": rad2deg(action_robot["motor_5"]),
        "gripper.pos": rad2deg(action_robot["motor_6"]),
    }

max_steps = True  # safety limit, set to True for infinite loop
step = 0

while max_steps is True or step < max_steps:
    # 1. Get observation from robot (includes images + joint state)
    raw_obs = robot.get_observation()
    obs = adapt_observation_to_dataset(raw_obs)
    # pprint(obs)
    # 2. Build frame matching dataset format
    obs_frame = build_inference_frame(
        observation=obs,
        ds_features=metadata.features,
        device=device
    )
    # pprint(obs_frame)
    
    # 3. Preprocess (normalise)
    obs_frame = preprocessor(obs_frame)
    
    # 4. Run policy inference
    with torch.no_grad():
        action_pred = policy.select_action(obs_frame)   # shape (6,)
    
    # 5. Postprocess (denormalise) – still in radians
    action_post = postprocessor(action_pred)
    
    # # 6. Convert radians → degrees for your robot
    # action_deg = rad2deg(action_rad.cpu().numpy())
    
    # # 7. Create action dict expected by robot
    # action_dict = {
    #     "shoulder_pan.pos": action_deg[0],
    #     "shoulder_lift.pos": action_deg[1],
    #     "elbow_flex.pos": action_deg[2],
    #     "wrist_flex.pos": action_deg[3],
    #     "wrist_roll.pos": action_deg[4],
    #     "gripper.pos": action_deg[5],
    # }
    action_robot = make_robot_action(action_post, metadata.features)
    action_final = action_to_robot_format(action_robot)
    # 8. Send to robot
    robot.send_action(action_final)
    
    step += 1
    print(f"Step {step} - action (deg): {action_robot}")
