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
# https://www.mintlify.com/huggingface/lerobot/tutorials/evaluate-policies


# Config
ROBOT_PORT = "/dev/ttyACM0"
MAIN_CAMERA_PATH = "/dev/video2"
SECONDARY_CAMERA_PATH = "/dev/video4"
DATASET_ID = "Sri-Ram-A/so100_keyboard_teleop_4"
MODEL_ID = "Sri-Ram-A/so100_keyboard_teleop_4_act"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load dataset metadata and create pre/post‑processors
metadata = LeRobotDatasetMetadata(DATASET_ID)
pprint(metadata)

# Access a single frame
dataset = LeRobotDataset(DATASET_ID)
frame = dataset[0]
print(f"Action shape: {frame['action'].shape}")
print(f"State shape: {frame['observation.state'].shape}")

#  Load trained policy from Hugging Face Hub
policy = ACTPolicy.from_pretrained(MODEL_ID)
policy.to(device)
policy.eval()
print("Policy loaded.")


# Access camera images (if available)
if metadata.camera_keys:
    print(f"Camera keys: {metadata.camera_keys}")
    camera_key = metadata.camera_keys[0]
    print(f"Image shape: {frame[camera_key].shape}")  # (C, H, W) in PyTorch format
# Camera keys: ['observation.images.main', 'observation.images.secondary_0']
# Image shape: torch.Size([3, 480, 640])

preprocessor, postprocessor = make_pre_post_processors(
    policy.config, dataset_stats=metadata.stats
)
print("Preprocessor:", preprocessor)
print("Postprocessor:", postprocessor)
# Preprocessor: DataProcessorPipeline(name='policy_preprocessor', steps=4: [RenameObservationsProcessorStep, AddBatchDimensionProcessorStep, ..., NormalizerProcessorStep])
# Postprocessor: DataProcessorPipeline(name='policy_postprocessor', steps=2: [UnnormalizerProcessorStep, DeviceProcessorStep])


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
# def rad2deg(rad):
#     return rad * 180.0 / np.pi

def adapt_observation_to_dataset(obs):
    return {
        # EXACT names required by dataset
        "shoulder_pan":   float(obs["shoulder_pan.pos"]),
        "shoulder_lift":  float(obs["shoulder_lift.pos"]),
        "elbow_flex":     float(obs["elbow_flex.pos"]),
        "wrist_flex":     float(obs["wrist_flex.pos"]),
        "wrist_roll":     float(obs["wrist_roll.pos"]),
        "gripper":        float(obs["gripper.pos"]),

        # Images (these are correct already)
        "main": obs["main"],
        "secondary_0": obs["secondary_0"],
    }

def action_to_robot_format(action_robot):
    return {
        "shoulder_pan.pos": action_robot["shoulder_pan"],
        "shoulder_lift.pos": action_robot["shoulder_lift"],
        "elbow_flex.pos": action_robot["elbow_flex"],
        "wrist_flex.pos": action_robot["wrist_flex"],
        "wrist_roll.pos": action_robot["wrist_roll"],
        "gripper.pos": action_robot["gripper"],
    }

max_steps = True  # safety limit, set to True for infinite loop
step = 0

while max_steps is True or step < max_steps:
    # 1. Get observation from robot (includes images + joint state)
    obs = robot.get_observation()
    obs = adapt_observation_to_dataset(obs)
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
    robot.send_action(action_final)
    
    step += 1
    print(f"Step {step} - action (deg): {action_robot}")

pprint(metadata.features)
print(metadata.stats["action"])