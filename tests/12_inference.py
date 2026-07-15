# - Created using Deepseek and https://www.mintlify.com/huggingface/lerobot/quickstart
import torch
from pprint import pprint
from pathlib import Path
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.configuration_opencv import Cv2Backends
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import time
from lerobot.cameras.opencv.configuration_opencv import ColorMode  # add this import
# https://www.mintlify.com/huggingface/lerobot/tutorials/evaluate-policies


# Config
ROBOT_PORT = "/dev/ttyACM0"
MAIN_CAMERA_PATH = Path("/dev/video2")
SECONDARY_CAMERA_PATH = Path("/dev/video4")
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
        backend=Cv2Backends.V4L2,
        color_mode=ColorMode.BGR,  # ← match what was recorded
    ),
    "secondary_0": OpenCVCameraConfig(
        index_or_path=SECONDARY_CAMERA_PATH,
        width=camera_width,
        height=camera_height,
        fps=fps,
        backend=Cv2Backends.V4L2,
        color_mode=ColorMode.BGR,  # ← match what was recorded
    ),
}

robot_cfg = SO100FollowerConfig(
    port=ROBOT_PORT,
    id="so100_follower",
    use_degrees=True,  # my robot expects degrees
    disable_torque_on_disconnect=True,
    cameras=camera_config,
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
        "shoulder_pan": float(obs["shoulder_pan.pos"]),
        "shoulder_lift": float(obs["shoulder_lift.pos"]),
        "elbow_flex": float(obs["elbow_flex.pos"]),
        "wrist_flex": float(obs["wrist_flex.pos"]),
        "wrist_roll": float(obs["wrist_roll.pos"]),
        "gripper": float(obs["gripper.pos"]),
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


while True:
    # for i in range(1):
    robot_observation = robot.get_observation()
    adapted_observation = adapt_observation_to_dataset(robot_observation)
    inference_frame = build_inference_frame(
        adapted_observation, device=device, ds_features=metadata.features
    )
    inference_frame['task'] = 'teleop' # same as below line
    # inference_frame['task'] = list(metadata.tasks.index)[0]
    preprocessed_inference_frame = preprocessor(inference_frame)
    with torch.no_grad():
        predicted_action = policy.select_action(preprocessed_inference_frame)
    postprocessed_action = postprocessor(predicted_action)
    robot_action = make_robot_action(postprocessed_action, metadata.features)
    formatted_action = action_to_robot_format(robot_action)
    robot.send_action(formatted_action)
    time.sleep(1 / 30)
