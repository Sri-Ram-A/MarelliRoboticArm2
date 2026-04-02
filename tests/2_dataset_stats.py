"""
Step 1: Basic Dataset Exploration
Understanding the structure of Sri-Ram-A/pnp1 dataset
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download, hf_hub_download
from pathlib import Path
from pprint import pprint

import torch
# Set your dataset info
DATASET_ID = "Sri-Ram-A/pnp1"
HF_TOKEN = os.getenv("HF_TOKEN")

# Download dataset metadata first (without downloading all videos)
print("STEP 1: Understanding LeRobot Dataset Structure")

# 1.1 Check dataset info.json
info_path = hf_hub_download(repo_id=DATASET_ID,filename="meta/info.json",repo_type="dataset",token=HF_TOKEN)
# '/home/srirama/hub/datasets--Sri-Ram-A--pnp1/snapshots/ed682a0b302da28ad60a10f1dc5968415cd50d9c/meta/info.json'
with open(info_path, 'r') as f:
    info = json.load(f)
pprint(info)
CACHE_DIR = Path("/.cache/huggingface/lerobot").expanduser()
# {'chunks_size': 1000,
#  'codebase_version': 'v3.0',
#  'data_files_size_in_mb': 100,
#  'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
#  'features': {'action': {'dtype': 'float32',
#                          'fps': 30,
#                          'names': ['motor_1',
#                                    'motor_2',
#                                    'motor_3',
#                                    'motor_4',
#                                    'motor_5',
#                                    'motor_6'],
#                          'shape': [6]},
#               'episode_index': {'dtype': 'int64',
#                                 'fps': 30,
#                                 'names': None,
#                                 'shape': [1]},
#               'frame_index': {'dtype': 'int64',
#                               'fps': 30,
#                               'names': None,
#                               'shape': [1]},
#               'index': {'dtype': 'int64',
#                         'fps': 30,
#                         'names': None,
#                         'shape': [1]},
#               'observation.images.main': {'dtype': 'video',
#                                           'info': {'has_audio': False,
#                                                    'video.codec': 'avc1',
#                                                    'video.fps': 30,
#                                                    'video.is_depth_map': False,
#                                                    'video.pix_fmt': 'yuv420p'},
#                                           'names': ['height',
#                                                     'width',
#                                                     'channel'],
#                                           'shape': [240, 320, 3]},
#               'observation.images.secondary_0': {'dtype': 'video',
#                                                  'info': {'has_audio': False,
#                                                           'video.codec': 'avc1',
#                                                           'video.fps': 30,
#                                                           'video.is_depth_map': False,
#                                                           'video.pix_fmt': 'yuv420p'},
#                                                  'names': ['height',
#                                                            'width',
#                                                            'channel'],
#                                                  'shape': [240, 320, 3]},
#               'observation.state': {'dtype': 'float32',
#                                     'fps': 30,
#                                     'names': ['motor_1',
#                                               'motor_2',
#                                               'motor_3',
#                                               'motor_4',
#                                               'motor_5',
#                                               'motor_6'],
#                                     'shape': [6]},
#               'task_index': {'dtype': 'int64',
#                              'fps': 30,
#                              'names': None,
#                              'shape': [1]},
#               'timestamp': {'dtype': 'float32',
#                             'fps': 30,
#                             'names': None,
#                             'shape': [1]}},
#  'fps': 30,
#  'robot_type': 'so-100',
#  'splits': {'train': '0:43'},
#  'total_episodes': 43,
#  'total_frames': 30441,
#  'total_tasks': 1,
#  'video_files_size_in_mb': 200,
#  'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'}


"""
Step 2: Load Dataset Using LeRobot's Dataset Class
This is the recommended way to work with LeRobot datasets
"""

print("STEP 2: Loading Dataset with LeRobot's Dataset Class")
from lerobot.datasets.lerobot_dataset import LeRobotDataset

print("\n2.1 Loading dataset (this may take a moment)...")
dataset = LeRobotDataset(
    repo_id=DATASET_ID,
    revision="main",
    root="/home/srirama/.cache/huggingface/lerobot",  # Cache location
    episodes=[0, 1, 2]  # Load first 3 episodes for testing
)
pprint(dataset)
print(f"\n✅ Dataset loaded successfully!")
print(f"  - Total episodes: {dataset.num_episodes}")
print(f"  - Total frames: {len(dataset)}")
print(f"  - Features: {list(dataset.features.keys())}")

# Get first sample
sample = dataset[0]
print(f"\n📦 First sample structure:")
for key in sample.keys():
    if isinstance(sample[key], dict):
        print(f"  - {key}: dict with keys {list(sample[key].keys())}")
    elif hasattr(sample[key], 'shape'):
        print(f"  - {key}: shape {sample[key].shape}, dtype {sample[key].dtype}")
    else:
        print(f"  - {key}: {type(sample[key])}")


"""
Step 3: Deep Dive into Action and Observation Data
Understanding what's actually stored in the dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tabulate import tabulate
from pprint import pprint
from lerobot.datasets.lerobot_dataset import LeRobotDataset

print("STEP 3: Analyzing Action and Observation Data")

# Dataset Indexing man
dataset = LeRobotDataset(
    repo_id=DATASET_ID,
    revision="main",
    root=CACHE_DIR
)

first_episode_idx = 0
last_episode_idx = info["total_frames"]-1

# Get all actions from every frame
actions = []
for i in range(first_episode_idx, last_episode_idx):
    sample = dataset[i]
    actions.append(sample["action"].numpy())

actions = np.array(actions)
print(f"\nActions shape: {actions.shape}")

action_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
action_stats = []
for i, name in enumerate(action_names):
    joint_actions = actions[:, i]
    action_stats.append([
        name,
        f"{joint_actions.min():6.4f}",
        f"{joint_actions.max():6.4f}",
        f"{joint_actions.mean():6.4f}",
        f"{joint_actions.std():6.4f}"
    ])
print("\nAction Statistics (per joint):")
print(tabulate(action_stats, headers=["Joint", "Min", "Max", "Mean", "Std"], tablefmt="grid"))
# +---------------+---------+--------+---------+--------+
# | Joint         |     Min |    Max |    Mean |    Std |
# +===============+=========+========+=========+========+
# | shoulder_pan  | -1.2927 | 1.1498 | -0.0961 | 0.704  |
# +---------------+---------+--------+---------+--------+
# | shoulder_lift | -0.7942 | 0.414  | -0.0598 | 0.236  |
# +---------------+---------+--------+---------+--------+
# | elbow_flex    | -0.5145 | 1.029  |  0.2506 | 0.3395 |
# +---------------+---------+--------+---------+--------+
# | wrist_flex    | -0.5346 | 1.5349 |  0.272  | 0.4416 |
# +---------------+---------+--------+---------+--------+
# | wrist_roll    | -1.4604 | 0.6011 | -0.111  | 0.5587 |
# +---------------+---------+--------+---------+--------+
# | gripper       | -0.0001 | 1.5001 |  1.1882 | 0.6086 |
# +---------------+---------+--------+---------+--------+

# Get All observations from every frame
observations = []
for i in range(first_episode_idx, last_episode_idx):
    sample = dataset[i]
    observations.append(sample["observation.state"].numpy())

observations = np.array(observations)
print(f"\nObservations shape: {observations.shape}")

obs_stats = []
for i, name in enumerate(action_names):
    joint_obs = observations[:, i]
    obs_stats.append([
        name,
        f"{joint_obs.min():6.4f}",
        f"{joint_obs.max():6.4f}",
        f"{joint_obs.mean():6.4f}",
        f"{joint_obs.std():6.4f}"
    ])
print("\nObservation Statistics (per joint):")
print(tabulate(obs_stats, headers=["Joint", "Min", "Max", "Mean", "Std"], tablefmt="grid"))
# +---------------+---------+--------+---------+--------+
# | Joint         |     Min |    Max |    Mean |    Std |
# +===============+=========+========+=========+========+
# | shoulder_pan  | -1.2643 | 1.1308 | -0.0899 | 0.6981 |
# +---------------+---------+--------+---------+--------+
# | shoulder_lift | -0.7365 | 0.4373 | -0.0049 | 0.2386 |
# +---------------+---------+--------+---------+--------+
# | elbow_flex    | -0.4312 | 1.0388 |  0.2888 | 0.3319 |
# +---------------+---------+--------+---------+--------+
# | wrist_flex    | -0.5063 | 1.5205 |  0.2784 | 0.4393 |
# +---------------+---------+--------+---------+--------+
# | wrist_roll    | -1.4592 | 0.603  | -0.1125 | 0.5594 |
# +---------------+---------+--------+---------+--------+
# | gripper       | -0      | 2.0929 |  1.7492 | 0.6769 |
# +---------------+---------+--------+---------+--------+

action_range = actions[:, :5]
max_abs = np.abs(action_range).max()

print("\nDetermining Action Units:")
if max_abs < 3.2:
    print(f"  Actions are in RADIANS (max absolute: {max_abs:.2f})")
elif max_abs > 100:
    print(f"  Actions are in DEGREES (max absolute: {max_abs:.1f})")
else:
    print(f"  Action range ambiguous: {max_abs:.2f}")

# Actions are in RADIANS (max absolute: 1.53)


print("STEP 4: Analyzing Camera Data")
camera_keys = [key for key in sample.keys() if key.startswith('observation.images.')]
print(f"\nCamera keys found: {camera_keys}")
# Camera keys found: ['observation.images.main', 'observation.images.secondary_0']

camera_info = []
for cam_key in camera_keys:
    img = sample[cam_key]
    is_normalized = "YES" if img.max() <= 1.0 else "NO"
    camera_info.append([
        cam_key.replace('observation.images.', ''),
        str(img.shape),
        str(img.dtype),
        f"[{img.min():.3f}, {img.max():.3f}]",
        is_normalized
    ])
print("\nCamera Analysis:")
print(tabulate(camera_info, headers=["Camera", "Shape", "Dtype", "Range", "Normalized"], tablefmt="grid"))
# +-------------+---------------------------+---------------+----------------+--------------+
# | Camera      | Shape                     | Dtype         | Range          | Normalized   |
# +=============+===========================+===============+================+==============+
# | main        | torch.Size([3, 240, 320]) | torch.float32 | [0.000, 0.824] | YES          |
# +-------------+---------------------------+---------------+----------------+--------------+
# | secondary_0 | torch.Size([3, 240, 320]) | torch.float32 | [0.000, 1.000] | YES          |
# +-------------+---------------------------+---------------+----------------+--------------+

fig, axes = plt.subplots(1, len(camera_keys), figsize=(15, 5))
if len(camera_keys) == 1:
    axes = [axes]
for i, cam_key in enumerate(camera_keys):
    img = sample[cam_key]
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    # Convert CHW → HWC
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
    # Normalize + type cast
    if img.max() <= 1.0:
        img_display = (img * 255).astype(np.uint8)
    else:
        img_display = img.astype(np.uint8)

    axes[i].imshow(img_display)
    axes[i].set_title(cam_key.replace('observation.images.', ''))
    axes[i].axis('off')
plt.tight_layout()
plt.savefig('dataset_sample_images.png', dpi=150)
print("\nSample images saved to dataset_sample_images.png")

print("STEP 5: Episode Quality Analysis")
action_diffs = np.diff(actions, axis=0)
action_jerk = np.diff(action_diffs, axis=0)

smoothness_stats = [
    ["Mean action difference", f"{np.abs(action_diffs).mean():.4f}"],
    ["Max action difference", f"{np.abs(action_diffs).max():.4f}"],
    ["Mean jerk", f"{np.abs(action_jerk).mean():.4f}"]
]
print("\nAction Smoothness Analysis:")
print(tabulate(smoothness_stats, headers=["Metric", "Value"], tablefmt="grid"))

if np.abs(action_diffs).mean() > 5:
    print("  Warning: Actions have large jumps - possible jerky movements")
else:
    print("  Actions are relatively smooth")
# +------------------------+---------+
# | Metric                 |   Value |
# +========================+=========+
# | Mean action difference |  0.004  |
# +------------------------+---------+
# | Max action difference  |  1.4974 |
# +------------------------+---------+
# | Mean jerk              |  0.0052 |
# +------------------------+---------+
#   Actions are relatively smooth