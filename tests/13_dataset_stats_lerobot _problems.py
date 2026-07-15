"""
Step 1: Basic Dataset Exploration
Understanding the structure of Sri-Ram-A/pnp1 dataset
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
from tabulate import tabulate
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch

# Set your dataset info
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_NAME = "so100_keyboard_teleop_4"
DATASET_REPO_ID = f"Sri-Ram-A/{DATASET_NAME}"  # Change as needed
DATASET_ROOT = Path("/home/srirama/Documents/sr_proj/RoboticArm/datasets")
DATASET_PATH = DATASET_ROOT / DATASET_NAME
# PosixPath('/home/srirama/Documents/sr_proj/RoboticArm/datasets/so100_keyboard_teleop_4')
# Download dataset metadata first (without downloading all videos)
print("STEP 1: Understanding LeRobot Dataset Structure")

# 1.1 Check dataset info.json
# info_path = hf_hub_download(repo_id=DATASET_ID,filename="meta/info.json",repo_type="dataset",token=HF_TOKEN)
info_path = r"/home/srirama/Documents/sr_proj/RoboticArm/datasets/so100_keyboard_teleop_4/meta/info.json"
# '/home/srirama/hub/datasets--Sri-Ram-A--pnp1/snapshots/ed682a0b302da28ad60a10f1dc5968415cd50d9c/meta/info.json'
with open(info_path, "r") as f:
    info = json.load(f)
pprint(info)
CACHE_DIR = Path("/.cache/huggingface/lerobot").expanduser()
# {'chunks_size': 1000,
#  'codebase_version': 'v3.0',
#  'data_files_size_in_mb': 100,
#  'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
#  'features': {'action': {'dtype': 'float32',
#                          'names': ['shoulder_pan',
#                                    'shoulder_lift',
#                                    'elbow_flex',
#                                    'wrist_flex',
#                                    'wrist_roll',
#                                    'gripper'],
#                          'shape': [6]},
#               'episode_index': {'dtype': 'int64', 'names': None, 'shape': [1]},
#               'frame_index': {'dtype': 'int64', 'names': None, 'shape': [1]},
#               'index': {'dtype': 'int64', 'names': None, 'shape': [1]},
#               'observation.images.main': {'dtype': 'video',
#                                           'info': {'has_audio': False,
#                                                    'video.channels': 3,
#                                                    'video.codec': 'av1',
#                                                    'video.fps': 30,
#                                                    'video.height': 480,
#                                                    'video.is_depth_map': False,
#                                                    'video.pix_fmt': 'yuv420p',
#                                                    'video.width': 640},
#                                           'names': ['height',
#                                                     'width',
#                                                     'channel'],
#                                           'shape': [480, 640, 3]},
#               'observation.images.secondary_0': {'dtype': 'video',
#                                                  'info': {'has_audio': False,
#                                                           'video.channels': 3,
#                                                           'video.codec': 'av1',
#                                                           'video.fps': 30,
#                                                           'video.height': 480,
#                                                           'video.is_depth_map': False,
#                                                           'video.pix_fmt': 'yuv420p',
#                                                           'video.width': 640},
#                                                  'names': ['height',
#                                                            'width',
#                                                            'channel'],
#                                                  'shape': [480, 640, 3]},
#               'observation.state': {'dtype': 'float32',
#                                     'names': ['shoulder_pan',
#                                               'shoulder_lift',
#                                               'elbow_flex',
#                                               'wrist_flex',
#                                               'wrist_roll',
#                                               'gripper'],
#                                     'shape': [6]},
#               'task_index': {'dtype': 'int64', 'names': None, 'shape': [1]},
#               'timestamp': {'dtype': 'float32', 'names': None, 'shape': [1]}},
#  'fps': 30,
#  'robot_type': 'so_follower',
#  'splits': {'train': '0:36'},
#  'total_episodes': 36,
#  'total_frames': 44477,
#  'total_tasks': 1,
#  'video_files_size_in_mb': 200,
#  'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'}

"""
Step 2: Load Dataset Using LeRobot's Dataset Class
This is the recommended way to work with LeRobot datasets
"""

print("STEP 2: Loading Dataset with LeRobot's Dataset Class")

print("\n2.1 Loading dataset (this may take a moment)...")
dataset = LeRobotDataset(
    repo_id=DATASET_REPO_ID,
    revision="main",
    root=DATASET_PATH,
    episodes=[0, 1, 2],  # Load first 3 episodes for testing
)
pprint(dataset)
print("\n✅ Dataset loaded successfully!")
print(f"  - Total episodes: {dataset.num_episodes}")
print(f"  - Total frames: {len(dataset)}")
print(f"  - Features: {list(dataset.features.keys())}")

# Get first sample
sample = dataset[0]
print("\n📦 First sample structure:")
for key in sample.keys():
    if isinstance(sample[key], dict):
        print(f"  - {key}: dict with keys {list(sample[key].keys())}")
    elif hasattr(sample[key], "shape"):
        print(f"  - {key}: shape {sample[key].shape}, dtype {sample[key].dtype}")
    else:
        print(f"  - {key}: {type(sample[key])}")


"""
Step 3: Deep Dive into Action and Observation Data
Understanding what's actually stored in the dataset
"""

print("STEP 3: Analyzing Action and Observation Data")

# Dataset Indexing man
dataset = LeRobotDataset(repo_id=DATASET_REPO_ID, revision="main", root=DATASET_PATH)

first_episode_idx = 0
last_episode_idx = info["total_frames"] - 1

# Get all actions from every frame
actions = []
for i in tqdm(range(first_episode_idx, last_episode_idx)):
    sample = dataset[i]
    actions.append(sample["action"].numpy())

actions = np.array(actions)
print(f"\nActions shape: {actions.shape}")

action_names = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
action_stats = []
for i, name in tqdm(enumerate(action_names)):
    joint_actions = actions[:, i]
    action_stats.append(
        [
            name,
            f"{joint_actions.min():6.4f}",
            f"{joint_actions.max():6.4f}",
            f"{joint_actions.mean():6.4f}",
            f"{joint_actions.std():6.4f}",
        ]
    )
print("\nAction Statistics (per joint):")
print(
    tabulate(
        action_stats, headers=["Joint", "Min", "Max", "Mean", "Std"], tablefmt="grid"
    )
)
# +---------------+-------+-------+----------+---------+
# | Joint         |   Min |   Max |     Mean |     Std |
# +===============+=======+=======+==========+=========+
# | shoulder_pan  |   -94 |    44 | -19.6493 | 31.7559 |
# +---------------+-------+-------+----------+---------+
# | shoulder_lift |   -66 |    36 |  -5.2913 | 15.7883 |
# +---------------+-------+-------+----------+---------+
# | elbow_flex    |   -24 |    30 |   7.7815 |  7.9494 |
# +---------------+-------+-------+----------+---------+
# | wrist_flex    |   -12 |   106 |  43.361  | 26.8827 |
# +---------------+-------+-------+----------+---------+
# | wrist_roll    |  -190 |   170 |  -8.8324 | 62.6642 |
# +---------------+-------+-------+----------+---------+
# | gripper       |     0 |   100 |  21.8058 | 27.0927 |
# +---------------+-------+-------+----------+---------+

# Get All observations from every frame
observations = []
for i in tqdm(range(first_episode_idx, last_episode_idx)):
    sample = dataset[i]
    observations.append(sample["observation.state"].numpy())

observations = np.array(observations)
print(f"\nObservations shape: {observations.shape}")

obs_stats = []
for i, name in enumerate(action_names):
    joint_obs = observations[:, i]
    obs_stats.append(
        [
            name,
            f"{joint_obs.min():6.4f}",
            f"{joint_obs.max():6.4f}",
            f"{joint_obs.mean():6.4f}",
            f"{joint_obs.std():6.4f}",
        ]
    )
print("\nObservation Statistics (per joint):")
print(
    tabulate(obs_stats, headers=["Joint", "Min", "Max", "Mean", "Std"], tablefmt="grid")
)

# +---------------+-----------+----------+----------+---------+
# | Joint         |       Min |      Max |     Mean |     Std |
# +===============+===========+==========+==========+=========+
# | shoulder_pan  |  -91.5165 |  43.7802 | -19.8337 | 31.4644 |
# +---------------+-----------+----------+----------+---------+
# | shoulder_lift |  -64.1758 |  29.8901 |  -1.5555 | 14.5486 |
# +---------------+-----------+----------+----------+---------+
# | elbow_flex    |  -18.7253 |  28.044  |  10.0366 |  7.66   |
# +---------------+-----------+----------+----------+---------+
# | wrist_flex    |  -10.1978 | 104.703  |  43.923  | 26.2955 |
# +---------------+-----------+----------+----------+---------+
# | wrist_roll    | -175.165  | 169.538  |  -8.8177 | 62.3534 |
# +---------------+-----------+----------+----------+---------+
# | gripper       |    2.4646 |  96.9193 |  25.583  | 23.828  |
# +---------------+-----------+----------+----------+---------+

action_range = actions[:, :5]
max_abs = np.abs(action_range).max()

print("\nDetermining Action Units:")
if max_abs < 3.2:
    print(f"  Actions are in RADIANS (max absolute: {max_abs:.2f})")
elif max_abs > 100:
    print(f"  Actions are in DEGREES (max absolute: {max_abs:.1f})")
else:
    print(f"  Action range ambiguous: {max_abs:.2f}")

# Actions are in DEGREES (max absolute: 190.0)

print("STEP 4: Analyzing Camera Data")
camera_keys = [key for key in sample.keys() if key.startswith("observation.images.")]
print(f"\nCamera keys found: {camera_keys}")
# Camera keys found: ['observation.images.main', 'observation.images.secondary_0']

camera_info = []
for cam_key in camera_keys:
    img = sample[cam_key]
    is_normalized = "YES" if img.max() <= 1.0 else "NO"
    camera_info.append(
        [
            cam_key.replace("observation.images.", ""),
            str(img.shape),
            str(img.dtype),
            f"[{img.min():.3f}, {img.max():.3f}]",
            is_normalized,
        ]
    )
print("\nCamera Analysis:")
print(
    tabulate(
        camera_info,
        headers=["Camera", "Shape", "Dtype", "Range", "Normalized"],
        tablefmt="grid",
    )
)

# +-------------+---------------------------+---------------+----------------+--------------+
# | Camera      | Shape                     | Dtype         | Range          | Normalized   |
# +=============+===========================+===============+================+==============+
# | main        | torch.Size([3, 480, 640]) | torch.float32 | [0.000, 1.000] | YES          |
# +-------------+---------------------------+---------------+----------------+--------------+
# | secondary_0 | torch.Size([3, 480, 640]) | torch.float32 | [0.000, 0.804] | YES          |
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
    axes[i].set_title(cam_key.replace("observation.images.", ""))
    axes[i].axis("off")
plt.tight_layout()
plt.savefig("dataset_sample_images.png", dpi=150)
print("\nSample images saved to dataset_sample_images.png")

print("STEP 5: Episode Quality Analysis")
action_diffs = np.diff(actions, axis=0)
action_jerk = np.diff(action_diffs, axis=0)

smoothness_stats = [
    ["Mean action difference", f"{np.abs(action_diffs).mean():.4f}"],
    ["Max action difference", f"{np.abs(action_diffs).max():.4f}"],
    ["Mean jerk", f"{np.abs(action_jerk).mean():.4f}"],
]
print("\nAction Smoothness Analysis:")
print(tabulate(smoothness_stats, headers=["Metric", "Value"], tablefmt="grid"))

if np.abs(action_diffs).mean() > 5:
    print("  Warning: Actions have large jumps - possible jerky movements")
else:
    print("  Actions are relatively smooth")
# +------------------------+----------+
# | Metric                 |    Value |
# +========================+==========+
# | Mean action difference |   0.1147 |
# +------------------------+----------+
# | Max action difference  | 186      |
# +------------------------+----------+
# | Mean jerk              |   0.0523 |
# +------------------------+----------+
#   Actions are relatively smooth

"""
Orbits report
```bash

Visual analysis skipped — install torch and transformers for embedding analysis
Array profile: absolute_position (6D, single-arm, gripper=[5])

Outlier detection: 1/36 episodes flagged (healthy)

Running proxy training (BC-MLP baseline)... (~2 min)
  Running policy-matched proxy (ACT-mini)... (~5-10 min)
  Proxy check: GO (val_loss=0.0425)
  Validation loss dropped to 0.0425 (from 0.9933) — data is learnable
  MLP baseline loss ratio: 0.015 (lower = more learnable)
  Worst episodes: #34, #30, #17
  (9s, 1000 steps, ACT-mini)

╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│  ORBIT Analysis: /home/srirama/Documents/sr_proj/RoboticArm/datasets/so100_keyboard_teleop_4                                            │
│  36 episodes · 6 action dims · SO_FOLLOWER · ACT                                                                                        │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Dataset Readiness: A (score: 87/100)
Ready to train — expect strong results

  ✓ Very few outlier episodes (1/36)
  ✓ Episode quality is high — very few flagged episodes
  ✓ Sufficient episodes (36) for act
  ✓ Good cross-episode consistency (0.71)
  ✓ Good consistency (0.62)
  ✓ Good policy fit (0.73)
  ✓ Proxy training signal: GO (val_loss=0.0425)
  ✗ 1 joint clipping
  ✗ Effort varies across episodes (CV=0.54) — force/torque usage is inconsistent
  ✗ High action divergence (0.28) — demos contradict each other in similar states

Top action: High action divergence — consider filtering to a single strategy or switching to Diffusion Policy

YOUR DATA AT A GLANCE
────────────────────────────────────────
  Episodes:       36    
  Coverage:       0.67  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░
  Signal Health:  0.92  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░

WHAT'S WORKING
  ✓ All 6 joints active — no dead servos detected
  ✓ Episode lengths consistent (CV=0.38)
  ✓ Episode count (36) meets minimum (30)
  ✓ Average smoothness good (1.00)

WHAT NEEDS ATTENTION
  ⚠ Episode 1 (low length (0.51)), Episode 4 (low completion (0.40)), Episode 5 (low completion (0.33))
    → Inspect episode 1 for quality issues
  ⚠ joint_3 has directional bias (96%)
  ⚠ joint_5 has directional bias (100%)
  ⚠ Clipping detected on joint_5
  ⚠ Episode lengths inconsistent (CV=0.38)

COMMUNITY COMPARISON
────────────────────────────────────────
  Compared against 1 real-world generic benchmarks

  Your dataset vs 1 similar moderate datasets on SO_FOLLOWER:
    Your episodes (36) are sufficient for your policy
    Your quality: 0.67 (average)

  Closest successful dataset: RoboTurk [REAL] [MEDIUM]
    100 episodes, 74% success (bc)

COMPATIBLE DATASETS FOR MIXING
────────────────────────────────────────

  RoboTurk ★★ [REAL] [MEDIUM]
    100 eps · 74% success · Strategy: Direct merge

POLICY FIT
────────────────────────────────────────
  act: 0.72
  diffusion_policy: 0.71
  dp3: 0.71

EPISODE QUALITY RANKING
────────────────────────────────────────
  Keep: 33/36 (91%)  │  Review: 3 (8%)  │  Remove: 0 (0%)

ACTION DIVERGENCE
────────────────────────────────────────
  Score: 0.28 (moderate)

TEMPORAL ALIGNMENT
────────────────────────────────────────
  ✓ State-action alignment: lag = -5 frames (negligible, <2% of episode)

SCALING ADVICE
────────────────────────────────────────
  Collect ~214 more demonstrations (50/environment)
  Estimated diversity: ~5 environments, ~7 demos each

CONSISTENCY DIAGNOSTICS
────────────────────────────────────────
  Jerk CV: 0.41 (moderate) — smoothness varies
  Curvature CV: 0.53 (high) — path shape varies
  Effort CV: 0.54 (high) — action magnitude varies
  Joint limit CV: 1.68 (high) — joint range usage varies
  Most inconsistent episodes: 1, 2, 3

NEXT STEPS (prioritized)
────────────────────────────────────────
  [HIGH] 1 joint clipping
  [HIGH] Effort varies across episodes (CV=0.54) — force/torque usage is inconsistent
  [HIGH] High action divergence (0.28) — demos contradict each other in similar states
  [HIGH] Collect 14 more episodes to reach the minimum recommended count of 50
  [MEDIUM] Inspect episode 1 for quality issues
  [MEDIUM] Review episodes 14, 22 — they appear significantly different from others (possible recording errors)

Run `orbit suggest /home/srirama/Documents/sr_proj/RoboticArm/datasets/so100_keyboard_teleop_4` to get a ready-to-run training command.

(lerobot) ~/Documents/sr_proj/RoboticArm$ orbit suggest /home/srirama/Documents/sr_proj/RoboticArm/datasets/so100_keyboard_teleop_4

              Recommended: Action Chunking Transformer (fit: 0.72)              
┌─────────────┬────────────────────────────────────────────────────────────────┐
│ Episodes    │ 36                                                             │
│ Cameras     │ 2 x 480p                                                       │
│ Action dims │ 6                                                              │
│ FPS         │ 30                                                             │
│ Alternative │ Diffusion Policy (fit: 0.71) — Close fit — scores 0.71 vs 0.72 │
└─────────────┴────────────────────────────────────────────────────────────────┘

Copy and run:

lerobot-train \
  --dataset.repo_id=/home/srirama/Documents/sr_proj/RoboticArm/datasets/so100_keyboard_teleop_4 \
  --policy.type=act \
  --batch_size=16 \
  --steps=100000 \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --log_freq=200 \
  --save_freq=20000 \
  --eval_freq=20000 \
  --policy.device=cuda \
  --output_dir=outputs/train/so100_keyboard_teleop_4-act

Training tips:
  • Loss should drop below 0.1 by step 20000
  • If loss plateaus above 0.2: your demonstrations may be too inconsistent
  • If robot doesn't move during eval: check action normalization stats

Save as script: echo the command above > train_so100_keyboard_teleop_4_act.sh

```
"""