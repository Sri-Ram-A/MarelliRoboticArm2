import torch
import numpy as np
from pathlib import Path
from pprint import pprint, pformat
from loguru import logger
from tabulate import tabulate
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
import json

# Set paths
MODEL_PATH = "Sri-Ram-A/act_pnp1"  # Your trained model on Hub
CACHE_DIR = Path("~/.cache/huggingface/lerobot").expanduser()


# 1.1 Load the model config
logger.info(f"Loading model from: {MODEL_PATH}")
policy = ACTPolicy.from_pretrained(MODEL_PATH)
logger.success("Loaded successfully using ACTPolicy.from_pretrained()")
config = policy.config
config_dict = config.__dict__
logger.info("\nKEY CONFIGURATION PARAMETERS:")
pprint(config_dict)
# {
#  'chunk_size': 60,
#  'device': 'cuda',
#  'dim_feedforward': 3200,
#  'dim_model': 512,
#  'dropout': 0.1,
#  'feedforward_activation': 'relu',
#  'input_features': {
#       'observation.images.main': 
#             PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>,shape=(3,240,320)),
#       'observation.images.secondary_0': 
#             PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>,shape=(3,240,320)),
#       'observation.state': 
#             PolicyFeature(type=<FeatureType.STATE: 'STATE'>,shape=(6,))
#   },
#  'kl_weight': 10.0,
#  'latent_dim': 32,
#  'license': None,
#  'n_action_steps': 30,
#  'n_decoder_layers': 1,
#  'n_encoder_layers': 4,
#  'n_heads': 8,
#  'n_obs_steps': 1,
#  'n_vae_encoder_layers': 4,
#  'normalization_mapping': {
#       'ACTION': <NormalizationMode.MEAN_STD: 'MEAN_STD'>,
#       'STATE': <NormalizationMode.MEAN_STD: 'MEAN_STD'>,
#       'VISUAL': <NormalizationMode.MEAN_STD: 'MEAN_STD'>
#   },
#  'optimizer_lr': 1e-05,
#  'optimizer_lr_backbone': 1e-05,
#  'optimizer_weight_decay': 0.0001,
#  'output_features': {'action': PolicyFeature(type=<FeatureType.ACTION: 'ACTION'>,shape=(6,))},
#  'pre_norm': False,
#  'pretrained_backbone_weights': 'ResNet18_Weights.IMAGENET1K_V1',
#  'pretrained_path': None,
#  'private': None,
#  'push_to_hub': True,
#  'replace_final_stride_with_dilation': 0,
#  'repo_id': 'Sri-Ram-A/act_pnp1',
#  'tags': None,
#  'temporal_ensemble_coeff': None,
#  'use_amp': False,
#  'use_peft': False,
#  'use_vae': True,
#  'vision_backbone': 'resnet18'
# }

# 1.2 Load the actual model
model = ACTPolicy(config)
model.from_pretrained(MODEL_PATH)
model.eval()  # Set to evaluation mode
logger.success("Model weights loaded successfully")

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
logger.info(f"Model moved to device: {device}")

# 1.3 Check model parameters
logger.info("\n📊 MODEL STATISTICS:")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

model_stats = [
    ["Metric", "Value"],
    ["Total parameters", f"{total_params:,}"],
    ["Trainable parameters", f"{trainable_params:,}"],
    ["Model size (MB)", f"{total_params * 4 / 1024 / 1024:.2f}"],
    ["Device", device],
]
print(tabulate(model_stats, headers="firstrow", tablefmt="grid"))

# 1.4 Inspect model architecture layers
logger.info("\nMODEL ARCHITECTURE:")
pprint(model)

# 1.5 Check input features
logger.info("\n📥 INPUT FEATURES (what model expects):")
input_features = config_dict.get('input_features', {})
for name, info in input_features.items():
    feature_type = info.type.value if hasattr(info, 'type') else info.get('type', 'UNKNOWN')
    shape = info.shape if hasattr(info, 'shape') else info.get('shape', '?')
    logger.info(f"  {name}: type={feature_type}, shape={shape}")

logger.info("\n📤 OUTPUT FEATURES (what model produces):")
output_features = config_dict.get('output_features', {})
for name, info in output_features.items():
    feature_type = info.type.value if hasattr(info, 'type') else info.get('type', 'UNKNOWN')
    shape = info.shape if hasattr(info, 'shape') else info.get('shape', '?')
    logger.info(f"  {name}: type={feature_type}, shape={shape}")


logger.info("BLOCK 2: Understanding Input Normalization")
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import PolicyProcessorPipeline
# 2.1 Load dataset to get statistics
logger.info("Loading dataset statistics...")
dataset = LeRobotDataset(
    repo_id="Sri-Ram-A/pnp1",
    revision="v3.0",
    root=CACHE_DIR
)
logger.success(f"Dataset loaded: {dataset.num_episodes} episodes, {len(dataset)} frames")

# 2.2 Check dataset statistics
logger.info("\nDATASET STATISTICS (for normalization):")

# Load stats from the dataset's metadata
stats_path = CACHE_DIR / "Sri-Ram-A/pnp1" / "meta" / "stats.json"
if stats_path.exists():
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    logger.success("Loaded dataset statistics from metadata")
    pprint(stats)
# {'action': {'count': [30441],
#             'max': [1.1308199197536886,
#                     0.5201464759789695,
#                     1.3072707891860824,
#                     1.6571038172781327,
#                     0.603001666842876,
#                     2.092860747006827],
#             'mean': [0.05253136086276792,
#                      0.009240809476844105,
#                      0.2345840234264244,
#                      0.319703936316955,
#                      -0.010927429148176505,
#                      1.3888065868738317],
#             'min': [-1.2643088383677605,
#                     -0.9927279349805109,
#                     -0.43115386356958824,
#                     -0.6567041053657785,
#                     -1.459171972436578,
#                     -0.00011301840588378823],
#             'std': [0.5543056595560558,
#                     0.237508631116915,
#                     0.3194999919505223,
#                     0.41215603219347796,
#                     0.2109952122837417,
#                     0.867070780206306]},
#  'episode_index': {'count': [30441],
#                    'max': [42],
#                    'mean': [19.357642653000887],
#                    'min': [0],
#                    'std': [12.776773053033503]},
#  'frame_index': {'count': [30441],
#                  'max': [2131],
#                  'mean': [419.1361321901383],
#                  'min': [0],
#                  'std': [346.1116576853445]},
#  'index': {'count': [30441],
#            'max': [2131],
#            'mean': [419.1361321901383],
#            'min': [0],
#            'std': [346.1116576853445]},
#  'observation.images.main': {'count': [2337868800],
#                              'max': [[[1.0]], [[1.0]], [[1.0]]],
#                              'mean': [[[0.518764934902502]],
#                                       [[0.5347091031820145]],
#                                       [[0.5081043256262725]]],
#                              'min': [[[0.0]], [[0.0]], [[0.0]]],
#                              'std': [[[0.18117525095056652]],
#                                      [[0.16838548667058947]],
#                                      [[0.17699329614542103]]]},
#  'observation.images.secondary_0': {'count': [2337868800],
#                                     'max': [[[1.0]], [[1.0]], [[1.0]]],
#                                     'mean': [[[0.5062656159732397]],
#                                              [[0.5158789962640765]],
#                                              [[0.4884097385790661]]],
#                                     'min': [[[0.0]], [[0.0]], [[0.0]]],
#                                     'std': [[[0.19373314314695336]],
#                                             [[0.19406729770202305]],
#                                             [[0.20492337837985525]]]},
#  'observation.state': {'count': [30441],
#                        'max': [1.1308199197536886,
#                                0.5201464759789695,
#                                1.3072707891860824,
#                                1.6571038172781327,
#                                0.603001666842876,
#                                2.092860747006827],
#                        'mean': [0.05253136086276792,
#                                 0.009240809476844105,
#                                 0.2345840234264244,
#                                 0.319703936316955,
#                                 -0.010927429148176505,
#                                 1.3888065868738317],
#                        'min': [-1.2643088383677605,
#                                -0.9927279349805109,
#                                -0.43115386356958824,
#                                -0.6567041053657785,
#                                -1.459171972436578,
#                                -0.00011301840588378823],
#                        'std': [0.5543056595560558,
#                                0.237508631116915,
#                                0.3194999919505223,
#                                0.41215603219347796,
#                                0.2109952122837417,
#                                0.867070780206306]},
#  'task_index': {'count': [30441],
#                 'max': [0],
#                 'mean': [0.0],
#                 'min': [0],
#                 'std': [0.0]},
#  'timestamp': {'count': [30441],
#                'max': [80.1346011999999],
#                'mean': [15.877849572753853],
#                'min': [0.0015633000002708286],
#                'std': [12.86891008622991]}}

# Display action statistics
logger.info("\nACTION STATISTICS (for normalization):")
action_stats = stats['action']
action_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
action_table = [["Joint", "Mean", "Std", "Min", "Max"]]
for i, name in enumerate(action_names):
    action_table.append([
        name,
        f"{action_stats['mean'][i]:.6f}",
        f"{action_stats['std'][i]:.6f}",
        f"{action_stats['min'][i]:.6f}",
        f"{action_stats['max'][i]:.6f}"
    ])
print(tabulate(action_table, headers="firstrow", tablefmt="grid"))

# Display observation statistics
logger.info("\nOBSERVATION STATISTICS (joint states):")
obs_stats = stats['observation.state']
obs_table = [["Joint", "Mean", "Std", "Min", "Max"]]
for i, name in enumerate(action_names):
    obs_table.append([
        name,
        f"{obs_stats['mean'][i]:.6f}",
        f"{obs_stats['std'][i]:.6f}",
        f"{obs_stats['min'][i]:.6f}",
        f"{obs_stats['max'][i]:.6f}"
    ])
print(tabulate(obs_table, headers="firstrow", tablefmt="grid"))

# Determine action units
max_rad = max([abs(action_stats['max'][i]) for i in range(5)])  # Exclude gripper
if max_rad < 3.2:
    logger.info(f"\nACTIONS ARE IN RADIANS (max: {max_rad:.4f} rad)")
else:
    logger.info(f"\nACTIONS ARE IN DEGREES (max: {max_rad:.1f}°)")
# ACTIONS ARE IN RADIANS (max: 1.6571 rad)


# Gripper analysis
gripper_mean = action_stats['mean'][5]
gripper_std = action_stats['std'][5]
logger.info(f"\nGRIPPER STATISTICS:")
logger.info(f"  Mean: {gripper_mean:.4f}")
logger.info(f"  Std: {gripper_std:.4f}")
logger.info(f"  Range: [{action_stats['min'][5]:.4f}, {action_stats['max'][5]:.4f}]")
if action_stats['max'][5] <= 1.0 and action_stats['min'][5] >= 0:
    logger.info("  → Gripper appears NORMALIZED (0-1 range)")
else:
    logger.info("  → Gripper appears in RAW units (likely percentage or radians)")
# Gripper appears in RAW units (likely percentage or radians) - 2.09286


logger.info("BLOCK 3: Creating Preprocessor and Testing Input Transformation")


# 2.3 Create preprocessor (to see how input is transformed)
logger.info("\n🔧 Creating preprocessing pipeline...")
from lerobot.processor.pipeline import DataProcessorPipeline
preprocessor = DataProcessorPipeline.from_pretrained(
    MODEL_PATH,
    config_filename="policy_preprocessor.json",
)
postprocessor = DataProcessorPipeline.from_pretrained(
    MODEL_PATH,
    config_filename="policy_postprocessor.json",
)
logger.success("Preprocessor created")

# 2.4 Test preprocessing on a sample observation
logger.info("\n📥 Testing preprocessing on a sample:")
sample_frame = dataset[0]

# Extract raw observation
raw_obs = {
    "observation.state": sample_frame["observation.state"].numpy(),
    "observation.images.main": sample_frame["observation.images.main"].numpy(),
    "observation.images.secondary_0": sample_frame["observation.images.secondary_0"].numpy(),
}

logger.info("Raw observation:")
logger.info(f"  State shape: {raw_obs['observation.state'].shape}")
logger.info(f"  Main camera shape: {raw_obs['observation.images.main'].shape}")
logger.info(f"  Secondary camera shape: {raw_obs['observation.images.secondary_0'].shape}")
logger.info(f"  State values (first 3): {raw_obs['observation.state'][:3]}")

# Preprocess
processed_obs = preprocessor(raw_obs)
pprint(processed_obs)

logger.info("BLOCK 3: Understanding Model Outputs")

# 3.1 Run inference on a sample observation
logger.info("Running inference on sample observation...")

with torch.no_grad():
    batch_obs = {k: v.unsqueeze(0).to(device) for k, v in processed_obs.items() if torch.is_tensor(v)}
    prediction = model.select_action(batch_obs)  # returns a single action, not a chunk
pprint(prediction)
# prediction = tensor([[-0.1788,  0.5382,  0.6477, -1.2776,  0.3722, -0.1723]])

# Let's analyze this output
action_tensor = prediction[0].cpu().numpy()  # Remove batch dimension
action_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']

logger.info("\nRaw model output (normalized values):")
for i, name in enumerate(action_names):
    logger.info(f"  {name:15s}: {action_tensor[i]:.6f}")

# Check the range
logger.info(f"\nOutput range: min={action_tensor.min():.4f}, max={action_tensor.max():.4f}")
logger.info(f"Output mean: {action_tensor.mean():.4f}")
logger.info(f"Output std: {action_tensor.std():.4f}")

# 1. First, denormalize using dataset stats
logger.info("STEP 1: DENORMALIZE USING DATASET STATS")

# Load stats
stats = dataset.meta.stats
action_mean = np.array(stats['action']['mean'])
action_std = np.array(stats['action']['std'])
logger.info("\nDataset stats for denormalization:")
for i, name in enumerate(action_names):
    logger.info(f"  {name:15s}: mean={action_mean[i]:.6f}, std={action_std[i]:.6f}")

# Denormalize: output_radians = (normalized * std) + mean
action_radians = action_tensor * action_std + action_mean
logger.info("\nDenormalized actions (RADIANS):")
for i, name in enumerate(action_names):
    logger.info(f"  {name:15s}: {action_radians[i]:.6f} rad")

# 2. Convert radians to degrees for robot
logger.info("STEP 2: CONVERT RADIANS TO DEGREES")
action_degrees = action_radians * 180 / np.pi
logger.info("\nConverted to DEGREES:")
for i, name in enumerate(action_names[:-1]):  # All except gripper
    logger.info(f"  {name:15s}: {action_degrees[i]:.2f}°")

# 3. Handle gripper specially
logger.info("STEP 3: HANDLE GRIPPER")
# Your gripper range in dataset is likely 0-1 (normalized) or 0-1.57 rad
gripper_rad = action_radians[5]
logger.info(f"Gripper in radians: {gripper_rad:.4f}")

# Convert to percentage (assuming 0 rad = 0%, 1.57 rad = 100%)
gripper_percent = (gripper_rad / 1.57) * 100
gripper_percent = np.clip(gripper_percent, 0, 100)
logger.info(f"Gripper as percentage: {gripper_percent:.1f}%")

# 4. Create robot action dictionary (the format your robot expects)
logger.info("STEP 4: FORMAT FOR ROBOT")
robot_action = {
    "shoulder_pan.pos": float(action_degrees[0]),
    "shoulder_lift.pos": float(action_degrees[1]),
    "elbow_flex.pos": float(action_degrees[2]),
    "wrist_flex.pos": float(action_degrees[3]),
    "wrist_roll.pos": float(action_degrees[4]),
    "gripper.pos": float(gripper_percent),
}

logger.info("\nFinal robot command:")
for key, value in robot_action.items():
    logger.info(f"  {key}: {value:.2f}")

# 5. Check if values are within safe ranges
logger.info("STEP 5: SAFETY CHECK")
# Approximate safe ranges for your robot (adjust based on your calibration)
safe_ranges = {
    "shoulder_pan": (-180, 180),
    "shoulder_lift": (-120, 120),
    "elbow_flex": (-120, 120),
    "wrist_flex": (-90, 90),
    "wrist_roll": (-180, 180),
    "gripper": (0, 100),
}

all_safe = True
for i, name in enumerate(action_names[:-1]):
    deg = action_degrees[i]
    min_safe, max_safe = safe_ranges[name]
    if deg < min_safe or deg > max_safe:
        logger.warning(f"  {name}: {deg:.1f}° is outside safe range [{min_safe}°, {max_safe}°]")
        all_safe = False
    else:
        logger.info(f"  ✓ {name}: {deg:.1f}° within safe range")

if gripper_percent < 0 or gripper_percent > 100:
    logger.warning(f"  gripper: {gripper_percent:.1f}% outside safe range [0%, 100%]")
    all_safe = False
else:
    logger.info(f"  ✓ gripper: {gripper_percent:.1f}% within safe range")

if all_safe:
    logger.success("\nAll actions within safe ranges - ready to send to robot!")
else:
    logger.warning("\nSome actions outside safe ranges - consider clipping before sending")

# 6. Complete function for inference to robot action
logger.info("COMPLETE INFERENCE FUNCTION")
def model_output_to_robot_action(model_output_tensor, dataset_stats):
    """
    Convert model output (normalized tensor) to robot action dict
    
    Args:
        model_output_tensor: Shape (1, 6) tensor from model.select_action()
        dataset_stats: Stats from dataset.meta.stats
    
    Returns:
        dict: Robot action with .pos keys
    """
    # Convert to numpy
    action_norm = model_output_tensor[0].cpu().numpy()
    
    # Denormalize
    action_mean = np.array(dataset_stats['action']['mean'])
    action_std = np.array(dataset_stats['action']['std'])
    action_rad = action_norm * action_std + action_mean
    
    # Convert joints to degrees
    action_deg = action_rad[:5] * 180 / np.pi
    
    # Convert gripper to percentage (assuming 1.57 rad = 100%)
    gripper_percent = (action_rad[5] / 1.57) * 100
    gripper_percent = np.clip(gripper_percent, 0, 100)
    
    # Format for robot
    robot_action = {
        "shoulder_pan.pos": float(action_deg[0]),
        "shoulder_lift.pos": float(action_deg[1]),
        "elbow_flex.pos": float(action_deg[2]),
        "wrist_flex.pos": float(action_deg[3]),
        "wrist_roll.pos": float(action_deg[4]),
        "gripper.pos": float(gripper_percent),
    }
    
    return robot_action

# Test the function
test_action = model_output_to_robot_action(prediction, stats)
logger.info("\nTest function output:")
for key, value in test_action.items():
    logger.info(f"  {key}: {value:.2f}")
logger.success("\nReady to use with robot!")

