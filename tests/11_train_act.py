"""
=============================================================================
ACT Policy Training Script for SO100 Keyboard Teleop Dataset
=============================================================================

Dataset:   Sri-Ram-A/so100_keyboard_teleop_4
Episodes:  36
Robot:     SO100 Follower (so_follower)
Cameras:   main (480x640) + secondary_0 (480x640)
Actions:   6 DOF in DEGREES (shoulder_pan, shoulder_lift, elbow_flex,
           wrist_flex, wrist_roll, gripper)

Why ACT over Diffusion for 36 episodes?
  - ACT is designed for precise manipulation with small datasets
  - Action chunking (predicting N future steps) handles temporal correlation
  - VAE objective captures multimodal behaviour
  - Diffusion needs many more samples to converge well

Normalization / Degrees pipeline:
  │  Raw dataset frames (degrees, raw pixels)                        │
  │        │                                                         │
  │  preprocessor(batch)   ← computed from dataset stats             │
  │        │   • actions/states: MEAN_STD normalised → [-~3, +~3]    │
  │        │   • images: MEAN_STD normalised (ImageNet stats)        │
  │        ▼                                                         │
  │  policy.forward(batch) → loss (model never sees raw degrees)     │
  │                                                                  │
  │  At inference time:                                              │
  │  policy.select_action(obs) → normalised action                   │
  │        │                                                         │
  │  postprocessor(action)                                           │
  │        │   • de-normalises back to DEGREES                       │
  │        ▼                                                         │
  │  robot.send_action(action_in_degrees)                         │

=============================================================================
"""

# %%
# BLOCK 1 ─ Imports
from pathlib import Path
from pprint import pprint
import torch
import torch.utils.data
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.types import NormalizationMode
from tqdm import tqdm

# BLOCK 2 ─ Configuration

# Dataset
DATASET_NAME = "so100_keyboard_teleop_4"
DATASET_REPO_ID = f"Sri-Ram-A/{DATASET_NAME}"
DATASET_FPS = 30  # recorded at 30 fps

# Output ─
OUTPUT_DIR = Path("outputs/train/act_so100")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training hyper-parameters
# With 36 episodes (~44k frames total) we keep the batch small and run many steps.
# Rule of thumb for ACT on real robot data: 50k–100k gradient steps.
TRAINING_STEPS = 80_000  # adjust up/down based on GPU time
BATCH_SIZE = 8  # small dataset → small batch prevents overfitting
LR = 1e-5  # ACT default; backbone gets same LR
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 10.0
NUM_WORKERS = 6  # dataloader workers

#  ACT architecture
CHUNK_SIZE = 50  # predict 50 steps ≈ 1.67 s at 30 fps (good for ~1s tasks)
N_ACTION_STEPS = 50  # how many of those 50 we actually execute per call
N_OBS_STEPS = 1  # CRITICAL: ACT uses only current observation, not history
DIM_MODEL = 512
N_HEADS = 8
N_ENC_LAYERS = 4
N_DEC_LAYERS = 1  # keep 1 (matches original ACT paper implementation)
DIM_FFN = 3200
USE_VAE = True
LATENT_DIM = 32
KL_WEIGHT = 10.0  # reconstruction_loss + KL_WEIGHT * kld_loss

#  Logging / checkpointing
LOG_FREQ = 200
SAVE_FREQ = 5_000

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"BLOCK 2: Config OK — training on {DEVICE}")


# BLOCK 3 ─ Load Dataset Metadata and Derive Policy Features
# LeRobotDatasetMetadata reads only the meta/info.json and stats.json files
# without loading any video frames into memory.  We use it to:
#   1. Extract feature shapes (state dim, action dim, image sizes)
#   2. Get dataset statistics (mean/std per feature) for the preprocessor
#
print("\nBLOCK 3: Loading dataset metadata …")
dataset_metadata = LeRobotDatasetMetadata(
    repo_id=DATASET_REPO_ID,
)
# Convert dataset features → policy Feature objects (carries dtype + shape)
all_features = dataset_to_policy_features(dataset_metadata.features)

# Split into input (observations) vs output (actions)
output_features = {
    k: ft for k, ft in all_features.items() if ft.type is FeatureType.ACTION
}
input_features = {k: ft for k, ft in all_features.items() if k not in output_features}

print(" Input features :")
pprint(input_features)
# {'observation.images.main': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>,shape=(3, 480, 640)),
#  'observation.images.secondary_0': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>,shape=(3, 480, 640)),
#  'observation.state': PolicyFeature(type=<FeatureType.STATE: 'STATE'>,shape=(6,))
# }

print(" Output features:")
pprint(output_features)
# {'action': PolicyFeature(type=<FeatureType.ACTION: 'ACTION'>, shape=(6,))}


action_ft = list(output_features.values())[0]
assert action_ft.shape == (6,), f"Expected 6-DOF action, got {action_ft.shape}"

#  Confirm camera shapes
for cam in ["observation.images.main", "observation.images.secondary_0"]:
    if cam in input_features:
        print(f"  Camera {cam}: shape={input_features[cam].shape}")
#   Camera observation.images.main: shape=(3, 480, 640)
#   Camera observation.images.secondary_0: shape=(3, 480, 640)

# BLOCK 4 ─ Build ACTConfig and Policy
#
# ACT key ideas:
#   • chunk_size = how many action steps the model predicts at once
#   • n_action_steps = how many we actually use before re-inferring
#   • use_vae = True trains the CVAE regularisation (helps with multimodality)
#   • normalization_mapping: MEAN_STD for states/actions means the model sees
#     zero-mean unit-variance values — YOUR DEGREES ARE HANDLED HERE.
#     The preprocessor computes:
#         normalised = (x - mean_degrees) / std_degrees
#     The postprocessor reverses it:
#         x_degrees = normalised * std_degrees + mean_degrees
#
print("\nBLOCK 4: Building ACT policy …")

# adjust import if needed
cfg = ACTConfig(
    #  I/O features ─
    input_features=input_features,
    output_features=output_features,
    #  Action chunking ─
    chunk_size=CHUNK_SIZE,
    n_action_steps=N_ACTION_STEPS,
    n_obs_steps=N_OBS_STEPS,  # ACT uses single observation step
    #  Transformer
    dim_model=DIM_MODEL,
    n_heads=N_HEADS,
    n_encoder_layers=N_ENC_LAYERS,
    n_decoder_layers=N_DEC_LAYERS,
    dim_feedforward=DIM_FFN,
    dropout=0.1,
    #  Vision backbone
    vision_backbone="resnet18",  # lightweight, good for 36 eps
    pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
    #  VAE ─
    use_vae=USE_VAE,
    latent_dim=LATENT_DIM,
    kl_weight=KL_WEIGHT,
    #  Normalisation
    # MEAN_STD normalises each feature channel independently using dataset stats.
    # This means degrees (range ~[-190, +190]) become roughly N(0,1).
    normalization_mapping={
        "VISUAL": NormalizationMode.MEAN_STD,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MEAN_STD,
    },
    #  Optimiser
    optimizer_lr=LR,
    optimizer_lr_backbone=LR_BACKBONE,
    optimizer_weight_decay=WEIGHT_DECAY,
)

pprint(cfg.image_features)

policy = ACTPolicy(cfg)
policy.train()
policy.to(DEVICE)

total_params = sum(p.numel() for p in policy.parameters()) / 1e6
print(f"  Policy parameters: {total_params:.1f} M")

# BLOCK 5 ─ Build Preprocessor and Postprocessor
# make_pre_post_processors uses dataset_metadata.stats to build:
#   preprocessor → called on every training batch before forward()
#   postprocessor → called at inference time after select_action()
# The stats come from stats.json in your dataset root, which LeRobot auto-
# computes when you create/finalise the dataset (mean, std, min, max per
# feature across all frames).
print("\nBLOCK 5: Building preprocessor / postprocessor …")
dataset_revision = "v3.0"
metadata = LeRobotDatasetMetadata(DATASET_REPO_ID)
pprint(metadata)
preprocessor, postprocessor = make_pre_post_processors(
    cfg,
    dataset_stats=metadata.stats, # type: ignore
)

for i, step in enumerate(preprocessor.steps):
    pprint(f"Step {i}: {step.__class__.__name__}")
    pprint(step)

for i, step in enumerate(postprocessor.steps):
    pprint(f"Step {i}: {step.__class__.__name__}")
    pprint(step)

# 'Step 0: RenameObservationsProcessorStep'
# RenameObservationsProcessorStep(rename_map={})
# 'Step 1: AddBatchDimensionProcessorStep'
# AddBatchDimensionProcessorStep(to_batch_action_processor=AddBatchDimensionActionStep(),
#                                to_batch_observation_processor=AddBatchDimensionObservationStep(),
#                                to_batch_complementary_data_processor=AddBatchDimensionComplementaryDataStep())
# 'Step 2: DeviceProcessorStep'
# DeviceProcessorStep(device='cpu', float_dtype=None)
# 'Step 3: NormalizerProcessorStep'
# NormalizerProcessorStep(features={'action': PolicyFeature(type=<FeatureType.ACTION: 'ACTION'>, shape=(6,)),
#                                   'observation.images.main': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 480, 640)),
#                                   'observation.images.secondary_0': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 480, 640)),
#                                   'observation.state': PolicyFeature(type=<FeatureType.STATE: 'STATE'>, shape=(6,))},
#                         norm_map={<FeatureType.ACTION: 'ACTION'>: <NormalizationMode.MEAN_STD: 'MEAN_STD'>,
#                                   <FeatureType.STATE: 'STATE'>: <NormalizationMode.MEAN_STD: 'MEAN_STD'>,
#                                   <FeatureType.VISUAL: 'VISUAL'>: <NormalizationMode.MEAN_STD: 'MEAN_STD'>},
#                         stats={'action': {'count': array([44477]),
#                                           'max': array([ 44.,  36.,  30., 106., 170., 100.]),
#                                           'mean': array([-19.64921206,  -5.29050976,   7.78123529,  43.36043356, -8.83301488,  21.80655179]),
#                                           'min': array([ -94.,  -66.,  -24.,  -12., -190.,    0.]),
#                                           'q01': array([-4.63188639e+01, -3.06883811e+01,  4.20995121e+00,  3.34694336e+00, -4.54644973e+01, -1.00000001e-10]),
#                                           'q10': array([-4.15948618e+01, -2.48824781e+01,  5.36758863e+00,  5.97271410e+00, -3.74821706e+01, -1.00000001e-10]),
#                                           'q50': array([-19.65496605,  -5.45661242,   7.66819607,  48.12446236, -7.93481527,  13.92888795]),
#                                           'q90': array([ 1.86853888, 13.24677393, 10.09128552, 70.59173646, 19.86448968, 58.42846699]),
#                                           'q99': array([ 4.83021756, 17.9447024 , 11.76326373, 73.7324893 , 22.5608679 , 67.96958746]),
#                                           'std': array([31.75556514, 15.78888885,  7.94941423, 26.88264617, 62.66358341, 27.0929065 ])},
#                                'episode_index': {'count': array([44477]),
#                                                  'max': array([35]),
#                                                  'mean': array([17.90990849]),
#                                                  'min': array([0]),
#                                                  'q01': array([17.90990849]),
#                                                  'q10': array([17.90990849]),
#                                                  'q50': array([17.90990849]),
#                                                  'q90': array([17.90990849]),
#                                                  'q99': array([17.90990849]),
#                                                  'std': array([10.47138417])},
#                                'frame_index': {'count': array([44477]),
#                                                'max': array([2878]),
#                                                'mean': array([704.13685725]),
#                                                'min': array([0]),
#                                                'q01': array([13.59966144]),
#                                                'q10': array([140.46017207]),
#                                                'q50': array([704.02510722]),
#                                                'q90': array([1267.75912411]),
#                                                'q99': array([1394.67405307]),
#                                                'std': array([516.60467865])},
#                                'index': {'count': array([44477]),
#                                          'max': array([44476]),
#                                          'mean': array([22238.]),
#                                          'min': array([0]),
#                                          'q01': array([21547.46280418]),
#                                          'q10': array([21674.32331481]),
#                                          'q50': array([22237.92090309]),
#                                          'q90': array([22801.62226686]),
#                                          'q99': array([22928.53719582]),
#                                          'std': array([12839.40395813])},
#                                'observation.images.main': {'count': array([7401]),
#                                                            'max': array([[[1.]], [[1.]], [[1.]]]),
#                                                            'mean': array([[[0.49461724]], [[0.52729661]], [[0.50856053]]]),
#                                                            'min': array([[[0.]], [[0.]], [[0.]]]),
#                                                            'q01': array([[[0.01148206]], [[0.0342861 ]], [[0.01785431]]]),
#                                                            'q10': array([[[0.17333939]], [[0.19758905]], [[0.13570303]]]),
#                                                            'q50': array([[[0.54634109]], [[0.57438039]], [[0.55968274]]]),
#                                                            'q90': array([[[0.67223193]], [[0.69470851]], [[0.67412841]]]),
#                                                            'q99': array([[[0.7326797 ]], [[0.75631402]], [[0.74644456]]]),
#                                                            'std': array([[[0.00983558]], [[0.00893308]], [[0.00844279]]])},
#                                'observation.images.secondary_0': {'count': array([7401]),
#                                                                   'max': array([[[1.]], [[1.]], [[1.]]]),
#                                                                   'mean': array([[[0.47281133]], [[0.51229607]], [[0.49637884]]]),
#                                                                   'min': array([[[0.]], [[0.]], [[0.]]]),
#                                                                   'q01': array([[[0.00067407]], [[0.028357  ]], [[0.01061038]]]),
#                                                                   'q10': array([[[0.10501999]], [[0.16531826]], [[0.15353425]]]),
#                                                                   'q50': array([[[0.51215447]], [[0.548981  ]], [[0.53380709]]]),
#                                                                   'q90': array([[[0.68493358]], [[0.69738687]], [[0.67372691]]]),
#                                                                   'q99': array([[[0.72548297]], [[0.73909955]], [[0.7143888 ]]]),
#                                                                   'std': array([[[0.00613318]], [[0.0056067 ]], [[0.00597626]]])},
#                                'observation.state': {'count': array([44477]),
#                                                      'max': array([ 43.78022003,  29.89011002,  28.04395676, 104.70330048, 169.53846741,  96.91928864]),
#                                                      'mean': array([-19.83360271,  -1.55477901,  10.03636843,  43.92240415,  -8.81835322,  25.58344147]),
#                                                      'min': array([ -91.51648712,  -64.17582703,  -18.72527504,  -10.19780254, -175.1648407 ,    2.46457171]),
#                                                      'q01': array([-46.05638115, -25.3437611 ,   6.11485266,   4.61507222, -44.80516969,   3.33113935]),
#                                                      'q10': array([-41.49288138, -19.27852513,   7.44596001,   7.14204352, -37.37604824,   3.96096673]),
#                                                      'q50': array([-19.85780063,  -1.13349616,   9.85105988,  48.53190748,  -7.7516359 ,  21.29441657]),
#                                                      'q90': array([ 1.24023705, 14.14789313, 12.84428234, 70.60142976, 19.74469568, 57.57270839]),
#                                                      'q99': array([ 4.41915271, 18.12275565, 14.13821149, 73.67940846, 22.50118072, 66.37150892]),
#                                                      'std': array([31.46413008, 14.54916525,  7.66009021, 26.2955687 , 62.35283993, 23.82802269])},
#                                'task_index': {'count': array([44477]),
#                                               'max': array([0]),
#                                               'mean': array([0.]),
#                                               'min': array([0]),
#                                               'q01': array([4.e-16]),
#                                               'q10': array([4.e-15]),
#                                               'q50': array([2.e-14]),
#                                               'q90': array([3.6e-14]),
#                                               'q99': array([3.96e-14]),
#                                               'std': array([0.])},
#                                'timestamp': {'count': array([44477]),
#                                              'max': array([95.93333333]),
#                                              'mean': array([23.47122858]),
#                                              'min': array([0.]),
#                                              'q01': array([0.45332205]),
#                                              'q10': array([4.68200574]),
#                                              'q50': array([23.46491014]),
#                                              'q90': array([42.25863747]),
#                                              'q99': array([46.4891351]),
#                                              'std': array([17.22015595])}},
#                         device='cpu',
#                         dtype=torch.float32,
#                         eps=1e-08,
#                         normalize_observation_keys=None)
# 'Step 0: UnnormalizerProcessorStep'
# UnnormalizerProcessorStep(features={'action': PolicyFeature(type=<FeatureType.ACTION: 'ACTION'>, shape=(6,))},
#                           norm_map={<FeatureType.ACTION: 'ACTION'>: <NormalizationMode.MEAN_STD: 'MEAN_STD'>,
#                                     <FeatureType.STATE: 'STATE'>: <NormalizationMode.MEAN_STD: 'MEAN_STD'>,
#                                     <FeatureType.VISUAL: 'VISUAL'>: <NormalizationMode.MEAN_STD: 'MEAN_STD'>},
#                           stats={'action': {'count': array([44477]),
#                                             'max': array([ 44.,  36.,  30., 106., 170., 100.]),
#                                             'mean': array([-19.64921206,  -5.29050976,   7.78123529,  43.36043356, -8.83301488,  21.80655179]),
#                                             'min': array([ -94.,  -66.,  -24.,  -12., -190.,    0.]),
#                                             'q01': array([-4.63188639e+01, -3.06883811e+01,  4.20995121e+00,  3.34694336e+00, -4.54644973e+01, -1.00000001e-10]),
#                                             'q10': array([-4.15948618e+01, -2.48824781e+01,  5.36758863e+00,  5.97271410e+00, -3.74821706e+01, -1.00000001e-10]),
#                                             'q50': array([-19.65496605,  -5.45661242,   7.66819607,  48.12446236, -7.93481527,  13.92888795]),
#                                             'q90': array([ 1.86853888, 13.24677393, 10.09128552, 70.59173646, 19.86448968, 58.42846699]),
#                                             'q99': array([ 4.83021756, 17.9447024 , 11.76326373, 73.7324893 , 22.5608679 , 67.96958746]),
#                                             'std': array([31.75556514, 15.78888885,  7.94941423, 26.88264617, 62.66358341, 27.0929065 ])},
#                                  'episode_index': {'count': array([44477]),
#                                                    'max': array([35]),
#                                                    'mean': array([17.90990849]),
#                                                    'min': array([0]),
#                                                    'q01': array([17.90990849]),
#                                                    'q10': array([17.90990849]),
#                                                    'q50': array([17.90990849]),
#                                                    'q90': array([17.90990849]),
#                                                    'q99': array([17.90990849]),
#                                                    'std': array([10.47138417])},
#                                  'frame_index': {'count': array([44477]),
#                                                  'max': array([2878]),
#                                                  'mean': array([704.13685725]),
#                                                  'min': array([0]),
#                                                  'q01': array([13.59966144]),
#                                                  'q10': array([140.46017207]),
#                                                  'q50': array([704.02510722]),
#                                                  'q90': array([1267.75912411]),
#                                                  'q99': array([1394.67405307]),
#                                                  'std': array([516.60467865])},
#                                  'index': {'count': array([44477]),
#                                            'max': array([44476]),
#                                            'mean': array([22238.]),
#                                            'min': array([0]),
#                                            'q01': array([21547.46280418]),
#                                            'q10': array([21674.32331481]),
#                                            'q50': array([22237.92090309]),
#                                            'q90': array([22801.62226686]),
#                                            'q99': array([22928.53719582]),
#                                            'std': array([12839.40395813])},
#                                  'observation.images.main': {'count': array([7401]),
#                                                              'max': array([[[1.]], [[1.]], [[1.]]]),
#                                                              'mean': array([[[0.49461724]], [[0.52729661]], [[0.50856053]]]),
#                                                              'min': array([[[0.]], [[0.]], [[0.]]]),
#                                                              'q01': array([[[0.01148206]], [[0.0342861 ]], [[0.01785431]]]),
#                                                              'q10': array([[[0.17333939]], [[0.19758905]], [[0.13570303]]]),
#                                                              'q50': array([[[0.54634109]], [[0.57438039]], [[0.55968274]]]),
#                                                              'q90': array([[[0.67223193]], [[0.69470851]], [[0.67412841]]]),
#                                                              'q99': array([[[0.7326797 ]], [[0.75631402]], [[0.74644456]]]),
#                                                              'std': array([[[0.00983558]], [[0.00893308]], [[0.00844279]]])},
#                                  'observation.images.secondary_0': {'count': array([7401]),
#                                                                     'max': array([[[1.]], [[1.]], [[1.]]]),
#                                                                     'mean': array([[[0.47281133]], [[0.51229607]], [[0.49637884]]]),
#                                                                     'min': array([[[0.]], [[0.]], [[0.]]]),
#                                                                     'q01': array([[[0.00067407]], [[0.028357  ]], [[0.01061038]]]),
#                                                                     'q10': array([[[0.10501999]], [[0.16531826]], [[0.15353425]]]),
#                                                                     'q50': array([[[0.51215447]], [[0.548981  ]], [[0.53380709]]]),
#                                                                     'q90': array([[[0.68493358]], [[0.69738687]], [[0.67372691]]]),
#                                                                     'q99': array([[[0.72548297]], [[0.73909955]], [[0.7143888 ]]]),
#                                                                     'std': array([[[0.00613318]], [[0.0056067 ]], [[0.00597626]]])},
#                                  'observation.state': {'count': array([44477]),
#                                                        'max': array([ 43.78022003,  29.89011002,  28.04395676, 104.70330048, 169.53846741,  96.91928864]),
#                                                        'mean': array([-19.83360271,  -1.55477901,  10.03636843,  43.92240415, -8.81835322,  25.58344147]),
#                                                        'min': array([ -91.51648712,  -64.17582703,  -18.72527504,  -10.19780254, -175.1648407 ,    2.46457171]),
#                                                        'q01': array([-46.05638115, -25.3437611 ,   6.11485266,   4.61507222, -44.80516969,   3.33113935]),
#                                                        'q10': array([-41.49288138, -19.27852513,   7.44596001,   7.14204352, -37.37604824,   3.96096673]),
#                                                        'q50': array([-19.85780063,  -1.13349616,   9.85105988,  48.53190748, -7.7516359 ,  21.29441657]),
#                                                        'q90': array([ 1.24023705, 14.14789313, 12.84428234, 70.60142976, 19.74469568, 57.57270839]),
#                                                        'q99': array([ 4.41915271, 18.12275565, 14.13821149, 73.67940846, 22.50118072, 66.37150892]),
#                                                        'std': array([31.46413008, 14.54916525,  7.66009021, 26.2955687 , 62.35283993, 23.82802269])},
#                                  'task_index': {'count': array([44477]),
#                                                 'max': array([0]),
#                                                 'mean': array([0.]),
#                                                 'min': array([0]),
#                                                 'q01': array([4.e-16]),
#                                                 'q10': array([4.e-15]),
#                                                 'q50': array([2.e-14]),
#                                                 'q90': array([3.6e-14]),
#                                                 'q99': array([3.96e-14]),
#                                                 'std': array([0.])},
#                                  'timestamp': {'count': array([44477]),
#                                                'max': array([95.93333333]),
#                                                'mean': array([23.47122858]),
#                                                'min': array([0.]),
#                                                'q01': array([0.45332205]),
#                                                'q10': array([4.68200574]),
#                                                'q50': array([23.46491014]),
#                                                'q90': array([42.25863747]),
#                                                'q99': array([46.4891351]),
#                                                'std': array([17.22015595])}},
#                           device=None,
#                           dtype=torch.float32,
#                           eps=1e-08,
#                           normalize_observation_keys=None)
# 'Step 1: DeviceProcessorStep'
# DeviceProcessorStep(device='cpu', float_dtype=None)


# BLOCK 6 ─ Build Dataset with Delta Timestamps (Action Chunking)
# delta_timestamps tells LeRobotDataset which temporal offsets to load.
# For ACT we need:
#   • observations at t=0 only (n_obs_steps=1 for ACT)
#   • actions from t=0 to t=(chunk_size-1)/fps  (the future chunk)
# ACTConfig exposes helper indices:
#   cfg.action_delta_indices      → [0, 1, …, chunk_size-1]
#   cfg.observation_delta_indices → [0]   (ACT uses single obs step)
#   cfg.image_features            → list of image feature keys in input_features

print("\nBLOCK 6: Building LeRobotDataset with delta_timestamps")

# https://www.mintlify.com/huggingface/lerobot/concepts/policies#temporal-structure
# Define timestamps in seconds directly (simpler)

delta_timestamps = {
    # Only current observation (t=0)
    "observation.state": [0.0],
    "observation.images.main": [0.0],
    "observation.images.secondary_0": [0.0],
    # Future actions for chunk (t=0 to t=chunk_size/fps)
    "action": [i / DATASET_FPS for i in range(CHUNK_SIZE)],
}

print("Delta timestamps configured:")
print(f"  Observations: {delta_timestamps['observation.state']}")
print(f"  Actions: {len(delta_timestamps['action'])} steps over {delta_timestamps['action'][-1]:.2f}s")

# Load dataset
dataset = LeRobotDataset(
    repo_id=DATASET_REPO_ID,
    delta_timestamps=delta_timestamps,
)
print(f"  Dataset frames available: {len(dataset)}")

sample = dataset[100]
processed = preprocessor(sample)

# BLOCK 7 ─ DataLoader and Optimizer
print("\nBLOCK 7: Building DataLoader and optimizer …")

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
    drop_last=True,
    persistent_workers=(NUM_WORKERS > 0),
)

print(f"  Batches per pass: {len(dataloader)}")
print(f"  Effective epoch size: {len(dataloader) * BATCH_SIZE} samples")

test_batch = next(iter(dataloader))
print("\nTest batch shapes:")
for key, value in test_batch.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: {value.shape}")

# Test batch shapes:
#   observation.images.main: torch.Size([16, 3, 480, 640])
#   observation.images.secondary_0: torch.Size([16, 3, 480, 640])
#   observation.state: torch.Size([16, 1, 6])
#   action: torch.Size([16, 50, 6])
#   timestamp: torch.Size([16])
#   frame_index: torch.Size([16])
#   episode_index: torch.Size([16])
#   index: torch.Size([16])
#   task_index: torch.Size([16])
#   observation.state_is_pad: torch.Size([16, 1])
#   observation.images.main_is_pad: torch.Size([16, 1])
#   observation.images.secondary_0_is_pad: torch.Size([16, 1])
#   action_is_pad: torch.Size([16, 50])

# ACTConfig provides a pre-configured AdamW optimizer preset tuned for ACT.
# It supports different LRs for backbone vs rest of the model.
optimizer = cfg.get_optimizer_preset().build(policy.get_optim_params())
# AdamW (
# Parameter Group 0
#     amsgrad: False
#     betas: (0.9, 0.999)
#     capturable: False
#     decoupled_weight_decay: True
#     differentiable: False
#     eps: 1e-08
#     foreach: None
#     fused: None
#     lr: 1e-05
#     maximize: False
#     weight_decay: 0.0001

# Parameter Group 1
#     amsgrad: False
#     betas: (0.9, 0.999)
#     capturable: False
#     decoupled_weight_decay: True
#     differentiable: False
#     eps: 1e-08
#     foreach: None
#     fused: None
#     lr: 1e-05
#     maximize: False
#     weight_decay: 0.0001
# )

print("BLOCK 7: DataLoader and optimizer ready")

# BLOCK 8 ─ Training Loop
#
# Flow per batch:
#   1. batch  = raw tensors from dataset  (degrees, raw-ish pixels)
#   2. batch  = preprocessor(batch)       → normalised tensors (model-friendly)
#   3. loss,_ = policy.forward(batch)     → ACT loss = L2 recon + KL * kl_weight
#   4. loss.backward() + optimizer.step()
#   5. At save checkpoints: save policy + preprocessor + postprocessor together
#      (postprocessor is needed at inference to de-normalise back to degrees)
#
print("\nBLOCK 8: Training …\n")


batch = next(iter(dataloader))
batch = preprocessor(batch)

for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)
assert batch["observation.images.main"].dim() == 4
assert batch["observation.state"].shape[1] == 1
assert batch["action"].shape[1] == CHUNK_SIZE


# Training loop
print("\nStarting training...\n")

step = 0
running_loss = 0.0
pbar = tqdm(total=TRAINING_STEPS, desc="Training", dynamic_ncols=True)

while step < TRAINING_STEPS:
    for batch in dataloader:
        # Move to device
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Preprocess (normalizes actions/states/images)
        batch = preprocessor(batch)
        
        # ── FIX ────────────────────────────────────────────────────────────
        # observation.state: [B, 1, 6] → [B, 6]
        # ACT's internal transformer cat fails with the extra time dimension.
        batch["observation.state"] = batch["observation.state"].squeeze(1)
        # ───────────────────────────────────────────────────────────────────
 
        # Forward pass
        loss, output_dict = policy.forward(batch)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        optimizer.zero_grad()
        
        # Logging
        running_loss += loss.item()
        step += 1
        pbar.update(1)
        
        if step % LOG_FREQ == 0:
            avg_loss = running_loss / LOG_FREQ
            log_dict = {"loss": f"{avg_loss:.4f}"}
            
            if "l1_loss" in output_dict:
                log_dict["l1"] = f"{output_dict['l1_loss']:.4f}"
            if "kl_loss" in output_dict:
                log_dict["kl"] = f"{output_dict['kl_loss']:.4f}"
            
            pbar.set_postfix(log_dict)
            running_loss = 0.0
        
        # Checkpoint
        if step % SAVE_FREQ == 0:
            ckpt_dir = OUTPUT_DIR / f"checkpoint_{step:07d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            policy.save_pretrained(ckpt_dir)
            preprocessor.save_pretrained(ckpt_dir)
            postprocessor.save_pretrained(ckpt_dir)
            
            pbar.write(f"Checkpoint saved to {ckpt_dir}")
        
        if step >= TRAINING_STEPS:
            break
        break

pbar.close()

print("\nBLOCK 8: Training complete!")

# BLOCK 9 ─ Save Final Model
print("\nBLOCK 9: Saving final model")

final_dir = OUTPUT_DIR / "final_model"
final_dir.mkdir(parents=True, exist_ok=True)

policy.save_pretrained(final_dir)
preprocessor.save_pretrained(final_dir)
postprocessor.save_pretrained(final_dir)

print(f"BLOCK 9: Model saved → {final_dir}")

# %%
