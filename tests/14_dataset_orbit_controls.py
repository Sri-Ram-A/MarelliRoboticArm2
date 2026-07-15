# New third_final Dataset
"""
(lerobot) ~/Documents/sr_proj/RoboticArm$ orbit analyze Sri-Ram-A/final_third --detail
Using cached embeddings
Array profile: absolute_position (6D, single-arm)

Outlier detection: 3/46 episodes flagged (healthy)

Running proxy training (BC-MLP baseline)... (~2 min)
  Running policy-matched proxy (ACT-mini)... (~5-10 min)
  Proxy check: GO (val_loss=0.0298)
  Validation loss dropped to 0.0298 (from 0.7584) — data is learnable
  MLP baseline loss ratio: 0.012 (lower = more learnable)
  Worst episodes: #0, #35, #33
  (9s, 1000 steps, ACT-mini)

╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│  ORBIT Analysis: Sri-Ram-A/final_third                                                                          │
│  45 episodes · 6 action dims · SO_FOLLOWER · ACT                                                                │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Dataset Readiness: A (score: 98/100)
Ready to train — expect strong results

  ✓ Episode quality is high — very few flagged episodes
  ✓ Sufficient episodes (45) for act
  ✓ Good cross-episode consistency (0.73)
  ✓ Good consistency (0.75)
  ✓ Good policy fit (0.80)
  ✓ Proxy training signal: GO (val_loss=0.0298)
  ✗ 2 joints clipping

Top action: 2 joints clipping

YOUR DATA AT A GLANCE
────────────────────────────────────────
  Episodes:       45    
  Coverage:       0.73  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░
  Signal Health:  0.83  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░

WHAT'S WORKING
  ✓ All 6 joints active — no dead servos detected
  ✓ Gripper transitions: 3.7/episode (sufficient for grasping tasks)
  ✓ Episode lengths consistent (CV=0.25)
  ✓ Directional bias on 5/6 joints — consistent with goal-directed task (expected)
  ✓ Episode count (45) meets minimum (30)
  ✓ Average smoothness good (1.00)

WHAT NEEDS ATTENTION
  ⚠ Episode 25 (low completion (0.03)), Episode 32 (low completion (0.10)), Episode 43 (low completion (0.07))
    → Inspect episode 25 for quality issues
  ⚠ Clipping detected on joint_3, joint_5

POLICY FIT
────────────────────────────────────────
  act: 0.80
  diffusion_policy: 0.77
  dp3: 0.77

EPISODE QUALITY RANKING
────────────────────────────────────────
  Keep: 42/46 (91%)  │  Review: 3 (6%)  │  Remove: 1 (2%)
  Worst episodes: #45 (score: 0.38), #27 (score: 0.67), #0 (score: 0.68)
  Removing them would improve quality: 0.79 → 0.80

ACTION DIVERGENCE
────────────────────────────────────────
  Score: 0.11 (low)

TEMPORAL ALIGNMENT
────────────────────────────────────────
  ✓ State-action alignment: lag = 5 frames (negligible, <2% of episode)

SCALING ADVICE
────────────────────────────────────────
  Collect ~155 more demonstrations (50/environment)
  Estimated diversity: ~4 environments, ~11 demos each

CONSISTENCY DIAGNOSTICS
────────────────────────────────────────
  Joint limit CV: 1.74 (high) — joint range usage varies
  Most inconsistent episodes: 26, 45, 31

NEXT STEPS (prioritized)
────────────────────────────────────────
  [HIGH] 2 joints clipping
  [HIGH] Collect 5 more episodes to reach the minimum recommended count of 50
  [MEDIUM] Inspect episode 25 for quality issues
  [MEDIUM] Review episodes 0, 44, 45 — they appear significantly different from others (possible recording errors)

Run `orbit suggest Sri-Ram-A/final_third` to get a ready-to-run training command.
 
             Recommended: Action Chunking Transformer (fit: 0.79)             
┌─────────────┬──────────────────────────────────────────────────────────────┐
│ Episodes    │ 45                                                           │
│ Cameras     │ 2 x 480p                                                     │
│ Action dims │ 6                                                            │
│ FPS         │ 30                                                           │
│ Alternative │ Diffusion Policy (fit: 0.76) — Close fit — scores 0.76 vs    │
│             │ 0.79                                                         │
└─────────────┴──────────────────────────────────────────────────────────────┘

Copy and run:

lerobot-train \
  --dataset.repo_id=Sri-Ram-A/final_third \
  --policy.type=act \
  --batch_size=16 \
  --steps=100000 \
  --policy.chunk_size=58 \
  --policy.n_action_steps=20 \
  --log_freq=200 \
  --save_freq=20000 \
  --eval_freq=20000 \
  --policy.device=cuda \
  --output_dir=outputs/train/final_third-act \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.random_order=true

Training tips:
  • Loss should drop below 0.1 by step 20000
  • If loss plateaus above 0.2: your demonstrations may be too inconsistent
  • If robot doesn't move during eval: check action normalization stats
"""