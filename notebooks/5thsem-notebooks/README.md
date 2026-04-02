Today is 5th Feb 2026
- python -c "import sys; print(sys.executable)"
- Trying : https://docs.phospho.ai/learn/ai-models#train-an-act-model-locally-with-lerobot
- Check if this is working (Works)
```bash
uv run lerobot/lerobot/scripts/train.py ^
  --dataset.repo_id=Sri-Ram-A/example_dataset ^
  --policy.type=act ^
  --output_dir=outputs/train/phoshobot_test ^
  --job_name=phosphobot_test ^
  --policy.device=cuda
```
API KEY : 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
Key information
Key ID : wandb_v1_LECCXd276BXfTCMuuY9wz2l7Ckv
Name : Untitled Key
Created : Feb 04, 2026
Owner : SRI RAM A

```bash
# Install wandb using uv not pip
(lerobot) C:\Users\SriRam.A\Documents\sr_proj\RoboticArm>uv add wandb
Resolved 247 packages in 4.01s
warning: The package `huggingface-hub==1.3.7` does not have an extra named `hf-transfer`
warning: The package `huggingface-hub==1.3.7` does not have an extra named `cli`
warning: The package `huggingface-hub==1.3.7` does not have an extra named `hf-transfer`
warning: The package `huggingface-hub==1.3.7` does not have an extra named `cli`
Audited 209 packages in 30ms

# login wandb using uv
(lerobot) C:\Users\SriRam.A\Documents\sr_proj\RoboticArm>uv run wandb login
wandb: Logging into https://api.wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)
wandb: Create a new API key at: https://wandb.ai/authorize?ref=models
wandb: Store your API key securely and do not share it.
wandb: Paste your API key and hit enter: 
wandb: No netrc file found, creating one.
wandb: Appending key for api.wandb.ai to your netrc file: C:\Users\SriRam.A\_netrc
wandb: Currently logged in as: srirama-ai23 (chandanamn-cs23-na) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin

```
uv pip install "huggingface_hub<0.24" # or else u will get an error

```bash
uv run lerobot/lerobot/scripts/train.py --dataset.repo_id=Sri-Ram-A/touch2 --policy.type=act --output_dir=outputs/train/phoshobot_test --job_name=phosphobot_test --policy.device=cuda --wandb.enable=true
```
Now I am getting error
```java
    dataset = LeRobotDataset(
  File "C:\Users\SriRam.A\Documents\sr_proj\RoboticArm\.venv\lib\site-packages\lerobot\common\datasets\lerobot_dataset.py", line 508, in __init__)
    timestamps = torch.stack(self.hf_dataset["timestamp"]).numpy()
TypeError: stack(): argument 'tensors' (position 1) must be tuple of Tensors, not Column
Download complete: : 86.5MB [01:02, 1.37MB/s]
```
Refer notebook : [C:\Users\SriRam.A\Documents\sr_proj\RoboticArm\aditya\tests\1.ipynb](C:\Users\SriRam.A\Documents\sr_proj\RoboticArm\aditya\tests\1.ipynb)

- Understand how the dataset is stored for lerobot v2.1 (Dont follow their instructions for installation)
https://docs.phospho.ai/learn/lerobot-dataset#how-is-a-lerobot-dataset-organized-on-disk
<dataset_name>/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       ├── observation.images.main/  (or your_camera_key_1)
│       │   ├── episode_000000.mp4
│       │   └── ...
│       ├── observation.images.secondary_0/ (or your_camera_key_2)
│       │   ├── episode_000000.mp4
│       │   └── ...
│       └── ...
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── tasks.jsonl
│   ├── episodes_stats.jsonl  (for v2.1) or stats.json (for v2.0)
│   └── README.md (often, for Hugging Face Hub)
└── README.md (top-level, for Hugging Face Hub)

uv run lerobot/lerobot/scripts/train.py --dataset.repo_id=lerobot/svla_so101_pickplace --policy.type=act --output_dir=outputs/train/svla_so101_pickplace --job_name=phosphobot_test --policy.device=cuda --wandb.enable=true
Same error

# Try with pusht dataset which is known to work
uv run lerobot/lerobot/scripts/train.py --dataset.repo_id=lerobot/pusht --policy.type=act --output_dir=outputs/train/pusht_test --job_name=pusht_test --policy.device=cpu --wandb.enable=true 

Now I am going to play around with this - https://github.com/phospho-app/lerobot

```bash
python lerobot/lerobot/scripts/visualize_dataset.py ^
    --repo-id lerobot/lerobot/svla_so101_pickplace ^
    --episode-index 0
```

(lerobot) C:\Users\SriRam.A\Documents\sr_proj\RoboticArm\aditya>uv pip show datasets
Using Python 3.10.19 environment at: C:\Users\SriRam.A\miniconda3\envs\lerobot
Name: datasets
Version: 3.6.0
Location: C:\Users\SriRam.A\miniconda3\envs\lerobot\Lib\site-packages
Requires: dill, filelock, fsspec, huggingface-hub, multiprocess, numpy, packaging, pandas, pyarrow, pyyaml, requests, tqdm, xxhash
Required-by: lerobot

uv add git+https://github.com/huggingface/lerobot
uv run python -m lerobot.scripts.train

cd lerobot
uv run python -m lerobot.scripts.train --dataset.repo_id=lerobot/svla_so101_pickplace --policy.type=act --output_dir=outputs/train/svla_so101_pickplace --job_name=phosphobot_test --policy.device=cuda 

# Leaving this ,Now for training - going back to lerobot fro training
<!-- https://huggingface.co/docs/lerobot/en/act -->

```bash
lerobot-train \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --policy.type=act \
  --output_dir=outputs/train/act_svla_so101_pickplace \
  --job_name=act_svla_so101_pickplace \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=Sri-Ram-A/act_svla_so101_pickplace_policy

```
- Asking GPT to train on modal using their docs - https://modal.com/docs/guide
- Recommended for automated environments: Environment variable
Set the WANDB_API_KEY environment variable to your API key. This is the most secure method for servers, Docker containers, or continuous integration (CI) systems, as it avoids exposing the key in plain text within your code or process lists.

- Refer this to see modal integration for training : [C:\Users\SriRam.A\Documents\sr_proj\RoboticArm\modal\1-train.py](C:\Users\SriRam.A\Documents\sr_proj\RoboticArm\modal\1-train.py)
- For correct training arguments i refered to - [](C:\Users\SriRam.A\Documents\sr_proj\RoboticArm\lerobot\src\lerobot\configs\train.py)
and also the dictionary output which prints in the terminal when u run lerobot-train ...
- You can off your laptop
```java
$ modal run --detach 1-train.py 
Note that running a local entrypoint in detached mode only keeps the last triggered Modal function alive after the parent process has been killed or disconnected.
✓ Initialized. View run at https://modal.com/apps/srirama-ai23/test-lerobot/ap-z7RgyESRMQ0oRslBb8dV9t
✓ Created objects.
├── 🔨 Created mount C:\Users\SriRam.A\Documents\sr_proj\RoboticArm\modal\1-train.py
└── 🔨 Created function train.
Converting dataset...
- Running (1/1 containers active)... View app at
https://modal.com/apps/srirama-ai23/test-lerobot/ap-z7RgyESRMQ0oRslBb8dV9t
Fetching 9 files:  33%|███▎      | 3/9 [00:01<00:02,  2.33it/s]Fetching 9 files:  89%|████████▉ | 8/9 [00:01<00:00,  4.97it/s]Fetching 9 files: 100%|██████████| 9/9 [00:01<00:00,  4.77it/s]
Dataset converteed sucessfully
Training...
# Whe user clicks Ctrl+C
✓ Shutting down Modal client.                                                                                         
The detached App will keep running. You can track its progress on the Dashboard:                                      
https://modal.com/apps/srirama-ai23/test-lerobot/ap-z7RgyESRMQ0oRslBb8dV9t
Stream App logs:
modal app logs ap-z7RgyESRMQ0oRslBb8dV9t
Stop the App:
modal app stop ap-z7RgyESRMQ0oRslBb8dV9t
```

- I got hugging face token error , so to reume training add 
```python
f"--resume=true",
f"--config_path=/outputs/train/{job_name}/checkpoints/last/pretrained_model/train_config.json"
```
- Now to push model to hugging face
modal volume cp lerobot-outputs /train .\lerobot-outputs\train

```bash
mkdir C:\Users\SriRam.A\Documents\sr_proj\RoboticArm\volumes\lerobot_outputs_local
modal volume get lerobot-outputs / C:\Users\SriRam.A\Documents\sr_proj\RoboticArm\volumes\lerobot_outputs_local
```

I actually tried this - modal volume get lerobot-rishav-outputs / lerobot-rishav-outputs
But I got this error in cmd : [Errno 13] Permission denied
```bash
huggingface-cli login
⚠️  Warning: 'huggingface-cli login' is deprecated. Use 'hf auth login' instead.
```
- https://huggingface.co/docs/lerobot/main/en/il_robots?utm_source=chatgpt.com#upload-policy-checkpoints

```bash
huggingface-cli upload ${HF_USER}/${JOB_NAME}${CKPT} \
  train/${JOB_NAME}/checkpoints/${CKPT}/pretrained_model
⚠️  Warning: 'huggingface-cli upload' is deprecated. Use 'hf upload' instead.

```
# particular folder from modal
modal volume get lerobot-rishav-outputs train/2-act_touch2/checkpoints/080000 volumes/lerobot-rishav-outputs/2_act_touch2_080000
modal volume get lerobot-outputs train/2-act_svla_so101_pickplace/checkpoints/100000 volumes/lerobot-outputs/2-act_svla_so101_pickplace_100000

(lerobot) C:\Users\SriRam.A\Documents\sr_proj\RoboticArm\modal\volumes\lerobot-rishav-outputs\2_act_touch2_080000\080000\pretrained_model > hf upload Sri-Ram-A/2-act_touch2_080000 .
[
  
](https://huggingface.co/Sri-Ram-A/2-act_touch2_080000/tree/main/.)
>hf upload Sri-Ram-A/2-act_svla_so101_pickplace_100000 \ volumes\lerobot-outputs\pretrained_model


(lerobot) C:\Users\SriRam.A\Documents\sr_proj\RoboticArm\modal\volumes\lerobot-outputs\2-act_svla_so101_pickplace_100000\100000\pretrained_model>hf upload Sri-Ram-A/2-act_svla_so101_pickplace_100000 .
https://huggingface.co/Sri-Ram-A/2-act_svla_so101_pickplace_100000/tree/main/.

- Inference : Control your robot with the ACT model
# Start server
python server.py --model_id Sri-Ram-A/2-act_svla_so101_pickplace_100000 --port=8090
python client.py
# Replace with <YOUR_HF_MODEL_ID>

# Inference gives "details" errror - shfiting to lerobot from 0-calibration.ipynb
lerobot-calibrate --robot.type=so100_follower --robot.port=COM3 --robot.id=so100
# Keyboard
! lerobot-teleoperate --robot.type=so100_follower --robot.port=COM9 --robot.id=so100 --teleop.type=keyboard
# Controller
! lerobot-teleoperate --robot.type=so100_follower --robot.port=COM3 --robot.id=so100 --teleop.type=gamepad
- Teleops not working getting error - StopIteration

python server.py --model_id=Sri-Ram-A/2-act_svla_so101_pickplace025000 --port=8090

lerobot-record  --robot.type=so100_follower  --dataset.repo_id=TANAY779/record_act  --policy.path=AdityaRege/so101-pick-place-act  --episodes=10 --port=COM9

lerobot-record  --robot.type=so100_follower --policy.path=AdityaRege/so101-pick-place-act --robot.port=COM9

lerobot-eval --policy.repo_id=AdityaRege/so101-pick-place-act --n_episodes=10

conda activate lerobot
python act_server.py --model_id=Sri-Ram-A/2-act_touch2025000 --port=8090      
python act_server.py --model_id=AdityaRege/so101-pick-place-act --port=8090      
