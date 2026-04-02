import modal

app = modal.App("lerobot-training")

# -------------------------------------------------
# Image
# -------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        "lerobot",
        "wandb",
        "huggingface_hub",
        "datasets",
    )
)
# -------------------------------------------------
# Persistent volume
# -------------------------------------------------
volume = modal.Volume.from_name(
    "lerobot-outputs",
    create_if_missing=True,
)


# -------------------------------------------------
# Training function
# -------------------------------------------------
@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 6,
    volumes={"/outputs": volume},
    secrets=[modal.Secret.from_name("wandb-secret"),
         modal.Secret.from_name("hf-secret")]
,
)
def train(
    steps: int = 25000,
    batch: int = 16,
    policy:str = "act",
    job_name: str = "act_svla_so101_pickplace",
):
    import subprocess
    import os
    os.getenv("WANDB_API_KEY")
    os.makedirs("/outputs/train", exist_ok=True)

    # ---------------------------------
    # 1. Convert dataset
    # ---------------------------------
    convert_cmd = [
        "python",
        "-m",
        "lerobot.datasets.v30.convert_dataset_v21_to_v30",
        "--repo-id=lerobot/svla_so101_pickplace",
    ]

    print("Converting dataset...")
    subprocess.run(convert_cmd, check=True)
    print("Dataset converteed sucessfully")

    # ---------------------------------
    # 2. Train
    # ---------------------------------
    train_cmd = [
        f"lerobot-train",
        f"--dataset.repo_id=lerobot/svla_so101_pickplace",
        f"--policy.type={policy}",
        f"--policy.device=cuda",
        f"--batch_size={batch}",
        f"--job_name={job_name}",
        f"--output_dir=/outputs/train/{job_name}",
        f"--wandb.enable=true",
        f"--policy.push_to_hub=false",
        f"--steps={steps}",
        f"--resume=true",
        f"--config_path=/outputs/train/{job_name}/checkpoints/last/pretrained_model/train_config.json",
        "--save_freq=10000"
    ]

    print("Training...")
    subprocess.run(train_cmd, check=True)
    print("Training completed")

# -------------------------------------------------
# Local entrypoint
# -------------------------------------------------
@app.local_entrypoint()
def main():
    train.remote(
        steps=1_00_000,
        batch=16,
        policy="act",
        job_name="2-act_svla_so101_pickplace",
    )
