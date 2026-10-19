"""Run training on a Modal GPU. Code is uploaded from this folder; runs/ and the Hugging Face cache live on Volumes.

Train (detached, survives closing the terminal):
    modal run --detach modal_app.py --config configs/release_12k.yaml

Interactive shell on the same GPU image with the same volumes (the SSH-style way in):
    modal shell modal_app.py::train
"""
import os
import subprocess

import modal

APP_DIR = "/root/tevro"
GPU = os.environ.get("TEVRO_GPU", "A100-80GB")  # release_12k needs ~46 GB at micro-batch 32
TIMEOUT_HOURS = 9  # hard cap on a training job

app = modal.App("tevro")
runs = modal.Volume.from_name("nirnay-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("nirnay-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync()
    .env({"HF_HOME": "/cache/huggingface", "PYTHONUNBUFFERED": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(".", APP_DIR, ignore=[".venv", ".git", "runs", "**/__pycache__", "*.pyc"])
)


@app.function(gpu=GPU, image=image, timeout=TIMEOUT_HOURS * 3600,
              volumes={f"{APP_DIR}/runs": runs, "/cache/huggingface": hf_cache})
def train(config: str, name: str = "", resume: str = ""):
    subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], check=False)
    cmd = ["python", "train.py", config]
    if name:
        cmd += ["--name", name]
    if resume:
        cmd += ["--resume", resume]
    try:
        subprocess.run(cmd, cwd=APP_DIR, check=True)
    finally:
        runs.commit()
        hf_cache.commit()


@app.function(gpu=GPU, image=image, timeout=20 * 60,  # eval only: 20 min cap
              volumes={f"{APP_DIR}/runs": runs, "/cache/huggingface": hf_cache})
def evaluate(checkpoint: str, max_questions: int = 5000, out: str = ""):
    """Evaluates a checkpoint on the test split (seed 0) and saves the printed metrics on the Volume."""
    result = subprocess.run(["python", "inference.py", checkpoint, "--split", "test",
                             "--max_questions", str(max_questions), "--batch_size", "16"],
                            cwd=APP_DIR, check=True, capture_output=True, text=True)
    print(result.stdout)
    if out:
        with open(f"{APP_DIR}/runs/{out}", "w") as f:
            f.write(result.stdout)
    runs.commit()
    hf_cache.commit()
    return result.stdout


@app.local_entrypoint()
def main(config: str, name: str = "", resume: str = ""):
    train.remote(config, name, resume)
