#!/bin/bash
# One-time setup for a rented Ubuntu GPU machine: system packages, uv, project environment, CUDA check.
set -e

cd "$(dirname "$0")"

if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

echo "Installing system packages..."
$SUDO apt-get update -y
$SUDO apt-get install -y tmux vim curl git

echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

echo "GPU / driver:"
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi not found: no NVIDIA driver on this machine."
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "Running uv sync..."
uv sync

# Make sure torch can use the GPU (otherwise training silently falls back to CPU)
uv run python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not visible to torch; check the driver version above.'
print('torch', torch.__version__, '| CUDA', torch.version.cuda, '|', torch.cuda.get_device_name(0))
"

echo "Setup complete. Run 'source \$HOME/.local/bin/env' to use uv in this shell."
