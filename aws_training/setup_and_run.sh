#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# AWS A10 GRPO Training — Setup & Launch Script
# Run this on a fresh Ubuntu 22.04 instance (g5.xlarge / g5.2xlarge)
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

echo "═══════════════════════════════════════════════════════════════"
echo "  GRPO Training Setup — AWS A16"
echo "═══════════════════════════════════════════════════════════════"

# ── 1. System dependencies ───────────────────────────────────────────────
echo "[1/7] Installing system dependencies..."
sudo apt-get update -y
sudo apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    python3-pip git wget curl \
    build-essential cmake \
    nvidia-driver-535 \
    unzip

# ── 2. CUDA toolkit (if not pre-installed) ──────────────────────────────
echo "[2/7] Checking CUDA..."
if ! command -v nvcc &> /dev/null; then
    echo "Installing CUDA 12.2..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update -y
    sudo apt-get install -y cuda-toolkit-12-2
    export PATH=/usr/local/cuda-12.2/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
    echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi
nvcc --version || echo "CUDA toolkit not found (driver may be enough)"
nvidia-smi

# ── 3. Python virtual environment ────────────────────────────────────────
echo "[3/7] Creating Python environment..."
python3.11 -m venv ~/grpo_env
source ~/grpo_env/bin/activate
pip install --upgrade pip setuptools wheel

# ── 4. Install PyTorch + training dependencies ──────────────────────────
echo "[4/7] Installing PyTorch + dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core training stack (pinned versions from proven existing repo)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" 2>&1 | tail -3 || \
    pip install unsloth 2>&1 | tail -3
pip install --no-deps "trl==0.24.0" "peft==0.11.1" "accelerate==0.33.0" transformers datasets bitsandbytes

# Hackathon-required toolkits
pip install aligntune sglang auditkit 2>&1 | tail -3

# Logging
pip install wandb 2>&1 | tail -1

echo "Dependencies installed."

# ── 5. Download training script ──────────────────────────────────────────
echo "[5/7] Setting up training directory..."
mkdir -p ~/grpo_training
cd ~/grpo_training

# Copy training script (will be uploaded separately)
# The script should be at ~/grpo_training/grpo_train.py

# ── 6. Download data ────────────────────────────────────────────────────
echo "[6/7] Downloading heldout eval data..."
mkdir -p ~/grpo_training/data
# Heldout data will be uploaded alongside the script
# Alternatively, the script loads from HF Hub directly.

# ── 7. Environment variables ─────────────────────────────────────────────
echo "[7/7] Setting environment variables..."
cat >> ~/.bashrc << 'ENVEOF'

# GRPO Training Environment
export HF_TOKEN="__HF_TOKEN__"
export HF_USERNAME="Suryansh7123"
export HF_REPO="Suryansh7123/llama3.2-3b-grpo-metamath"
export OUTPUT_DIR="/tmp/grpo_output"
export WANDB_PROJECT="grpo-llama3b-metamath"

# CUDA
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENVEOF

# Replace placeholder with actual token
sed -i "s/__HF_TOKEN__/${HF_TOKEN:-}/g" ~/.bashrc

source ~/.bashrc

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  To start training:"
echo "    source ~/grpo_env/bin/activate"
echo "    cd ~/grpo_training"
echo "    nohup python grpo_train.py > training_stdout.log 2>&1 &"
echo "    tail -f training_stdout.log"
echo ""
echo "  To monitor:"
echo "    tail -f /tmp/grpo_output/training_log.txt"
echo "    tail -f /tmp/grpo_output/metrics.jsonl"
echo "═══════════════════════════════════════════════════════════════"
