#!/usr/bin/env bash
#
# startup.sh — Google Cloud VM startup script for GRPO training
# Targets: NVIDIA L4 or T4 GPU on Debian/Ubuntu-based GCP images
#
# DESIGN: Two-phase execution for fast iterations
#   Phase 1 (ONE-TIME): Install Python, venv, deps, download model + dataset
#   Phase 2 (EVERY RESTART): Download latest training script, start training
#
# A marker file ~/grpo_env/.setup_done gates Phase 1.
# To force a full re-setup, delete that file and reboot.
# To iterate on training code, just reset the VM — Phase 1 is skipped entirely.
#
set -eo pipefail
export HOME="${HOME:-/root}"
export USER="${USER:-root}"
exec > >(tee -a /var/log/startup-script.log) 2>&1

echo "=============================================="
echo "  GRPO Training VM — Startup $(date -u)"
echo "=============================================="

# -------------------------------------------------------------------
# Helper: read GCP metadata (custom metadata is NOT auto-injected as env vars)
# -------------------------------------------------------------------
get_metadata() {
    local key="$1"
    curl -s -f -H "Metadata-Flavor: Google" \
        "http://metadata.google.internal/computeMetadata/v1/instance/attributes/${key}" 2>/dev/null || echo ""
}

# -------------------------------------------------------------------
# 0. Configuration from metadata
# -------------------------------------------------------------------
export HF_TOKEN="$(get_metadata HF_TOKEN)"
export HF_USERNAME="$(get_metadata HF_USERNAME)"
export HF_REPO="$(get_metadata HF_REPO)"
export WANDB_PROJECT="$(get_metadata WANDB_PROJECT)"
export WANDB_API_KEY="$(get_metadata WANDB_API_KEY)"
export MODEL_NAME="$(get_metadata MODEL_NAME)"
export DATASET_NAME="$(get_metadata DATASET_NAME)"
TRAINING_SCRIPT_URL="$(get_metadata TRAINING_SCRIPT_URL)"
TRAINING_SCRIPT_PATH="$HOME/train_grpo.py"
SETUP_MARKER="$HOME/grpo_env/.setup_done"
# Extra CLI args for the training script (override hyperparameters without changing code)
EXTRA_TRAIN_ARGS="$(get_metadata EXTRA_TRAIN_ARGS)"
# Force full re-setup (set to "1" to delete marker and re-cache model/dataset)
FORCE_SETUP="$(get_metadata FORCE_SETUP)"

echo "HF_USERNAME=${HF_USERNAME:-(none)}"
echo "HF_REPO=${HF_REPO:-(none)}"
echo "MODEL_NAME=${MODEL_NAME:-(default)}"
echo "TRAINING_SCRIPT_URL=${TRAINING_SCRIPT_URL:-(none)}"
echo "EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-(none)}"

# -------------------------------------------------------------------
# 1. ONE-TIME: Install Python 3.10
# -------------------------------------------------------------------
install_python() {
    # Detect any available Python 3.10+ (venv built with whichever version is present)
    if command -v python3.10 &>/dev/null; then
        PY_VER="3.10"
    elif command -v python3.11 &>/dev/null; then
        PY_VER="3.11"
    elif command -v python3.12 &>/dev/null; then
        PY_VER="3.12"
    else
        echo "[…] Installing Python 3.10 …"
        export DEBIAN_FRONTEND=noninteractive
        sudo apt-get update -y -qq
        sudo apt-get install -y -qq software-properties-common curl wget git \
            build-essential libssl-dev zlib1g-dev libffi-dev \
            jq unzip ca-certificates gnupg lsb-release \
            python3.10 python3.10-venv python3.10-dev
        PY_VER="3.10"
        sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
    fi
    echo "[✓] Using Python ${PY_VER}: $(python${PY_VER} --version)"
}

# -------------------------------------------------------------------
# 2. ONE-TIME: Create virtual environment
# -------------------------------------------------------------------
setup_venv() {
    local VENV="$HOME/grpo_env"
    if [[ ! -d "$VENV" ]]; then
        echo "[…] Creating virtual environment at $VENV …"
        # Use whatever python3 is available (3.10, 3.11, 3.12, …)
        python3 -m venv "$VENV"
    else
        echo "[✓] Virtual environment already exists at $VENV"
    fi
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    echo "[✓] Activated $VENV (python: $(python --version 2>&1))"
}

# -------------------------------------------------------------------
# 3. ONE-TIME: Install Python dependencies
# -------------------------------------------------------------------
install_deps() {
    PIP="$HOME/grpo_env/bin/pip"
    PY="$HOME/grpo_env/bin/python"
    echo "[…] Upgrading pip …"
    "$PIP" install --upgrade pip setuptools wheel -q

    echo "[…] Checking for pre-installed torch …"
    if "$PY" -c "import torch; print(torch.__version__)" 2>/dev/null; then
        echo "[✓] PyTorch already installed (DL VM image)"
    else
        echo "[…] Installing torch …"
        "$PIP" install torch --index-url https://download.pytorch.org/whl/cu121 -q
    fi

    echo "[…] Installing training stack …"
    "$PIP" install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" 2>&1 | tail -3
    # Pin transformers to satisfy unsloth-zoo's constraint (<=5.17.0)
    "$PIP" install "transformers<=5.17.0" 2>&1 | tail -3
    "$PIP" install trl peft accelerate 2>&1 | tail -3

    echo "[…] Installing supporting libraries …"
    "$PIP" install datasets bitsandbytes wandb 2>&1 | tail -3

    echo "[…] Installing Lexsi Labs toolkits (optional, hackathon compliance) …"
    "$PIP" install "aligntune @ git+https://github.com/LexsiLabs/aligntune.git" 2>&1 | tail -3 || echo "[!] aligntune failed (optional)"
    "$PIP" install "auditkit @ git+https://github.com/LexsiLabs/auditkit.git" 2>&1 | tail -3 || echo "[!] auditkit failed (optional)"
    "$PIP" install sglang 2>&1 | tail -3 || echo "[!] sglang failed (optional)"

    echo "[✓] All Python dependencies installed"
}

# -------------------------------------------------------------------
# 4. ONE-TIME: Pre-download model & dataset into HF cache
#    This is the expensive step — we cache it so restarts skip it.
# -------------------------------------------------------------------
pre_download_assets() {
    echo "[…] Pre-downloading model & dataset (one-time, ~5-10 min) …"
    "$HOME/grpo_env/bin/python" << 'PYEOF'
import os, sys

model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
dataset_name = os.environ.get("DATASET_NAME", "meta-math/MetaMathQA")
hf_token = os.environ.get("HF_TOKEN", "")

if not hf_token:
    print("[cache] WARNING: HF_TOKEN not set — gated models will fail")
else:
    print(f"[cache] HF_TOKEN present ({len(hf_token)} chars)")

kwargs = {"trust_remote_code": True}
if hf_token:
    kwargs["token"] = hf_token

print(f"[cache] Downloading tokenizer for {model_name} ...")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model_name, **kwargs)
print(f"[cache] Tokenizer cached: {tok.__class__.__name__}")

print(f"[cache] Downloading model weights for {model_name} ...")
from transformers import AutoModelForCausalLM
import torch
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cpu",
    **kwargs,
)
print(f"[cache] Model cached: {model.num_parameters():,} params")

print(f"[cache] Downloading dataset {dataset_name} ...")
from datasets import load_dataset
ds = load_dataset(dataset_name, split="train")
print(f"[cache] Dataset cached: {len(ds)} rows")

print("[cache] All assets cached successfully.")
PYEOF
    echo "[✓] Model and dataset cached"
}

# -------------------------------------------------------------------
# 5. Verify GPU (every run, but fast)
# -------------------------------------------------------------------
verify_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        echo "--- GPU Info ---"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
        echo "-----------------"
    else
        echo "[!] nvidia-smi not available"
    fi
}

# -------------------------------------------------------------------
# 6. EVERY RUN: Download latest training script from GCS
#    This is small (a few KB) and fast.
# -------------------------------------------------------------------
download_training_script() {
    if [[ -z "$TRAINING_SCRIPT_URL" ]]; then
        echo "[!] TRAINING_SCRIPT_URL not set — skipping"
        return
    fi

    echo "[…] Downloading training script from $TRAINING_SCRIPT_URL …"
    # Use gcloud storage (works on DL VM) instead of gsutil (may be broken)
    if command -v gcloud &>/dev/null; then
        gcloud storage cp "$TRAINING_SCRIPT_URL" "$TRAINING_SCRIPT_PATH"
    else
        gsutil cp "$TRAINING_SCRIPT_URL" "$TRAINING_SCRIPT_PATH"
    fi
    chmod +x "$TRAINING_SCRIPT_PATH" 2>/dev/null || true
    echo "[✓] Training script saved to $TRAINING_SCRIPT_PATH"
}

# -------------------------------------------------------------------
# 7. EVERY RUN: Start training with nohup
# -------------------------------------------------------------------
start_training() {
    if [[ ! -f "$TRAINING_SCRIPT_PATH" ]]; then
        echo "[!] No training script at $TRAINING_SCRIPT_PATH — cannot start"
        return
    fi

    # Kill any previously started training processes (e.g. from earlier VM resets)
    # so we don't accumulate multiple trainers competing for the GPU.
    if pgrep -f "train_grpo.py" >/dev/null 2>&1; then
        echo "[…] Killing existing training processes before starting new one …"
        pkill -f "train_grpo.py" 2>/dev/null || true
        sleep 2
    fi

    echo "[…] Starting training (background) …"
    VENV_PY="$HOME/grpo_env/bin/python"

    if [[ ! -x "$VENV_PY" ]]; then
        echo "[!] Venv Python not found at $VENV_PY — aborting"
        return
    fi

    # Use flock to ensure only ONE instance launches training, even if GCP
    # triggers the startup script twice (it runs on boot AND metadata updates,
    # sometimes within ~1s of each other). The exclusive lock guarantees
    # the first launcher kills any existing process and starts fresh, while
    # the second launcher finds training already running and skips.
    # Use flock to serialize training launches across concurrent startup-script
    # runs. GCP triggers the script on boot AND metadata updates, sometimes
    # within ~1s. The exclusive lock ensures only one launcher starts training.
    # We HOLD the lock for 10s after launch so the process becomes visible to
    # the second launcher's pgrep check before the lock is released.
    LOCK_FILE="$HOME/.grpo_training.lock"
    (
        flock -n 9 || {
            # Another launcher holds the lock — wait for it to finish, then verify
            echo "[…] Another launcher is starting training — waiting..."
            flock 9  # blocks until the other launcher releases the lock
            # Now verify the other launcher actually started training
            sleep 2
            if pgrep -f "train_grpo.py" >/dev/null 2>&1; then
                echo "[✓] Training started by the other launcher — skipping"
            else
                echo "[!] Other launcher finished but no training running — launching now"
                > "$HOME/training.log"
                nohup "$VENV_PY" -u "$TRAINING_SCRIPT_PATH" $EXTRA_TRAIN_ARGS >> "$HOME/training.log" 2>&1 &
                echo "[✓] Training started (PID $!). Logs: $HOME/training.log"
            fi
            exit 0
        }

        # We hold the exclusive lock — kill any stale process, then launch
        if pgrep -f "train_grpo.py" >/dev/null 2>&1; then
            echo "[…] Killing existing training process"
            pkill -9 -f "train_grpo.py" 2>/dev/null || true
            sleep 2
        fi

        > "$HOME/training.log"
        nohup "$VENV_PY" -u "$TRAINING_SCRIPT_PATH" $EXTRA_TRAIN_ARGS >> "$HOME/training.log" 2>&1 &
        TRAIN_PID=$!
        echo "[✓] Training started (PID $TRAIN_PID). Logs: $HOME/training.log"
        echo "    Metrics upload to GCS for external monitoring."

        # Hold the lock for 10s so the second launcher's flock blocks until
        # we're done, then sees the running process and skips.
        sleep 10
    ) 9>"$LOCK_FILE"

    # Print first few lines of log to confirm startup (non-blocking)
    sleep 5
    echo "─── Training log (first 20 lines) ──────────────────────────────"
    head -20 "$HOME/training.log" 2>/dev/null || echo "    (log not yet available)"
    echo "────────────────────────────────────────────────────────────────"
}

# -------------------------------------------------------------------
# Main — two-phase execution
# -------------------------------------------------------------------
main() {
    if [[ "$FORCE_SETUP" == "1" ]]; then
        echo "[…] FORCE_SETUP=1 — removing marker, re-running Phase 1"
        rm -f "$SETUP_MARKER"
    fi

    if [[ -f "$SETUP_MARKER" ]]; then
        echo "=============================================="
        echo "  FAST RESTART — skipping Phase 1 (setup)"
        echo "  Set FORCE_SETUP=1 to force full re-setup"
        echo "=============================================="
        source "$HOME/grpo_env/bin/activate"
    else
        echo "=============================================="
        echo "  FIRST RUN — Phase 1: full setup (~10-15 min)"
        echo "=============================================="
        install_python
        setup_venv
        install_deps
        pre_download_assets
        touch "$SETUP_MARKER"
        echo "=============================================="
        echo "  Phase 1 complete — marker: $SETUP_MARKER"
        echo "=============================================="
    fi

    # Phase 2: always runs (fast — just fetch code + start)
    verify_gpu
    download_training_script
    start_training

    echo "=============================================="
    echo "  Startup complete $(date -u)"
    echo "=============================================="
}

main "$@"
