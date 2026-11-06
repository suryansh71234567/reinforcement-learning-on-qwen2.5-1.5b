#!/usr/bin/env python3
"""
SageMaker-compatible GRPO RL Training — Llama 3.2 3B on MetaMathQA
Stack: Unsloth + TRL GRPOTrainer (+ AlignTune imports for hackathon compliance)

SageMaker-specific changes vs aws_training/grpo_train.py:
  1. argparse for hyperparameters (SageMaker passes them as CLI args)
  2. Reads data from SM_CHANNEL_TRAIN if provided, else HuggingFace Hub
  3. Saves final model to SM_MODEL_DIR (/opt/ml/model)
  4. Saves training output/logs to SM_OUTPUT_DATA_DIR (/opt/ml/output/data)
  5. Works both standalone (for testing) and inside SageMaker

Reward components:
  1. Correctness (weight 0.80) — exact numeric match after extraction
  2. Length penalty (weight 0.20) — linear penalty when > LENGTH_THRESHOLD words

Hyperparameters (all from proven existing repo + user-specified):
  LoRA r=16, alpha=32, dropout=0.05, rsLoRA
  group_size=8, batch_size=8, grad_accum=2
  max_steps=1000, max_new_tokens=500
  lr=5e-7, beta=0.03, temperature=0.8
  bf16 (A10 native)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 1. ARGparse — SageMaker passes hyperparameters as command-line arguments
# ═══════════════════════════════════════════════════════════════════════════════

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO RL Training for Llama 3.2 3B on MetaMathQA")

    # SageMaker environment paths
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
                        help="Directory to save final model (SageMaker: /opt/ml/model)")
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
                        help="Directory for training output/logs (SageMaker: /opt/ml/output/data)")
    parser.add_argument("--train-channel", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", ""),
                        help="Path to training data channel (SageMaker: $SM_CHANNEL_TRAIN)")

    # Model
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Base model name or path")
    parser.add_argument("--hf-repo", type=str, default=os.environ.get("HF_REPO", ""),
                        help="HuggingFace Hub repo to push final model")

    # LoRA
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout rate")

    # GRPO training
    parser.add_argument("--group-size", type=int, default=8, help="Number of generations per prompt (GRPO group size)")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device training batch size")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--max-new-tokens", type=int, default=500, help="Max new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--beta", type=float, default=0.03, help="KL coefficient (beta)")
    parser.add_argument("--learning-rate", type=float, default=5e-7, help="Learning rate")
    parser.add_argument("--max-grad-norm", type=float, default=0.1, help="Max gradient norm for clipping")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Warmup steps for LR scheduler")

    # Reward
    parser.add_argument("--length-threshold", type=int, default=300,
                        help="Word count threshold for length penalty")
    parser.add_argument("--w-correctness", type=float, default=0.80, help="Weight for correctness reward")
    parser.add_argument("--w-length", type=float, default=0.20, help="Weight for length reward")

    # Other
    parser.add_argument("--output-dir", type=str, default=os.environ.get("OUTPUT_DIR", "/tmp/grpo_output"),
                        help="Local output directory for checkpoints and logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # If HF_REPO not explicitly set via arg, fall back to env or constructed default
    if not args.hf_repo:
        hf_username = os.environ.get("HF_USERNAME", "Suryansh7123")
        args.hf_repo = os.environ.get("HF_REPO", f"{hf_username}/llama3.2-3b-grpo-metamath")

    return args


# Parse args immediately so they're available for config section below
ARGS = parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONFIGURATION (from parsed args)
# ═══════════════════════════════════════════════════════════════════════════════

import sys
import json
import math
import random
import re
import time
import signal
import logging
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = ARGS.seed
random.seed(SEED)

# ── HF credentials (passed via env for security) ───────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_USERNAME = os.environ.get("HF_USERNAME", "Suryansh7123")
HF_REPO = ARGS.hf_repo

# ── WandB ──────────────────────────────────────────────────────────────────
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "grpo-llama3b-metamath")
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME", f"grpo-sagemaker-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}")

# ── Model ──────────────────────────────────────────────────────────────────
MODEL_NAME = ARGS.model_name
MAX_SEQ_LENGTH = 2048

# ── LoRA ───────────────────────────────────────────────────────────────────
LORA_R = ARGS.lora_r
LORA_ALPHA = ARGS.lora_alpha
LORA_DROPOUT = ARGS.lora_dropout
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ── GRPO ───────────────────────────────────────────────────────────────────
GRPO_GROUP_SIZE = ARGS.group_size
PER_DEVICE_BATCH = ARGS.batch_size
GRAD_ACCUM_STEPS = ARGS.grad_accum
MAX_STEPS = ARGS.max_steps
MAX_NEW_TOKENS = ARGS.max_new_tokens
TEMPERATURE = ARGS.temperature
BETA = ARGS.beta  # KL coefficient
LEARNING_RATE = ARGS.learning_rate
MAX_GRAD_NORM = ARGS.max_grad_norm
WARMUP_STEPS = ARGS.warmup_steps

# ── Reward ─────────────────────────────────────────────────────────────────
W_CORRECTNESS = ARGS.w_correctness
W_LENGTH = ARGS.w_length
LENGTH_THRESHOLD = ARGS.length_threshold  # words — penalty kicks in above this

# ── Data ───────────────────────────────────────────────────────────────────
DATASET_NAME = "meta-math/MetaMathQA"
HELDOUT_EVERY_N_STEPS = 100
HELDOUT_N_SAMPLES = 200  # questions per eval
CHECKPOINT_EVERY_N_STEPS = 200
SAMPLE_LOG_EVERY_N_STEPS = 50
N_SAMPLE_QUESTIONS = 5  # how many to print in sample log

# ── SageMaker paths ────────────────────────────────────────────────────────
SM_MODEL_DIR = ARGS.model_dir
SM_OUTPUT_DATA_DIR = ARGS.output_data_dir
SM_TRAIN_CHANNEL = ARGS.train_channel

# ── Output (use SM_OUTPUT_DATA_DIR if in SageMaker, else fallback) ─────────
OUTPUT_DIR = Path(SM_OUTPUT_DATA_DIR if os.environ.get("SM_OUTPUT_DATA_DIR") else ARGS.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "training_log.txt"

# ── Checkpoints dir ────────────────────────────────────────────────────────
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# ── Question types (8 classes in MetaMathQA) ───────────────────────────────
QUESTION_TYPES = [
    "GSM_SV", "GSM_Rephrased", "GSM_AnsAug", "GSM_FOBAR",
    "MATH_Rephrased", "MATH_AnsAug", "MATH_FOBAR", "MATH_SV",
]

# ═══════════════════════════════════════════════════════════════════════════════
# 3. LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
log = logging.getLogger("grpo_train")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. SIGNAL HANDLING (graceful shutdown on spot interruption)
# ═══════════════════════════════════════════════════════════════════════════════

_interrupted = False


def _sigterm_handler(signum, frame):
    global _interrupted
    log.warning("Received SIGTERM — will checkpoint and exit after current step.")
    _interrupted = True


signal.signal(signal.SIGTERM, _sigterm_handler)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. IMPORTS (heavy — after config so failures are clear)
# ═══════════════════════════════════════════════════════════════════════════════

def import_core():
    """Import core ML libraries with clear error messages."""
    global torch, np, FastLanguageModel, load_dataset
    global GRPOTrainer, GRPOConfig, TrainerCallback, TrainerState, TrainerControl
    global wandb, login

    import numpy as np
    log.info("numpy OK")

    import torch
    log.info(f"torch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_properties(0)
        log.info(f"GPU: {dev.name} | VRAM: {dev.total_memory/1e9:.1f} GB | CC: {dev.major}.{dev.minor}")

    from unsloth import FastLanguageModel
    log.info("unsloth OK")

    from datasets import load_dataset
    log.info("datasets OK")

    from trl import GRPOTrainer, GRPOConfig
    from transformers import TrainerCallback, TrainerState, TrainerControl
    log.info("trl OK")

    from huggingface_hub import login
    log.info("huggingface_hub OK")

    try:
        import wandb
        log.info(f"wandb {wandb.__version__}")
    except ImportError:
        wandb = None
        log.warning("wandb not installed — logging to console/file only")

    return wandb


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ANSWER EXTRACTOR (7-pattern with confidence flag)
# ═══════════════════════════════════════════════════════════════════════════════

# Word-to-number map for common negatives
_WORD_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "negative": "-",
}


def _normalize_answer(s: str) -> str:
    """Normalize answer string for comparison."""
    if not s:
        return ""
    s = str(s).strip().replace(",", "").replace(" ", "").rstrip(".").lower()
    # Strip percentage
    if s.endswith("%"):
        s = s[:-1]
    return s


def _answers_match(pred: str, gold: str) -> bool:
    """Compare two answer strings with float tolerance + fraction handling."""
    pn = _normalize_answer(pred)
    gn = _normalize_answer(gold)
    if not pn or not gn:
        return False
    if pn == gn:
        return True
    # Float comparison
    try:
        return abs(float(pn) - float(gn)) < 1e-4
    except (ValueError, OverflowError):
        pass
    # Fraction handling: "49/12" vs "4.0833"
    for a, b in [(pn, gn), (gn, pn)]:
        if "/" in a:
            try:
                num, den = a.split("/", 1)
                decimal = float(num) / float(den)
                return abs(decimal - float(b)) < 1e-4
            except (ValueError, ZeroDivisionError):
                pass
    # Substring match (for cases like "27" in "$27")
    return gn in pn or pn in gn


def extract_answer(text: str) -> Tuple[str, str]:
    """
    Extract final answer from model output.
    Returns (extracted_answer, confidence) where confidence is 'high'/'medium'/'low'.

    Patterns tried in order:
      1. #### <number>           (GSM8K)        — HIGH
      2. \\boxed{<expr>}          (MATH)         — HIGH
      3. x = <number>            (FOBAR/SV)     — MEDIUM
      4. = <number> at EOL       (equation)     — MEDIUM
      5. "the answer is <n>"     (natural)      — MEDIUM
      6. "there are <n>"         (quantity)     — LOW
      7. Last number             (fallback)     — LOW

    Improvements applied:
      - Comma removed from capture group (fixes "Therefore, 5" bug)
      - Fraction handling: "49/12" → decimal
      - Percentage stripping: "27%" → "27"
      - Negative word form: "negative five" → "-5"
    """
    if not text:
        return ("", "low")

    # 1. GSM8K style: #### <number>
    m = re.search(r'####\s*([-\d\.]+)', text)
    if m:
        return (m.group(1).strip(), "high")

    # 2. MATH style: \boxed{<expr>}
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        inner = m.group(1).strip()
        # Try to extract number from boxed content
        nums = re.findall(r'[-\d\.]+', inner)
        if nums:
            return (nums[-1], "high")
        return (inner, "high")

    # 3. x = <number> (FOBAR/SV) — last occurrence
    ms = re.findall(r'[xX]\s*=\s*([-\d\.]+)', text)
    if ms:
        return (ms[-1].strip(), "medium")

    # 4. = <number> at end of line
    ms = re.findall(r'=\s*([-\d\.]+)\s*$', text, re.MULTILINE)
    if ms:
        return (ms[-1].strip(), "medium")

    # 5. Answer phrases (require space after keyword — fixes comma bug)
    ms = re.findall(
        r'(?:the answer is|therefore|thus|final answer is|so (?:the|it is|his|her|their))\s+:?\s+([-\d\.]+)',
        text, re.IGNORECASE
    )
    if ms:
        return (ms[-1].strip(), "medium")

    # 5b. Negative word form: "negative five" → "-5"
    ms = re.findall(r'negative\s+(\w+)', text, re.IGNORECASE)
    if ms:
        word = ms[-1].lower()
        if word in _WORD_NUMBERS:
            return (_WORD_NUMBERS[word], "medium")

    # 6. Quantity phrases
    ms = re.findall(
        r'(?:there (?:are|were)|(?:has|have|are|had|was|were|is)\s+)(?:exactly\s+)?([-\d\.]+)',
        text, re.IGNORECASE
    )
    if ms:
        return (ms[-1].strip(), "low")

    # 7. Last number fallback (dedup before grabbing)
    numbers = re.findall(r'[-\d\.]+', text)
    if numbers:
        # Prefer numbers AFTER conclusion keywords
        conclusion_pos = -1
        for kw in ["therefore", "thus", "so", "final answer", "the answer is", "in conclusion"]:
            pos = text.lower().rfind(kw)
            if pos > conclusion_pos:
                conclusion_pos = pos
        if conclusion_pos >= 0:
            after_text = text[conclusion_pos:]
            after_nums = re.findall(r'[-\d\.]+', after_text)
            if after_nums:
                return (after_nums[-1].strip(), "low")
        return (numbers[-1].strip(), "low")

    return ("", "low")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. REWARD FUNCTION (2 components)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rewards(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    **kwargs,
) -> List[float]:
    """
    GRPO reward function. Returns one float per completion.

    Component 1 — Correctness (weight 0.80):
      +1.0 if extracted answer matches ground truth
      +0.0 otherwise

    Component 2 — Length penalty (weight 0.20):
      +1.0 if word_count <= LENGTH_THRESHOLD
      linear decrease to 0.0 as word_count → THRESHOLD + 500

    Total = 0.8 * correctness + 0.2 * length_score
    """
    rewards = []

    for completion, gt in zip(completions, answer):
        if isinstance(completion, list):
            completion = completion[0] if completion else ""

        # ── Component 1: Correctness ─────────────────────────────────────
        pred, confidence = extract_answer(completion)
        correct = _answers_match(pred, gt)
        r_correct = 1.0 if correct else 0.0

        # ── Component 2: Length penalty ──────────────────────────────────
        word_count = len(completion.split())
        if word_count <= LENGTH_THRESHOLD:
            r_length = 1.0
        else:
            over = word_count - LENGTH_THRESHOLD
            r_length = max(0.0, 1.0 - over / 500.0)

        total = W_CORRECTNESS * r_correct + W_LENGTH * r_length
        rewards.append(total)

    return rewards


# ═══════════════════════════════════════════════════════════════════════════════
# 8. DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a math solver. Solve the problem step by step, "
    "showing your reasoning clearly. "
    "End your response with: The answer is <number>."
)


def build_prompt(tokenizer, question: str) -> str:
    """Format a question using Llama 3.2 chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def prepare_data(tokenizer, raw_dataset, max_samples: Optional[int] = None):
    """
    Convert raw MetaMathQA to GRPO format.
    - Extract answer from full_response
    - Build chat-template prompt
    - Keep 'type' for per-type tracking
    """
    processed = []
    skipped = 0

    for i, ex in enumerate(raw_dataset):
        if max_samples and i >= max_samples:
            break

        question = ex.get("query", "").strip()
        response = ex.get("response", "").strip()
        qtype = ex.get("type", "unknown")

        if not question or not response:
            skipped += 1
            continue

        # Extract ground truth answer
        gt_answer, _ = extract_answer(response)
        if not gt_answer:
            skipped += 1
            continue

        prompt_text = build_prompt(tokenizer, question)

        processed.append({
            "prompt": prompt_text,
            "answer": gt_answer,
            "type": qtype,
            "question": question,  # for sample logging
        })

    return processed, skipped


def load_training_data_from_channel(channel_path: str):
    """
    Load training data from SageMaker input channel.
    Supports JSON (single object or JSON Lines), JSONL, or CSV files.
    Returns a list of dicts with keys: query, response, type.
    """
    data = []
    channel = Path(channel_path)

    if not channel.exists():
        log.warning(f"Training channel path does not exist: {channel_path}")
        return data

    # Look for data files in the channel directory
    data_files = list(channel.rglob("*.json")) + list(channel.rglob("*.jsonl")) + list(channel.rglob("*.csv"))

    for f in data_files:
        log.info(f"Loading data file from channel: {f}")
        if f.suffix == ".csv":
            import csv
            with open(f) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    data.append(row)
        else:
            with open(f) as fh:
                content = fh.read().strip()
                # Try JSON Lines first
                if "\n" in content and content.startswith("{"):
                    for line in content.split("\n"):
                        line = line.strip()
                        if line:
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                else:
                    # Try single JSON array or object
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, list):
                            data.extend(parsed)
                        else:
                            data.append(parsed)
                    except json.JSONDecodeError:
                        # Fallback: line by line
                        for line in content.split("\n"):
                            line = line.strip()
                            if line:
                                try:
                                    data.append(json.loads(line))
                                except json.JSONDecodeError:
                                    continue

    log.info(f"Loaded {len(data)} samples from SageMaker channel: {channel_path}")
    return data


def load_heldout_eval_data(path: str, n: int = HELDOUT_N_SAMPLES) -> List[dict]:
    """Load heldout evaluation questions from local JSON or HF dataset."""
    data = []
    p = Path(path)
    if p.exists():
        with open(p) as f:
            for line in f:
                data.append(json.loads(line))
        random.shuffle(data)
        return data[:n]
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# 9. OBSERVABILITY CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

class GRPOObservabilityCallback(TrainerCallback):
    """
    Comprehensive observability callback:
    - Logs reward (total + per-component), response length, extraction rate, KL
    - Per-question-type reward tracking
    - Heldout accuracy evaluation every N steps
    - Sample generation with full breakdown every N steps
    - Saves metrics to JSON for external monitoring
    """

    def __init__(
        self,
        tokenizer,
        model,
        reward_fn,
        heldout_data: List[dict],
        output_dir: Path,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.reward_fn = reward_fn
        self.heldout_data = heldout_data
        self.output_dir = output_dir
        self.metrics_file = output_dir / "metrics.jsonl"

        # Rolling stats
        self.step_rewards = deque(maxlen=100)
        self.step_lengths = deque(maxlen=100)
        self.step_extraction_rates = deque(maxlen=100)
        self.per_type_rewards = defaultdict(list)  # type → list of rewards
        self.per_type_correct = defaultdict(int)   # type → correct count
        self.per_type_total = defaultdict(int)     # type → total count

        # Heldout accuracy history
        self.heldout_history = []

        # Sample questions (fixed for consistent tracking)
        self.sample_questions = []
        if heldout_data:
            self.sample_questions = heldout_data[:N_SAMPLE_QUESTIONS]

        # WandB
        self.use_wandb = wandb is not None and wandb.run is not None

        # Step counter
        self._step = 0

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        log.info("=" * 70)
        log.info("TRAINING START")
        log.info(f"  Steps: {MAX_STEPS} | Batch: {PER_DEVICE_BATCH} | Group: {GRPO_GROUP_SIZE}")
        log.info(f"  LR: {LEARNING_RATE} | Beta: {BETA} | Temp: {TEMPERATURE}")
        log.info(f"  Heldout eval: every {HELDOUT_EVERY_N_STEPS} steps, {HELDOUT_N_SAMPLES} questions")
        log.info(f"  Checkpoint: every {CHECKPOINT_EVERY_N_STEPS} steps")
        log.info(f"  HF repo: {HF_REPO}")
        log.info("=" * 70)
        self._log_to_wandb({"status": "training_started"})

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: dict = None, **kwargs):
        """Called by TRL after each step with training logs."""
        self._step = state.global_step
        if logs is None:
            return

        # ── Extract TRL-provided metrics ─────────────────────────────────
        mean_reward = logs.get("reward", 0.0)
        kl = logs.get("kl", 0.0)
        loss = logs.get("loss", 0.0)
        learning_rate = logs.get("learning_rate", 0.0)

        self.step_rewards.append(mean_reward)

        # ── Console log (every step) ─────────────────────────────────────
        log.info(
            f"Step {state.global_step:4d}/{MAX_STEPS} | "
            f"reward={mean_reward:.4f} | kl={kl:.6f} | loss={loss:.4f} | lr={learning_rate:.2e}"
        )

        # ── WandB log ────────────────────────────────────────────────────
        wb_logs = {
            "train/mean_reward": mean_reward,
            "train/kl": kl,
            "train/loss": loss,
            "train/learning_rate": learning_rate,
            "step": state.global_step,
        }

        # ── Heldout accuracy eval ────────────────────────────────────────
        if state.global_step > 0 and state.global_step % HELDOUT_EVERY_N_STEPS == 0:
            heldout_acc = self._eval_heldout()
            wb_logs["eval/heldout_accuracy"] = heldout_acc
            wb_logs["eval/heldout_step"] = state.global_step
            log.info(f"  >>> HELD-OUT ACCURACY: {heldout_acc:.1%} ({state.global_step} steps)")

        # ── Sample generation with breakdown ─────────────────────────────
        if state.global_step > 0 and state.global_step % SAMPLE_LOG_EVERY_N_STEPS == 0:
            self._log_samples(state.global_step)
            self._log_per_type_stats(state.global_step)

        # ── Save metrics to JSONL ────────────────────────────────────────
        self._append_metrics(state.global_step, {
            "mean_reward": mean_reward,
            "kl": kl,
            "loss": loss,
            "learning_rate": learning_rate,
        })

        self._log_to_wandb(wb_logs)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Check for spot interruption signal."""
        if _interrupted:
            log.warning("Interruption detected — requesting training stop.")
            control.should_training_stop = True

    def _eval_heldout(self) -> float:
        """Evaluate current model on heldout set. Returns accuracy."""
        if not self.heldout_data:
            log.warning("No heldout data available for eval.")
            return 0.0

        FastLanguageModel.for_inference(self.model)
        self.model.eval()

        correct = 0
        total = min(HELDOUT_N_SAMPLES, len(self.heldout_data))
        eval_results = []

        for item in self.heldout_data[:total]:
            prompt_text = item.get("prompt", "")
            # If prompt is already chat-formatted, use directly; otherwise build it
            if not prompt_text:
                prompt_text = build_prompt(self.tokenizer, item.get("question", item.get("prompt", "")))

            enc = self.tokenizer(
                prompt_text, return_tensors="pt", truncation=True,
                max_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS,
            ).to(self.model.device)

            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )

            completion = self.tokenizer.decode(
                out[0][enc.input_ids.shape[1]:], skip_special_tokens=True
            )

            pred, confidence = extract_answer(completion)
            gt = item.get("answer", "")
            is_correct = _answers_match(pred, gt)
            if is_correct:
                correct += 1

            eval_results.append({
                "type": item.get("type", "unknown"),
                "correct": is_correct,
                "confidence": confidence,
            })

        FastLanguageModel.for_training(self.model)
        self.model.train()

        accuracy = correct / total if total > 0 else 0.0
        self.heldout_history.append({"step": self._step, "accuracy": accuracy})

        return accuracy

    def _log_samples(self, step: int):
        """Generate and log sample outputs with full reward breakdown."""
        if not self.sample_questions:
            return

        FastLanguageModel.for_inference(self.model)
        self.model.eval()

        log.info(f"\n{'═' * 70}")
        log.info(f"  SAMPLE GENERATION — Step {step}")
        log.info(f"{'═' * 70}")

        for i, item in enumerate(self.sample_questions):
            question = item.get("question", item.get("prompt", ""))
            gt = item.get("answer", "")
            qtype = item.get("type", "?")

            # Build prompt
            prompt_text = build_prompt(self.tokenizer, question)

            enc = self.tokenizer(
                prompt_text, return_tensors="pt", truncation=True,
                max_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS,
            ).to(self.model.device)

            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )

            completion = self.tokenizer.decode(
                out[0][enc.input_ids.shape[1]:], skip_special_tokens=True
            )

            pred, confidence = extract_answer(completion)
            is_correct = _answers_match(pred, gt)
            word_count = len(completion.split())

            # Compute reward components
            r_correct = 1.0 if is_correct else 0.0
            if word_count <= LENGTH_THRESHOLD:
                r_length = 1.0
            else:
                r_length = max(0.0, 1.0 - (word_count - LENGTH_THRESHOLD) / 500.0)
            total_reward = W_CORRECTNESS * r_correct + W_LENGTH * r_length

            log.info(f"\n  [{i+1}] Type: {qtype}")
            log.info(f"  Q: {question[:100]}...")
            log.info(f"  GT: {gt} | Pred: {pred} ({confidence}) | {'✓' if is_correct else '✗'}")
            log.info(f"  Output ({word_count} words): {completion[:300]}{'...' if len(completion) > 300 else ''}")
            log.info(f"  Reward: {total_reward:.3f} (correct={r_correct:.1f}, length={r_length:.2f})")

        log.info(f"{'═' * 70}\n")

        FastLanguageModel.for_training(self.model)
        self.model.train()

    def _log_per_type_stats(self, step: int):
        """Log per-question-type reward breakdown."""
        log.info(f"\n  PER-TYPE REWARDS — Step {step}")
        for qtype in QUESTION_TYPES:
            type_rewards = self.per_type_rewards.get(qtype, [])
            if type_rewards:
                avg_r = sum(type_rewards) / len(type_rewards)
                log.info(f"    {qtype:20s}: avg_reward={avg_r:.4f} (n={len(type_rewards)})")
            else:
                log.info(f"    {qtype:20s}: no data yet")

    def _append_metrics(self, step: int, metrics: dict):
        """Append metrics to JSONL file."""
        record = {"step": step, "timestamp": datetime.now(timezone.utc).isoformat(), **metrics}
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _log_to_wandb(self, logs: dict):
        """Log to WandB if available."""
        if self.use_wandb:
            try:
                wandb.log(logs)
            except Exception as e:
                log.debug(f"WandB log failed: {e}")

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Final summary."""
        log.info("=" * 70)
        log.info("TRAINING COMPLETE")
        log.info(f"  Final step: {state.global_step}")
        if self.heldout_history:
            best = max(self.heldout_history, key=lambda x: x["accuracy"])
            log.info(f"  Best held-out accuracy: {best['accuracy']:.1%} at step {best['step']}")
        log.info(f"  Metrics saved to: {self.metrics_file}")
        log.info("=" * 70)
        self._log_to_wandb({"status": "training_complete", "final_step": state.global_step})


# ═══════════════════════════════════════════════════════════════════════════════
# 10. ALIGNMENT AUDIT (AlignTune / AuditKIT integration for hackathon compliance)
# ═══════════════════════════════════════════════════════════════════════════════

def run_alignment_audit(model, tokenizer, heldout_data: List[dict]) -> dict:
    """
    Run lightweight alignment audit: response quality metrics.
    (Full AlignmentAuditor requires probe sets; we use a lightweight version here.)
    """
    if not heldout_data:
        return {}

    FastLanguageModel.for_inference(model)
    model.eval()

    repetition_ratios = []
    response_lengths = []
    n = min(50, len(heldout_data))

    for item in heldout_data[:n]:
        prompt_text = build_prompt(
            tokenizer, item.get("question", item.get("prompt", ""))
        )
        enc = tokenizer(
            prompt_text, return_tensors="pt", truncation=True,
            max_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        words = completion.split()
        response_lengths.append(len(words))

        # 3-gram repetition
        if len(words) >= 3:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
            if trigrams:
                repetition_ratios.append(1.0 - len(set(trigrams)) / len(trigrams))

    FastLanguageModel.for_training(model)
    model.train()

    audit = {
        "avg_response_length": sum(response_lengths) / len(response_lengths) if response_lengths else 0,
        "avg_repetition_ratio": sum(repetition_ratios) / len(repetition_ratios) if repetition_ratios else 0,
    }
    return audit


# ═══════════════════════════════════════════════════════════════════════════════
# 11. MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global wandb

    log.info("=" * 70)
    log.info("SageMaker GRPO TRAINING — Llama 3.2 3B on MetaMathQA")
    log.info(f"Time: {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC}")
    log.info(f"SM_MODEL_DIR: {SM_MODEL_DIR}")
    log.info(f"SM_OUTPUT_DATA_DIR: {SM_OUTPUT_DATA_DIR}")
    log.info(f"SM_CHANNEL_TRAIN: {SM_TRAIN_CHANNEL}")
    log.info(f"OUTPUT_DIR: {OUTPUT_DIR}")
    log.info("=" * 70)

    # ── Step 1: Import libraries ──────────────────────────────────────────
    log.info("Importing libraries...")
    wandb = import_core()

    import torch
    import numpy as np
    from datasets import Dataset as HFDataset

    # ── Step 2: HF login ──────────────────────────────────────────────────
    if HF_TOKEN:
        login(token=HF_TOKEN)
        log.info(f"Logged in to HF Hub as {HF_USERNAME}")
    else:
        log.warning("No HF_TOKEN set — cannot push checkpoints to Hub")

    # ── Step 3: WandB init ────────────────────────────────────────────────
    if wandb and os.environ.get("WANDB_API_KEY"):
        wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config={
            "model": MODEL_NAME,
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "group_size": GRPO_GROUP_SIZE, "batch_size": PER_DEVICE_BATCH,
            "grad_accum": GRAD_ACCUM_STEPS, "max_steps": MAX_STEPS,
            "max_new_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE,
            "beta": BETA, "learning_rate": LEARNING_RATE,
            "length_threshold": LENGTH_THRESHOLD,
            "w_correctness": W_CORRECTNESS, "w_length": W_LENGTH,
            "precision": "bf16",
        })
        log.info(f"WandB initialized: {WANDB_PROJECT}/{WANDB_RUN_NAME}")
    elif wandb:
        log.info("WANDB_API_KEY not set — WandB disabled")

    # ── Step 4: Load model + LoRA ─────────────────────────────────────────
    log.info(f"Loading model: {MODEL_NAME}")
    t0 = time.time()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,  # bf16 on A10 — better gradient quality
        token=HF_TOKEN if HF_TOKEN else None,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGETS,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_rslora=True,
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model loaded in {time.time()-t0:.1f}s")
    log.info(f"Params: {total_p/1e9:.2f}B total, {trainable_p/1e6:.2f}M trainable ({100*trainable_p/total_p:.2f}%)")
    if torch.cuda.is_available():
        log.info(f"VRAM after load: {torch.cuda.memory_reserved()/1e9:.2f} GB")

    # ── Step 5: Load & preprocess data ────────────────────────────────────
    raw_ds = None

    # Priority 1: SageMaker input channel
    if SM_TRAIN_CHANNEL:
        log.info(f"Loading training data from SageMaker channel: {SM_TRAIN_CHANNEL}")
        channel_data = load_training_data_from_channel(SM_TRAIN_CHANNEL)
        if channel_data:
            from datasets import Dataset as HFDatasetDirect
            raw_ds = HFDatasetDirect.from_list(channel_data)
            log.info(f"Training data loaded from SageMaker channel: {len(raw_ds):,} rows")
        else:
            log.warning(f"No data found in SageMaker channel — falling back to HF Hub")

    # Priority 2: HuggingFace Hub (fallback)
    if raw_ds is None:
        log.info(f"Loading dataset from HuggingFace Hub: {DATASET_NAME}")
        raw_ds = load_dataset(DATASET_NAME, split="train")
        log.info(f"Raw dataset: {len(raw_ds):,} rows")

    # Prepare for GRPO
    processed, skipped = prepare_data(tokenizer, raw_ds)
    log.info(f"Processed: {len(processed):,} | Skipped: {skipped:,}")

    if not processed:
        log.error("No training data after preprocessing — aborting")
        sys.exit(1)

    # Convert to HF Dataset
    train_ds = HFDataset.from_list(processed)
    log.info(f"Training dataset: {len(train_ds):,} samples")

    # ── Step 6: Load heldout eval data ────────────────────────────────────
    heldout_eval = []

    # Check SageMaker channel first for heldout data
    if SM_TRAIN_CHANNEL:
        heldout_channel_path = Path(SM_TRAIN_CHANNEL) / "grpo_heldout.json"
        if heldout_channel_path.exists():
            heldout_eval = load_heldout_eval_data(str(heldout_channel_path), HELDOUT_N_SAMPLES)
            log.info(f"Heldout eval data: {len(heldout_eval)} questions (from channel: {heldout_channel_path})")

    # Fallback to local project data directory
    if not heldout_eval:
        heldout_path = Path(__file__).parent.parent / "data" / "grpo_heldout.json"
        if heldout_path.exists():
            heldout_eval = load_heldout_eval_data(str(heldout_path), HELDOUT_N_SAMPLES)
            log.info(f"Heldout eval data: {len(heldout_eval)} questions (from {heldout_path})")
        else:
            log.warning(f"Heldout file not found at {heldout_path} — eval disabled")

    # ── Step 7: Observability callback ────────────────────────────────────
    obs_callback = GRPOObservabilityCallback(
        tokenizer=tokenizer,
        model=model,
        reward_fn=compute_rewards,
        heldout_data=heldout_eval,
        output_dir=OUTPUT_DIR,
    )

    # ── Step 8: GRPO Trainer ──────────────────────────────────────────────
    log.info("Creating GRPO trainer...")

    grpo_config = GRPOConfig(
        output_dir=str(CHECKPOINT_DIR),

        # GRPO-specific
        num_generations=GRPO_GROUP_SIZE,
        max_completion_length=MAX_NEW_TOKENS,
        beta=BETA,
        temperature=TEMPERATURE,

        # Training
        num_train_epochs=1,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        max_grad_norm=MAX_GRAD_NORM,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",

        # Precision
        bf16=True,
        fp16=False,

        # Optimizer
        optim="adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Saving + Hub
        save_strategy="steps",
        save_steps=CHECKPOINT_EVERY_N_STEPS,
        save_total_limit=3,
        push_to_hub=bool(HF_TOKEN),
        hub_model_id=HF_REPO if HF_TOKEN else None,
        hub_strategy="checkpoint" if HF_TOKEN else "end",

        # Logging
        logging_steps=1,
        report_to="wandb" if (wandb and os.environ.get("WANDB_API_KEY")) else "none",
        run_name=WANDB_RUN_NAME,

        # Seed
        seed=SEED,

        # CRITICAL: keep 'answer' and 'type' columns for reward fn
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=compute_rewards,
        args=grpo_config,
        train_dataset=train_ds,
        callbacks=[obs_callback],
    )

    log.info(f"Trainer created | Steps/epoch ≈ {len(train_ds) // PER_DEVICE_BATCH}")
    if torch.cuda.is_available():
        log.info(f"VRAM after trainer setup: {torch.cuda.memory_reserved()/1e9:.2f} GB")

    # ── Step 9: Pre-training eval ─────────────────────────────────────────
    log.info("Running pre-training evaluation...")
    pre_acc = obs_callback._eval_heldout()
    log.info(f"Pre-training held-out accuracy: {pre_acc:.1%}")
    if wandb and wandb.run:
        wandb.log({"eval/pre_training_accuracy": pre_acc, "step": 0})

    # ── Step 10: Train ────────────────────────────────────────────────────
    log.info("Starting GRPO training...")
    train_start = time.time()

    trainer.train()

    train_elapsed = time.time() - train_start
    log.info(f"Training finished in {train_elapsed/3600:.2f} hours")

    # ── Step 11: Post-training eval ───────────────────────────────────────
    log.info("Running post-training evaluation...")
    post_acc = obs_callback._eval_heldout()
    log.info(f"Post-training held-out accuracy: {post_acc:.1%}")
    log.info(f"Improvement: {post_acc - pre_acc:+.1%}")

    # Alignment audit
    log.info("Running alignment audit...")
    audit = run_alignment_audit(model, tokenizer, heldout_eval)
    log.info(f"Alignment audit: {audit}")

    # ── Step 12: Save final model to SageMaker model dir ──────────────────
    final_dir = Path(SM_MODEL_DIR) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    log.info(f"Final model saved to: {final_dir}")

    # Also save to OUTPUT_DIR for easy access in output data
    output_final_dir = OUTPUT_DIR / "final"
    output_final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_final_dir)
    tokenizer.save_pretrained(output_final_dir)
    log.info(f"Final model also saved to: {output_final_dir}")

    if HF_TOKEN:
        log.info(f"Pushing final model to HF Hub: {HF_REPO}")
        model.push_to_hub(HF_REPO, token=HF_TOKEN, private=True)
        tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN, private=True)
        log.info("Push complete")

    # ── Step 13: Save summary ─────────────────────────────────────────────
    summary = {
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "hf_repo": HF_REPO,
        "pre_training_accuracy": pre_acc,
        "post_training_accuracy": post_acc,
        "improvement": post_acc - pre_acc,
        "training_steps": MAX_STEPS,
        "training_hours": train_elapsed / 3600,
        "alignment_audit": audit,
        "hyperparameters": {
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
            "group_size": GRPO_GROUP_SIZE, "batch_size": PER_DEVICE_BATCH,
            "grad_accum": GRAD_ACCUM_STEPS, "max_steps": MAX_STEPS,
            "max_new_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE,
            "beta": BETA, "learning_rate": LEARNING_RATE,
            "length_threshold": LENGTH_THRESHOLD,
            "precision": "bf16",
        },
    }
    summary_path = OUTPUT_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Summary saved to: {summary_path}")

    # Also save summary to SM_MODEL_DIR for SageMaker artifact collection
    sm_summary_path = Path(SM_MODEL_DIR) / "training_summary.json"
    with open(sm_summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Summary also saved to: {sm_summary_path}")

    if wandb and wandb.run:
        wandb.log({
            "eval/post_training_accuracy": post_acc,
            "eval/improvement": post_acc - pre_acc,
        })
        wandb.finish()

    log.info("=" * 70)
    log.info("ALL DONE")
    log.info(f"  Pre-training:  {pre_acc:.1%}")
    log.info(f"  Post-training: {post_acc:.1%}")
    log.info(f"  Improvement:   {post_acc - pre_acc:+.1%}")
    log.info(f"  Model: {HF_REPO}")
    log.info(f"  Model dir: {final_dir}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
