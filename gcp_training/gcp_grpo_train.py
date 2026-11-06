#!/usr/bin/env python3
"""
GRPO RL Training — Llama 3.2 3B on MetaMathQA
GCP L4/T4 (24GB) — fp16, 4-bit quantization, LoRA r=16, heavy observability

Stack: Unsloth + TRL GRPOTrainer (+ AlignTune imports for hackathon compliance)
Data: meta-math/MetaMathQA (HuggingFace) or local file path
Base: meta-llama/Llama-3.2-3B-Instruct

Reward components:
  1. Correctness (weight 0.80) — exact numeric match after extraction
  2. Length penalty (weight 0.20) — linear penalty when > LENGTH_THRESHOLD words

Hyperparameters (all configurable via CLI, defaults below):
  LoRA r=16, alpha=32, dropout=0.05, rsLoRA
  group_size=4, batch_size=4, grad_accum=2
  max_steps=1000, max_new_tokens=500
  lr=5e-7, beta=0.03, temperature=0.8
  fp16 (L4/T4 do not support bf16 well), 4-bit quantization
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 1. ARGUMENT PARSING (ALL hyperparameters configurable)
# ═══════════════════════════════════════════════════════════════════════════════

import argparse
import os
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


def parse_args():
    """Parse all hyperparameters from CLI with defaults matching the proven config."""
    parser = argparse.ArgumentParser(
        description="GRPO Training — GCP L4/T4 with fp16 + 4-bit quantization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="HuggingFace model ID or local model path")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length for the model")

    # ── LoRA ──────────────────────────────────────────────────────────────
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")

    # ── GRPO ──────────────────────────────────────────────────────────────
    parser.add_argument("--group_size", type=int, default=4,
                        help="Number of generations per prompt (GRPO group size)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device training batch size")
    parser.add_argument("--grad_accum", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum training steps")
    parser.add_argument("--max_new_tokens", type=int, default=500,
                        help="Maximum new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature for generation")
    parser.add_argument("--beta", type=float, default=0.03,
                        help="KL divergence coefficient in GRPO")
    parser.add_argument("--learning_rate", type=float, default=5e-7,
                        help="Peak learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=0.1,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--warmup_steps", type=int, default=50,
                        help="Learning rate warmup steps")

    # ── Reward ────────────────────────────────────────────────────────────
    parser.add_argument("--w_correctness", type=float, default=0.80,
                        help="Weight for correctness reward component")
    parser.add_argument("--w_length", type=float, default=0.20,
                        help="Weight for length penalty reward component")
    parser.add_argument("--length_threshold", type=int, default=300,
                        help="Word count threshold — penalty kicks in above this")

    # ── Data ──────────────────────────────────────────────────────────────
    parser.add_argument("--dataset", type=str, default="meta-math/MetaMathQA",
                        help="HuggingFace dataset name OR local file path (JSON/JSONL/CSV/Parquet)")
    parser.add_argument("--data_split", type=str, default="train",
                        help="Dataset split to use (ignored for local files)")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Maximum number of training samples (None = all)")
    parser.add_argument("--heldout_path", type=str, default=None,
                        help="Path to heldout eval JSON/JSONL file")
    parser.add_argument("--heldout_n_samples", type=int, default=200,
                        help="Number of heldout eval questions")
    parser.add_argument("--heldout_every_n_steps", type=int, default=100,
                        help="Heldout eval frequency in steps")

    # ── Checkpoints & Logging ─────────────────────────────────────────────
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=200,
                        help="Checkpoint save frequency in steps")
    parser.add_argument("--sample_log_every_n_steps", type=int, default=50,
                        help="Sample generation log frequency in steps")
    parser.add_argument("--n_sample_questions", type=int, default=5,
                        help="Number of questions to log in sample generation")

    # ── Output & Hub ──────────────────────────────────────────────────────
    parser.add_argument("--output_dir", type=str, default="./gcp_grpo_output",
                        help="Local output directory for checkpoints and final model")
    parser.add_argument("--hf_repo", type=str, default=None,
                        help="HuggingFace Hub repo ID to push model (e.g., 'username/model-name')")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push final model to HuggingFace Hub")
    parser.add_argument("--hub_private", action="store_true", default=True,
                        help="Push to Hub as private repo")

    # ── Precision & Quantization ──────────────────────────────────────────
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use fp16 precision (default for L4/T4)")
    parser.add_argument("--no_fp16", action="store_true", default=False,
                        help="Disable fp16 (use bf16 instead)")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="Load model in 4-bit quantization for memory efficiency")

    # ── Misc ──────────────────────────────────────────────────────────────
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()


args = parse_args()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONFIGURATION (from parsed args + environment)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Reproducibility ─────────────────────────────────────────────────────
SEED = args.seed
random.seed(SEED)

# ── HF credentials (passed via env for security) ───────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_USERNAME = os.environ.get("HF_USERNAME", "Suryansh7123")
HF_REPO = args.hf_repo or os.environ.get("HF_REPO", f"{HF_USERNAME}/llama3.2-3b-grpo-metamath-gcp")

# ── WandB ──────────────────────────────────────────────────────────────
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "grpo-llama3b-metamath-gcp")
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME", f"grpo-l4-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}")

# ── Model ──────────────────────────────────────────────────────────────
MODEL_NAME = args.model_name
MAX_SEQ_LENGTH = args.max_seq_length

# ── LoRA ───────────────────────────────────────────────────────────────
LORA_R = args.lora_r
LORA_ALPHA = args.lora_alpha
LORA_DROPOUT = args.lora_dropout
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ── GRPO ───────────────────────────────────────────────────────────────
GRPO_GROUP_SIZE = args.group_size
PER_DEVICE_BATCH = args.batch_size
GRAD_ACCUM_STEPS = args.grad_accum
MAX_STEPS = args.max_steps
MAX_NEW_TOKENS = args.max_new_tokens
TEMPERATURE = args.temperature
BETA = args.beta  # KL coefficient
LEARNING_RATE = args.learning_rate
MAX_GRAD_NORM = args.max_grad_norm
WARMUP_STEPS = args.warmup_steps

# ── Reward ─────────────────────────────────────────────────────────────
W_CORRECTNESS = args.w_correctness
W_LENGTH = args.w_length
LENGTH_THRESHOLD = args.length_threshold  # words — penalty kicks in above this

# ── Data ───────────────────────────────────────────────────────────────
DATASET_NAME = args.dataset
DATA_SPLIT = args.data_split
MAX_TRAIN_SAMPLES = args.max_train_samples
HELDOUT_PATH = args.heldout_path
HELDOUT_EVERY_N_STEPS = args.heldout_every_n_steps
HELDOUT_N_SAMPLES = args.heldout_n_samples
CHECKPOINT_EVERY_N_STEPS = args.checkpoint_every_n_steps
SAMPLE_LOG_EVERY_N_STEPS = args.sample_log_every_n_steps
N_SAMPLE_QUESTIONS = args.n_sample_questions

# ── Output ─────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "training_log.txt"

# ── Precision ──────────────────────────────────────────────────────────
USE_FP16 = args.fp16 and not args.no_fp16
USE_BF16 = not USE_FP16
LOAD_IN_4BIT = args.load_in_4bit

# ── Question types (8 classes in MetaMathQA) ───────────────────────────
QUESTION_TYPES = [
    "GSM_SV", "GSM_Rephrased", "GSM_AnsAug", "GSM_FOBAR",
    "MATH_Rephrased", "MATH_AnsAug", "MATH_FOBAR", "MATH_SV",
]

# ═══════════════════════════════════════════════════════════════════════════════
# 3. LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

# Also write to /dev/console (GCP serial port) for observability without SSH.
# This ensures training progress is visible in serial port output.
_console_handler = None
try:
    _console_handler = logging.FileHandler("/dev/console", mode="a")
    _console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
except (OSError, PermissionError):
    _console_handler = None  # /dev/console may not exist locally

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
        *([_console_handler] if _console_handler else []),
    ],
)
log = logging.getLogger("gcp_grpo_train")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. SIGNAL HANDLING (graceful shutdown on preemption)
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

    # Seed torch/numpy for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_properties(0)
        log.info(f"GPU: {dev.name} | VRAM: {dev.total_memory / 1e9:.1f} GB | CC: {dev.major}.{dev.minor}")

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
    # Only use substring match for multi-digit answers to avoid false positives
    if len(pn) >= 2 and len(gn) >= 2 and len(pn) <= len(gn) + 2:
        return gn in pn
    return False


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
            return ("-" + _WORD_NUMBERS[word], "medium")

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


class MetricsTracker:
    """
    Shared accumulator: reward function writes per-type stats here,
    observability callback reads/clears it at each logging step.
    Thread-safe enough for single-GPU GRPO (reward fn and callback
    never run concurrently).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.type_correct: Dict[str, int] = defaultdict(int)
        self.type_total: Dict[str, int] = defaultdict(int)
        self.type_reward_sum: Dict[str, float] = defaultdict(float)
        self.type_length_sum: Dict[str, int] = defaultdict(int)
        self.n_samples = 0
        self.total_correct = 0
        self.total_reward = 0.0

    def record(self, qtype: str, correct: bool, reward: float, word_count: int):
        self.type_total[qtype] += 1
        self.type_reward_sum[qtype] += reward
        self.type_length_sum[qtype] += word_count
        if correct:
            self.type_correct[qtype] += 1
        self.n_samples += 1
        self.total_correct += int(correct)
        self.total_reward += reward

    def snapshot(self) -> dict:
        """Return per-type stats and reset accumulators."""
        types = sorted(self.type_total.keys())
        per_type = {}
        for t in types:
            n = self.type_total[t]
            per_type[t] = {
                "n": n,
                "accuracy": self.type_correct[t] / n if n > 0 else 0.0,
                "avg_reward": self.type_reward_sum[t] / n if n > 0 else 0.0,
                "avg_length": self.type_length_sum[t] / n if n > 0 else 0.0,
            }
        overall = {
            "n": self.n_samples,
            "accuracy": self.total_correct / self.n_samples if self.n_samples > 0 else 0.0,
            "avg_reward": self.total_reward / self.n_samples if self.n_samples > 0 else 0.0,
        }
        snap = {"overall": overall, "per_type": per_type}
        self.reset()
        return snap


# Global tracker instance
_metrics = MetricsTracker()


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

    Also records per-type stats to the global MetricsTracker so the
    observability callback can report per-category accuracy.
    """
    # 'type' is passed through from the dataset row (remove_unused_columns=False)
    qtypes = kwargs.get("type", ["unknown"] * len(completions))
    rewards = []

    for idx, (completion, gt) in enumerate(zip(completions, answer)):
        if isinstance(completion, list):
            completion = completion[0] if completion else ""
        if isinstance(completion, dict):
            completion = completion.get("content", "")

        qtype = qtypes[idx] if idx < len(qtypes) else "unknown"

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

        # ── Record for observability ─────────────────────────────────────
        _metrics.record(qtype, correct, total, word_count)

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


def load_training_data(dataset_name: str, split: str = "train", max_samples: Optional[int] = None):
    """
    Load training data from HuggingFace Hub OR local file path.
    Supports: JSON, JSONL, CSV, Parquet files.
    """
    from datasets import load_dataset, Dataset as HFDataset

    # Check if it's a local file path
    local_path = Path(dataset_name)
    if local_path.exists():
        log.info(f"Loading training data from local file: {dataset_name}")
        suffix = local_path.suffix.lower()

        if suffix == ".jsonl" or suffix == ".json":
            raw_ds = load_dataset("json", data_files=str(local_path), split="train")
        elif suffix == ".csv":
            raw_ds = load_dataset("csv", data_files=str(local_path), split="train")
        elif suffix == ".parquet":
            raw_ds = load_dataset("parquet", data_files=str(local_path), split="train")
        else:
            # Try auto-detection
            raw_ds = load_dataset(str(local_path), split="train")
        log.info(f"Local dataset loaded: {len(raw_ds):,} rows")
        return raw_ds
    else:
        log.info(f"Loading training data from HuggingFace Hub: {dataset_name}")
        raw_ds = load_dataset(dataset_name, split=split)
        log.info(f"HuggingFace dataset loaded: {len(raw_ds):,} rows")
        return raw_ds


def load_heldout_eval_data(path: str, n: int = HELDOUT_N_SAMPLES) -> List[dict]:
    """Load heldout evaluation questions from local JSON/JSONL or HF dataset."""
    data = []
    p = Path(path)
    if p.exists():
        suffix = p.suffix.lower()
        with open(p) as f:
            if suffix == ".jsonl":
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            else:
                # JSON array
                data = json.load(f)
        random.shuffle(data)
        return data[:n]
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# 9. OBSERVABILITY CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

# Top-level import needed for the class definition below (module-level code
# runs before main(), so imports inside import_core() are not yet available).
from transformers import TrainerCallback, TrainerState, TrainerControl


class GRPOObservabilityCallback(TrainerCallback):
    """
    Comprehensive observability callback:
    - Per-step: reward, loss, KL, LR + moving averages + trend arrows
    - Per-question-type accuracy & reward breakdown (from MetricsTracker)
    - Heldout accuracy eval every N steps (with per-type breakdown)
    - Sample generation with full reward breakdown every N steps
    - Improvement/stagnation/degradation health signal
    - Saves metrics to JSONL for external monitoring
    - WandB logging of all metrics
    """

    # Thresholds for the health signal (fraction of moving-avg change)
    TREND_IMPROVING = 0.001   # reward up > 0.1% per step → improving
    TREND_DEGRADING = -0.002  # reward down > 0.2% per step → degrading

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

        # Rolling stats for trend detection (window=20 steps)
        self.step_rewards = deque(maxlen=20)
        self.step_losses = deque(maxlen=20)
        self.step_kls = deque(maxlen=20)

        # Per-type rolling accuracy (heldout eval history)
        self.type_accuracy_history: Dict[str, deque] = {
            t: deque(maxlen=50) for t in QUESTION_TYPES
        }

        # Heldout accuracy history
        self.heldout_history = []

        # Best reward seen so far (for "is this the best" indicator)
        self.best_reward = float("-inf")
        self.best_reward_step = 0

        # Sample questions (fixed for consistent tracking)
        self.sample_questions = []
        if heldout_data:
            self.sample_questions = heldout_data[:N_SAMPLE_QUESTIONS]

        # WandB
        self.use_wandb = wandb is not None and wandb.run is not None

        # Step counter
        self._step = 0

        # Per-type table print frequency (every N steps to avoid log spam)
        self._type_table_interval = 10

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        log.info("=" * 72)
        log.info("  GCP GRPO TRAINING START")
        log.info(f"  Steps: {MAX_STEPS} | Batch: {PER_DEVICE_BATCH} | Group: {GRPO_GROUP_SIZE}")
        log.info(f"  LR: {LEARNING_RATE} | Beta: {BETA} | Temp: {TEMPERATURE}")
        log.info(f"  Precision: {'fp16' if USE_FP16 else 'bf16'} | 4-bit: {LOAD_IN_4BIT}")
        log.info(f"  Heldout eval: every {HELDOUT_EVERY_N_STEPS} steps, {HELDOUT_N_SAMPLES} questions")
        log.info(f"  Checkpoint:  every {CHECKPOINT_EVERY_N_STEPS} steps")
        log.info(f"  Per-type table: every {self._type_table_interval} steps")
        log.info(f"  HF repo: {HF_REPO}")
        log.info("=" * 72)
        self._log_to_wandb({"status": "training_started"})

    def _trend_symbol(self, current: float, history: deque) -> str:
        """Return ↑ / → / ↓ based on comparison to moving average."""
        if len(history) < 3:
            return "→"
        avg = sum(history) / len(history)
        if current > avg * (1 + self.TREND_IMPROVING):
            return "↑"
        elif current < avg * (1 + self.TREND_DEGRADING):
            return "↓"
        return "→"

    def _health_signal(self) -> Tuple[str, str]:
        """
        Determine if the model is genuinely improving.
        Returns (signal, detail) where signal is IMPROVING / STAGNANT / DEGRADING.
        """
        if len(self.step_rewards) < 5:
            return "WARMUP", "collecting data"
        recent = list(self.step_rewards)
        # Compare first half avg vs second half avg
        mid = len(recent) // 2
        first_half = sum(recent[:mid]) / max(mid, 1)
        second_half = sum(recent[mid:]) / max(len(recent) - mid, 1)
        delta = second_half - first_half
        if delta > self.TREND_IMPROVING * len(recent):
            return "IMPROVING", f"reward ↑ {delta:+.4f} (window)"
        elif delta < self.TREND_DEGRADING * len(recent):
            return "DEGRADING", f"reward ↓ {delta:+.4f} (window)"
        return "STAGNANT", f"reward → {delta:+.4f} (window)"

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

        # ── Update rolling stats ─────────────────────────────────────────
        self.step_rewards.append(mean_reward)
        self.step_losses.append(loss)
        self.step_kls.append(kl)

        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            self.best_reward_step = state.global_step

        # ── Get per-type stats from the reward function accumulator ──────
        metrics_snap = _metrics.snapshot()

        # ── Trend arrows ─────────────────────────────────────────────────
        r_trend = self._trend_symbol(mean_reward, self.step_rewards)
        l_trend = self._trend_symbol(loss, self.step_losses)

        # ── Moving averages ──────────────────────────────────────────────
        r_avg = sum(self.step_rewards) / len(self.step_rewards) if self.step_rewards else mean_reward
        l_avg = sum(self.step_losses) / len(self.step_losses) if self.step_losses else loss

        # ── Health signal ────────────────────────────────────────────────
        health, health_detail = self._health_signal()

        # ── Console log (every step — rich single line) ──────────────────
        pct = state.global_step / MAX_STEPS * 100
        log.info(
            f"Step {state.global_step:4d}/{MAX_STEPS} ({pct:5.1f}%) | "
            f"reward={mean_reward:.4f}{r_trend}(avg{r_avg:.4f}) | "
            f"loss={loss:.4f}{l_trend}(avg{l_avg:.4f}) | "
            f"kl={kl:.6f} | lr={learning_rate:.2e}"
        )
        log.info(
            f"  HEALTH: {health} ({health_detail}) | "
            f"Best reward: {self.best_reward:.4f} @step {self.best_reward_step}"
        )

        # ── WandB log ────────────────────────────────────────────────────
        wb_logs = {
            "train/mean_reward": mean_reward,
            "train/reward_moving_avg": r_avg,
            "train/kl": kl,
            "train/loss": loss,
            "train/loss_moving_avg": l_avg,
            "train/learning_rate": learning_rate,
            "train/health": {"IMPROVING": 1, "STAGNANT": 0, "WARMUP": 0, "DEGRADING": -1}.get(health, 0),
            "step": state.global_step,
        }

        # ── Per-type stats to WandB ──────────────────────────────────────
        overall_acc = metrics_snap["overall"]["accuracy"]
        wb_logs["train/batch_accuracy"] = overall_acc
        wb_logs["train/batch_avg_reward"] = metrics_snap["overall"]["avg_reward"]
        for qtype, stats in metrics_snap["per_type"].items():
            safe = qtype.replace("/", "_")
            wb_logs[f"train/per_type/{safe}/accuracy"] = stats["accuracy"]
            wb_logs[f"train/per_type/{safe}/avg_reward"] = stats["avg_reward"]
            wb_logs[f"train/per_type/{safe}/avg_length"] = stats["avg_length"]

        # ── Heldout accuracy eval ────────────────────────────────────────
        if state.global_step > 0 and state.global_step % HELDOUT_EVERY_N_STEPS == 0:
            heldout_acc, heldout_per_type = self._eval_heldout()
            wb_logs["eval/heldout_accuracy"] = heldout_acc
            wb_logs["eval/heldout_step"] = state.global_step
            log.info(f"  >>> HELD-OUT ACCURACY: {heldout_acc:.1%} ({state.global_step} steps)")
            for qtype, acc in sorted(heldout_per_type.items()):
                safe = qtype.replace("/", "_")
                wb_logs[f"eval/heldout_per_type/{safe}_accuracy"] = acc
                log.info(f"        {qtype:20s}: {acc:.1%}")

        # ── Per-type table (every N steps) ──────────────────────────────
        if state.global_step > 0 and state.global_step % self._type_table_interval == 0:
            self._log_per_type_table(state.global_step, metrics_snap)

        # ── Sample generation with breakdown ─────────────────────────────
        if state.global_step > 0 and state.global_step % SAMPLE_LOG_EVERY_N_STEPS == 0:
            self._log_samples(state.global_step)

        # ── Upload metrics to GCS (every 10 steps) ───────────────────────
        if state.global_step > 0 and state.global_step % 10 == 0:
            self._upload_metrics_to_gcs(state.global_step)

        # ── Save metrics to JSONL ────────────────────────────────────────
        self._append_metrics(state.global_step, {
            "mean_reward": mean_reward,
            "reward_moving_avg": r_avg,
            "kl": kl,
            "loss": loss,
            "loss_moving_avg": l_avg,
            "learning_rate": learning_rate,
            "health": health,
            "batch_accuracy": overall_acc,
            "per_type": {k: {"accuracy": v["accuracy"], "avg_reward": v["avg_reward"], "n": v["n"]}
                          for k, v in metrics_snap["per_type"].items()},
        })

        self._log_to_wandb(wb_logs)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Check for preemption signal."""
        if _interrupted:
            log.warning("Interruption detected — requesting training stop.")
            control.should_training_stop = True

    def _eval_heldout(self) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate current model on heldout set.
        Returns (overall_accuracy, per_type_accuracy_dict).
        """
        if not self.heldout_data:
            log.warning("No heldout data available for eval.")
            return 0.0, {}

        FastLanguageModel.for_inference(self.model)
        self.model.eval()
        try:
            correct = 0
            total = min(HELDOUT_N_SAMPLES, len(self.heldout_data))
            type_correct: Dict[str, int] = defaultdict(int)
            type_total: Dict[str, int] = defaultdict(int)

            for item in self.heldout_data[:total]:
                # Always build chat-formatted prompt from the raw question
                raw_question = item.get("query", item.get("original_question", item.get("prompt", "")))
                prompt_text = build_prompt(self.tokenizer, raw_question)

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
                qtype = item.get("type", "unknown")
                type_total[qtype] += 1
                if is_correct:
                    correct += 1
                    type_correct[qtype] += 1
        finally:
            FastLanguageModel.for_training(self.model)
            self.model.train()

        accuracy = correct / total if total > 0 else 0.0
        self.heldout_history.append({"step": self._step, "accuracy": accuracy})

        per_type_acc = {}
        for t in type_total:
            per_type_acc[t] = type_correct[t] / type_total[t]
            if t not in self.type_accuracy_history:
                self.type_accuracy_history[t] = deque(maxlen=50)
            self.type_accuracy_history[t].append(per_type_acc[t])

        return accuracy, per_type_acc

    def _log_samples(self, step: int):
        """Generate and log sample outputs with full reward breakdown."""
        if not self.sample_questions:
            return

        FastLanguageModel.for_inference(self.model)
        self.model.eval()
        try:
            log.info(f"\n{'═' * 72}")
            log.info(f"  SAMPLE GENERATION — Step {step}")
            log.info(f"{'═' * 72}")

            for i, item in enumerate(self.sample_questions):
                question = item.get("original_question", item.get("prompt", ""))
                gt = item.get("answer", "")
                qtype = item.get("type", "?")

                # Always build chat-formatted prompt from the raw question
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

                log.info(f"\n  [{i + 1}] Type: {qtype}")
                log.info(f"  Q: {question[:100]}...")
                log.info(f"  GT: {gt} | Pred: {pred} ({confidence}) | {'✓' if is_correct else '✗'}")
                log.info(f"  Output ({word_count} words): {completion[:300]}{'...' if len(completion) > 300 else ''}")
                log.info(f"  Reward: {total_reward:.3f} (correct={r_correct:.1f}, length={r_length:.2f})")

            log.info(f"{'═' * 72}\n")
        finally:
            FastLanguageModel.for_training(self.model)
            self.model.train()

    def _log_per_type_table(self, step: int, snap: dict):
        """Log a formatted per-question-type accuracy & reward table."""
        per_type = snap["per_type"]
        if not per_type:
            return

        log.info(f"\n  ┌─────────────────────────────────────────────────────────────────────┐")
        log.info(f"  │ PER-TYPE BREAKDOWN — Step {step:4d}                                    │")
        log.info(f"  ├──────────────────┬──────────┬──────────┬──────────┬────────────────┤")
        log.info(f"  │ Type             │ Accuracy │ AvgReward│ AvgLen   │ Trend          │")
        log.info(f"  ├──────────────────┼──────────┼──────────┼──────────┼────────────────┤")

        for qtype in QUESTION_TYPES:
            stats = per_type.get(qtype)
            if stats and stats["n"] > 0:
                # Trend from heldout history if available
                hist = self.type_accuracy_history.get(qtype, deque())
                if len(hist) >= 3:
                    recent_avg = sum(list(hist)[-3:]) / 3
                    older_avg = sum(list(hist)[:-3]) / max(len(list(hist)) - 3, 1)
                    delta = recent_avg - older_avg
                    trend_str = f"↑{delta:+.3f}" if delta > 0.01 else (f"↓{delta:+.3f}" if delta < -0.01 else f"→{delta:+.3f}")
                else:
                    trend_str = "n/a"
                log.info(
                    f"  │ {qtype:16s} │ {stats['accuracy']:7.1%} │ {stats['avg_reward']:8.4f} │ "
                    f"{stats['avg_length']:7.1f}  │ {trend_str:14s} │"
                )
            else:
                log.info(f"  │ {qtype:16s} │    n/a   │    n/a   │    n/a   │      n/a       │")

        overall = snap["overall"]
        log.info(f"  ├──────────────────┼──────────┼──────────┼──────────┼────────────────┤")
        log.info(
            f"  │ {'OVERALL':16s} │ {overall['accuracy']:7.1%} │ {overall['avg_reward']:8.4f} │ "
            f"{'':7s}  │ {'':14s} │"
        )
        log.info(f"  └──────────────────┴──────────┴──────────┴──────────┴────────────────┘\n")

    def _append_metrics(self, step: int, metrics: dict):
        """Append metrics to JSONL file."""
        record = {"step": step, "timestamp": datetime.now(timezone.utc).isoformat(), **metrics}
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _upload_metrics_to_gcs(self, step: int):
        """Upload latest metrics snapshot to GCS for external monitoring."""
        gcs_dir = os.environ.get("GCS_METRICS_DIR", "")
        if not gcs_dir:
            return
        try:
            import subprocess
            snapshot = {
                "step": step,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reward_moving_avg": sum(self.step_rewards) / len(self.step_rewards) if self.step_rewards else 0,
                "loss_moving_avg": sum(self.step_losses) / len(self.step_losses) if self.step_losses else 0,
                "best_reward": self.best_reward,
                "best_reward_step": self.best_reward_step,
                "heldout_history": self.heldout_history[-10:],
            }
            local_tmp = str(self.output_dir / "latest_metrics.json")
            with open(local_tmp, "w") as f:
                json.dump(snapshot, f, default=str)
            subprocess.run(
                ["gcloud", "storage", "cp", local_tmp, f"{gcs_dir}/latest_metrics.json"],
                capture_output=True, timeout=30,
            )
        except Exception as e:
            log.debug(f"GCS metrics upload failed: {e}")

    def _log_to_wandb(self, logs: dict):
        """Log to WandB if available."""
        if self.use_wandb:
            try:
                wandb.log(logs)
            except Exception as e:
                log.debug(f"WandB log failed: {e}")

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Final summary."""
        log.info("=" * 72)
        log.info("  GCP GRPO TRAINING COMPLETE")
        log.info(f"  Final step: {state.global_step}")
        log.info(f"  Best reward: {self.best_reward:.4f} @ step {self.best_reward_step}")
        if self.heldout_history:
            best = max(self.heldout_history, key=lambda x: x["accuracy"])
            log.info(f"  Best held-out accuracy: {best['accuracy']:.1%} at step {best['step']}")
        log.info(f"  Metrics saved to: {self.metrics_file}")
        log.info("=" * 72)
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
    try:
        repetition_ratios = []
        response_lengths = []
        n = min(50, len(heldout_data))

        for item in heldout_data[:n]:
            prompt_text = build_prompt(
                tokenizer, item.get("query", item.get("original_question", item.get("prompt", "")))
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
                trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
                if trigrams:
                    repetition_ratios.append(1.0 - len(set(trigrams)) / len(trigrams))
    finally:
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
    log.info("GCP GRPO TRAINING — Llama 3.2 3B on MetaMathQA")
    log.info(f"GPU: L4/T4 | Precision: {'fp16' if USE_FP16 else 'bf16'} | 4-bit: {LOAD_IN_4BIT}")
    log.info(f"Time: {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC}")
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
            "precision": "fp16" if USE_FP16 else "bf16",
            "load_in_4bit": LOAD_IN_4BIT,
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
        dtype=torch.float16 if USE_FP16 else torch.bfloat16,
        load_in_4bit=LOAD_IN_4BIT,  # 4-bit for L4/T4 memory efficiency
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
    log.info(f"Model loaded in {time.time() - t0:.1f}s")
    log.info(f"Params: {total_p / 1e9:.2f}B total, {trainable_p / 1e6:.2f}M trainable ({100 * trainable_p / total_p:.2f}%)")
    log.info(f"VRAM after load: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # ── Step 5: Load & preprocess data ────────────────────────────────────
    log.info(f"Loading dataset: {DATASET_NAME}")
    raw_ds = load_training_data(DATASET_NAME, split=DATA_SPLIT, max_samples=MAX_TRAIN_SAMPLES)
    log.info(f"Raw dataset: {len(raw_ds):,} rows")

    # Prepare for GRPO
    processed, skipped = prepare_data(tokenizer, raw_ds, max_samples=MAX_TRAIN_SAMPLES)
    log.info(f"Processed: {len(processed):,} | Skipped: {skipped:,}")

    if not processed:
        log.error("No training data after preprocessing — aborting")
        sys.exit(1)

    # Convert to HF Dataset
    train_ds = HFDataset.from_list(processed)
    log.info(f"Training dataset: {len(train_ds):,} samples")

    # ── Step 6: Load heldout eval data ────────────────────────────────────
    heldout_eval = []
    # Resolve heldout path: explicit arg > parent data dir > skip
    if HELDOUT_PATH:
        heldout_file = Path(HELDOUT_PATH)
    else:
        heldout_file = Path(__file__).parent.parent / "data" / "grpo_heldout.json"

    if heldout_file.exists():
        heldout_eval = load_heldout_eval_data(str(heldout_file), HELDOUT_N_SAMPLES)
        log.info(f"Heldout eval data: {len(heldout_eval)} questions (from {heldout_file})")
    else:
        log.warning(f"Heldout file not found at {heldout_file} — eval disabled")

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
        output_dir=str(OUTPUT_DIR / "checkpoints"),

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

        # Precision — fp16 for L4/T4 (bf16 not well supported)
        fp16=USE_FP16,
        bf16=USE_BF16,

        # Optimizer
        optim="adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Saving + Hub
        save_strategy="steps",
        save_steps=CHECKPOINT_EVERY_N_STEPS,
        save_total_limit=3,
        push_to_hub=bool(HF_TOKEN) and args.push_to_hub,
        hub_model_id=HF_REPO if (HF_TOKEN and args.push_to_hub) else None,
        hub_strategy="checkpoint" if (HF_TOKEN and args.push_to_hub) else "end",

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
        processing_class=tokenizer,
        reward_funcs=compute_rewards,
        args=grpo_config,
        train_dataset=train_ds,
        callbacks=[obs_callback],
    )

    log.info(f"Trainer created | Steps/epoch ≈ {len(train_ds) // PER_DEVICE_BATCH}")
    log.info(f"VRAM after trainer setup: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # ── Step 9: Pre-training eval ─────────────────────────────────────────
    log.info("Running pre-training evaluation...")
    pre_acc, pre_per_type = obs_callback._eval_heldout()
    log.info(f"Pre-training held-out accuracy: {pre_acc:.1%}")
    if wandb and wandb.run:
        wandb.log({"eval/pre_training_accuracy": pre_acc, "step": 0})

    # ── Step 10: Train ────────────────────────────────────────────────────
    log.info("Starting GRPO training...")
    train_start = time.time()

    trainer.train()

    train_elapsed = time.time() - train_start
    log.info(f"Training finished in {train_elapsed / 3600:.2f} hours")

    # ── Step 11: Post-training eval ───────────────────────────────────────
    log.info("Running post-training evaluation...")
    post_acc, post_per_type = obs_callback._eval_heldout()
    log.info(f"Post-training held-out accuracy: {post_acc:.1%}")
    log.info(f"Improvement: {post_acc - pre_acc:+.1%}")

    # Alignment audit
    log.info("Running alignment audit...")
    audit = run_alignment_audit(model, tokenizer, heldout_eval)
    log.info(f"Alignment audit: {audit}")

    # ── Step 12: Save final model locally + push to Hub ───────────────────
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    log.info(f"Final model saved locally to: {final_dir}")

    if HF_TOKEN and args.push_to_hub:
        log.info(f"Pushing final model to HF Hub: {HF_REPO}")
        model.push_to_hub(HF_REPO, token=HF_TOKEN, private=args.hub_private)
        tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN, private=args.hub_private)
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
            "w_correctness": W_CORRECTNESS, "w_length": W_LENGTH,
            "precision": "fp16" if USE_FP16 else "bf16",
            "load_in_4bit": LOAD_IN_4BIT,
        },
    }
    summary_path = OUTPUT_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Summary saved to: {summary_path}")

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
    log.info(f"  Model (local): {final_dir}")
    if HF_TOKEN and args.push_to_hub:
        log.info(f"  Model (Hub):   {HF_REPO}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
