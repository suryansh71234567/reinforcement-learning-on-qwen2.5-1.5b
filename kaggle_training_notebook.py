import json

nb = {
  "cells": [],
  "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.12"},
    "accelerator": "GPU",
    "gpuClass": "T4"
  },
  "nbformat": 4,
  "nbformat_minor": 4
}

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    lines = text.splitlines(keepends=True)
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}

# ═══════════════════════════════════════════════════════════════════════════
# CELL 1: Title & Config
# ═══════════════════════════════════════════════════════════════════════════
nb["cells"].append(md("""# GRPO Training — Llama 3.2 3B on MetaMathQA (Kaggle T4)

**Stack:** Unsloth + TRL GRPOTrainer | **GPU:** T4 (16GB) | **Precision:** fp16 + 4-bit

**Observability:** reward tracking, heldout eval, per-type stats, alignment audit, WandB

## Config
- Model: meta-llama/Llama-3.2-3B-Instruct
- LoRA: r=16, alpha=32, dropout=0.05, rsLoRA
- GRPO: group_size=4, batch_size=4, grad_accum=2
- Training: max_steps=1000, lr=5e-7, beta=0.03, temperature=0.8
- Reward: correctness (0.80) + length penalty (0.20)
- Data: meta-math/MetaMathQA (395K questions)
"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 2: Install dependencies
# ═══════════════════════════════════════════════════════════════════════════
nb["cells"].append(code("""
import subprocess, sys, time, os

def pip_install(pkg):
    print(f"Installing: {pkg}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", pkg],
        capture_output=True, text=True
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"FAILED ({elapsed:.1f}s): {result.stderr[-500:]}")
    else:
        print(f"OK ({elapsed:.1f}s)")

# Core stack (pinned versions from proven existing repo)
pip_install("unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git")
pip_install("trl==0.24.0")
pip_install("peft==0.11.1")
pip_install("accelerate==0.33.0")
pip_install("transformers==4.44.0")
pip_install("datasets")
pip_install("bitsandbytes")

# Hackathon toolkits
pip_install("aligntune @ git+https://github.com/lexi-labs/aligntune.git")
pip_install("sglang==0.3.0")
pip_install("auditkit @ git+https://github.com/lexi-labs/auditkit.git")

# Logging
pip_install("wandb")

print("\\nAll installs complete.")
"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 3: Environment check
# ═══════════════════════════════════════════════════════════════════════════
nb["cells"].append(code("""
import torch, os, sys, json, re, math, random, signal, logging, time
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any

print("=" * 60)
print("ENVIRONMENT")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_mem/1e9:.1f} GB)")
print("=" * 60)
"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 4: Configuration
# ═══════════════════════════════════════════════════════════════════════════
nb["cells"].append(code("""
# ═══ HF credentials ═══
HF_TOKEN = os.environ.get("HF_TOKEN", "hf_YOUR_TOKEN_HERE")
HF_USERNAME = "Suryansh7123"
HF_REPO = f"{HF_USERNAME}/llama3.2-3b-grpo-metamath"

# ═══ WandB ═══
WANDB_PROJECT = "grpo-llama3b-metamath"
WANDB_RUN_NAME = f"grpo-t4-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")

# ═══ Reproducibility ═══
SEED = 42
random.seed(SEED)

# ═══ Model ═══
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH = 2048

# ═══ LoRA ═══
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ═══ GRPO (SMALL for T4) ═══
GRPO_GROUP_SIZE = 4   # reduced from 8 for T4 memory
PER_DEVICE_BATCH = 4  # reduced from 8 for T4 memory
GRAD_ACCUM_STEPS = 2
MAX_STEPS = 1000
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.8
BETA = 0.03
LEARNING_RATE = 5e-7
MAX_GRAD_NORM = 0.1
WARMUP_STEPS = 50

# ═══ Reward ═══
W_CORRECTNESS = 0.80
W_LENGTH = 0.20
LENGTH_THRESHOLD = 300

# ═══ Data ═══
DATASET_NAME = "meta-math/MetaMathQA"
HELDOUT_EVERY_N_STEPS = 100
HELDOUT_N_SAMPLES = 200
CHECKPOINT_EVERY_N_STEPS = 200
SAMPLE_LOG_EVERY_N_STEPS = 50
N_SAMPLE_QUESTIONS = 5

# ═══ Output ═══
OUTPUT_DIR = Path("/kaggle/working/grpo_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══ Question types ═══
QUESTION_TYPES = [
    "GSM_SV", "GSM_Rephrased", "GSM_AnsAug", "GSM_FOBAR",
    "MATH_Rephrased", "MATH_AnsAug", "MATH_FOBAR", "MATH_SV",
]

print("Config loaded.")
print(f"  Model: {MODEL_NAME}")
print(f"  Group: {GRPO_GROUP_SIZE} | Batch: {PER_DEVICE_BATCH} | Steps: {MAX_STEPS}")
print(f"  LR: {LEARNING_RATE} | Beta: {BETA} | Temp: {TEMPERATURE}")
print(f"  Output: {OUTPUT_DIR}")
"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 5: Answer Extractor (7-pattern with confidence)
# ═══════════════════════════════════════════════════════════════════════════
nb["cells"].append(code("""
_WORD_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "negative": "-",
}

def _normalize_answer(s):
    if not s: return ""
    s = str(s).strip().replace(",", "").replace(" ", "").rstrip(".").lower()
    if s.endswith("%"): s = s[:-1]
    return s

def _answers_match(pred, gold):
    pn, gn = _normalize_answer(pred), _normalize_answer(gold)
    if not pn or not gn: return False
    if pn == gn: return True
    try:
        return abs(float(pn) - float(gn)) < 1e-4
    except (ValueError, OverflowError): pass
    for a, b in [(pn, gn), (gn, pn)]:
        if "/" in a:
            try:
                num, den = a.split("/", 1)
                return abs(float(num)/float(den) - float(b)) < 1e-4
            except (ValueError, ZeroDivisionError): pass
    return gn in pn or pn in gn

def extract_answer(text):
    if not text: return ("", "low")
    m = re.search(r'####\\s*([-\\d\\.]+)', text)
    if m: return (m.group(1).strip(), "high")
    m = re.search(r'\\\\boxed\\{([^}]+)\\}', text)
    if m:
        inner = m.group(1).strip()
        nums = re.findall(r'[-\\d\\.]+', inner)
        return (nums[-1], "high") if nums else (inner, "high")
    ms = re.findall(r'[xX]\\s*=\\s*([-\\d\\.]+)', text)
    if ms: return (ms[-1].strip(), "medium")
    ms = re.findall(r'=\\s*([-\\d\\.]+)\\s*$', text, re.MULTILINE)
    if ms: return (ms[-1].strip(), "medium")
    ms = re.findall(r'(?:the answer is|therefore|thus|final answer is|so (?:the|it is|his|her|their))\\s*:?\\s*([-\\d\\.]+)', text, re.IGNORECASE)
    if ms: return (ms[-1].strip(), "medium")
    ms = re.findall(r'negative\\s+(\\w+)', text, re.IGNORECASE)
    if ms:
        word = ms[-1].lower()
        if word in _WORD_NUMBERS: return (_WORD_NUMBERS[word], "medium")
    ms = re.findall(r'(?:there (?:are|were)|(?:has|have|are|had|was|were|is)\\s+)(?:exactly\\s+)?([-\\d\\.]+)', text, re.IGNORECASE)
    if ms: return (ms[-1].strip(), "low")
    numbers = re.findall(r'[-\\d\\.]+', text)
    if numbers:
        conclusion_pos = -1
        for kw in ["therefore", "thus", "so", "final answer", "the answer is", "in conclusion"]:
            pos = text.lower().rfind(kw)
            if pos > conclusion_pos: conclusion_pos = pos
        if conclusion_pos >= 0:
            after_nums = re.findall(r'[-\\d\\.]+', text[conclusion_pos:])
            if after_nums: return (after_nums[-1].strip(), "low")
        return (numbers[-1].strip(), "low")
    return ("", "low")

# Quick test
test_cases = [
    ("The answer is 42.", "42"),
    ("#### 27", "27"),
    ("\\\\boxed{15}", "15"),
    ("x = 100 so x = 100", "100"),
    ("negative five", "-5"),
]
for text, expected in test_cases:
    pred, conf = extract_answer(text)
    match = _answers_match(pred, expected)
    print(f"  '{text[:30]}' -> {pred} ({conf}) {'✓' if match else '✗'}")
print("Answer extractor: OK")
"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 6: Reward function
# ═══════════════════════════════════════════════════════════════════════════
nb["cells"].append(code("""
def compute_rewards(prompts, completions, answer, **kwargs):
    rewards = []
    for completion, gt in zip(completions, answer):
        if isinstance(completion, list): completion = completion[0] if completion else ""
        pred, confidence = extract_answer(completion)
        correct = _answers_match(pred, gt)
        r_correct = 1.0 if correct else 0.0
        word_count = len(completion.split())
        if word_count <= LENGTH_THRESHOLD:
            r_length = 1.0
        else:
            over = word_count - LENGTH_THRESHOLD
            r_length = max(0.0, 1.0 - over / 500.0)
        total = W_CORRECTNESS * r_correct + W_LENGTH * r_length
        rewards.append(total)
    return rewards

print("Reward function: OK")
"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 7: Load model (4-bit + LoRA for T4)
# ═══════════════════════════════════════════════════════════════════════════
nb["cells"].append(code("""
from unsloth import FastLanguageModel
from huggingface_hub import login

# Login to HF
login(token=HF_TOKEN)
print(f"Logged in as {HF_USERNAME}")

print(f"Loading model: {MODEL_NAME}")
t0 = time.time()

# 4-bit quantization for T4 (16GB)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.float16,  # T4 doesn't support bf16
    load_in_4bit=True,    # 4-bit for T4 memory
    token=HF_TOKEN,
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
print(f"Model loaded in {time.time()-t0:.1f}s")
print(f"Params: {total_p/1e9:.2f}B total, {trainable_p/1e6:.2f}M trainable ({100*trainable_p/total_p:.2f}%)")
print(f"VRAM after load: {torch.cuda.memory_reserved()/1e9:.2f} GB")
print("=" * 60)
"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 8: Load & preprocess data
# ═══════════════════════════════════════════════════════════════════════════
nb["cells"].append(code("""
from datasets import load_dataset, Dataset as HFDataset

SYSTEM_PROMPT = (
    "You are a math solver. Solve the problem step by step, "
    "showing your reasoning clearly. "
    "End your response with: The answer is <number>."
)

def build_prompt(tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

print(f"Loading dataset: {DATASET_NAME}")
raw_ds = load_dataset(DATASET_NAME, split="train")
print(f"Raw dataset: {len(raw_ds):,} rows")

processed = []
skipped = 0
for ex in raw_ds:
    question = ex.get("query", "").strip()
    response = ex.get("response", "").strip()
    qtype = ex.get("type", "unknown")
    if not question or not response:
        skipped += 1
        continue
    gt_answer, _ = extract_answer(response)
    if not gt_answer:
        skipped += 1
        continue
    prompt_text = build_prompt(tokenizer, question)
    processed.append({
        "prompt": prompt_text,
        "answer": gt_answer,
        "type": qtype,
        "question": question,
    })

print(f"Processed: {len(processed):,} | Skipped: {skipped:,}")

train_ds = HFDataset.from_list(processed)
print(f"Training dataset: {len(train_ds):,} samples")
print("=" * 60)
"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 9: Observability callback
# ═══════════════════════════════════════════════════════════════════════════
nb["cells"].append(code("""
from trl import GRPOTrainer, GRPOConfig
from transformers import TrainerCallback, TrainerState, TrainerControl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("grpo_kaggle")

# Load heldout eval data
heldout_eval = []
heldout_path = "/kaggle/input/metamathqa-grpo/grpo_heldout.json"
if not Path(heldout_path).exists():
    # Try HF dataset
    try:
        ho_ds = load_dataset("meta-math/MetaMathQA", split="train[:200]")
        for ex in ho_ds:
            q = ex.get("query","").strip()
            r = ex.get("response","").strip()
            gt, _ = extract_answer(r)
            if q and gt:
                heldout_eval.append({"prompt": build_prompt(tokenizer, q), "answer": gt, "type": ex.get("type","unknown"), "question": q})
    except:
        pass
else:
    with open(heldout_path) as f:
        for line in f:
            heldout_eval.append(json.loads(line))
    random.shuffle(heldout_eval)
    heldout_eval = heldout_eval[:HELDOUT_N_SAMPLES]

print(f"Heldout eval data: {len(heldout_eval)} questions")

class GRPOObservabilityCallback(TrainerCallback):
    def __init__(self, tokenizer, model, reward_fn, heldout_data, output_dir):
        self.tokenizer = tokenizer
        self.model = model
        self.reward_fn = reward_fn
        self.heldout_data = heldout_data
        self.output_dir = output_dir
        self.metrics_file = output_dir / "metrics.jsonl"
        self.step_rewards = deque(maxlen=100)
        self.step_lengths = deque(maxlen=100)
        self.per_type_rewards = defaultdict(list)
        self.per_type_correct = defaultdict(int)
        self.per_type_total = defaultdict(int)
        self.heldout_history = []
        self.sample_questions = heldout_data[:N_SAMPLE_QUESTIONS] if heldout_data else []
        self.use_wandb = False
        self._step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        log.info("=" * 70)
        log.info("TRAINING START")
        log.info(f"  Steps: {MAX_STEPS} | Batch: {PER_DEVICE_BATCH} | Group: {GRPO_GROUP_SIZE}")
        log.info(f"  LR: {LEARNING_RATE} | Beta: {BETA} | Temp: {TEMPERATURE}")
        log.info("=" * 70)

    def on_log(self, args, state, control, logs=None, **kwargs):
        self._step = state.global_step
        if logs is None: return
        mean_reward = logs.get("reward", 0.0)
        kl = logs.get("kl", 0.0)
        loss = logs.get("loss", 0.0)
        lr = logs.get("learning_rate", 0.0)
        self.step_rewards.append(mean_reward)
        log.info(f"Step {state.global_step:4d}/{MAX_STEPS} | reward={mean_reward:.4f} | kl={kl:.6f} | loss={loss:.4f} | lr={lr:.2e}")

        wb_logs = {"train/mean_reward": mean_reward, "train/kl": kl, "train/loss": loss, "train/learning_rate": lr, "step": state.global_step}

        if state.global_step > 0 and state.global_step % HELDOUT_EVERY_N_STEPS == 0:
            heldout_acc = self._eval_heldout()
            wb_logs["eval/heldout_accuracy"] = heldout_acc
            log.info(f"  >>> HELD-OUT ACCURACY: {heldout_acc:.1%} ({state.global_step} steps)")

        if state.global_step > 0 and state.global_step % SAMPLE_LOG_EVERY_N_STEPS == 0:
            self._log_samples(state.global_step)
            self._log_per_type_stats(state.global_step)

        self._append_metrics(state.global_step, {"mean_reward": mean_reward, "kl": kl, "loss": loss, "learning_rate": lr})

        if self.use_wandb:
            try: wb.log(wb_logs)
            except: pass

    def on_step_end(self, args, state, control, **kwargs):
        pass

    def _eval_heldout(self):
        if not self.heldout_data: return 0.0
        FastLanguageModel.for_inference(self.model)
        self.model.eval()
        correct = 0
        total = min(HELDOUT_N_SAMPLES, len(self.heldout_data))
        for item in self.heldout_data[:total]:
            prompt_text = item.get("prompt", "")
            if not prompt_text:
                prompt_text = build_prompt(self.tokenizer, item.get("question", ""))
            enc = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS).to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id, use_cache=True)
            completion = self.tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
            pred, conf = extract_answer(completion)
            if _answers_match(pred, item.get("answer", "")): correct += 1
        FastLanguageModel.for_training(self.model)
        self.model.train()
        accuracy = correct / total if total > 0 else 0.0
        self.heldout_history.append({"step": self._step, "accuracy": accuracy})
        return accuracy

    def _log_samples(self, step):
        if not self.sample_questions: return
        FastLanguageModel.for_inference(self.model)
        self.model.eval()
        log.info(f"{'═' * 70}\\n  SAMPLE GENERATION — Step {step}\\n{'═' * 70}")
        for i, item in enumerate(self.sample_questions):
            question = item.get("question", "")
            gt = item.get("answer", "")
            qtype = item.get("type", "?")
            prompt_text = build_prompt(self.tokenizer, question)
            enc = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS).to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id, use_cache=True)
            completion = self.tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
            pred, conf = extract_answer(completion)
            is_correct = _answers_match(pred, gt)
            word_count = len(completion.split())
            r_correct = 1.0 if is_correct else 0.0
            r_length = 1.0 if word_count <= LENGTH_THRESHOLD else max(0.0, 1.0 - (word_count - LENGTH_THRESHOLD)/500.0)
            total_reward = W_CORRECTNESS * r_correct + W_LENGTH * r_length
            log.info(f"\\n  [{i+1}] Type: {qtype}\\n  Q: {question[:100]}...\\n  GT: {gt} | Pred: {pred} ({conf}) | {'✓' if is_correct else '✗'}\\n  Output ({word_count} words): {completion[:300]}{'...' if len(completion) > 300 else ''}\\n  Reward: {total_reward:.3f} (correct={r_correct:.1f}, length={r_length:.2f})")
        log.info(f"{'═' * 70}\\n")
        FastLanguageModel.for_training(self.model)
        self.model.train()

    def _log_per_type_stats(self, step):
        log.info(f"\\n  PER-TYPE REWARDS — Step {step}")
        for qtype in QUESTION_TYPES:
            type_rewards = self.per_type_rewards.get(qtype, [])
            if type_rewards:
                avg_r = sum(type_rewards) / len(type_rewards)
                log.info(f"    {qtype:20s}: avg_reward={avg_r:.4f} (n={len(type_rewards)})")
            else:
                log.info(f"    {qtype:20s}: no data yet")

    def _append_metrics(self, step, metrics):
        record = {"step": step, "timestamp": datetime.now(timezone.utc).isoformat(), **metrics}
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(record) + "\\n")

    def on_train_end(self, args, state, control, **kwargs):
        log.info("=" * 70)
        log.info("TRAINING COMPLETE")
        log.info(f"  Final step: {state.global_step}")
        if self.heldout_history:
            best = max(self.heldout_history, key=lambda x: x["accuracy"])
            log.info(f"  Best held-out accuracy: {best['accuracy']:.1%} at step {best['step']}")
        log.info("=" * 70)

obs_callback = GRPOObservabilityCallback(
    tokenizer=tokenizer, model=model, reward_fn=compute_rewards,
    heldout_data=heldout_eval, output_dir=OUTPUT_DIR,
)
print("Observability callback: OK")
"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 10: WandB init
# ═══════════════════════════════════════════════════════════════════════════
nb["cells"].append(code("""
wb = None
try:
    import wandb
    if WANDB_API_KEY:
        wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config={
            "model": MODEL_NAME, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "group_size": GRPO_GROUP_SIZE, "batch_size": PER_DEVICE_BATCH,
            "grad_accum": GRAD_ACCUM_STEPS, "max_steps": MAX_STEPS,
            "max_new_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE,
            "beta": BETA, "learning_rate": LEARNING_RATE,
            "length_threshold": LENGTH_THRESHOLD, "precision": "fp16-4bit",
        })
        obs_callback.use_wandb = True
        print(f"WandB initialized: {WANDB_PROJECT}/{WANDB_RUN_NAME}")
    else:
        print("WANDB_API_KEY not set — WandB disabled")
except ImportError:
    print("wandb not installed — logging to console only")
"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 11: Create GRPO Trainer
# ═══════════════════════════════════════════════════════════════════════════
nb["cells"].append(code("""
print("Creating GRPO trainer...")

grpo_config = GRPOConfig(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    num_generations=GRPO_GROUP_SIZE,
    max_completion_length=MAX_NEW_TOKENS,
    beta=BETA,
    temperature=TEMPERATURE,
    num_train_epochs=1,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    max_grad_norm=MAX_GRAD_NORM,
    warmup_steps=WARMUP_STEPS,
    lr_scheduler_type="cosine",
    bf16=False,       # T4 doesn't support bf16
    fp16=True,        # T4 uses fp16
    optim="adamw_8bit",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    save_strategy="steps",
    save_steps=CHECKPOINT_EVERY_N_STEPS,
    save_total_limit=3,
    push_to_hub=True,
    hub_model_id=HF_REPO,
    hub_strategy="checkpoint",
    logging_steps=1,
    report_to="wandb" if (obs_callback.use_wandb) else "none",
    run_name=WANDB_RUN_NAME,
    seed=SEED,
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

print(f"Trainer created | Steps/epoch ≈ {len(train_ds) // PER_DEVICE_BATCH}")
print(f"VRAM after trainer setup: {torch.cuda.memory_reserved()/1e9:.2f} GB")
print("=" * 60)
"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 12: Pre-training eval
# ═══════════════════════════════════════════════════════════════════════════
nb["cells"].append(code("""
print("Running pre-training evaluation...")
pre_acc = obs_callback._eval_heldout()
print(f"Pre-training held-out accuracy: {pre_acc:.1%}")
if obs_callback.use_wandb:
    wb.log({"eval/pre_training_accuracy": pre_acc, "step": 0})
print("=" * 60)
"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 13: TRAIN
# ═══════════════════════════════════════════════════════════════════════════
nb["cells"].append(code("""
print("Starting GRPO training...")
train_start = time.time()

trainer.train()

train_elapsed = time.time() - train_start
print(f"Training finished in {train_elapsed/3600:.2f} hours")
print("=" * 60)
"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 14: Post-training eval
# ═══════════════════════════════════════════════════════════════════════════
nb["cells"].append(code("""
print("Running post-training evaluation...")
post_acc = obs_callback._eval_heldout()
print(f"Post-training held-out accuracy: {post_acc:.1%}")
print(f"Improvement: {post_acc - pre_acc:+.1%}")

# Save final model
final_dir = OUTPUT_DIR / "final"
final_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"Final model saved to: {final_dir}")

# Push to HF Hub
print(f"Pushing final model to HF Hub: {HF_REPO}")
model.push_to_hub(HF_REPO, token=HF_TOKEN, private=True)
tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN, private=True)
print("Push complete")

# Save summary
summary = {
    "model": MODEL_NAME, "dataset": DATASET_NAME, "hf_repo": HF_REPO,
    "pre_training_accuracy": pre_acc, "post_training_accuracy": post_acc,
    "improvement": post_acc - pre_acc, "training_steps": MAX_STEPS,
    "training_hours": train_elapsed / 3600,
    "hyperparameters": {
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
        "group_size": GRPO_GROUP_SIZE, "batch_size": PER_DEVICE_BATCH,
        "grad_accum": GRAD_ACCUM_STEPS, "max_steps": MAX_STEPS,
        "max_new_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE,
        "beta": BETA, "learning_rate": LEARNING_RATE,
        "length_threshold": LENGTH_THRESHOLD, "precision": "fp16-4bit",
    },
}
with open(OUTPUT_DIR / "training_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

if obs_callback.use_wandb:
    wb.log({"eval/post_training_accuracy": post_acc, "eval/improvement": post_acc - pre_acc})
    wb.finish()

print("=" * 70)
print("ALL DONE")
print(f"  Pre-training:  {pre_acc:.1%}")
print(f"  Post-training: {post_acc:.1%}")
print(f"  Improvement:   {post_acc - pre_acc:+.1%}")
print(f"  Model: {HF_REPO}")
print("=" * 70)
"""))

# ═══════════════════════════════════════════════════════════════════════════
# CELL 15: Alignment audit (post-training)
# ═══════════════════════════════════════════════════════════════════════════
nb["cells"].append(code("""
print("Running alignment audit...")

FastLanguageModel.for_inference(model)
model.eval()

repetition_ratios = []
response_lengths = []
n = min(50, len(heldout_eval))

for item in heldout_eval[:n]:
    prompt_text = build_prompt(tokenizer, item.get("question", ""))
    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id)
    completion = tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    words = completion.split()
    response_lengths.append(len(words))
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
print(f"Alignment audit: {audit}")
print("=" * 60)
"""))

# Write notebook
path = "C:/coding vs/syntax_error_rl/kaggle_training_notebook.ipynb"
with open(path, "w") as f:
    json.dump(nb, f, indent=1)

with open(path) as f:
    v = json.load(f)
print(f"Written: {len(v['cells'])} cells, valid JSON: OK")
print(f"Path: {path}")
