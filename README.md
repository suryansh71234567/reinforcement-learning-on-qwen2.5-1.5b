# Syntax Error 2026 — GRPO RL Training for Math Reasoning

Reinforcement learning pipeline that fine-tunes **Qwen2.5-1.5B-Instruct** on **MetaMathQA** using **GRPO** (Group Relative Policy Optimization) with **Unsloth** + **TRL**.

**Result: 41.5% → 64.5% accuracy (+23 points) on a 200-question heldout test set.**

---

## Overview

This project trains a small open-source LLM to solve grade-school and competition math problems via RL. Instead of supervised fine-tuning on fixed answers, the model samples multiple solutions per question and receives a reward only when the final numeric answer is correct — teaching it to self-correct arithmetic and reasoning errors.

| | Base | GRPO-Trained |
|---|---|---|
| **Accuracy** | 41.5% (83/200) | **64.5% (129/200)** |
| **Learned (wrong→right)** | — | **57** |
| **Forgotten (right→wrong)** | — | **11** |
| **Net change** | — | **+46** |

### Per-type improvement

| Type | Base | Trained | Delta |
|------|------|---------|-------|
| GSM_AnsAug | 44% | 96% | **+52** |
| GSM_Rephrased | 32% | 80% | **+48** |
| MATH_AnsAug | 32% | 60% | **+28** |
| MATH_SV | 36% | 56% | +20 |
| MATH_Rephrased | 32% | 48% | +16 |
| GSM_FOBAR | 64% | 76% | +12 |
| GSM_SV | 48% | 60% | +12 |
| MATH_FOBAR | 44% | 40% | -4 |

---

## Repository structure

```
syntax_error_rl/
├── gcp_training/              # GCP VM training (L4/T4)
│   ├── gcp_grpo_train.py      # Main GRPO trainer (1450 lines, full observability)
│   ├── startup.sh             # Two-phase VM startup script
│   └── enable_billing.sh
├── aws_training/              # AWS training variant
│   ├── grpo_train.py
│   └── setup_and_run.sh
├── sagemaker_training/        # SageMaker training variant
│   ├── sagemaker_grpo_train.py
│   ├── launch_sagemaker_job.py
│   └── requirements.txt
├── kaggle_training_notebook.ipynb   # Kaggle notebook version
├── _make_rl_notebook.py       # Generates the Kaggle RL notebook
├── _make_sft_notebook.py      # Generates the SFT notebook
│
├── ollama_eval.py             # Run any Ollama model on the 200-question test set
├── compare_results.py         # Compare base vs trained results
├── plot_comparison.py         # Generate comparison charts
├── merge_lora.py              # Merge LoRA adapter into base model
├── eval_local.py              # Local GPU eval (transformers + 4-bit)
├── eval_trained.py            # Trained adapter eval (transformers + 4-bit)
├── baseline_eval.py           # Baseline eval via Ollama
├── hackathon_eval.py          # Hackathon eval via Ollama
├── diagnostic_eval.py         # Prompt-format diagnostics
├── prompt_test.py             # Prompt testing utility
│
├── data/
│   ├── hackathon_test_200.json          # 200-question heldout test set
│   ├── comparison_results_base.json     # Base model outputs
│   ├── comparison_results_trained.json  # Trained model outputs
│   ├── comparison_analysis.json         # Full analysis (learned/forgotten/per-type)
│   ├── comparison_report.md             # Written comparison report
│   ├── comparison_plot.png              # Comparison visualization
│   ├── hackathon_eval_*.md              # Baseline mistake analysis
│   └── hackathon_eval_results_*.json    # Baseline eval results
│
├── Modelfile                  # Ollama model definition (trained model)
├── .env                       # HF token (gitignored)
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install torch transformers trl peft accelerate datasets bitsandbytes
```

For Unsloth (recommended):
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install "transformers<=5.17.0"
```

### 2. Configure HuggingFace token

Create a `.env` file (gitignored):
```
HF_TOKEN=hf_...
```

### 3. Download the trained adapter

```bash
# The adapter is on HuggingFace Hub
# Suryansh7123/qwen2.5-1.5b-grpo-metamath
```

---

## Training

### GCP (L4/T4 GPU)

```bash
# Phase 1 (one-time): install deps, cache model + dataset
# Phase 2 (every restart): download script, start training
bash gcp_training/startup.sh
```

Logs: `$HOME/training.log` and `$HOME/grpo_output/metrics.jsonl`

### AWS

```bash
bash aws_training/setup_and_run.sh
```

### SageMaker

```bash
python sagemaker_training/launch_sagemaker_job.py
```

### Kaggle

Open `kaggle_training_notebook.ipynb` and run.

### Key hyperparameters

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen2.5-1.5B-Instruct |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| rsLoRA | yes |
| Group size | 4 |
| Batch size | 4 |
| Grad accum | 2 |
| Max steps | 1000 |
| Max new tokens | 500 |
| Learning rate | 5e-7 |
| KL coefficient (beta) | 0.03 |
| Temperature | 0.8 |
| Precision | fp16 + 4-bit quantization |

### Reward function

| Component | Weight | Description |
|-----------|--------|-------------|
| Correctness | 0.80 | +1.0 if extracted answer matches ground truth |
| Length penalty | 0.20 | Linear penalty above 300 words |

---

## Evaluation

### Quick eval via Ollama (fast)

```bash
# Serve the trained model
ollama create trained-model -f Modelfile

# Run on 200 questions
python ollama_eval.py trained-model trained

# Run base model for comparison
python ollama_eval.py qwen2.5:1.5b base

# Compare
python compare_results.py
python plot_comparison.py
```

### Local GPU eval (slower, no Ollama)

```bash
python eval_local.py          # Both models, 4-bit quantization
python eval_trained.py        # Trained adapter only
```

### Merge LoRA into a standalone model

```bash
python merge_lora.py
# Output: merged_model/ (HF format, ~2.9 GB)
```

---

## Results files

| File | Description |
|------|-------------|
| `data/comparison_results_base.json` | Base model responses + correctness (200 questions) |
| `data/comparison_results_trained.json` | Trained model responses + correctness |
| `data/comparison_analysis.json` | Learned/forgotten questions, per-type stats |
| `data/comparison_report.md` | Full written report |
| `data/comparison_plot.png` | Bar charts: per-type, learned/forgotten, overall |
| `data/hackathon_eval_*.md` | Pre-training mistake analysis (36.5% baseline) |

---

## Key findings

1. **GRPO works on small models** — +23 points on a 1.5B model with only ~18M trainable params (LoRA).
2. **Augmented-answer formats improved most** — GSM_AnsAug (+52) and GSM_Rephrased (+48) saw the largest gains, suggesting the model learned to handle varied problem phrasings.
3. **Few regressions** — only 11 questions went right→wrong (out of 200), and only MATH_FOBAR regressed (-4 pts).
4. **Reward signal is sufficient** — correctness-only reward (no process reward) was enough to drive large gains.
5. **Training is reproducible** — scripts for GCP, AWS, SageMaker, and Kaggle all use the same hyperparameters and reward function.

---

## License

MIT
# reinforcement-learning-on-qwen2.5-1.5b
