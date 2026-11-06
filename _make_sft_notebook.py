import json

nb = {
  "cells": [],
  "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.12"}
  },
  "nbformat": 4,
  "nbformat_minor": 4
}

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    lines = text.splitlines(keepends=True)
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}

nb["cells"].append(md(
"""# SFT Compatibility Test

**Goal:** Verify AlignTune + Unsloth can load Llama 3.2 3B, load MetaMathQA data, and run SFT.

**No real training** — just 3 steps to confirm the pipeline.

Sections:
1. Install dependencies
2. Environment check
3. Import check
4. Locate model and data
5. Load model (4-bit + LoRA)
6. Load data (tiny sample)
7. Prepare SFT format data
8. Create SFT trainer
9. Run 3 training steps
10. Inference test
11. SGLang check"""
))

nb["cells"].append(code(
"""import subprocess, sys, time

def pip_install(pkg):
    print(f"\\nInstalling: {pkg}")
    t0 = time.time()
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"FAILED ({elapsed:.1f}s): {result.stderr[-500:]}")
        raise RuntimeError(f"pip install {pkg} failed")
    print(f"OK ({elapsed:.1f}s)")

pip_install("aligntune")
pip_install("datasets")
pip_install("huggingface_hub")
print("\\nAll installs complete.")
"""
))

nb["cells"].append(code(
"""import torch, os, sys, json

print("=" * 60)
print("ENVIRONMENT")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_mem/1e9:.1f} GB)")

print(f"\\n/kaggle/input:")
kaggle_input = "/kaggle/input"
if os.path.exists(kaggle_input):
    for item in sorted(os.listdir(kaggle_input)):
        print(f"  {item}/")
else:
    print("  (not found)")
print("=" * 60)
"""
))

nb["cells"].append(code(
"""print("=" * 60)
print("IMPORT CHECK")
print("=" * 60)

errors = []
for mod_name in ["torch", "transformers", "peft", "trl", "datasets", "bitsandbytes"]:
    try:
        m = __import__(mod_name)
        print(f"  {mod_name}: {getattr(m, '__version__', 'ok')}")
    except Exception as e:
        errors.append(f"{mod_name}: {e}")
        print(f"  {mod_name}: FAILED - {e}")

try:
    from unsloth import FastLanguageModel
    print("  unsloth: ok")
except Exception as e:
    errors.append(f"unsloth: {e}")
    print(f"  unsloth: FAILED - {e}")

try:
    from aligntune.core.backend_factory import create_sft_trainer
    print("  aligntune: ok")
except Exception as e:
    errors.append(f"aligntune: {e}")
    print(f"  aligntune: FAILED - {e}")

try:
    import sglang as sgl
    print("  sglang: ok")
except Exception as e:
    errors.append(f"sglang: {e}")
    print(f"  sglang: FAILED - {e}")

try:
    import auditkit as ak
    print("  auditkit: ok")
except Exception as e:
    errors.append(f"auditkit: {e}")
    print(f"  auditkit: FAILED - {e}")

print()
print("=" * 60)
if errors:
    print(f"RESULT: {len(errors)} FAILED")
    for e in errors:
        print(f"  - {e}")
else:
    print("RESULT: ALL IMPORTS OK")
print("=" * 60)
"""
))

nb["cells"].append(code(
"""print("=" * 60)
print("LOCATING MODEL AND DATA")
print("=" * 60)

model_path = None
data_path = None
kaggle_input = "/kaggle/input"

if os.path.exists(kaggle_input):
    for d in os.listdir(kaggle_input):
        if "llama" in d.lower() and "3.2" in d.lower():
            model_dir = os.path.join(kaggle_input, d)
            for sub in os.listdir(model_dir):
                if "3b" in sub.lower():
                    candidate = os.path.join(model_dir, sub)
                    if os.path.isdir(candidate):
                        files = os.listdir(candidate)
                        if any(f.endswith(('.safetensors', '.bin')) for f in files):
                            model_path = candidate
                            break
            if model_path:
                break
    for d in os.listdir(kaggle_input):
        if "metamathqa" in d.lower() or "grpo" in d.lower():
            data_dir = os.path.join(kaggle_input, d)
            if os.path.isdir(data_dir):
                for f in os.listdir(data_dir):
                    if f == "grpo_train.json":
                        data_path = os.path.join(data_dir, f)

print(f"Model: {model_path or 'NOT FOUND'}")
print(f"Data: {data_path or 'NOT FOUND'}")
if not model_path:
    model_path = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"Fallback: {model_path}")
if not data_path:
    data_path = "meta-math/MetaMathQA"
    print(f"Fallback: {data_path}")
print("=" * 60)
"""
))

nb["cells"].append(code(
"""print("=" * 60)
print("LOADING MODEL (4-bit)")
print("=" * 60)

import torch
from unsloth import FastLanguageModel

t0 = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=torch.bfloat16,
)
print(f"Model loaded: {time.time()-t0:.1f}s")
total = sum(p.numel() for p in model.parameters())
print(f"Total params: {total/1e9:.2f}B")
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print("Model: OK")
print("=" * 60)
"""
))

nb["cells"].append(code(
"""print("=" * 60)
print("LOADING DATA")
print("=" * 60)

from datasets import load_dataset

if data_path.endswith(".json") or data_path.endswith(".jsonl"):
    with open(data_path) as f:
        raw_data = [json.loads(line) for line in f]
    print(f"Loaded {len(raw_data)} rows from file")
else:
    ds = load_dataset(data_path, split="train[:1000]")
    raw_data = ds
    print(f"Loaded {len(raw_data)} rows from HF")

sample = raw_data[0] if isinstance(raw_data, list) else raw_data[0]
print(f"Keys: {list(sample.keys())}")
print(f"  prompt: {str(sample.get('prompt',''))[:80]}")
print(f"  answer: {sample.get('answer','')}")

SAMPLE_SIZE = 50
tiny_data = raw_data[:SAMPLE_SIZE] if isinstance(raw_data, list) else raw_data.select(range(min(SAMPLE_SIZE, len(raw_data))))
print(f"Using {SAMPLE_SIZE} rows")
print("Data: OK")
print("=" * 60)
"""
))

nb["cells"].append(code(
"""print("=" * 60)
print("PREPARING SFT DATA")
print("=" * 60)

# Convert GRPO format to Alpaca-like for SFT
sft_data = []
for row in tiny_data:
    sft_data.append({
        "instruction": row["prompt"],
        "input": "",
        "output": row["full_response"],
    })

sft_data_path = "/kaggle/working/sft_tiny_data.jsonl"
with open(sft_data_path, 'w') as f:
    for row in sft_data:
        f.write(json.dumps(row) + "\\n")
print(f"Saved {len(sft_data)} rows to {sft_data_path}")
print("SFT data: OK")
print("=" * 60)
"""
))

nb["cells"].append(code(
"""print("=" * 60)
print("APPLYING LoRA")
print("=" * 60)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"LoRA: {trainable/1e6:.2f}M / {total/1e9:.2f}B ({100*trainable/total:.2f}%)")
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print("LoRA: OK")
print("=" * 60)
"""
))

nb["cells"].append(code(
"""print("=" * 60)
print("CREATING SFT TRAINER")
print("=" * 60)

try:
    from aligntune.core.backend_factory import create_sft_trainer
    trainer = create_sft_trainer(
        model_name=model_path,
        dataset_name=sft_data_path,
        backend="unsloth",
        num_epochs=1,
        max_steps=3,
        batch_size=1,
        learning_rate=2e-5,
        max_seq_length=2048,
    )
    print(f"AlignTune SFT: OK ({type(trainer).__name__})")
except Exception as e:
    print(f"AlignTune SFT: FAILED - {e}")
    print("FALLBACK: raw TRL SFT")
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset as HFDataset
    def fmt(row):
        return f"### Instruction:\\n{row['instruction']}\\n\\n### Response:\\n{row['output']}"
    sft_rows = [{"text": fmt(r)} for r in sft_data]
    train_ds = HFDataset.from_list(sft_rows)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=SFTConfig(
            max_seq_length=2048,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            max_steps=3,
            logging_steps=1,
            output_dir="/kaggle/working/sft_fallback",
        ),
    )
    print(f"TRL SFT: OK ({type(trainer).__name__})")
print("=" * 60)
"""
))

nb["cells"].append(code(
"""print("=" * 60)
print("TRAINING (3 steps)")
print("=" * 60)

t0 = time.time()
trainer.train()
print(f"Training: OK ({time.time()-t0:.1f}s)")
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print("=" * 60)
"""
))

nb["cells"].append(code(
"""print("=" * 60)
print("INFERENCE TEST")
print("=" * 60)

test_prompt = "What is 15 + 27?"
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.0, do_sample=False)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Q: {test_prompt}")
print(f"A: {response[:300]}")
print("Inference: OK")
print("=" * 60)
"""
))

nb["cells"].append(code(
"""print("=" * 60)
print("SGLANG CHECK")
print("=" * 60)

import sglang as sgl
print(f"SGLang: {getattr(sgl, '__version__', 'unknown')}")
print(f"sgl.Engine: {hasattr(sgl, 'Engine')}")
print("SGLang: OK")
print("=" * 60)
"""
))

nb["cells"].append(code(
"""print("=" * 60)
print("ALL SFT TESTS COMPLETE")
print("=" * 60)
print("If no FAILED above, the pipeline is ready.")
print("=" * 60)
"""
))

path = "C:/coding vs/syntax_error_rl/sft_compatibility_test/sft_compatibility_test.ipynb"
with open(path, 'w') as f:
    json.dump(nb, f, indent=1)

with open(path) as f:
    v = json.load(f)
print(f"Written: {len(v['cells'])} cells, valid JSON: OK")
