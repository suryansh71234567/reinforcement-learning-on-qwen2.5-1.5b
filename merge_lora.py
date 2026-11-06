#!/usr/bin/env python3
"""Merge LoRA adapter with base model and save for GGUF conversion."""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

# Load HF token from .env
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not found")
    sys.exit(1)

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TRAINED_ADAPTER = r"C:\coding vs\syntax_error_rl\trained_adapter"
OUTPUT_DIR = r"C:\coding vs\syntax_error_rl\merged_model"

def main():
    print(f"Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        token=HF_TOKEN,
    )

    print(f"Loading LoRA adapter: {TRAINED_ADAPTER}")
    model = PeftModel.from_pretrained(model, TRAINED_ADAPTER)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done! Merged model saved.")
    print(f"Files: {os.listdir(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
