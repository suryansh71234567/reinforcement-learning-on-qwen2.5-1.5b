#!/usr/bin/env python3
"""Run the GRPO-trained model (LoRA adapter) on 200 questions."""

import json
import re
import time
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Load HF token from .env
from pathlib import Path
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

TEST_FILE = r"C:\coding vs\syntax_error_rl\data\hackathon_test_200.json"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TRAINED_ADAPTER = r"C:\coding vs\syntax_error_rl\trained_adapter"
OUTPUT_FILE = r"C:\coding vs\syntax_error_rl\data\comparison_results_trained.json"

SYSTEM_PROMPT = (
    "You are a math solver. Solve the problem step by step. "
    "Show your reasoning clearly and end with the final answer."
)


def extract_answer(text):
    """Extract final answer using multi-pattern approach. Returns (answer, method)."""
    # 1. GSM8K style: #### <number>
    m = re.search(r'####\s*([-\d\.]+)', text)
    if m:
        return m.group(1).strip(), "#### marker"

    # 2. MATH style: \boxed{...}
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        return m.group(1).strip(), "boxed"

    # 3. x = NUMBER (FOBAR/SV problems)
    ms = re.findall(r'[xX]\s*=\s*([-\d\.]+)', text)
    if ms:
        return ms[-1].strip(), "x = N"

    # 4. = NUMBER at end of line
    ms = re.findall(r'=\s*([-\d\.]+)\s*$', text, re.MULTILINE)
    if ms:
        return ms[-1].strip(), "= N at EOL"

    # 5. Answer phrase
    ms = re.findall(
        r'(?:the answer is|therefore|thus|final answer is|so (?:the|his|her|their|it is))\s*:?\s+([-\d\.]+)',
        text, re.IGNORECASE
    )
    if ms:
        return ms[-1].strip(), "answer phrase"

    # 6. Quantity phrase
    ms = re.findall(
        r'(?:there (?:are|were)|(?:has|have|are|had|was|were|is)\s+)(?:exactly\s+)?([-\d\.]+)',
        text, re.IGNORECASE
    )
    if ms:
        return ms[-1].strip(), "quantity phrase"

    # 7. Last number fallback
    ns = re.findall(r'[-\d\.]+', text)
    if ns:
        return ns[-1].strip(), "last number"

    return text.strip().split('\n')[-1][:50], "raw last line"


def normalize(ans):
    if ans is None:
        return ""
    return str(ans).strip().replace(",", "").rstrip(".").replace(" ", "").lower()


def check_correct(model_ans, gt):
    mn = normalize(model_ans)
    gn = normalize(gt)
    if not mn:
        return False
    if mn == gn:
        return True
    try:
        return abs(float(mn) - float(gn)) < 1e-6
    except (ValueError, OverflowError):
        pass
    return gn in mn or mn in gn


def load_questions():
    questions = []
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def main():
    questions = load_questions()
    print(f"Loaded {len(questions)} questions")
    print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    print("Loading base model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=HF_TOKEN,
    )
    model.eval()
    print(f"Base model loaded. VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, TRAINED_ADAPTER, token=HF_TOKEN)
    model.eval()
    print(f"Adapter loaded. VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Run inference
    results = []
    correct_count = 0
    start = time.time()

    for i, q in enumerate(questions):
        # Use plain prompt with system message (matching existing scripts)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q["prompt"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=192,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted, method = extract_answer(response)
        ground_truth = q["answer"]
        is_correct = check_correct(predicted, ground_truth)
        if is_correct:
            correct_count += 1

        results.append({
            "index": i,
            "type": q.get("type", "unknown"),
            "prompt": q["prompt"][:200],
            "ground_truth": ground_truth,
            "predicted": predicted,
            "extraction_method": method,
            "correct": is_correct,
            "response": response[:500],
        })

        if (i + 1) % 10 == 0:
            acc = correct_count / (i + 1)
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(questions)}] acc={acc:.3f} ({correct_count}/{i+1}) time={elapsed:.0f}s", flush=True)

    total_time = time.time() - start
    accuracy = correct_count / len(questions)
    print(f"\nDONE: {correct_count}/{len(questions)} = {accuracy:.3f} in {total_time:.0f}s")

    # Save results
    output = {
        "model": TRAINED_ADAPTER,
        "accuracy": accuracy,
        "results": results,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
