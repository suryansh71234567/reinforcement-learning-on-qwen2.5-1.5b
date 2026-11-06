#!/usr/bin/env python3
"""Run both base and GRPO-trained models on the same 200 questions and compare."""

import json
import re
import time
import os
import sys

# Set HF token from environment
# Load from .env file
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
    print("ERROR: HF_TOKEN not found in .env file or environment")
    sys.exit(1)
os.environ["HF_TOKEN"] = HF_TOKEN

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

TEST_FILE = r"C:\coding vs\syntax_error_rl\data\hackathon_test_200.json"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TRAINED_ADAPTER = "Suryansh7123/qwen2.5-1.5b-grpo-metamath"
OUTPUT_FILE = r"C:\coding vs\syntax_error_rl\data\comparison_results.json"

SYSTEM_PROMPT = "You are a math solver. Solve the problem step by step, showing your reasoning clearly. End your response with: The answer is <number>."


def extract_answer(text):
    """Extract numerical answer from model response."""
    # Try "The answer is <number>" pattern first
    m = re.search(r'[Tt]he answer is[:\s]+([-\d.,]+)', text)
    if m:
        try:
            return float(m.group(1).replace(',', ''))
        except ValueError:
            pass
    # Try \boxed{<number>}
    m = re.search(r'\\boxed\{([-\d.,]+)\}', text)
    if m:
        try:
            return float(m.group(1).replace(',', ''))
        except ValueError:
            pass
    # Fall back to last number in response
    nums = re.findall(r'[-]?\d[\d,]*\.?\d*', text)
    if nums:
        try:
            return float(nums[-1].replace(',', ''))
        except ValueError:
            pass
    return None


def load_questions():
    questions = []
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def run_model(model, tokenizer, questions, model_name):
    """Run inference on all questions with a given model."""
    results = []
    correct_count = 0
    start = time.time()

    for i, q in enumerate(questions):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q["prompt"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted = extract_answer(response)
        ground_truth = float(q["answer"])

        is_correct = predicted is not None and abs(predicted - ground_truth) < 0.01
        if is_correct:
            correct_count += 1

        results.append({
            "index": i,
            "type": q.get("type", "unknown"),
            "prompt": q["prompt"][:200],
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": is_correct,
            "response": response[:500],
        })

        if (i + 1) % 25 == 0:
            acc = correct_count / (i + 1)
            elapsed = time.time() - start
            print(f"  [{model_name}] {i+1}/{len(questions)}: acc={acc:.3f} ({correct_count}/{i+1}) time={elapsed:.0f}s")

    total_time = time.time() - start
    accuracy = correct_count / len(questions)
    print(f"  [{model_name}] DONE: {correct_count}/{len(questions)} = {accuracy:.3f} in {total_time:.0f}s")
    return results, accuracy


def main():
    questions = load_questions()
    print(f"Loaded {len(questions)} questions")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    print("\n=== Loading BASE model ===")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=HF_TOKEN,
    )
    base_model.eval()

    base_results, base_acc = run_model(base_model, tokenizer, questions, "BASE")
    del base_model
    torch.cuda.empty_cache()

    # Load trained model (base + LoRA adapter)
    print("\n=== Loading TRAINED model ===")
    trained_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=HF_TOKEN,
    )
    trained_model = PeftModel.from_pretrained(trained_model, TRAINED_ADAPTER, token=HF_TOKEN)
    trained_model.eval()

    trained_results, trained_acc = run_model(trained_model, tokenizer, questions, "TRAINED")

    # Compare
    print("\n=== COMPARISON ===")
    print(f"Base accuracy:    {base_acc:.3f}")
    print(f"Trained accuracy: {trained_acc:.3f}")
    print(f"Improvement:      {trained_acc - base_acc:+.3f}")

    # Per-type comparison
    from collections import defaultdict
    base_type = defaultdict(lambda: {"correct": 0, "total": 0})
    trained_type = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in base_results:
        base_type[r["type"]]["total"] += 1
        if r["correct"]:
            base_type[r["type"]]["correct"] += 1

    for r in trained_results:
        trained_type[r["type"]]["total"] += 1
        if r["correct"]:
            trained_type[r["type"]]["correct"] += 1

    print("\nPer-type comparison:")
    for t in sorted(base_type.keys()):
        b = base_type[t]
        tr = trained_type[t]
        b_acc = b["correct"] / b["total"] if b["total"] else 0
        tr_acc = tr["correct"] / tr["total"] if tr["total"] else 0
        print(f"  {t:>15}: base={b['correct']:>2}/{b['total']:<2} ({b_acc:.3f}) -> trained={tr['correct']:>2}/{tr['total']:<2} ({tr_acc:.3f}) | {tr_acc - b_acc:+.3f}")

    # Find questions that changed
    learned = []
    forgotten = []
    for b, tr in zip(base_results, trained_results):
        if not b["correct"] and tr["correct"]:
            learned.append({"index": b["index"], "type": b["type"], "prompt": b["prompt"], "ground_truth": b["ground_truth"]})
        elif b["correct"] and not tr["correct"]:
            forgotten.append({"index": b["index"], "type": b["type"], "prompt": b["prompt"], "ground_truth": b["ground_truth"], "base_pred": b["predicted"], "trained_pred": tr["predicted"]})

    print(f"\nLearned (wrong->right): {len(learned)}")
    print(f"Forgotten (right->wrong): {len(forgotten)}")

    if learned:
        print("\n=== LEARNED (sample) ===")
        for l in learned[:10]:
            print(f"  [{l['type']}] Q: {l['prompt'][:80]}")
            print(f"    Answer: {l['ground_truth']}")

    if forgotten:
        print("\n=== FORGOTTEN (sample) ===")
        for f in forgotten[:10]:
            print(f"  [{f['type']}] Q: {f['prompt'][:80]}")
            print(f"    Answer: {f['ground_truth']} | Base pred: {f['base_pred']} | Trained pred: {f['trained_pred']}")

    # Save results
    output = {
        "base_accuracy": base_acc,
        "trained_accuracy": trained_acc,
        "improvement": trained_acc - base_acc,
        "base_results": base_results,
        "trained_results": trained_results,
        "learned": learned,
        "forgotten": forgotten,
        "per_type": {
            t: {
                "base": base_type[t],
                "trained": trained_type[t],
            }
            for t in base_type
        },
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
