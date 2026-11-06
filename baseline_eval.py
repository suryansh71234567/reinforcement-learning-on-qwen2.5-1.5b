#!/usr/bin/env python3
"""
Baseline evaluation script — runs Qwen model on MetaMathQA heldout set via Ollama.
Compares model output against ground truth answers to establish pre-training baseline.

Usage:
    python baseline_eval.py --model qwen3:4b --test-file data/smoke_test.json
    python baseline_eval.py --model qwen3:4b --test-file data/baseline_test.json
    python baseline_eval.py --model qwen3:4b --test-file data/grpo_heldout.json --max-samples 100
"""

import json
import re
import sys
import time
import argparse
import urllib.request
import urllib.error

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def ollama_generate(model, prompt, system=None, temperature=0.0, max_tokens=2048):
    """Call Ollama API to generate a response. Returns (response, thinking) tuple."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",  # Keep model loaded for 30 min to avoid reloading
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }
    if system:
        payload["system"] = system

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    # Qwen3 outputs both 'response' and 'thinking' fields
    return result.get("response", ""), result.get("thinking", "")


def extract_answer(text, thinking=""):
    """Extract the final answer from model output. Tries multiple formats."""
    # Combine response + thinking for extraction (Qwen3 puts reasoning in thinking)
    full_text = text + "\n" + thinking if thinking else text

    # GSM8K style: #### <number>
    match = re.search(r'####\s*([-\d,\.]+)', full_text)
    if match:
        return match.group(1).strip()

    # MATH style: \boxed{...}
    match = re.search(r'\\boxed\{([^}]+)\}', full_text)
    if match:
        return match.group(1).strip()

    # "x = NUMBER" or "X = NUMBER" at end of line (FOBAR problems)
    matches = re.findall(r'[xX]\s*=\s*([-\d,\.]+)', full_text)
    if matches:
        return matches[-1].strip()

    # "= NUMBER" at end of line (last equation = answer)
    matches = re.findall(r'=\s*([-\d,\.]+)\s*$', full_text, re.MULTILINE)
    if matches:
        return matches[-1].strip()

    # "The answer is: X" / "Therefore, X" / "Thus, the answer is X"
    matches = re.findall(r'(?:the answer is|therefore|thus|final answer is|so (?:the|his|her|their|it is))\s*[:]*\s*([-\d,\.]+)', full_text, re.IGNORECASE)
    if matches:
        return matches[-1].strip().rstrip('.')

    # "there are NUMBER ways" / "are NUMBER" / "has NUMBER" / "is NUMBER"
    matches = re.findall(r'(?:there (?:are|were)|(?:has|have|are|had|was|were|is)\s+)(?:exactly\s+)?([-\d,\.]+)\s*(?:bars|pickles?|ways|left|remaining|total|each|people|miles|dollars|cents|steps|slices|cakes|oranges)?', full_text, re.IGNORECASE)
    if matches:
        return matches[-1].strip().rstrip('.')

    # Fallback: last number in the text
    numbers = re.findall(r'[-\d,\.]+', full_text)
    if numbers:
        return numbers[-1].strip()

    return full_text.strip().split('\n')[-1]  # last line as fallback


def normalize_answer(ans):
    """Normalize answer for comparison."""
    if ans is None:
        return ""
    ans = str(ans).strip()
    # Remove commas from numbers
    ans = ans.replace(",", "")
    # Remove trailing dots
    ans = ans.rstrip(".")
    # Remove whitespace
    ans = ans.replace(" ", "")
    # Lowercase
    ans = ans.lower()
    return ans


def check_correct(model_answer, ground_truth):
    """Check if model answer matches ground truth."""
    model_norm = normalize_answer(model_answer)
    truth_norm = normalize_answer(ground_truth)

    # Reject empty model answers
    if not model_norm:
        return False

    if model_norm == truth_norm:
        return True

    # Try numeric comparison
    try:
        model_float = float(model_norm)
        truth_float = float(truth_norm)
        return abs(model_float - truth_float) < 1e-6
    except (ValueError, OverflowError):
        pass

    # Check if ground truth is contained in model answer or vice versa
    if truth_norm in model_norm or model_norm in truth_norm:
        return True

    return False


def run_eval(model, test_file, system_prompt=None, max_samples=None):
    """Run full evaluation."""
    with open(test_file, 'r') as f:
        test_data = [json.loads(line) for line in f]

    if max_samples:
        test_data = test_data[:max_samples]

    print(f"\n{'='*60}")
    print(f"Baseline Evaluation: {model}")
    print(f"Test set: {test_file} ({len(test_data)} questions)")
    print(f"{'='*60}\n")

    results = []
    correct_count = 0
    total_time = 0

    system_msg = system_prompt or (
        "You are a math solver. Solve the problem step by step. "
        "Show your reasoning clearly and end with the final answer."
    )

    for i, item in enumerate(test_data):
        prompt = item['prompt']
        ground_truth = item['answer']
        q_type = item['type']

        start = time.time()
        try:
            response, thinking = ollama_generate(model, prompt, system=system_msg)
        except Exception as e:
            response = f"ERROR: {e}"
            thinking = ""
        elapsed = time.time() - start
        total_time += elapsed

        model_answer = extract_answer(response, thinking)
        is_correct = check_correct(model_answer, ground_truth)
        if is_correct:
            correct_count += 1

        result = {
            'index': i,
            'type': q_type,
            'prompt': prompt[:100] + '...' if len(prompt) > 100 else prompt,
            'ground_truth': ground_truth,
            'model_answer': model_answer,
            'correct': is_correct,
            'time': elapsed,
        }
        results.append(result)

        status = "[OK]" if is_correct else "[--]"
        print(f"[{i+1:3d}/{len(test_data)}] {status} | {q_type:15s} | "
              f"GT: {str(ground_truth)[:20]:20s} | Got: {str(model_answer)[:20]:20s} | "
              f"{elapsed:.1f}s")

    # Summary
    accuracy = correct_count / len(test_data) * 100 if test_data else 0
    avg_time = total_time / len(test_data) if test_data else 0

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Accuracy: {correct_count}/{len(test_data)} = {accuracy:.1f}%")
    print(f"Avg time per question: {avg_time:.1f}s")
    print(f"Total time: {total_time:.0f}s")
    print()

    # Per-type breakdown
    from collections import defaultdict
    type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in results:
        type_stats[r['type']]['total'] += 1
        if r['correct']:
            type_stats[r['type']]['correct'] += 1

    print("Per-type accuracy:")
    for t in sorted(type_stats.keys()):
        s = type_stats[t]
        acc = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0
        print(f"  {t:15s}: {s['correct']:3d}/{s['total']:3d} = {acc:.1f}%")

    # Save detailed results
    output_file = test_file.replace('.json', '_results.json')
    output_data = {
        'model': model,
        'test_file': test_file,
        'total_questions': len(test_data),
        'correct': correct_count,
        'accuracy': accuracy,
        'avg_time': avg_time,
        'total_time': total_time,
        'per_type': {t: s for t, s in type_stats.items()},
        'results': results,
    }
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline evaluation via Ollama')
    parser.add_argument('--model', default='qwen2.5:4b', help='Ollama model name')
    parser.add_argument('--test-file', default='data/baseline_test.json', help='Test JSON file')
    parser.add_argument('--system-prompt', default=None, help='Custom system prompt')
    parser.add_argument('--max-samples', type=int, default=None, help='Max questions to test')
    args = parser.parse_args()

    run_eval(args.model, args.test_file, args.system_prompt, args.max_samples)
