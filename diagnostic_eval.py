#!/usr/bin/env python3
"""
Diagnostic evaluation script — deep analysis of model behavior before training.
Tests Llama 3.2 3B on MetaMathQA with full response visibility, error categorization,
chat template comparison, and response quality metrics.

Usage:
    python diagnostic_eval.py --model llama3.2:3b --test-file data/baseline_test.json
    python diagnostic_eval.py --model llama3.2:3b --test-file data/smoke_test.json
"""

import json
import re
import sys
import time
import argparse
import urllib.request
import urllib.error
from collections import defaultdict

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


SYSTEM_PROMPT = (
    "You are a math solver. Solve the problem step by step. "
    "Show your reasoning clearly and end with the final answer."
)

# Llama 3.2 Instruct chat template (simplified)
LLAMA_CHAT_TEMPLATE = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

"""


def ollama_generate(model, prompt, system=None, temperature=0.0, max_tokens=2048):
    """Call Ollama API to generate a response."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
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
    return result.get("response", ""), result.get("thinking", "")


def ollama_generate_chat(model, system_prompt, user_message, temperature=0.0, max_tokens=2048):
    """Call Ollama using chat template format."""
    url = "http://localhost:11434/api/generate"
    # Use the chat template format that Llama 3.2 expects
    full_prompt = LLAMA_CHAT_TEMPLATE.format(
        system_prompt=system_prompt,
        user_message=user_message
    )
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result.get("response", ""), result.get("thinking", "")


def extract_answer(text, thinking=""):
    """Extract the final answer from model output. Tries multiple formats.
    Returns (extracted_answer, method_used)."""
    full_text = text + "\n" + thinking if thinking else text

    # GSM8K style: #### <number>
    match = re.search(r'####\s*([-\d,\.]+)', full_text)
    if match:
        return match.group(1).strip(), "#### marker"

    # MATH style: \boxed{...}
    match = re.search(r'\\boxed\{([^}]+)\}', full_text)
    if match:
        return match.group(1).strip(), "boxed"

    # "x = NUMBER" or "X = NUMBER" at end of line (FOBAR problems)
    matches = re.findall(r'[xX]\s*=\s*([-\d\.]+)', full_text)
    if matches:
        return matches[-1].strip(), "x = N"

    # "= NUMBER" at end of line (last equation = answer)
    matches = re.findall(r'=\s*([-\d\.]+)\s*$', full_text, re.MULTILINE)
    if matches:
        return matches[-1].strip(), "= N at EOL"

    # "The answer is: X" / "Therefore, X" / "Thus, the answer is X"
    # Use word boundary + require at least one space/colon after phrase
    matches = re.findall(
        r'(?:the answer is|therefore|thus|final answer is|so (?:the|his|her|their|it is))\s*:?\s+([-\d\.]+)',
        full_text, re.IGNORECASE
    )
    if matches:
        return matches[-1].strip().rstrip('.'), "answer phrase"

    # "there are NUMBER ways" / "are NUMBER" / "has NUMBER" / "is NUMBER"
    matches = re.findall(
        r'(?:there (?:are|were)|(?:has|have|are|had|was|were|is)\s+)(?:exactly\s+)?([-\d\.]+)',
        full_text, re.IGNORECASE
    )
    if matches:
        return matches[-1].strip().rstrip('.'), "quantity phrase"

    # Fallback: last number in the text
    numbers = re.findall(r'[-\d,\.]+', full_text)
    if numbers:
        return numbers[-1].strip(), "last number"

    return full_text.strip().split('\n')[-1], "raw last line"


def normalize_answer(ans):
    """Normalize answer for comparison."""
    if ans is None:
        return ""
    ans = str(ans).strip()
    ans = ans.replace(",", "")
    ans = ans.rstrip(".")
    ans = ans.replace(" ", "")
    ans = ans.lower()
    return ans


def check_correct(model_answer, ground_truth):
    """Check if model answer matches ground truth."""
    model_norm = normalize_answer(model_answer)
    truth_norm = normalize_answer(ground_truth)
    if not model_norm:
        return False
    if model_norm == truth_norm:
        return True
    try:
        model_float = float(model_norm)
        truth_float = float(truth_norm)
        return abs(model_float - truth_float) < 1e-6
    except (ValueError, OverflowError):
        pass
    if truth_norm in model_norm or model_norm in truth_norm:
        return True
    return False


def categorize_error(response, extracted_answer, ground_truth, is_correct):
    """Categorize the type of error the model made."""
    if is_correct:
        return "CORRECT"

    norm_extracted = normalize_answer(extracted_answer)
    norm_gt = normalize_answer(ground_truth)

    # No answer extracted
    if not norm_extracted or norm_extracted == "raw last line":
        return "NO_ANSWER_EXTRACTED"

    # Check if numbers are close but not exact (arithmetic error)
    try:
        ext_float = float(norm_extracted)
        gt_float = float(norm_gt)
        if abs(ext_float - gt_float) / max(abs(gt_float), 1) < 0.1:
            return "ARITHMETIC_CLOSE"
        # Check if it's off by a factor (multiplication/division error)
        if ext_float != 0 and gt_float != 0:
            ratio = ext_float / gt_float
            if ratio in (2, 0.5, 3, 0.333, 4, 0.25, 10, 0.1):
                return "FACTOR_ERROR"
        # Check if off by addition/subtraction
        if abs(ext_float - gt_float) < max(abs(gt_float) * 0.5, 100):
            return "ARITHMETIC_OFF"
    except (ValueError, OverflowError):
        pass

    # Check if response is very short (under-reasoning)
    if len(response.split()) < 20:
        return "UNDER_REASONING"

    # Check if the response contains the right reasoning but wrong final answer
    # (look for keywords from the question in the response)
    response_lower = response.lower()
    gt_lower = norm_gt

    # Check for format issues
    if '####' not in response and '\\boxed' not in response:
        return "WRONG_ANSWER_NL_FORMAT"

    return "WRONG_ANSWER"


def estimate_token_count(text):
    """Rough token estimate (~4 chars per token for English)."""
    return len(text) // 4


def run_diagnostic(model, test_file, use_chat_template=False, max_samples=None):
    """Run diagnostic evaluation with full analysis."""
    with open(test_file, 'r') as f:
        test_data = [json.loads(line) for line in f]

    if max_samples:
        test_data = test_data[:max_samples]

    mode = "CHAT_TEMPLATE" if use_chat_template else "PLAIN_PROMPT"
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC EVALUATION: {model}")
    print(f"Mode: {mode}")
    print(f"System prompt: {SYSTEM_PROMPT[:80]}...")
    print(f"Test set: {test_file} ({len(test_data)} questions)")
    print(f"{'='*80}\n")

    results = []
    correct_count = 0
    total_time = 0
    error_categories = defaultdict(int)
    extraction_methods = defaultdict(int)
    response_lengths = []

    for i, item in enumerate(test_data):
        prompt = item['prompt']
        ground_truth = item['answer']
        q_type = item['type']

        start = time.time()
        try:
            if use_chat_template:
                response, thinking = ollama_generate_chat(model, SYSTEM_PROMPT, prompt)
            else:
                response, thinking = ollama_generate(model, prompt, system=SYSTEM_PROMPT)
        except Exception as e:
            response = f"ERROR: {e}"
            thinking = ""
        elapsed = time.time() - start
        total_time += elapsed

        extracted_answer, method = extract_answer(response, thinking)
        is_correct = check_correct(extracted_answer, ground_truth)
        error_type = categorize_error(response, extracted_answer, ground_truth, is_correct)

        if is_correct:
            correct_count += 1

        error_categories[error_type] += 1
        extraction_methods[method] += 1
        response_lengths.append(len(response))

        result = {
            'index': i,
            'type': q_type,
            'prompt': prompt,
            'ground_truth': ground_truth,
            'response': response,
            'extracted_answer': extracted_answer,
            'extraction_method': method,
            'correct': is_correct,
            'error_type': error_type,
            'response_length': len(response),
            'est_tokens': estimate_token_count(response),
            'time': elapsed,
        }
        results.append(result)

        # Print detailed per-question output
        status = "[OK]" if is_correct else f"[{error_type}]"
        print(f"[{i+1:3d}/{len(test_data)}] {status:25s} | {q_type:15s} | "
              f"GT: {str(ground_truth)[:15]:15s} | Got: {str(extracted_answer)[:15]:15s} | "
              f"method: {method:15s} | {len(response):5d} chars | {elapsed:.1f}s")

        if not is_correct:
            # Show the full response for wrong answers
            print(f"  RESPONSE: {response[:300]}{'...' if len(response) > 300 else ''}")
            print()

    # ============ SUMMARY ============
    accuracy = correct_count / len(test_data) * 100 if test_data else 0
    avg_time = total_time / len(test_data) if test_data else 0
    avg_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Accuracy: {correct_count}/{len(test_data)} = {accuracy:.1f}%")
    print(f"Avg time: {avg_time:.1f}s | Total: {total_time:.0f}s")
    print(f"Avg response length: {avg_length:.0f} chars (~{int(avg_length//4)} tokens)")

    # Per-type breakdown
    print(f"\n{'='*80}")
    print(f"PER-TYPE BREAKDOWN")
    print(f"{'='*80}")
    type_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'avg_len': 0, 'avg_time': 0})
    for r in results:
        type_stats[r['type']]['total'] += 1
        type_stats[r['type']]['avg_len'] += r['response_length']
        type_stats[r['type']]['avg_time'] += r['time']
        if r['correct']:
            type_stats[r['type']]['correct'] += 1

    for t in sorted(type_stats.keys()):
        s = type_stats[t]
        acc = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0
        avg_l = s['avg_len'] / s['total']
        avg_t = s['avg_time'] / s['total']
        print(f"  {t:15s}: {s['correct']:2d}/{s['total']:2d} = {acc:5.1f}% | "
              f"avg {avg_l:.0f} chars, {avg_t:.1f}s")

    # Error categories
    print(f"\n{'='*80}")
    print(f"ERROR CATEGORIES")
    print(f"{'='*80}")
    for cat, count in sorted(error_categories.items(), key=lambda x: -x[1]):
        print(f"  {cat:30s}: {count:3d} ({count/len(test_data)*100:.1f}%)")

    # Extraction methods
    print(f"\n{'='*80}")
    print(f"ANSWER EXTRACTION METHODS USED")
    print(f"{'='*80}")
    for method, count in sorted(extraction_methods.items(), key=lambda x: -x[1]):
        print(f"  {method:20s}: {count:3d} ({count/len(test_data)*100:.1f}%)")

    # Response length distribution
    print(f"\n{'='*80}")
    print(f"RESPONSE LENGTH DISTRIBUTION")
    print(f"{'='*80}")
    buckets = [(0, 100), (100, 300), (300, 600), (600, 1000), (1000, 2000), (2000, 99999)]
    for lo, hi in buckets:
        count = sum(1 for l in response_lengths if lo <= l < hi)
        label = f"{lo}-{hi} chars" if hi < 99999 else f"{lo}+ chars"
        print(f"  {label:15s}: {count:3d} ({count/len(test_data)*100:.1f}%)")

    # Detailed wrong answer analysis
    print(f"\n{'='*80}")
    print(f"WRONG ANSWER DETAILS (for reward function design)")
    print(f"{'='*80}")
    wrong_by_type = defaultdict(list)
    for r in results:
        if not r['correct']:
            wrong_by_type[r['type']].append(r)

    for t in sorted(wrong_by_type.keys()):
        items = wrong_by_type[t]
        print(f"\n  --- {t} ({len(items)} wrong) ---")
        for r in items[:3]:  # Show first 3 of each type
            print(f"    Q: {r['prompt'][:80]}...")
            print(f"    GT: {r['ground_truth']} | Got: {r['extracted_answer']} | Error: {r['error_type']}")
            print(f"    Response: {r['response'][:200]}...")
            print()

    # Save full results
    output_file = test_file.replace('.json', f'_diagnostic_{mode.lower()}.json')
    output_data = {
        'model': model,
        'mode': mode,
        'test_file': test_file,
        'system_prompt': SYSTEM_PROMPT,
        'total_questions': len(test_data),
        'correct': correct_count,
        'accuracy': accuracy,
        'avg_time': avg_time,
        'total_time': total_time,
        'avg_response_length': avg_length,
        'per_type': {t: {**s, 'accuracy': s['correct']/s['total']*100 if s['total'] > 0 else 0}
                     for t, s in type_stats.items()},
        'error_categories': dict(error_categories),
        'extraction_methods': dict(extraction_methods),
        'results': results,
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nFull diagnostic saved to: {output_file}")

    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diagnostic evaluation')
    parser.add_argument('--model', default='llama3.2:3b', help='Ollama model name')
    parser.add_argument('--test-file', default='data/baseline_test.json', help='Test JSON file')
    parser.add_argument('--chat-template', action='store_true', help='Use Llama chat template')
    parser.add_argument('--compare', action='store_true', help='Run both modes and compare')
    parser.add_argument('--max-samples', type=int, default=None, help='Max questions to test')
    args = parser.parse_args()

    if args.compare:
        print("\n" + "="*80)
        print("COMPARISON MODE: Plain prompt vs Chat template")
        print("="*80)
        acc_plain = run_diagnostic(args.model, args.test_file, use_chat_template=False,
                                   max_samples=args.max_samples)
        acc_chat = run_diagnostic(args.model, args.test_file, use_chat_template=True,
                                  max_samples=args.max_samples)
        print(f"\n{'='*80}")
        print(f"COMPARISON RESULT")
        print(f"{'='*80}")
        print(f"  Plain prompt:  {acc_plain:.1f}%")
        print(f"  Chat template: {acc_chat:.1f}%")
        print(f"  Difference:    {acc_chat - acc_plain:+.1f}%")
    else:
        run_diagnostic(args.model, args.test_file, use_chat_template=args.chat_template,
                       max_samples=args.max_samples)
