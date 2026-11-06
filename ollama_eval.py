#!/usr/bin/env python3
"""Run models via Ollama API on 200 questions and compare."""

import json
import re
import time
import os
import sys
import urllib.request

TEST_FILE = r"C:\coding vs\syntax_error_rl\data\hackathon_test_200.json"
OUTPUT_FILE = r"C:\coding vs\syntax_error_rl\data\comparison_results.json"
OLLAMA_URL = "http://localhost:11434/api/generate"

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

    # 5. Answer phrase (require space after keyword)
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
    """Normalize answer for comparison."""
    if ans is None:
        return ""
    return str(ans).strip().replace(",", "").rstrip(".").replace(" ", "").lower()


def check_correct(model_ans, gt):
    """Check if extracted answer matches ground truth."""
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


def ollama_generate(model, prompt, system=None, max_tokens=512):
    """Call Ollama API. Matches existing eval scripts (keep_alive, no chat template)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",  # Keep model loaded for 30 min to avoid reloading
        "options": {"temperature": 0.0, "num_predict": max_tokens}
    }
    if system:
        payload["system"] = system
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result.get("response", "")


def run_model(model_name, questions):
    results = []
    correct_count = 0
    start = time.time()

    for i, q in enumerate(questions):
        try:
            response = ollama_generate(model_name, q["prompt"], SYSTEM_PROMPT)
        except Exception as e:
            print(f"  ERROR on question {i}: {e}")
            response = ""

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
    print(f"  DONE: {correct_count}/{len(questions)} = {accuracy:.3f} in {total_time:.0f}s")
    return results, accuracy


def main():
    questions = load_questions()
    print(f"Loaded {len(questions)} questions")

    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:1.5b"
    output_key = sys.argv[2] if len(sys.argv) > 2 else "base"

    print(f"\n=== Running {model_name} ===")
    results, accuracy = run_model(model_name, questions)

    output_file = OUTPUT_FILE.replace(".json", f"_{output_key}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "accuracy": accuracy, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
