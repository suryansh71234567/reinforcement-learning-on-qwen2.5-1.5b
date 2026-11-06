#!/usr/bin/env python3
"""
Test different prompt instructions to find what works best for GSM8K.
Tests 7 prompt styles on the 5-question smoke test with both 1B and 3B.
"""

import json
import re
import sys
import time
import urllib.request

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Fix extraction (no comma in capture groups)
def extract_answer(text):
    """Extract final answer. Returns (answer, method)."""
    # GSM8K style: #### <number>
    m = re.search(r'####\s*([-\d\.]+)', text)
    if m: return m.group(1).strip(), "####"

    # \boxed{...}
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m: return m.group(1).strip(), "boxed"

    # x = NUMBER
    ms = re.findall(r'[xX]\s*=\s*([-\d\.]+)', text)
    if ms: return ms[-1].strip(), "x=N"

    # = NUMBER at EOL
    ms = re.findall(r'=\s*([-\d\.]+)\s*$', text, re.MULTILINE)
    if ms: return ms[-1].strip(), "=N EOL"

    # answer phrase (require space after keyword)
    ms = re.findall(
        r'(?:the answer is|therefore|thus|final answer is)\s*:?\s+([-\d\.]+)',
        text, re.IGNORECASE
    )
    if ms: return ms[-1].strip(), "phrase"

    # quantity phrase
    ms = re.findall(
        r'(?:there (?:are|were)|(?:has|have|are|had|was|were|is)\s+)(?:exactly\s+)?([-\d\.]+)',
        text, re.IGNORECASE
    )
    if ms: return ms[-1].strip(), "quantity"

    # last number
    ns = re.findall(r'[-\d\.]+', text)
    if ns: return ns[-1].strip(), "last_num"

    return text.strip().split('\n')[-1][:50], "raw"


def normalize(ans):
    return str(ans).strip().replace(",", "").rstrip(".").replace(" ", "").lower()


def check_correct(model_ans, gt):
    mn = normalize(model_ans)
    gn = normalize(gt)
    if not mn: return False
    if mn == gn: return True
    try:
        return abs(float(mn) - float(gn)) < 1e-6
    except: pass
    return gn in mn or mn in gn


def ollama_gen(model, prompt, system=None, max_tokens=1024):
    payload = {
        "model": model, "prompt": prompt, "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.0, "num_predict": max_tokens}
    }
    if system: payload["system"] = system
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request("http://localhost:11434/api/generate",
                                data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result.get("response", "")


# 7 different prompt styles to test
PROMPTS = {
    "baseline": "You are a math solver. Solve the problem step by step. Show your reasoning clearly and end with the final answer.",

    "minimal": "Solve this math problem.",

    "direct_steps": "Solve this math problem step by step. Show all your work and calculations clearly.",

    "think_aloud": "Think through this math problem carefully, one step at a time. Explain your reasoning as you go.",

    "teacher": "You are a math teacher explaining to a student. Solve this problem clearly, showing each step of your work.",

    "concise": "Solve this math problem. Be precise with your calculations. Give the final numerical answer at the end.",

    "structured": "Solve the following math problem. First, identify what is given and what is asked. Then work through the solution step by step. Finally, state the answer clearly.",
}

# Load smoke test
with open("data/smoke_test.json") as f:
    test_data = [json.loads(line) for line in f]

models = ["llama3.2:1b", "llama3.2:3b"]

for model in models:
    print(f"\n{'='*80}")
    print(f"MODEL: {model}")
    print(f"{'='*80}")

    for prompt_name, system_prompt in PROMPTS.items():
        correct = 0
        results = []

        for item in test_data:
            response = ollama_gen(model, item["prompt"], system=system_prompt)
            ans, method = extract_answer(response)
            is_correct = check_correct(ans, item["answer"])
            if is_correct:
                correct += 1
            results.append({
                "type": item["type"],
                "gt": item["answer"],
                "got": ans,
                "method": method,
                "correct": is_correct,
                "response_len": len(response),
                "response_tail": response[-120:].replace('\n', ' | ') if len(response) > 120 else response.replace('\n', ' | ')
            })

        acc = correct / len(test_data) * 100
        avg_len = sum(r["response_len"] for r in results) / len(results)

        status = "✅" if acc >= 80 else ("⚠️" if acc >= 60 else "❌")
        print(f"\n  {status} [{prompt_name:15s}] {correct}/{len(test_data)} = {acc:.0f}%  avg_len={avg_len:.0f} chars")
        print(f"      System: \"{system_prompt[:70]}...\"" if len(system_prompt) > 70 else f"      System: \"{system_prompt}\"")

        for r in results:
            mark = "✓" if r["correct"] else "✗"
            print(f"      {mark} {r['type']:15s} GT={r['gt']:>8s} Got={r['got']:>10s} [{r['method']:10s}] ...{r['response_tail'][-60:]}")

print(f"\n{'='*80}")
print("SUMMARY: Which prompt gives best accuracy + shortest responses + reliable extraction?")
print("="*80)
