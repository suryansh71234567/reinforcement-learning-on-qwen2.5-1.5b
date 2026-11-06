#!/usr/bin/env python3
"""Compare base vs trained model results and analyze what the model learned."""

import json
import os
from collections import defaultdict

BASE_FILE = r"C:\coding vs\syntax_error_rl\data\comparison_results_base.json"
TRAINED_FILE = r"C:\coding vs\syntax_error_rl\data\comparison_results_trained.json"
OUTPUT_FILE = r"C:\coding vs\syntax_error_rl\data\comparison_analysis.json"


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    base = load_results(BASE_FILE)
    trained = load_results(TRAINED_FILE)

    base_results = base["results"]
    trained_results = trained["results"]

    base_acc = base["accuracy"]
    trained_acc = trained["accuracy"]

    print("=" * 60)
    print("BASELINE vs TRAINED MODEL COMPARISON")
    print("=" * 60)
    print(f"Base model:    {base_acc:.3f} ({int(base_acc * 200)}/200)")
    print(f"Trained model: {trained_acc:.3f} ({int(trained_acc * 200)}/200)")
    print(f"Improvement:   {trained_acc - base_acc:+.3f}")

    # Per-type comparison
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

    print("\n" + "-" * 60)
    print("PER-TYPE COMPARISON")
    print("-" * 60)
    print(f"{'Type':>15} | {'Base':>12} | {'Trained':>12} | {'Delta':>8}")
    print("-" * 60)
    for t in sorted(base_type.keys()):
        b = base_type[t]
        tr = trained_type[t]
        b_acc = b["correct"] / b["total"] if b["total"] else 0
        tr_acc = tr["correct"] / tr["total"] if tr["total"] else 0
        print(f"{t:>15} | {b['correct']:>2}/{b['total']:<2} ({b_acc:.2f}) | {tr['correct']:>2}/{tr['total']:<2} ({tr_acc:.2f}) | {tr_acc - b_acc:+.2f}")

    # Find questions that changed
    learned = []
    forgotten = []
    for b, tr in zip(base_results, trained_results):
        if not b["correct"] and tr["correct"]:
            learned.append({
                "index": b["index"],
                "type": b["type"],
                "prompt": b["prompt"],
                "ground_truth": b["ground_truth"],
                "trained_pred": tr["predicted"],
                "trained_response": tr["response"],
            })
        elif b["correct"] and not tr["correct"]:
            forgotten.append({
                "index": b["index"],
                "type": b["type"],
                "prompt": b["prompt"],
                "ground_truth": b["ground_truth"],
                "base_pred": b["predicted"],
                "trained_pred": tr["predicted"],
                "trained_response": tr["response"],
            })

    print("\n" + "-" * 60)
    print("WHAT THE MODEL LEARNED")
    print("-" * 60)
    print(f"Learned (wrong -> right):   {len(learned)}")
    print(f"Forgotten (right -> wrong): {len(forgotten)}")
    print(f"Net change:                 {len(learned) - len(forgotten):+d}")

    if learned:
        print("\n--- LEARNED (sample) ---")
        for l in learned[:15]:
            print(f"\n[{l['index']}] [{l['type']}] Q: {l['prompt'][:100]}")
            print(f"    Answer: {l['ground_truth']} | Trained pred: {l['trained_pred']}")

    if forgotten:
        print("\n--- FORGOTTEN (sample) ---")
        for f in forgotten[:15]:
            print(f"\n[{f['index']}] [{f['type']}] Q: {f['prompt'][:100]}")
            print(f"    Answer: {f['ground_truth']} | Base pred: {f['base_pred']} | Trained pred: {f['trained_pred']}")

    # Save analysis
    output = {
        "base_accuracy": base_acc,
        "trained_accuracy": trained_acc,
        "improvement": trained_acc - base_acc,
        "learned_count": len(learned),
        "forgotten_count": len(forgotten),
        "learned": learned,
        "forgotten": forgotten,
        "per_type": {
            t: {
                "base": base_type[t],
                "trained": trained_type[t],
                "base_acc": base_type[t]["correct"] / base_type[t]["total"] if base_type[t]["total"] else 0,
                "trained_acc": trained_type[t]["correct"] / trained_type[t]["total"] if trained_type[t]["total"] else 0,
            }
            for t in base_type
        },
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nAnalysis saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
