# Base vs GRPO-Trained Model Comparison

**Date:** 2026-11-05
**Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
**Trained model:** `Suryansh7123/qwen2.5-1.5b-grpo-metamath` (LoRA r=16, alpha=32, rsLoRA)
**Test set:** 200 heldout questions (25 per type × 8 MetaMathQA types)
**Inference:** Ollama (local RTX 2050, 4GB VRAM)

---

## Overall Results

| Model | Correct | Accuracy |
|-------|---------|----------|
| Base | 83/200 | **41.5%** |
| Trained | 129/200 | **64.5%** |
| **Improvement** | **+46** | **+23.0 pts** |

---

## Per-Type Accuracy

| Type | Base | Trained | Delta |
|------|------|---------|-------|
| GSM_AnsAug | 11/25 (44%) | 24/25 (96%) | **+52** |
| GSM_Rephrased | 8/25 (32%) | 20/25 (80%) | **+48** |
| MATH_AnsAug | 8/25 (32%) | 15/25 (60%) | **+28** |
| MATH_SV | 9/25 (36%) | 14/25 (56%) | **+20** |
| MATH_Rephrased | 8/25 (32%) | 12/25 (48%) | **+16** |
| GSM_FOBAR | 16/25 (64%) | 19/25 (76%) | **+12** |
| GSM_SV | 12/25 (48%) | 15/25 (60%) | **+12** |
| MATH_FOBAR | 11/25 (44%) | 10/25 (40%) | **-4** |

**7 of 8 types improved.** Only MATH_FOBAR regressed slightly (-4 pts).

---

## What the Model Learned

| Metric | Count |
|--------|-------|
| **Learned** (wrong → right) | **57** |
| **Forgotten** (right → wrong) | **11** |
| **Net change** | **+46** |

### Biggest gains
- **GSM_AnsAug**: 44% → 96% (+52 pts) — the model learned to handle augmented-answer GSM problems
- **GSM_Rephrased**: 32% → 80% (+48 pts) — rephrased word problems improved dramatically
- **MATH_AnsAug**: 32% → 60% (+28 pts) — augmented MATH problems improved substantially

### Regressions (forgotten)
- **MATH_FOBAR**: 44% → 40% (-4 pts) — slight regression on FOBAR-format MATH problems
- 11 questions that the base model got right are now wrong (out of 200)

### Example learned questions
1. `[MATH_SV]` "$100_{10}$ in base $b$ has exactly X digits..." → Answer: 5 (was wrong, now correct)
2. `[GSM_Rephrased]` "If Lisa initially had 36 candies..." → Answer: 4 (was wrong, now correct)
3. `[GSM_AnsAug]` "A library cabinet houses five ancient scrolls..." → Answer: 20655 (was wrong, now correct)
4. `[MATH_AnsAug]` "Round $54.\overline{54}$ to the nearest hundredth." → Answer: 54.55 (was wrong, now correct)
5. `[GSM_FOBAR]` "Vince owns a hair salon and he earns $x per head..." → Answer: 18 (was wrong, now correct)

---

## Training Loss / Reward Curve

**Not available.** Training logs are on the GCP VM (`$HOME/training.log` and `metrics.jsonl`), which is not accessible from this machine (no gcloud/SSH configured). The training script does upload metrics to GCS every 10 steps, but the bucket is not accessible either.

If you can provide the training log file, I can plot the reward/loss curve.

---

## Files Generated

| File | Description |
|------|-------------|
| `data/comparison_results_base.json` | Base model results (200 questions) |
| `data/comparison_results_trained.json` | Trained model results (200 questions) |
| `data/comparison_analysis.json` | Full analysis (learned, forgotten, per-type) |
| `data/comparison_plot.png` | Comparison visualization |
| `merged_model/` | Merged LoRA + base model (2.9 GB, HF format) |
| `trained_adapter/` | Original LoRA adapter |
| Ollama: `trained-model` | Trained model imported into Ollama |

---

## Conclusion

The GRPO training was **highly effective**: +23 percentage points overall (41.5% → 64.5%), with 57 questions learned and only 11 forgotten. The model showed dramatic improvement on augmented-answer and rephrased problem formats, which are the core MetaMathQA categories. The only regression was a minor -4 point drop on MATH_FOBAR problems.
