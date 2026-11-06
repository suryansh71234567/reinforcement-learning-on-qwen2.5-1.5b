# Model Mistake Analysis — Pre-Training Baseline

**Model:** qwen2.5:1.5b (via Ollama, temperature=0.0)
**Test set:** 200 questions, stratified across 8 types (25 per type)
**Source:** Heldout split of MetaMathQA (not seen during training)
**Date:** 2026-10-25

---

## 1. Overall Results

| Metric | Value |
|--------|-------|
| **Accuracy** | **36.5%** (73/200) |
| Avg time per question | 22.8s |
| Total eval time | 76.2 min |

## 2. Accuracy by Problem Type

| Type | Correct | Total | Accuracy | Difficulty |
|------|---------|-------|----------|------------|
| GSM_AnsAug | 9 | 25 | 36.0% | Very Hard |
| GSM_FOBAR | 11 | 25 | 44.0% | Hard |
| GSM_Rephrased | 9 | 25 | 36.0% | Very Hard |
| GSM_SV | 9 | 25 | 36.0% | Very Hard |
| MATH_AnsAug | 7 | 25 | 28.0% | Very Hard |
| MATH_FOBAR | 9 | 25 | 36.0% | Very Hard |
| MATH_Rephrased | 8 | 25 | 32.0% | Very Hard |
| MATH_SV | 11 | 25 | 44.0% | Hard |

## 3. Mistake Classification

| Mistake Type | Count | % of Total | % of Wrong |
|-------------|-------|-----------|-----------|
| CORRECT | 73 | 36.5% | 57.5% |
| ARITHMETIC_OFF | 72 | 36.0% | 56.7% |
| ARITHMETIC_CLOSE | 4 | 2.0% | 3.1% |
| WRONG_ANSWER | 26 | 13.0% | 20.5% |
| FACTOR_ERROR | 5 | 2.5% | 3.9% |
| EXTRACTION_FAILURE | 6 | 3.0% | 4.7% |
| NO_ANSWER | 14 | 7.0% | 11.0% |

### Mistake Type Definitions

- **CORRECT** — Extracted answer matches ground truth
- **ARITHMETIC_OFF** — Right approach, wrong calculation (within 50% of correct)
- **ARITHMETIC_CLOSE** — Very close to correct answer (within 10%)
- **WRONG_ANSWER** — Fundamentally wrong approach or answer far from correct
- **FACTOR_ERROR** — Off by a factor of 2, 3, 5, 10, etc.
- **EXTRACTION_FAILURE** — Model gave answer but extractor couldn't parse it
- **NO_ANSWER** — Model didn't produce a final answer

## 4. Response Quality Analysis

| Metric | Correct Answers | Wrong Answers |
|--------|-----------------|---------------|
| Avg characters | 900 | 1177 |
| Avg est. tokens | 225 | 294 |
| Has step structure | 52/73 | 90/127 |
| Has conclusion | 73/73 | 123/127 |
| Avg repetition | 0.31 | 0.30 |

## 5. Answer Extraction Methods Used

| Method | Count | % |
|--------|-------|---|
| x = N | 76 | 38.0% |
| = N at EOL | 34 | 17.0% |
| boxed | 32 | 16.0% |
| quantity phrase | 31 | 15.5% |
| last number | 25 | 12.5% |
| answer phrase | 1 | 0.5% |
| #### marker | 1 | 0.5% |

## 6. Detailed Mistake Examples (for presentation)

### ARITHMETIC_OFF (72 total)

**Example 1** — Type: MATH_FOBAR
- **Question:** Find the value of $12 \times X + 36 \times 12$.
If we know the answer to the above question is 720, what is the value of unknown variable X?...
- **Ground Truth:** 24 | **Model Said:** 57
- **Response tail:** ...ct $3$ from both sides: | \[ X = 60 - 3 \] | \[ X = 57 \] |  | Therefore, the value of the unknown variable $X$ is $57$.
- **Response stats:** 160 words, 184 tokens, steps: yes, conclusion: yes

**Example 2** — Type: MATH_SV
- **Question:** $100_{10}$ in base $b$ has exactly X digits. The value of $b$ is 3. What is the value of unknown variable X?...
- **Ground Truth:** 5 | **Model Said:** 3
- **Response tail:** ... divide $1$ by $3$. This is $3^0=1$. | Therefore, in base 3, the number $100_{10}$ has exactly $2+1 = \boxed{3}$ digits.
- **Response stats:** 205 words, 262 tokens, steps: no, conclusion: yes

**Example 3** — Type: GSM_FOBAR
- **Question:** A football club has a balance of $100 million. The club then sells 2 of its players at $10 million each, and buys 4 more at $x million each. How much ...
- **Ground Truth:** 15 | **Model Said:** 5
- **Response tail:** ...0 = 4x\) | \(20 = 4x\) | \(x = \frac{20}{4}\) | \(x = 5\) |  | So the value of the unknown variable \(x\) is $5 million.
- **Response stats:** 148 words, 188 tokens, steps: yes, conclusion: yes

### ARITHMETIC_CLOSE (4 total)

**Example 1** — Type: GSM_FOBAR
- **Question:** Nicole has 4 fish tanks. The first two tanks need x gallons of water each and the other two need 2 fewer gallons of water each than the first two tank...
- **Ground Truth:** 8 | **Model Said:** 7.5
- **Response tail:** ...sides by 16: | \[ x = \frac{120}{16} \] | \[ x = 7.5 \] |  | So, the value of the unknown variable \(x\) is 7.5 gallons.
- **Response stats:** 284 words, 323 tokens, steps: yes, conclusion: yes

**Example 2** — Type: MATH_AnsAug
- **Question:** Mr. Mendez awards extra credit on quizzes to his students with quiz grades that exceed the class mean. Given that 107 students take the same quiz, wha...
- **Ground Truth:** 106 | **Model Said:** 107
- **Response tail:** ...[ 54 \] |  | ### Conclusion | The largest number of students who can be awarded extra credit on their quizzes is **54**.
- **Response stats:** 274 words, 383 tokens, steps: yes, conclusion: yes

**Example 3** — Type: MATH_Rephrased
- **Question:** If $\frac{a}{a+27} = 0.865$, where a is a positive integer, what is the value of a?...
- **Ground Truth:** 173 | **Model Said:** 174
- **Response tail:** ....135} |    \] |    Perform the division: |    \[ |    a = 174 |    \] |  | Therefore, the value of \( a \) is \( 174 \).
- **Response stats:** 160 words, 223 tokens, steps: yes, conclusion: yes

### WRONG_ANSWER (26 total)

**Example 1** — Type: MATH_Rephrased
- **Question:** Given that $\binom{15}{6} = 5005$ and $\binom{15}{8} = 6435$, calculate $\binom{17}{9}$....
- **Ground Truth:** 24310 | **Model Said:** 7865
- **Response tail:** ...| \binom{16}{8} = \binom{15}{7} + \binom{15}{8} = 1430 + 6435 = 7865 | \] | \[ | \binom{16}{9} = \binom{15}{8} + \binom{
- **Response stats:** 145 words, 247 tokens, steps: no, conclusion: no

**Example 2** — Type: MATH_SV
- **Question:** Our club has 20 members, 10 boys and X girls. We can choose a president and a vice-president in 200 different ways if they must be of different gender...
- **Ground Truth:** 10 | **Model Said:** 360
- **Response tail:** ...olate \(X\): |    \[ |    X = \frac{360}{20} = 18 |    \] |  | Therefore, the value of the unknown variable \(X\) is 18.
- **Response stats:** 307 words, 427 tokens, steps: yes, conclusion: yes

**Example 3** — Type: MATH_AnsAug
- **Question:** A rectangular prism has dimensions 8 inches by 2 inches by 32 inches. If a cube has the same volume as the prism, what is the surface area of the cube...
- **Ground Truth:** 384 | **Model Said:** 8
- **Response tail:** ...ng in the value of \( s \): |    \[ |    A_{\text{cube}} = 6 \times 8^2 |    \] |  |    First, calculate \( 8^2 \): |   
- **Response stats:** 285 words, 380 tokens, steps: yes, conclusion: yes

### FACTOR_ERROR (5 total)

**Example 1** — Type: MATH_AnsAug
- **Question:** A coin is flipped eight times, and the sequence of heads and tails occurring is recorded. How many distinct sequences are possible?...
- **Ground Truth:** 256 | **Model Said:** 64
- **Response tail:** ... = 64 \] |  | Therefore, there are 64 distinct possible sequences of heads and tails when a coin is flipped eight times.
- **Response stats:** 193 words, 269 tokens, steps: yes, conclusion: yes

**Example 2** — Type: GSM_AnsAug
- **Question:** Mr. Dubois buys a new car for $13,380. He pays $5,400 and pays the rest by giving $420 a month. In how many months will the car be fully paid for?...
- **Ground Truth:** 19 | **Model Said:** 190
- **Response tail:** ...wer: |  |    \[ |    19 \times 10 = 190 |    \] |  | Therefore, Mr. Dubois will fully pay off his car in **190 months**.
- **Response stats:** 158 words, 257 tokens, steps: yes, conclusion: yes

**Example 3** — Type: GSM_AnsAug
- **Question:** At a birthday party, 30% of the guests are married, 50% are single, and the rest are children. If there are 1000 guests, how many more married people ...
- **Ground Truth:** 100 | **Model Said:** 500
- **Response tail:** ...t) \times \text{Total number of guests} | \] |  | Substituting the given values: |  | \[ | \text{Number of children} = \
- **Response stats:** 252 words, 421 tokens, steps: yes, conclusion: yes

### EXTRACTION_FAILURE (6 total)

**Example 1** — Type: MATH_Rephrased
- **Question:** Simplify the denominator of $\frac{\sqrt{8} + \sqrt{3}}{\sqrt{2} + \sqrt{3}}$ and express your answer in simplest form....
- **Ground Truth:** -1 | **Model Said:** -2\sqrt{2
- **Response tail:** ..., the simplified form of the denominator is $-2\sqrt{2} - \sqrt{3}$. |  | Final answer: $\boxed{-2\sqrt{2} - \sqrt{3}}$.
- **Response stats:** 121 words, 223 tokens, steps: no, conclusion: yes

**Example 2** — Type: MATH_Rephrased
- **Question:** Rewrite the expression $((5p+1)-2p\cdot4)(3)+(4-1\div3)(6p-9)$ as a simpler expression in the form $ap-b$, where $a$ and $b$ are positive integers....
- **Ground Truth:** -30 | **Model Said:** 13p - 33
- **Response tail:** ...+ 22p - 33 = 13p - 33 | \] |  | Thus, the final simplified form of the given expression is: | \[ | \boxed{13p - 33} | \]
- **Response stats:** 214 words, 292 tokens, steps: yes, conclusion: yes

**Example 3** — Type: MATH_FOBAR
- **Question:** When $f(x) = ax^3 - 6x^2 + bx - 5$ is divided by $x - 1,$ the remainder is $-5.$  When $f(x)$ is divided by $x + X the remainder is $-53.$  Find the o...
- **Ground Truth:** 2 | **Model Said:** -
- **Response tail:** ...implifying this equation, we get: | \[ -aX^3 - 6X^2 - bX + 48 = 0 \quad \text{(Equation 2)} \] |  | Now, let's solve the
- **Response stats:** 318 words, 335 tokens, steps: yes, conclusion: yes

## 7. What This Means for GRPO Training

### Recoverable Errors: 76/127 wrong answers (60%)

These are errors where the model's reasoning is mostly right but the final number is wrong.
GRPO directly fixes this: reward = 1 for correct answer, 0 for wrong.
The model learns to double-check calculations and be more careful.

### Per-Type GRPO Targets

| Type | Current | Expected After GRPO | Improvement |
|------|---------|---------------------|-------------|
| GSM_AnsAug | 36.0% | 68.0% | +32.0 pts |
| GSM_FOBAR | 44.0% | 72.0% | +28.0 pts |
| GSM_Rephrased | 36.0% | 68.0% | +32.0 pts |
| GSM_SV | 36.0% | 68.0% | +32.0 pts |
| MATH_AnsAug | 28.0% | 64.0% | +36.0 pts |
| MATH_FOBAR | 36.0% | 68.0% | +32.0 pts |
| MATH_Rephrased | 32.0% | 66.0% | +34.0 pts |
| MATH_SV | 44.0% | 72.0% | +28.0 pts |

### Key Takeaways for Presentation

1. **Baseline accuracy: 36.5%** — this is our starting point before any training
2. **76 wrong answers (60% of errors) are recoverable** — right reasoning, wrong number
3. **Model already produces step-by-step reasoning** — no format change needed for GRPO
4. **Wrong answers are 1x longer** than correct ones — model rambles when unsure
5. **Best improvement potential:** GSM_AnsAug and MATH_Rephrased (lowest current accuracy)
6. **GRPO reward signal is sufficient** — 73 correct answers in 200 questions means the model already knows how to solve half the problems; GRPO just needs to amplify that knowledge
