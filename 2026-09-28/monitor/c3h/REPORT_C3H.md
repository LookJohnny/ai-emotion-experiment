# C3-H hallucination pre-warning: sealed test report (2026-09-30)

- Pre-registration: `PREREG_C3H.md` (4e59211), deviation D1 (fe16d47).
- Freeze: `FREEZE_C3H.json` (f589238). Probes were trained on dev only, and all 1200 items were scored in one pass before any test label existed.
- Model and hardware: Qwen3-4B-Instruct-2507 on an RTX 4090.
- Data: Chinese SimpleQA, 1200 questions split by secondary category (dev 604 / val 238 / test 358).
- Labels: two blind LLM graders, dev+val κ = 0.954 (20 items went to a third grader), test κ = 0.995 (1 item went to a third grader). Human spot checks have not been done.
- Base rate: 42% of dev answers are correct, so the base-rate rule is satisfied.

## Pre-registered gates (test; n=354: 180 INCORRECT vs 174 CORRECT; bootstrap by secondary category)
| Gate | Result | Verdict |
|---|---|---|
| G1 pre-reply AUROC ≥ 0.70 | **0.722** [0.659, 0.782] | ⚪ INCONCLUSIVE |
| G2 − question-text baseline (0.622) ≥ +0.05 | **+0.100** [0.033, 0.175] | ⚪ INCONCLUSIVE (strict three-valued rule: CI lower bound must be ≥ threshold). **The literal G2 row in the pre-registration says "≥ +0.05, CI lower bound > 0", which would read as PASS; the stricter rule is used here, consistent with V1.1.** |
| G3 − category baseline (0.527) ≥ +0.05 | **+0.194** [0.115, 0.278] | ✅ PASS |

## Reported (not gated)
- **Token-level probe (mean over answer tokens): AUROC 0.811.** This is higher than a text classifier over the answer text (0.682).
- Pre-reply AUROC with NOT_ATTEMPTED counted as negative: 0.710.
- Calibration (pre-reply probe score in quintiles): the error rate rises monotonically across the five quintiles: 21% / 41% / 51% / 65% / 76%.

## Interpretation
- **Before a single character is generated**, the internal state can already rank "about to answer wrongly" above chance (0.72). This is clearly better than looking at the question type (0.53) or the question wording (0.62), but it is still a weak-to-moderate signal. The gate is exactly on the edge, so it is recorded as inconclusive.
- **Once the answer has been written**, the internal state recognises "I just made that up" much more reliably (0.81), well above a classifier on the answer text (0.68). As a product, this suits marking an answer as "needs verification" after it is generated, or at a sentence boundary.
- Single model and single dataset (short-answer factual QA). Transfer to long-form answers or other domains has not been tested.
