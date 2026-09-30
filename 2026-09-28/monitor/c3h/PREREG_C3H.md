# C3-H pre-registration: hallucination pre-warning (2026-09-30)

Written before any generation, label, or probe. Model and hardware are the same as V1.1: Qwen3-4B-Instruct-2507 in BF16, RTX 4090 (AutoDL bjb1).

## Concept
**C3-H: the model is about to give a confident but wrong factual answer (hallucination).** Before the first character of the reply is generated, the model's internal state is used to predict whether this answer will be wrong.

## Data
- **Chinese SimpleQA** (OpenStellarTeam, 2024, public). It has 3000 short factual questions, each with a gold answer, in 6 primary categories and 99 secondary categories.
- **Sample:** 1200 questions drawn with `random.Random(7)`. Each secondary category contributes at most `ceil(1200 × its share)` questions.
- **Split:** by **secondary category**, with `random.Random(8)` shuffling the category list and allocating it in order. Target question shares are dev 50% / val 20% / test 30%. The resulting category list and item ids are written to `c3h/items.jsonl` and committed before generation.
- **Prompt:**
  - system: 你是知识问答助手，请直接、简洁地回答问题。 ("You are a knowledge Q&A assistant; answer the question directly and concisely.")
  - user: the question.
  - Generation is greedy with max_new_tokens=64.

## Labels
- Two blind LLM raters see the question, the gold answer, and the model's answer. Each gives one of **CORRECT / INCORRECT / NOT_ATTEMPTED**, following the Chinese SimpleQA grading definitions.
  - Semantically equivalent answers are correct.
  - If the answer contains the correct answer plus a contradicting error, it is incorrect.
  - If the model explicitly says it doesn't know, the item is not attempted.
- The two labels must agree to be used. If they disagree, a third LLM rater breaks the tie. Inter-rater agreement and κ are reported.
- The test split is rated only after `FREEZE_C3H.json` is committed.

## Probes (same method as V1.1)
- **C3H-pre:**
  - Feature: the state at the last prompt position, with candidate layers 12/16/19/22/26.
  - Layer choice: 5-fold CV grouped by secondary category.
  - Classifier: LR with C=0.05.
  - Training contrast: INCORRECT (1) vs CORRECT (0).
- **C3H-token (report only):** the mean over the answer tokens.

## Gates (test; bootstrap by secondary category, 2000 draws, 95% CI; three-valued: PASS / FAIL / INCONCLUSIVE)
| # | Content | Threshold |
|---|---|---|
| G1 | C3H-pre AUROC (INCORRECT vs CORRECT) | ≥ 0.70 |
| G2 | G1 − question-text baseline (char 1–3-gram TF-IDF + LR, trained on dev) | ≥ +0.05, CI lower bound > 0 |
| G3 | G1 − category baseline (the dev error rate of the question's primary category) | ≥ +0.05, CI lower bound > 0 |

**Reported, not gated:**
- AUROC with NOT_ATTEMPTED items treated as negatives.
- C3H-token AUROC.
- Answer-text baseline: TF-IDF on the model's answer.
- Calibration curve.

**Base-rate rule:** if the dev correct rate is < 10% or > 90%, stop and report without training.

## Deviations

- **D1 (2026-09-30, before any probe result was viewed and before any test grading):** `build_c3h.py` created item_id from the first 12 characters of the source id. The Chinese SimpleQA source ids collide on that prefix, so 11 questions shared 3 ids. Grades are unaffected, because graders worked on packet-level numbered items (q####) with the full question text. Only the mapping back to items collapsed. Fix: re-key every item as `h{line index:04d}` and keep the original id in `src_id`. The packet order was reproduced and verified item by item against each packet's question and answer text. Sample, split and grades are unchanged. The probe-training job that had just started was stopped and its outputs deleted; it had produced no results.
