# Reader-model experiment report: monitoring Kimi K3 (closed API) with frozen Qwen3-4B probes (2026-09-30)

Pre-registration: `READER_PREREG.md` (commit 0c39495, before any generation). Deviation D1: 1 empty C1 reply excluded. Numbers: `report_reader.json` (`reader_eval.py`).

## Data
- **C1:** 143 Kimi replies (12 test templates × 4 history tones × 3, minus 1 empty). Two blind raters, r = 0.988, binary agreement 143/143. **18 positives** (≥ 15, so the base-rate rule did not trigger and val was not added).
  - All 18 positives follow sarcastic (8/36) or insulting (10/36) histories. Polite and curt histories gave 0/71: Kimi sometimes carries over a hostile persona from earlier turns but never starts one.
- **C3-H:** 358 answers. Two graders, κ = 0.984, 2 tiebreaks.
  - Kimi: CORRECT 285 / INCORRECT 53 / NOT_ATTEMPTED 20 (accuracy 79.6%; Qwen3-4B on the same questions is about 49%).

## Gates (bootstrap 95% CI by template / category; strict rule: CI lower bound ≥ threshold)
| # | Result | Verdict |
|---|---|---|
| R1 C1 reader AUROC | **0.973** [0.957, 0.991] (thr 0.80) | **PASS** |
| R2 C1 − text classifier (0.914) | +0.060 [0.001, 0.140] | INCONCLUSIVE (the literal wording "≥ .05 and CI lower bound > 0" would pass) |
| R3 C3-H reader AUROC | **0.737** [0.662, 0.802] (thr 0.70) | INCONCLUSIVE |
| R4 C3-H − answer-text classifier (0.636) | +0.102 [0.022, 0.193] | INCONCLUSIVE (literal wording would pass) |

## Reported metrics
- **C1:**
  - Kimi's own logprobs: NLL AUROC 0.777, entropy 0.651. The reader is clearly better than both.
  - At the demo threshold of 0.2372: recall 0.944, FPR 0.088.
  - Reading Kimi (0.973) is not worse than self-reading (0.929). The positives here are more explicit, so this does not mean the reader is better.
  - Prompt-side C1-pre predicting Kimi hostility: AUROC 0.746 overall, and only 0.571 within hostile histories. As expected, **there is no pre-reply warning for a closed model**; only reading the output works.
  - Exploratory (not pre-registered): within the sarcastic and insulting histories only (n = 72, 18 positives), reader AUROC is 0.944. So the reader is judging the reply itself, not just the history tone.
- **C3-H:**
  - Reader − Kimi NLL: +0.089 [−0.041, 0.214].
  - Reader − Kimi top-5 entropy: +0.136 [0.016, 0.256].
  - Drop versus self-read (0.811): −0.074.
  - **Split by whether Qwen itself knows the answer:**
    - Qwen correct: reader AUROC **0.868** (n = 174, only 7 Kimi errors, so few positives).
    - Qwen wrong: **0.612** (n = 161, 45 Kimi errors).
  - The reader's own knowledge is the bottleneck. On questions it does not know itself, it cannot tell whether someone else answered correctly.
  - At the demo threshold of 0.9301: recall 0.642, FPR 0.305. That threshold is calibrated for Qwen's own answers and is too high a false-alarm rate for Kimi, so it would need recalibration.

## Conclusions
1. **Behavioural concepts (hostility) transfer across models.** A 4B local reader reading closed-model text gets AUROC 0.97 (PASS), clearly better than Kimi's own logprobs. Its advantage over a text classifier is +0.06 and not yet firmly established.
2. **Knowledge concepts (hallucination) are limited by what the reader knows.** Reading a stronger model: 0.74 overall (INCONCLUSIVE), 0.87 on questions the reader knows, 0.61 on those it does not. It is still better than the text baseline and Kimi's entropy.
3. **Product implications** (consistent with the tiers in `API_MONITORING_RESEARCH.md`):
   - For closed models, a "reader proxy" monitoring hostility and tone is viable.
   - Hallucination monitoring needs a reader at least as knowledgeable as the target model, or retrieval / an LLM judge.
   - Pre-reply warning is not possible for closed models.

## Limitations
- C1 positives come from red-team-style hostile histories (Kimi carrying over a persona). Kimi never produced hostility from a polite start.
- 18 positives from 12 templates, so the CIs are wide.
- Kimi runs at temperature 1 (fixed by the API).
- Labels come from LLM raters and have not had human spot checks.
