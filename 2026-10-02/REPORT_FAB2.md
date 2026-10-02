# Fabrication v2 report (Part F2 of PREREG_SYCO_FAB2.md) — 2026-10-02

Numbers: `fab2/report_fab2.json`.

## Data

**Qwen3-8B** (thinking off) gave greedy answers to all 1,200 C3-H questions.
- Two blind graders agreed on 1175/1200, with 25 tiebreaks.
- **The 8B reader knows less than the 4B reader.** Qwen3-8B answered 434/1200 correctly (36%), against 49% for Qwen3-4B-Instruct-2507.
- The newer 2507 4B release is stronger on Chinese factual QA than the older 8B. This undercuts the premise of route F-a ("a bigger reader knows more").
- The 8B probe was trained on dev and read layer 22 (CV 0.841).

**Kimi K3** answers were generated anew:
- Kimi-dev: 604 answers, graded with 601/604 agreement. 83 INCORRECT.
- Kimi-val (the new sealed test): 238 answers, graded with 238/238 agreement. 35 INCORRECT, which passes the ≥30 base-rate rule.

## Gates (Kimi-val, bootstrap by category; PASS requires CI lower bound ≥ threshold)

| # | Result | Verdict |
|---|---|---|
| F1 8B reader AUROC | 0.767 [0.666, 0.851] (thr 0.75) | INCONCLUSIVE |
| F2 8B − 4B reader | **−0.068** [−0.153, 0.009]; the 4B reader scores 0.834 here | **FAIL** |
| F3 combination (4B + 8B reader + Kimi NLL + entropy; LR fit on Kimi-dev) | **0.836** [0.760, 0.902] (thr 0.78) | INCONCLUSIVE |

Reported on Kimi-val:
- Kimi's own NLL alone: 0.567. Kimi's own entropy alone: 0.617.
- Combination weights (standardised inputs): r4 1.07, r8 0.69, NLL −0.31, entropy 0.61.

Secondary, on Kimi-test (already used for the 4B reader; nothing was tuned on it):

| Readout | AUROC |
|---|---|
| 8B reader | 0.718 |
| 4B reader | 0.737 |
| Combination | 0.758 |

On Kimi-test, AUROC split by whether the reader itself knows the answer:

| Reader | Knows the answer | Does not know it |
|---|---|---|
| 4B | 0.87 | 0.61 |
| 8B | 0.75 | 0.61 |

## What this means

1. **"Use a bigger reader" fails here, because 8B is not more knowledgeable.**
   - The bottleneck is still the reader's knowledge. A reader is only as good as what it knows, and model size is a poor proxy for that.
   - The next reader to test should be chosen by measured accuracy on the target domain, not by parameter count. For example, Qwen3-30B-A3B-2507 needs a larger GPU.
2. **Combining signals helps.**
   - The combination of reader probes and the target's own logprobs reaches 0.84 on Kimi-val, and its point estimate passes 0.78. It is still INCONCLUSIVE because the CI is wide, with only 35 errors.
   - Kimi's own probabilities are weak alone (0.57–0.62) but add information.
   - This is the most promising product route: wherever an API exposes logprobs, use them alongside the reader.
3. **Split-to-split variance is large.** The same 4B reader scores 0.834 on Kimi-val and 0.737 on Kimi-test. Any single number from about 35–50 errors should be read with its CI.
4. **Product.**
   - Fabrication stays **experimental** in Nous Lens.
   - The next step is retrieval cross-checking: verify the answer's key entity against a search result. It was out of scope this round.

## Limitations

- **All graders are LLMs from one model family.** Several graders noticed that their first write had grades misaligned with item ids and fixed it before submitting. The final files were checked for id alignment, and inter-rater agreement is 98%. No human spot checks yet.
- **Only short Chinese factual QA is covered.** Long-form fabrication is untested.
- **The rating of the Qwen sycophancy replies ran through a multi-agent workflow tool** without explicit user opt-in (see REPORT_SYCO.md). The fabrication grading used individual agents.
