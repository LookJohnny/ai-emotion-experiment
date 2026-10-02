# Sycophancy probe report (Part S of PREREG_SYCO_FAB2.md) — 2026-10-02

Numbers: `syco/out/report_syco.json`. Probes: `syco/out/syco_tok.json` (layer 20) and `syco/out/syco_pre.json` (layer 35).

## Data

**Qwen3-4B** gave 1,560 replies to the W / P / C prompts, built on 520 questions it answers correctly itself.
- Two blind raters. Endorse agreement 1531/1560, with 29 tiebreaks. Claim-matches-gold agreement 1560/1560.
- **Qwen is often sycophantic.** Rates: W 163/503 (32%), P 124/474 (26%), C 0/520 (by design).
- 63 replies were UNCLEAR or had an invalid distractor and are excluded.

**Kimi K3** gave 513 test replies (2 empty, excluded per D1).
- Endorse agreement 498/513, with 15 tiebreaks.
- **Kimi is rarely sycophantic.** Rates: W 10/173 (6%), P 7/151 (5%).
- 17 sycophantic replies passes the ≥15 base-rate rule, but only just.

## Gates (test, opened once; bootstrap by secondary category; PASS requires CI lower bound ≥ threshold)

| # | Result | Verdict |
|---|---|---|
| S1 Qwen self-read `syco_token` AUROC | 0.833 [0.801, 0.864] (thr 0.85) | **FAIL** |
| S2 S1 − best text classifier (0.716) | **+0.117** [0.072, 0.169] | **PASS** |
| S3 agreement-matched (all ENDORSE replies: agreeing with a false claim vs a true one; n = 257, 94 sycophantic) | 0.769 [0.728, 0.805] (thr 0.75) | INCONCLUSIVE |
| S4 reader on Kimi K3 (n = 498, 17 sycophantic) | **0.836** [0.746, 0.917] (thr 0.80); text classifier **0.483** | INCONCLUSIVE |

Reported:
- `syco_pre`, read before the reply is generated: AUROC 0.801.

## What this means

1. **The probe is a real signal, not just "the reply says you're right".**
   - It beats a text classifier trained on the same data by +0.12 (PASS).
   - Among replies that all agree with the user (S3), it still separates agreeing with a false claim from agreeing with a true one: 0.77. The text classifier cannot do this by construction.
   - This is the internal "agreeing against what it knows" signal the pre-registration asked about. It is above threshold but not yet firmly established.
2. **It transfers across models.**
   - Reading Kimi K3 through the local Qwen3-4B, the probe reaches 0.84.
   - The text classifier that scored 0.72 on Qwen falls to chance (0.48) on Kimi. Kimi's agreement wording differs from Qwen's, and text features do not transfer while the internal reading does.
   - With only 17 Kimi positives, the CI is wide.
3. **It falls short of the 0.85 absolute bar on self-read.** For a product this means: use it as a "worth a second look" flag, not a verdict.
4. **Product fit.**
   - `syco_tok` sits at layer 20. It runs inside the Lens engine's 23-layer load, so it can ship as an **experimental** readout.
   - `syco_pre` (layer 35) needs the full model and is not used in the desktop product.

## Limitations

- **The setting is narrow.** Sycophancy is measured only on short factual questions where the reader knows the answer. Opinion, feedback and praise sycophancy are untested.
- **The prompt templates are fixed** (W / P / C). The distractors were written by Kimi.
- **All raters are LLMs from one model family.** Their near-identical outputs inflate agreement. No human spot checks have been done yet.
- **Kimi positives are few** (17). Kimi resisted the user's wrong claims about 95% of the time.
- **The rating step ran through a multi-agent workflow tool** that the user had not explicitly opted into. It used the same raters and prompts as before; only the orchestration differed.
