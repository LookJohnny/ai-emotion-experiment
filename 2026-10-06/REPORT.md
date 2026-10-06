# E02 report: emotion vs content inside three LLMs (heads, dimensions, layers)

2026-10-06. Mac M4, teacher forcing only, no generation and no LLM raters. Models:

- Llama-3.2-1B-Instruct (L);
- Qwen3-1.7B (Q1);
- Qwen3-4B-Instruct-2507 (Q4).

Data: E01's 960 mixed-event records. The method was fixed in `PLAN.md` (f32ddd3) before any data was extracted. Patterns were looked for on explore (80 scenarios). The 10 claims were committed in `CLAIMS.md` (d368ffa) before any holdout feature existed, then checked once on holdout (40 scenarios, unseen topics). Checker: `checks.py` → `out/checks_holdout.txt`.

## What held on holdout

| # | Pattern | L | Q1 | Q4 |
|---|---|---|---|---|
| H1 | **Carried emotion peaks mid-depth, then fades.** At the anchor (text identical across emotions), the best layer sits at relative depth 0.43–0.56 in all three models and drops by 0.04–0.08 AUROC towards the last layer | ✅ 0.47 | ✅ 0.56 | ✅ 0.43 |
| H2 | **Tiny variance, near-perfect readability.** At the peak, emotion explains 3–5% of the variance (content explains about 90%), yet it is read with AUROC ≥ 0.99 | ✅ | ❌ 0.0501 (threshold < 0.05) | ✅ |
| H4 | **Not aligned to coordinates.** After standardisation, the anger direction is *more evenly spread* than permutation nulls at every layer of every model | ✅ | ✅ | ✅ |
| H5 | **A valence split appears from about a quarter of the depth.** Below that depth, joy, sadness and anger look equally similar. From there on, sadness and anger are always closer to each other than to joy | ✅ +0.09 | ✅ +0.05 | ✅ +0.09 |
| H6 | The head map is stable across topics (Spearman) | | | ✅ ρ 0.93 / 0.98 |
| H7 | **A purely correlational head map finds E01's causal steering heads.** 8 of E01's 12 heads fall in the top 10% of 1152 (p < 0.0001 against random heads from the same layer window); E01's key head L14H25 ranks 1st on explore and in the top 10 on holdout | | | ✅ |
| H9 | **The product probes are orthogonal to the speaker's own emotion.** Hostility, fabrication and sycophancy show no alignment with joy / sadness / anger, at 18 of 18 cells | | | ✅ |
| H10 | The user-distress probe leans toward anger (cos 0.085 vs null 0.075, small) | | | ✅ |

## What did not hold

- **H3, "emotion and content subspaces are no closer than random"**: fails in 5 of 6 model × site cells.
  - The cosines stay small (≤ 0.11, i.e. at least 84°), but holdout is higher than explore.
  - Diagnostic, run after the check and not a re-test: explore subsampled to the same 40 scenarios gives the same rise (0.06–0.115).
  - So the isotropic random-subspace null was the wrong null for estimates from 40 scenarios. **Undetermined**, not "overlapping".
- **H8, "Q4 emotion heads sit mid-depth"**: 0.684 against a threshold of 0.70. It was already not universal on explore (L's best heads sit late).
- **H2 for Q1**: misses by 0.0001. Reported as a fail.

## What it means

1. **Emotion and content share the same tokens but live in different "volumes".** Content dominates the variance. Emotion is a thin but sharp signal: a few percent of the variance, read almost perfectly. Two practical consequences:
   - a probe has to look for a direction;
   - variance-based tools (PCA, top neurons) will mostly see content, which helps explain why E01's top-neuron ablation failed.
2. **The model carries the opening's emotion forward in the middle of its depth, and does so in all three models.** Mid-depth is where readers and steering should act. E01's causal heads (L14–L22 of 36) sit exactly in this band.
3. **Emotion is not stored in "emotion neurons" or special coordinates.** In all three models it is spread more evenly than chance. Together with E01's census (70–76% of neurons emotion-selective) and failed ablation, this points to a distributed code.
4. **There is a shared "not neutral" axis plus a valence axis that forms around 1/4 depth.** This pattern is universal across the three models. A product consequence: a readout trained on anger will partly fire on sadness in middle and late layers.
5. **The safety probes measure something other than the AI's own mood.** Hostility toward the user is not "the AI is angry"; fabrication is not anger. E01's "anger steering raised fabrication" is therefore not explained by a shared linear direction at L19. It is more likely a behavioural link: angry text embellishes.
6. **Cheap correlational maps can stand in for expensive causal search.** Here, a head map that cost about 5 minutes on a laptop recovered most of the heads that E01's GPU patching found. This matters for adapting Nous probes to new models.

## Limits

- Three small models; Llama-1B is weak in Chinese.
- One dataset of short mixed-event first-person texts.
- Decodability is not causation. H7 is the only causal tie-in (E01's heads).
- Holdout = 40 scenarios. The test split was used by E01's steering test, though never by these analyses.
- Not done: the link to **generated** output, i.e. whether the mid-depth state predicts word choice, length or fabrication in what the model writes next. That needs generation plus rating, the next step.

## Files

- Code: `feats.py`, `analyze.py`, `geometry.py`, `summarize.py`, `checks.py`.
- Outputs: `out/*.json`, `out/checks_holdout.txt`.
- Features (about 1 GB, not committed): `feats/*.npz`, regenerable with `feats.py`.
