# E02 plan: emotion vs content inside the model, across heads, dimensions, layers and models (2026-10-06)

Written before any feature was extracted. **This is an exploratory study.**

- Patterns are looked for on the **explore** set only.
- Each pattern we claim is written down as a check before the **holdout** set is opened.
- Only patterns that hold on holdout are reported as findings. Everything else is reported as "seen on explore only".

## Questions, and how E01 leaves them open

E01 (Qwen3-4B) found three things:
- emotion is carried into later text by a few middle-layer attention heads (L14–L22; key head L14H25);
- neurons are dense and mixed;
- ablation does not remove emotion, while steering does change it.

E01 left these open:

1. **Do emotion heads and content heads overlap?**
2. **Where does emotion separate from content in the residual stream?**
3. **Is emotion aligned with a few coordinate dimensions, or spread out?**
4. **Do these patterns hold in other models?**

Plus one question tied to E01's finding that anger steering raised fabrication: **5. How do the emotion directions relate geometrically to the product probes (hostility, fabrication, sycophancy)?**

## Data

- `2026-09-28/data/formal_960.jsonl`: 120 scenarios × 4 emotions × 2 paraphrases.
- Input: chat prefix (user prompt with the facts) + `style_lead` (carries the emotion) + `common_anchor`. The anchor is token-identical within a scenario, so any emotion signal at the anchor was **carried there** by the model; it is not in the anchor's own words.
- **Explore**: discovery + validation splits (80 scenarios, 640 records).
- **Holdout**: the test split (40 scenarios, 320 records; topics 工作协作, 兴趣活动). These records were used for E01's steering test, but never in any analysis below.

## Features

Teacher forcing, one forward pass per record. For every layer, mean over two sites:

- **lead**: the style_lead tokens;
- **anchor**: the common_anchor tokens.

Two kinds of vector at each site:

- residual stream (layer output);
- each attention head's output (that head's slice of the o_proj input).

## Analyses (fixed now)

**A. Head map.** For each (layer, head), at the anchor site:

- `eta_emo`: share of variance explained by emotion within scenario. Computed as the sum of squares of the emotion means after removing the scenario means, divided by the total sum of squares.
- `eta_scn`: share of variance explained by scenario.
- `emo_auc`: macro one-vs-rest AUROC for the 4 emotions, standardise + logistic regression (C = 0.05), 5-fold CV grouped by scenario.
- Classes of head:
  - **emotion head**: emo_auc ≥ 0.75 and eta_emo > eta_scn;
  - **content head**: eta_scn ≥ 0.5 and emo_auc < 0.6;
  - **mixed**: everything else with emo_auc ≥ 0.75.

**B. Layer profile.** Same measures on the residual stream, at both sites. Plus the principal angles between:

- the emotion subspace (3 LDA directions, within-scenario);
- the top-3 scenario PCs.

**C. Basis alignment.** For the anger−neutral mean difference at each layer:

- participation ratio and the share of its norm in the top 1% of coordinates;
- the same for random directions (null);
- its overlap with "massive activation" dimensions (coordinates whose mean |activation| exceeds 20× the median).

**D. Other models.** Run A–C on Qwen3-1.7B and Llama-3.2-1B-Instruct, with depth normalised to 0–1.

**E. Concept geometry (Qwen3-4B).** At each product probe's layer, the cosine between:

- the probe's raw-space weight (coef / scale);
- the emotion directions (anger / joy / sadness − neutral, residual, lead site).

Null: random directions.

## How patterns get confirmed

After exploring, `CLAIMS.md` lists each claimed pattern as a check with a number. It is committed before holdout features are analysed. Holdout checks always include:

- the head map ranks (emo_auc) correlate between explore and holdout (Spearman ρ ≥ 0.6);
- E01's 12 causal heads sit in the top 10% of the holdout emo_auc ranking more often than chance (one-sided permutation, p < 0.05).

## Limits known now

- Three models, all small.
- Llama-3.2-1B is weak in Chinese.
- Decodability is not causation; E01's causal heads are the only causal anchor.
- One dataset of mixed-event Chinese text.

## Deviations

- 2026-10-06, after seeing Llama-3.2-1B explore numbers only. No holdout features had been extracted.
  1. **The head classes in A cannot work at the anchor site.** The anchor's own words differ by scenario, so eta_scn > eta_emo for every head by construction. Two changes follow:
     - heads are ranked by emo_auc, as the confirmation checks already do;
     - one more measure is reported: `within_share = eta_emo / (1 − eta_scn)`, the share of within-scenario variance explained by emotion.
     
     The classes are still reported at the lead site.
  2. **A null for B's subspace cosine.** It is the mean and 95th percentile of the same statistic for random 3-dimensional subspace pairs in the same dimension, over 200 draws.
