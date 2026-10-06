# E02 claims to check on holdout (written after explore, before any holdout feature exists)

**Explore** = discovery + validation (80 scenarios). **Holdout** = test split (40 scenarios, topics 工作协作 / 兴趣活动).

- Each check uses the same code (`analyze.py`, `geometry.py`) on `feats/<model>_holdout.npz`.
- A claim is **confirmed** only if its check passes on holdout.
- **Models**: L = Llama-3.2-1B-Instruct, Q1 = Qwen3-1.7B, Q4 = Qwen3-4B-Instruct-2507.
- **Anchor site** = the common_anchor tokens. They are identical across emotions within a scenario, so emotion found there was carried by the model.

## Layer profile (B)

**H1 — carried emotion peaks mid-depth, then fades** (all three models). At the anchor site, the residual emo_auc reaches its maximum at relative depth 0.35–0.65. The last layer is lower than the peak by ≥ 0.02.

| Explore | L | Q1 | Q4 |
|---|---|---|---|
| Peak depth | 0.47 | 0.56 | 0.49 |
| Drop to last layer | 0.055 | 0.034 | 0.087 |

**H2 — small variance, high readability** (all three). At the peak layer, eta_emo < 0.05 while emo_auc ≥ 0.95.

| Explore | L | Q1 | Q4 |
|---|---|---|---|
| eta_emo at peak | 0.021 | 0.040 | 0.010 |

**H3 — emotion and content subspaces are no closer than chance** (all three, both sites). For at most 20% of layers does the emotion-vs-content cosine exceed the random-subspace p95 (L/Q1 0.077, Q4 0.074).

## Dimensions (C)

**H4 — emotion is not aligned to coordinates** (all three, lead site). After per-coordinate standardisation, the anger − neutral direction's top-1% coordinate share is below the permutation null's p95 at ≥ 90% of layers.

## Emotion geometry

**H5 — a valence split appears from about a quarter of the depth** (all three, anchor site). At every layer with relative depth ≥ 0.25:

  cos(sadness, anger) − cos(joy, anger) > 0

Here each emotion's direction is its mean minus the neutral mean. Explore minimum gap over those layers: L +0.02, Q1 +0.04, Q4 +0.09.

## Heads (A), Qwen3-4B

**H6 — the head map is stable.** Spearman ρ between explore and holdout across all 1152 heads at the anchor site is ≥ 0.6. Checked for both within_share and emo_auc.

**H7 — the correlational map finds E01's causal heads.** Count how many of E01's 12 frozen steering heads (`2026-09-28/stageD/FREEZE_5b.json`) are in the holdout top 10%. Compare with random 12-head sets from E01's search window L12–27, using a one-sided permutation (10,000 draws); p < 0.05. Checked for within_share and for emo_auc.

On explore, which overlaps the data E01 selected those heads on and is therefore partly circular:

| Explore | Heads in top 10% | p |
|---|---|---|
| within_share | 9 / 12 | < 0.0001 |
| emo_auc | 7 / 12 | 0.0006 |

**H8 — emotion heads sit mid-depth in Qwen3-4B.** At least 70% of the top-5% within_share heads have relative depth 0.3–0.7 (explore 0.81). On explore this is **not** universal: Q1 0.64, L 0.36. It is not claimed for those models.

## Product probes vs emotion (E), Qwen3-4B

**H9 — the hostility (c1_token), fabrication (c3h_token) and sycophancy (syco_token) probes are orthogonal to the speaker's own emotion directions.** |cos| ≤ the permutation null's |p95|, for all 3 emotions × 2 sites. Explore: all 18 within the null.

**H10 — the user-distress probe (c2) leans toward anger at the lead site.** cos(c2, anger) > the null's |p95|. Explore: 0.099 vs 0.080.

## Not claimed (seen on explore, too weak or not universal)

- Massive-activation dimensions carry a slightly larger share of the anger direction in Qwen's first ~6 layers (raw scale only).
- Head depth concentration in Q1 and L.
