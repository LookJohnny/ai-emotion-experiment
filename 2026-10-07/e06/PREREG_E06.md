# E06 pre-registration: two-stage head selection vs GCM (attribution patching) vs ITI, on Qwen3-14B

Written 2026-10-07. At the time of writing:
- no E06 selection had been run;
- no E06 generation or rating existed;
- the model was still downloading to a fresh instance.

## Question

The closest prior art selects steering heads in two other ways:

- **GCM**, Sankaranarayanan et al., arXiv 2602.16080. Heads are ranked by their causal mediation of the contrast between the model's own responses. The cheap variant uses attribution patching: 2 forward passes and 1 backward pass score every head.
- **ITI**, Li et al., arXiv 2306.03341. Each head gets a linear probe, and heads are ranked by probe accuracy.

**Does our two-stage selection (E04) steer better than these, at equal push, on the same model and data?**

We also test one variant borrowed from GCM: our sign screen using the model's **self-generated** responses in place of the LLM-drafted reference suffixes.

## Model and data

- **Model**: Qwen3-14B, bf16, `enable_thinking=False`, greedy decoding, 64 new tokens.
- **Data**: E01's `formal_960.jsonl`.
  - Selection uses **discovery** only: 60 scenarios × paraphrases A and B, emotions anger and neutral.
  - Evaluation uses **test**: 40 scenarios, paraphrase A.

The Qwen3-14B test split was used once before, in E04. E06 regenerates and re-rates every condition, the E04 two-stage condition included, in one fresh blinded batch, so all comparisons are within E06.

## Self-generated responses (discovery)

For each discovery scenario and paraphrase, generate a greedy 64-token continuation from:
- the **neutral** input (neutral lead + anchor): r_orig;
- the **anger** input (anger lead + anchor): r_contrast.

## Head sets (12 heads each, frozen before any test generation)

| Set | How it is selected |
|---|---|
| `twostage` | E04's frozen set, unchanged: `2026-10-07/e04/screen/SCREEN_FREEZE.json`, field `twostage` |
| `gcm_atp` | Rank every head by attribution-patching indirect effect, then take the top 12. For each discovery pair: IE_hat(z) = ∇_z [log π(r_contrast \| p_orig) − log π(r_orig \| p_orig)] · (z_contrast − z_orig), averaged over pairs. z is the head's o_proj-input slice at the **anchor token positions**. The anchor is the only position range aligned between the two inputs, because the leads differ in length. This adaptation is fixed now |
| `iti` | A per-head logistic-regression probe, anger vs neutral, on the head output at the last anchor token (the last prompt token). Use a 4:1 train/validation split by scenario, rank heads by validation accuracy, take the top 12. This follows ITI's selection rule |
| `twostage_self` | E04's map top 60, then E04's sign screen with the reference suffixes replaced by the self-generated r_contrast and r_orig. Keep effect > 2 SE, take the top 12 by effect |

**Freeze**: all four sets, with timing for each selection method, go into `E06_FREEZE.json`. It is committed before test generation.

## Steering and conditions (test)

- Steering is continuous, as in E04: a difference-in-means vector per head (discovery, anger − neutral, anchor mean) is added at the anchor positions and at every generated position.
- Every head-set condition is rescaled to the same summed residual-change norm as `twostage` at that λ.
- **λ is fixed in advance, not selected.**
  - Primary: λ = 8, E04's frozen value.
  - Secondary: λ = 4.
- **Conditions**:
  - `none`;
  - `primary_prefix`;
  - at each λ: `twostage`, `gcm_atp`, `iti`, `twostage_self`, and `rheads_j` (random 12 heads, seeds 301–303, each head's own vector).
- Total: 40 × (2 + 2 × (4 + 3)) = 640 continuations, rated by 2 blind LLM raters (E01 continuation-fragment protocol), shuffled together.

## Hypotheses and decision rules

All tests are on anger at λ = 8 unless stated, with scenario-level means and a bootstrap over scenarios (10,000 draws, 95% CI).

### H1 (main): twostage vs gcm_atp

| Outcome | Rule |
|---|---|
| **better** | CI lower bound > 0 |
| **non-inferior** | CI lower bound > −0.5, and not better |
| **worse** | CI upper bound < 0 |
| **inconclusive** | otherwise |

### Other hypotheses

| | Comparison | Pass if |
|---|---|---|
| **H2** | twostage vs iti | CI lower bound > 0 |
| **H3** | twostage_self vs twostage | Reported with CI and the same four-way labels |
| **H4** | each method − mean(rheads) | Reported with CI |

**Also reported**:
- the λ = 4 versions of H1–H4;
- fluency and factual-consistency change against `none` for each method;
- the overlap between head sets;
- the wall-clock cost of each selection method on the same GPU.

### Interpretation fixed now

| Result | Reading |
|---|---|
| H1 better | Strong evidence for the patent's inventive step and the paper's main claim |
| H1 non-inferior | Our method matches GCM without backward passes. Its claim narrows to forward-only selection |
| H1 worse | Our selection is dominated by GCM's attribution patching on this task. The patent's core claim should be reconsidered before filing |

## Known limits

- One model, one attribute (anger).
- LLM raters.
- Test scenarios were used in E04.
- GCM is adapted to anchor-position patching because our leads differ in length.
