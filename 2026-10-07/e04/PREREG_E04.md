# E04 pre-registration: two-stage head selection (map prefilter + sign screen), confirmatory, on Qwen3-14B

Written 2026-10-07. At the time of writing:
- no Qwen3-14B feature, screen, generation or rating existed;
- the model was still downloading.

## Why

E03 (Qwen3-8B) gave two results:

1. **Confirmatory negative.** Steering the anchor map's top 12 heads changed nothing.
2. **Exploratory finding (after the test was generated, so it does not count as confirmation).** The map finds causally *relevant* heads: 50 of the top 60 had a significant effect, against 24 of 60 random heads. It does not find their *sign*: 15 of the top 60 opposed the steering direction, and in the top 12 the pushers and opposers cancelled.

E04 tests the method E03 suggested, on a model never used before. That method:

1. **Map prefilter**: the top 60 heads by within_share at the anchor.
2. **Sign screen**: a teacher-forced single-head steering test; keep heads that move the output in the intended direction.

## Model and data

- **Model**: Qwen3-14B (ModelScope `Qwen/Qwen3-14B`), bf16, `enable_thinking=False`. Greedy decoding, 64 new tokens. One 48 GB RTX 4090.
- **Data**: E01's `formal_960.jsonl`, same splits as before.
  - discovery (60 scenarios): map, vectors and screen;
  - validation (20 scenarios): λ;
  - test (40 scenarios): a single final evaluation.
- No part of this model has seen any split.

## Stage 1: map (discovery A + B)

`feats.py` + `map_select.py`, unchanged from E03:

- within_share at the anchor site for every head;
- steering vectors: per head, anger − neutral mean of the o_proj input slice over anchor tokens.

## Stage 2: sign screen (discovery only), frozen before any generation

Script: `screen.py`. Inputs are discovery scenarios × paraphrases A + B (120 inputs), each the neutral lead + anchor.

- **Effect of a head**: the mean over inputs of the change, versus no steering, in [mean log p(anger reference suffix) − mean log p(neutral reference suffix)]. Steering is applied at the anchor and every suffix position.
- **Push**: every screened head is steered with its own vector, rescaled so its residual-stream change ‖W_O[:, h]·v‖ equals that of the map's rank-1 head at λ = 8.
- **Two-stage set (`twostage`)**: among the map's top 60, keep heads with effect > 2 SE (SE over inputs). Take up to 12 by largest effect.
- **Screen-only set (`rscreen`)**: the same screen and rule applied to 60 heads drawn at random (seed 401) from outside the map's top 60. This control spends the same screening budget without the map.
- **Freeze**: both sets, all screen numbers and the hashes go into `SCREEN_FREEZE.json`, committed before validation generation.
- **Stop rule**: if `twostage` has fewer than 4 heads, stop and report; do not generate.

## Conditions (validation and test)

Steering adds the vector at the anchor and at every generated position. The recipient input is the neutral lead + anchor.

| Condition | What it is |
|---|---|
| `none` | no steering |
| `primary_prefix` | the anger lead itself; this is the ceiling |
| `target` | `twostage` heads, λ·v_h. Its summed residual-change norm T defines the push for every head-set control below |
| `maponly` | the map's top 12, which is E03's method, rescaled to total push T |
| `rscreen` | the screen-only set, each head's own vector, rescaled to total push T |
| `rheads_j` | 12 random heads outside `twostage` (seeds 301–305), each head's own vector, rescaled to total push T |
| `rdir_i` | `twostage` heads with random directions, norm-matched per head (seeds 101–105) |

## Validation (choose λ)

- 20 scenarios, paraphrase A.
- λ grid {1, 2, 4, 8}.
- rdir seeds 101–102 and rheads seeds 301–302 at validation.
- Two blind LLM raters, using E01's continuation-fragment protocol.

**Quality gates** against `none` are three-valued:
- fluency threshold −0.3;
- consistency threshold −5 pp;
- **FAIL** if the mean is worse than the threshold, **PASS** if the CI lower bound is above it, otherwise **INCONCLUSIVE**.

**Pick**: among λ that do not FAIL, the largest `target − mean(rdir)` on anger. If two are within 0.1, take the smaller λ. If none qualify, stop.

**Freeze**: `FREEZE_E04.json`, written before test generation.

## Test (opened once): 40 scenarios, paraphrase A

| Gate | Test (anger unless stated) | Requirement |
|---|---|---|
| **G1** | target − mean(rdir) | ≥ 0.5 / 4 and CI lower bound > 0 |
| **G2** | target − none | CI lower bound > 0 |
| **G3** | target − mean(rheads) | CI lower bound > 0 |
| **G4** (the two-stage method beats the map alone, i.e. E03's method) | target − maponly | CI lower bound > 0 |
| **G5** (specificity) | (target − none) on anger minus the same on joy, and minus the same on sadness | each CI lower bound > 0 |
| **G6** (quality) | fluency and consistency | three-valued; FAIL blocks |

All tests use scenario-level means and a bootstrap over scenarios (10,000 draws, 95% CI).

**Secondary, reported with CI, not a gate**: `target − rscreen`, i.e. whether the map prefilter adds value over screening random heads.

**Interpretation, fixed now**:

| Outcome | Reading |
|---|---|
| G1–G4 PASS | The two-stage method selects steerable heads in a new model, and beats map-only selection |
| G1–G3 PASS, G4 FAIL | The heads steer, but are no better than the map alone in this model |
| Secondary CI lower bound > 0 | The map prefilter adds value beyond screening |
| Secondary CI spans 0 | The map's role is narrowed to reducing screening cost: it found the screen candidates among 60 heads instead of 1,600 |

## Known limits

- Raters are LLMs; human calibration has not been done.
- The test split's scenarios were seen by other models (E01 4B, E03 8B), never by Qwen3-14B.
- The screen's effect metric uses reference suffixes, which are LLM-drafted text.
