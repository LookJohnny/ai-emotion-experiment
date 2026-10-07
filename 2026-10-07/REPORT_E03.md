# E03 report: does the anchor head map pick steerable heads in Qwen3-8B?

2026-10-07. RTX 4090.

| Step | Commit |
|---|---|
| Pre-registration | 40b384a |
| Map frozen | 4c6c61a |
| λ frozen | f820871 |
| Test opened once | 50bca4f |

Raters: two blind LLM streams. Rater agreement on validation: r = 0.93 for anger, 0.73–0.91 for the other dimensions.

## Confirmatory result: negative

At the frozen λ = 1, on the test split (40 scenarios, 520 continuations):

| Contrast (anger, /4) | Mean | 95% CI | Gate |
|---|---|---|---|
| Map heads − random directions | +0.04 | [−0.05, 0.12] | G1 ✗ |
| Map heads − none | +0.03 | [−0.09, 0.14] | G2 ✗ |
| Map heads − random heads (same total push) | −0.01 | [−0.11, 0.09] | G3 ✗ |
| Specificity vs joy / vs sadness | +0.03 / −0.04 | both CIs span 0 | G4 ✗ |
| Fluency / factual consistency | −0.01 / 0.00 | | G5 ✓ |
| Anger lead itself (ceiling) | +1.10 | [0.79, 1.41] | |

**Verdict.** Steering the 12 top heads of the anchor map does not change Qwen3-8B's expressed anger. The raters were sensitive enough to detect a +1.10 difference.

On validation, larger λ did not help either:
- the map heads stayed at baseline;
- random heads steered with their own true vectors rose in anger (λ = 8: random heads − map heads −0.59, CI [−0.85, −0.31]).

## Exploratory follow-up: why

This was run after the test was generated and does not change the verdict. It is a teacher-forced screen on validation (A + B, 40 inputs). The effect measure is the change in mean log p(anger reference suffix) − log p(neutral reference suffix) relative to no steering. Every configuration got the same total push. Data: `explore_screen.json`; code: `explore8b.py`.

**1. The map finds causally relevant heads, but not their sign.**
- Single-head screen at equal push: 50 of the map's top 60 heads have an effect beyond 2 SE, against 24 of 60 random heads (Fisher p = 1.7 × 10⁻⁶).
- 15 of the top-60 heads push the *opposite* way: adding the anger − neutral vector makes anger less likely.
- The map's top 12 contain two strong opposers, which cancel the pushers:
  - opposers: **L17H19** −0.56 and **L22H16** −0.41;
  - pushers: L19H29 +0.29 and L11H29 +0.13;
  - five late-layer heads (L27–L35) with near-zero effect.
- Group results: the map top 12 as a whole gives about 0; its mid-depth members alone give **−1.15** (the opposers dominate).

**2. E01's causal heads keep their role across model sizes, by index.**
- E01 chose 12 heads by activation patching in Qwen3-4B. The same layer/head indices, steered in Qwen3-8B with 8B's own vectors, give **+1.15 ± 0.09**, the largest effect of any group.
- Five of the ten strongest single heads in 8B are E01 indices: L14H26, L17H16, L19H29, L22H19 and L16H3.
- The map's own first-ranked head in 8B is L14H25, also E01's key head in 4B.

**3. What this means for the method.** Readability at the anchor marks the heads that *take part* in carrying emotion. That includes heads that push emotion forward and heads that counter-regulate it. Picking the top-k by readability alone mixes the two, and they cancel.

A sign-aware second stage would separate them: a teacher-forced single-head screen (about 3 minutes for 120 heads on one 4090), keeping heads whose steering moves the output the right way. This two-stage method (map prefilter, then sign screen) is **a hypothesis from this exploration**. It needs its own confirmatory test on fresh data or a fresh model.

## Implications

- **Patent:** claim 1 as drafted (top-k by the carry score alone, used for steering) is not supported on the 8B model. Options:
  - narrow the claims to the two-stage method;
  - add within-family index transfer as a dependent claim.

  Both need confirmation before they can be relied on.
- **E02 stays as reported.** It only claimed that E01's causal heads fall in the map's top 10%, and in 8B they again sit at ranks 0–36.
- **The confirmatory negative stands.** Any new method gets a new pre-registration.
