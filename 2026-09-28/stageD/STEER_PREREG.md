# §5b Continuous-steering causal test — pre-registration (2026-09-29, written before any §5b generation or rating)

## Why this revision
§5 (anchor-only intervention) failed on validation: even full anchor substitution moved rated anger by only +0.06 [−0.04, 0.16]
(prefix ceiling +1.21). A one-shot change at the anchor positions cannot steer the generated text, because the neutral lead stays
in context. §5b therefore intervenes **continuously**: at the anchor positions and at every generated position.
No test data has been opened; the test split is still sealed.

## Object (frozen from B3, discovery only)
The 12 attention heads selected in B3 by the pre-registered rule (k for 50% of window effect ≤ 16, bootstrap Jaccard 0.78):
L16H12, L17H16, L14H25, L19H28, L17H4, L19H29, L18H8, L22H19, L16H3, L14H26, L20H11, L16H17.

## Steering vector (discovery only)
For each head h: v_h = mean over discovery records of the head's output (o_proj input slice h, averaged over anchor tokens)
for **anger** minus the same for **neutral**. Intervention: add λ·v_h to that slice at the anchor positions and at every generated position.

## Inputs
Recipient: neutral style_lead + anchor_fact + "我的看法是：" (same as §5). Greedy decoding, 64 new tokens.

## Conditions
- none (no steering); primary_prefix (the anger lead itself = ceiling)
- target: λ·v_h on the 12 heads
- random_i: the same 12 heads, random Gaussian directions per head, scaled so each head's residual-stream change norm
  ‖W_o[:, h]·vec‖ equals the target's (validation: 2 seeds; test: 5 seeds 101–105)
- shuffled: v_h recomputed with emotion labels shuffled within scenario (seed 201), norm-matched per head the same way

## Validation selection (validation split, paraphrases A+B, 20 scenarios)
λ grid {1, 2, 4, 8}. Two blind LLM raters (same fragment protocol as §5).
Quality gates vs none: fluency difference CI lower > −0.3; factual-consistency difference CI lower > −5 pp.
Pick, among λ passing the quality gates, the largest target − random (anger); if within 0.1, the smaller λ.
If no λ passes: do not freeze, do not open test, report.

## Test (after freeze; paraphrase A, 40 scenarios; 9 conditions = 360 continuations)
Gates (all must hold; LLM raters are provisional, not the human raters of the original plan):
1. target − mean(random) on anger ≥ 0.5 /4 and 95% CI lower > 0
2. target − none on anger: CI lower > 0
3. target − shuffled on anger: CI lower > 0
4. specificity: rise in anger > rise in sadness and in joy (target − none), CI lower > 0
5. quality: factual-consistency CI lower > −5 pp; fluency CI lower > −0.3
A too-wide CI → "inconclusive", not pass or fail.

---
## Deviation record (2026-09-29, after validation blind rating, BEFORE any test-split generation)

**What happened**: no λ passed the validation quality gates, so under the original rule the test split could not be opened.
All four λ failed on the factual-consistency gate (CI lower bound > −5 pp). With 20 validation scenarios, the CI half-width is about ±14 pp even when the mean change is exactly 0.00 (λ=2), so this gate **could not be passed at this sample size**. That is a power error made at pre-registration, not evidence of harm.

**Validation results (for the record)**: target − random (anger) for λ=1/2/4/8 was +0.13 / **+0.88 [0.62, 1.17]** / +2.43 / +2.88.
Fluency change was −0.01 / **−0.03 [−0.25, 0.20]** / −0.44 [−0.69, −0.20] / −1.11. Consistency change was +0.01 / **0.00 [−0.14, 0.14]** / −0.05 / −0.14.

**Change (approved by the user)**: quality gates become three-valued, both in validation selection and at test:
PASS if the CI lower bound is above the threshold; **FAIL if the point estimate (mean) is already worse than the threshold**; otherwise INCONCLUSIVE (mean acceptable but CI too wide), which does not block. (This is the rule approved by the user: "means not worse + a CI too wide counts as undetermined".)
Applied to validation: λ=1 (fluency mean −0.01, consistency +0.01) and λ=2 (−0.03, 0.00) are INCONCLUSIVE on consistency and PASS/INCONCLUSIVE on fluency → not failing; λ=4 (fluency mean −0.44 < −0.3) and λ=8 (−1.11) FAIL fluency. Among non-failing λ, the largest target − random is **λ=2** → frozen.

*Correction note*: a first draft of this paragraph (written minutes earlier, before any test generation) used "FAIL if CI upper < threshold", under which λ=4 would not have failed; it contradicted the rule the user approved and was replaced by this version before re-freezing.

**Known limitation of a control (noted before the test)**: the within-scenario label-shuffle vector keeps part of the true anger direction; after norm-matching (×8.4) it also elicits anger, so target − shuffled underestimates specificity. The test still reports it, and its gate is interpreted with this caveat.

The test split (work, hobby) had not been generated, read or rated at the time of this record.
