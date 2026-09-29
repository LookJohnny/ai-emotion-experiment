# E01 Final Report: Localizing, causally testing and reading out Chinese emotional expression in an LLM

Model: Qwen3-4B-Instruct-2507 (BF16, HF commit `cdbee75f`) · Hardware: one RTX 5090 · Period: 2026-09-28 – 09-29
Status: research prototype. **All emotion scores come from blind LLM raters (two independent streams per round, inter-rater r ≈ 0.72–0.98). They have not been validated by human raters.**

---

## 1. Conclusions in one paragraph

In Qwen3-4B, the emotional tone of Chinese expression does **not** depend on a handful of "emotion neurons". Emotion information is spread across most MLP neurons (70–76% per layer). Erasing the most anger-selective 1024 neurons leaves the tone almost unchanged. The unit that works is a **small set of attention heads**: 12 heads in the middle layers (L14–L22) carry the emotion of the opening sentence into later text. If a steering vector is added **continuously** on these 12 heads, on the sealed test set the anger of the continuation rises by **+1.19/4** relative to random directions matched in magnitude, **[0.96, 1.42]**. That is on par with directly writing an angry opening sentence (+1.01). The effect is specific to anger and does not hurt fluency. **Two frozen gates were not met**, however: the factual-error rate rose by 6 points on the point estimate (CI spans 0), and the difference against the label-shuffled control was not significant, because that control has a known design flaw. On the product side, a real-time readout of the model's internal state tracks the emotion the model actually expresses (ρ 0.54–0.71 with blind ratings), and just before a reply starts it can already predict the upcoming tone (ρ 0.75–0.90).

---

## 2. Data

- **960 formal records** = 120 scenarios × 4 emotions (joy / sadness / anger / neutral) × 2 paraphrases, plus 48 control records. Each record contains: an opening sentence with the emotion (style_lead) + a fact anchor identical word for word within a scenario + a reference continuation. Paraphrase B contains no emotion-category words.
- **Splits (by topic)**: discovery 60 (shopping, travel, family) / validation 20 (study) / **test 40 (work collaboration, hobbies), sealed until 09-29 11:25**.
- **Emotion fit**: v1 events were purely good or purely bad news, so four emotions were forced onto them (in the blind check only 6/120 scenarios had all three emotions rated ≥3). v2 rewrote every event as a **"three-element mixed event"**: a real gain, the loss of something the speaker valued, and an improper decision by someone else. After 8 rounds of independent blind rating, **all 120/120 scenarios score ≥3 on all three emotions**.
- Details: `data/README.md`, `data/manifest.json`.

## 3. Stage A: does emotion carry over into the continuation? (ceiling)

| Validation (vs neutral opening) | Effect /4 | 95% CI |
|---|---|---|
| Joy opening → joy | +0.35 ~ +0.43 | lower bound 0.19–0.25 |
| **Anger opening → anger** | **+1.08** | [0.71, 1.45] |
| Discovery confirmation: joy | +0.20 | [0.08, 0.32] ✗ |
| **Discovery confirmation: anger** | **+1.15** | [0.96, 1.33] ✓ |

The joyful opening barely carries over: the model brings the "loss" in the event back up by itself. An 8-cell grid of anchor endings × lengths could not rescue it. → **The main contrast was changed to anger vs. neutral** (plan v2.1, documented; details in `stageA/STAGE_A_REPORT.md`).

## 4. Stage B: where is the emotion? (discovery set, teacher-forced recovery, i.e. how close the model gets to the donor's probability of generating the same continuation)

| Finding | Evidence |
|---|---|
| Emotion is concentrated in the middle-layer residual stream | B1: replacing the whole anchor residual peaks at anger L17 / sadness L18 / joy L19, recovery 0.32–0.36 (cross-emotion 0.05–0.11) |
| **Carried by attention, not MLPs** | B2: attention outputs replaced simultaneously across L0–35 recover **0.65**; **MLP replaced alone across layers is negative** (−0.30) |
| **12 attention heads do most of the transport** | B3: within L12–27, 12 heads reach 50% of the window effect and 24 heads reach 80%; bootstrap Jaccard 0.78. Key head **L14H25**: affects almost only anger; 48% of its attention falls on the opening sentence |
| A single residual direction also works | B4: direction at L17 recovers 0.11 (λ=1) → 0.24 (λ=4) with negative cross-emotion effect; random direction ≈0 |
| Neurons: dense, mixed, "whose emotion" coded separately | Census of 350k neurons: 70–76% emotion-selective per layer but each explains ~9% of variance; some strongly tuned neurons are stable across topics (L12#8223, L14#6962); "I am angry" and "the user describes something infuriating" overlap only at r≈0.2 |
| **Erasing anger neurons does not work** | Ablation: top-32/256/1024 → anger barely changes (only top-256 beats random, −0.063); the wording changes but the tone does not |

Details: `stageB/CENSUS_REPORT.md`, `stageB/ABLATION_REPORT.md`, `stageB/results/`.

## 5. Stage C/D: causal test (preregistered, frozen, sealed test set)

### 5.1 §5 anchor-only intervention (negative, test set not opened)
Replacing the 12 heads **only at the anchor positions** barely changed the generated text on validation: target − random +0.04 [−0.07, 0.16]. Even **replacing the entire anchor** gave only +0.06 (the opening-sentence ceiling is +1.21). Reason: the neutral opening sentence is still in context and the model keeps attending to it. **Conclusion: a one-off intervention at a local position can shift probabilities (the B2 teacher-forced level) but cannot change generated behaviour.**

### 5.2 §5b continuous steering (the main test)
**Design** (`stageD/STEER_PREREG.md`, written before any result): on the 12 heads, add λ · v_h at the anchor positions **and at every generated token**, where v_h is the mean difference of head outputs between anger and neutral on the discovery set. Controls: random directions per head matched in magnitude (5 seeds); a vector from label-shuffled data.

**Validation** (20 scenarios × A/B, 2 raters): clear dose-response.

| λ | target − random (anger) | fluency | factual consistency |
|---|---|---|---|
| 1 | +0.13 | −0.01 | +0.01 |
| **2** | **+0.88** [0.62, 1.17] | −0.03 | 0.00 |
| 4 | +2.43 | **−0.44** | −0.05 |
| 8 | +2.88 | **−1.11** | −0.14 |

**Pre-registration deviation (user-approved, recorded and pushed before the test set was opened, commit `8d19eb3`)**: the original quality gate (consistency CI lower bound > −5pp) could not be passed by any λ at n=20, even with a true effect of 0. It was changed to three levels: "fail if the mean is worse than the threshold; pass if the CI lower bound is above it; otherwise undecided". → **λ=2 frozen**.

**Post-freeze bug fix**: `final_eval.py` crashed on its first line (an inline comment swallowed three statements). The fix only moved the comment; no logic changed; the change was made before any test statistic was computed. sha `73ebb0b` → `da7fa14`, recorded in STEER_PREREG.md.

**Test set** (40 scenarios, paraphrase A, 360 continuations, 2 raters, r 0.84–0.97):

| Gate | Result | Verdict |
|---|---|---|
| target − random ≥ 0.5 and CI > 0 | **+1.19** [0.96, 1.42] | ✅ |
| target − no intervention CI > 0 | **+1.19** [0.95, 1.43] | ✅ |
| Specificity (anger rise > joy / sadness rise) | +1.38 / +0.96, CI both > 0 | ✅ |
| Fluency | +0.06 [−0.13, 0.28] | ✅ |
| Factual consistency (mean must not be worse than −5pp) | **−6.2pp** [−19, +5] | ❌ |
| target − label-shuffled | +0.19 [−0.13, 0.49] | ❌ |

Reference: angry opening (real ceiling) +1.01; mean anger 0.35 with no intervention, **1.54 with steering**.

**How to read the two failures:**
- *Factual consistency*: the fabrication rate rose from 20% to 26%. The CI spans 0, but by the frozen rule a mean worse than −5pp counts as a fail. For context, **an angry opening by itself also raises it to 25%** (stage A discovery set: anger continuations fabricate facts 13 points more often). Angry text more readily "embellishes the facts"; this may be a property of the emotion itself rather than a side effect of steering. The pre-registered rule still counts it as a fail, however.
- *Label-shuffled control*: shuffling labels only among the 4 emotions still leaves part of the true anger direction in the vector. After magnitude matching (×8.4) it also produces anger (1.35). This defect **was written into the pre-registration before the test**. Target vs. random directions (+1.19) and specificity show that the effect is not "any perturbation makes it angrier", but the rule's verdict is still a fail.

**Overall verdict: not all gates passed.** Steering 12 attention heads **strongly, specifically and fluently** changes the model's expression of anger on the sealed test set. It fails the factual-consistency non-inferiority gate, and the label-shuffled control is not interpretable.

## 6. Product prototype: real-time emotion readout (`product/`)

| Component | Result |
|---|---|
| Per-token readout (L19 probe, v2) | Blind-rating ρ: anger 0.71 / joy 0.62 / sadness 0.54; pure-task replies are no longer misread as angry (0.28 → 0.01) |
| **Pre-reply readout** (L22 probe, v2, predicts upcoming tone just before the reply starts) | Test ρ: anger 0.75 / joy **0.82** / sadness 0.90; beats a "user message only" baseline only on joy (+0.09) |
| Service | FastAPI + WebSocket streaming, 31–45 tok/s, first token 0.05 s; outputs VRM expression weights (happy/sad/angry/neutral) per token |

Implication: the prototype can make **the face move before the text** (e.g. on a complaint, "anger 0.38" is readable before the first character). The v2 readout is more conservative, so expression-mapping thresholds need tuning before connecting to SoulForge `/live`.

## 7. Limitations

1. **The raters are LLMs**, not people. Every score needs human spot checks before it becomes a confirmed conclusion.
2. **Single model** (Qwen3-4B), not replicated on other models.
3. **Main contrast changed after seeing data** (joy → anger), confirmed on the discovery set; **§5b has one pre-registration deviation plus one post-freeze bug fix** (both recorded and timestamped).
4. The data was drafted by LLMs and checked by LLM blind raters. Human review of `review_sheet.csv` has not been done.
5. Sample size is small (validation 20, test 40 scenarios); non-inferiority gates are underpowered.
6. §6 (free-generation transfer) was **not run**. §5b uses an input with a neutral opening sentence, not a pure dialogue setting.

## 8. Recommended next steps

- **Product**: tune expression thresholds → connect to SoulForge `/live` → a human perception A/B test ("readout-driven expression" vs "text sentiment analysis").
- **Research**: human raters on a subsample (~100 items) to calibrate the LLM raters; add a factual-consistency constraint to steering (e.g. restrict it to the first N tokens); replace the label-shuffled control with one orthogonal to the true direction; replicate on 2–3 more models; train an SAE on L12–27 to find features rather than neurons.

## Appendix: Key files

`EXPERIMENT.md` (plan v2.1) · `data/` · `stageA/STAGE_A_REPORT.md` · `stageB/{CENSUS,ABLATION}_REPORT.md` ·
`stageD/STEER_PREREG.md` (design + deviation + fix record) · `stageD/FREEZE_5b.json` · `stageD/rating_steer_test/final_results.json` ·
`product/` (readout / pre-readout / service / demo)
