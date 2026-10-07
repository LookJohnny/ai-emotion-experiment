# E06 report: two-stage head selection vs GCM attribution patching vs ITI (Qwen3-14B). Our method is dominated by GCM

2026-10-07. One 48 GB RTX 4090; the instance was shut down after generation.

| Step | Commit |
|---|---|
| Pre-registration | ae874a1 |
| Code | aae18c7 |
| Selection frozen | cdd9bb8 |
| Test | opened once |

Two blind LLM raters, anger agreement r = 0.94. 40 test scenarios, 640 continuations, every condition rated in one shuffled batch.

## Selection (discovery only) and cost on the same GPU

| Set | Heads (12) | Selection cost |
|---|---|---|
| twostage (E04, frozen) | L15H14 L18H14 L16H25 L17H18 L15H11 L27H22 L22H28 L16H17 L25H16 L16H11 L16H29 L18H13 | about 30 s map + 757 s sign screen of 60 heads |
| **gcm_atp** | L21H16 L19H10 L20H1 L22H28 L21H25 L19H12 L21H34 L24H9 L23H0 L24H7 L20H25 L20H4 | **33 s for all 1,600 heads** (2 forward + 1 backward per pair, 120 pairs), plus 286 s of self-generation |
| iti | L3H18 … L12H2 | 6 s. **All of the top 30 tied at validation accuracy 1.0**; the pre-committed tie-break (layer, head) picked the earliest layers |
| twostage_self | L22H28 L19H10 L16H25 L21H16 L21H34 L20H39 L24H17 L15H14 L17H18 L13H7 L25H16 L22H27 | 757 s, plus the same self-generation |

Overlap: twostage ∩ gcm = 1 head; twostage_self ∩ gcm = 4; twostage_self ∩ twostage = 5.

## Results (anger, /4; scenario bootstrap)

**Mean anger by condition:**

| Condition | λ = 8 (primary) | λ = 4 |
|---|---|---|
| **gcm_atp** | **3.01** | **2.28** |
| twostage_self | 2.30 | 1.36 |
| twostage | 2.03 | 1.21 |
| iti | 0.50 | 0.43 |
| random heads | 0.43 | 0.39 |
| none | 0.33 | 0.33 |
| anger lead (ceiling) | 1.38 | 1.38 |

**Pre-registered hypotheses:**

| Hypothesis | λ = 8 | λ = 4 |
|---|---|---|
| **H1** twostage − gcm_atp | **−0.99 [−1.38, −0.64] → worse** | −1.06 [−1.39, −0.74] → worse |
| H2 twostage − iti | +1.53 [1.23, 1.81] → pass | +0.79 → pass |
| H3 twostage_self − twostage | +0.28 [0.03, 0.54] → better | +0.15 [−0.08, 0.36] → non-inferior |

**Quality change against none** (fluency /5, factual consistency):

| Method | λ = 8 | λ = 4 |
|---|---|---|
| gcm_atp | −0.04, −3.7 pp | −0.21, −6.2 pp |
| twostage | −0.34, −6.2 pp | −0.08, +1.3 pp |
| twostage_self | −0.28, −15 pp | −0.14, −7.5 pp |
| iti | −0.38, −16 pp | −0.23, +5 pp |

## What this means

1. **On this task, GCM's attribution patching picks clearly better steering heads than our two-stage method.** The effect is about +1.0 larger at both λ, the cost to fluency and facts is smaller, and selection is about 20× faster. By the interpretation fixed in the pre-registration, our selection is dominated, and **the patent's core claim should be reconsidered before filing.**
2. **Why GCM wins (likely).** It scores a head by its effect on the model's own long continuation, with a gradient through the whole downstream network, over all 1,600 heads. Our stage 1 is a correlational filter: it keeps heads where emotion is readable at the anchor. GCM's top heads (L19–L24) are mostly *not* in that filter. Only 1 of 12 overlaps our set.
3. **Borrowing from GCM helps our method** (H3: self-generated references, +0.28 at λ = 8). But it does not close the gap.
4. **ITI, as run here, is uninformative.** Anger versus neutral is trivially separable at the last prompt token, so probe accuracy saturates and cannot rank heads. This is a known weakness of accuracy-based selection, not a fair test of probe methods in general.
5. **What still stands from E02–E05:**
   - the anchor design and the descriptive findings (mid-depth carry, low variance with high readability, the valence split);
   - the anchor map as a cheap **forward-only** diagnostic, recovering E01's patching heads 8 of 12 in the top 10%;
   - the sign problem: readable heads include opposers (E03);
   - index transfer within a family (E05).

   None of these is a better steering selector than GCM.

## Limits

- One model, one attribute.
- LLM raters.
- The test scenarios were used in E04.
- GCM was adapted to patch only at anchor positions, because our leads differ in length. This adaptation, if anything, handicaps GCM, and it still won.
