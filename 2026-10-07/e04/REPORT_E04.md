# E04 report: two-stage head selection (map prefilter + sign screen) on Qwen3-14B. Confirmatory: all gates pass

2026-10-07. One 48 GB RTX 4090.

| Step | Commit |
|---|---|
| Pre-registration | d1958bd (before any 14B data) |
| Map | 73bd081 |
| Screen frozen | 5bed906 |
| λ frozen | ee2118e |
| Test | opened once |

Raters: two blind LLM streams. Agreement on validation: anger r = 0.96, joy 0.88, sadness 0.75, fluency 0.59, fact error 0.97.

## Selection (discovery only)

- **Map**: residual peak at relative depth 0.436.
- **Sign screen**: single heads at equal push.
  - Map top 60: 27 significantly positive, 13 significantly negative.
  - 60 random heads: 6 positive, 19 negative.
- **Two-stage set**: the 12 strongest positive heads from the map's top 60. L15H14 L18H14 L16H25 L17H18 L15H11 L27H22 L22H28 L16H17 L25H16 L16H11 L16H29 L18H13.
- **Screen-only set**: the 6 positive heads found among the random 60.

**λ chosen on validation by the pre-registered rule**: 8. It had the largest two-stage − random-direction effect on anger: +1.50, against +1.15 at λ = 4.

## Test (40 scenarios, 600 continuations, λ = 8)

| Contrast (anger, /4) | Mean | 95% CI | Gate |
|---|---|---|---|
| Two-stage − random directions | **+1.90** | [1.60, 2.18] | G1 ✓ |
| Two-stage − none | **+1.76** | [1.41, 2.09] | G2 ✓ |
| Two-stage − random heads (same total push) | **+1.74** | [1.40, 2.05] | G3 ✓ |
| Two-stage − map-only top 12 (E03's method) | **+0.71** | [0.40, 1.04] | G4 ✓ |
| Specificity: vs joy / vs sadness | +2.00 / +1.38 | both CIs > 0 | G5 ✓ |
| Quality: fluency / factual consistency | −0.29 / −0.05 | both INCONCLUSIVE (mean not worse than the threshold) | G6 ✓ |
| Secondary: two-stage − screen-only | +1.81 | [1.46, 2.13] | the map prefilter adds value |
| Ceiling: anger lead − none | +1.11 | [0.76, 1.44] | |

**Mean anger by condition:**

| Condition | Mean anger |
|---|---|
| none | 0.46 |
| anger lead | 1.58 |
| **two-stage** | **2.23** |
| map-only | 1.51 |
| random heads | 0.49 |
| random directions | 0.33 |
| screen-only | 0.41 |

**Verdict.** On a model never used before, two-stage selection picks heads whose continuous steering raises expressed anger strongly and specifically. The effect is larger than writing an angry opening sentence. It beats random heads at equal push, random directions, and the map alone (E03's method).

- The map prefilter matters. Screening 60 random heads found only 6 weak positive heads, and steering them did nothing.
- On this model the map alone also works partly (+1.05 over none). E03 showed it can fail when opposing heads sit in the top 12, which is what the second stage removes.

## Costs and caveats

- **Fluency** −0.29 and **factual errors +5 pp** (target 16% vs none 11%) at λ = 8. Both are inside the pre-registered tolerance but close to its edge. E01 saw the same pattern: angry text embellishes.
- In applications, λ is a product choice. On validation, λ = 4 gave about +1.15 with smaller costs.
- Raters are LLMs; human calibration has not been done.
- One dataset: E01's Chinese mixed-event texts.
- Screen cost: about 24 minutes for 120 heads on one 4090.

## Where this leaves the method

| Experiment | Model | Finding |
|---|---|---|
| E02 | 4B | The map recovers causal heads (top 10%) |
| E03 | 8B | The map's top 12 alone fails, because opposing heads cancel; exploratory analysis found the map selects relevance, not sign |
| E04 | 14B | Map prefilter + sign screen passes every pre-registered gate |

The patent's core claim should be the two-stage method.
