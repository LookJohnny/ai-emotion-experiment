# E05 report: within-family index transfer (Qwen3-4B two-stage heads → Qwen3-8B). Partial pass

2026-10-07. One RTX 4090.

| Step | Commit |
|---|---|
| Pre-registration | 842e62d |
| Selection frozen | 4c59aba |
| λ frozen | d2544cc |
| Test | opened once |

Raters: two blind LLM streams.

## Selection (discovery only)

| Set | Heads |
|---|---|
| **Transfer set** (two-stage method on Qwen3-4B) | L22H19 L19H29 L16H12 L17H16 L17H18 L22H17 L16H3 L14H26 L15H27 L12H27 L19H28 L15H4 |
| **Native set** (two-stage method on Qwen3-8B) | L14H26 L19H29 L17H16 L22H19 L18H9 L15H18 L11H30 L15H17 L22H7 L22H17 L16H3 L10H16 |

- **Overlap**: 6 of 12 indices are the same in both sets.
- **Descriptive**: 7 of the 4B transfer heads are among the 12 heads E01 found in 4B by activation patching. The two-stage method and patching largely agree.

## Validation: why λ = 1

On validation, transfer and native heads steered almost identically at every λ (difference within ±0.2):

| λ | target − random directions | factual-consistency change | quality gate |
|---|---|---|---|
| 1 | +0.25 | +2.5 pp | passes |
| 2 | +1.29 | −7.5 pp | fails |
| 4 | +2.09 | −15 pp | fails |
| 8 | +2.95 | −20 pp | fails |

On Qwen3-8B, λ ≥ 2 makes the continuations add facts beyond the pre-registered −5 pp limit, so the rule chose **λ = 1**.

Qwen3-14B (E04) tolerated λ = 8 (−5 pp, inconclusive). The quality cost depends on the model.

## Test (40 scenarios, 560 continuations, λ = 1)

| Gate | Result | Verdict |
|---|---|---|
| G1: transfer − random directions ≥ 0.5 | +0.25 [0.11, 0.39] | **FAIL**: positive but small |
| G2: transfer − none | +0.24 [0.08, 0.40] | PASS |
| G3: transfer − random heads (same push) | +0.21 [0.08, 0.35] | PASS |
| **G4: transfer not worse than native by 0.5** | **+0.06 [−0.04, 0.18]** | **PASS**; transfer was even slightly ahead |
| G5: specificity vs joy / vs sadness | +0.35 [0.14, 0.56] / −0.21 [−0.55, 0.16] | **FAIL**: the sadness CI spans 0 |
| G6: quality | fluency +0.01, consistency 0.00 | PASS |

Native − none was +0.18 [0.03, 0.33]. The anger lead itself (ceiling) was +1.11.

**Overall verdict: not all gates passed.**

## What this establishes

1. **Transferred heads steer as well as heads selected in the target model.** This was the question the dependent claim rests on, and the non-inferiority gate passed on a sealed test.
2. **Both sets beat random heads at equal push.** So the transferred indices carry real head identity, not just any push in the anger direction.
3. **What did not pass is the size and specificity of the effect at the only λ this model allowed.** At λ = 1 the push is weak:
   - transfer and native both raise anger by about 0.2;
   - the rise is not clearly above the rise in sadness.

   This is a property of steering Qwen3-8B within the factual-consistency limit, not of the transfer: native selection was no better.
4. **For the patent:** support the dependent claim with G4 (non-inferiority) and G2/G3, and state the λ / quality trade-off honestly. Do not claim large effects for index transfer on 8B.

## Limits

- One family pair.
- LLM raters.
- The same dataset as E01–E04.
- Qwen3-8B's test split had been used in E03 for other conditions.
