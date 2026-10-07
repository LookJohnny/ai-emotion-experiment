# E05 pre-registration: within-family transfer of steering heads by index (Qwen3-4B → Qwen3-8B), confirmatory

Written 2026-10-07. At the time of writing:
- no Qwen3-4B two-stage screen had been run;
- no Qwen3-8B two-stage screen had been run;
- no E05 generation or rating existed.

## Hypothesis and its origin

E03's exploratory screen produced the hypothesis. E01 had selected 12 heads in Qwen3-4B by activation patching. Steered at the same layer and head indices in Qwen3-8B (same shape, 36 × 32), with 8B's own vectors, these heads gave the largest teacher-forced effect in that screen. That screen used Qwen3-8B **validation** inputs only, teacher-forced, with no generation and no rating.

**E05 tests the patent's dependent claim as written.** Heads chosen in model 1 by the two-stage method (E04's method), applied by index to model 2 with model 2's own vectors, steer model 2.

## Models and data

- **Source**: Qwen3-4B-Instruct-2507.
- **Target**: Qwen3-8B, with `enable_thinking=False`.
- Both in bf16, greedy decoding, 64 new tokens.
- **Data**: E01's `formal_960.jsonl`.

| Split | Used for |
|---|---|
| discovery | maps, vectors and screens in both models |
| validation | λ for the target model |
| test | the single final evaluation |

**Test-split history**: the Qwen3-8B test split was used in E03 for other conditions (map-only heads). None of E05's conditions have been generated or rated on it, and the hypothesis was formed without it.

## Selection (discovery only), frozen before any E05 generation

1. **Source two-stage set (`transfer`)**: run E04's `map_select.py` + `screen.py` unchanged on Qwen3-4B. Keep up to 12 heads by the E04 rule: effect > 2 SE among the map's top 60, ranked by effect. The **indices** of these heads are the transfer set.
2. **Target native two-stage set (`native`)**: the same pipeline on Qwen3-8B, using E03's frozen 8B map (`2026-10-07/map`, unchanged).
3. **Vectors**: for every steered head in the target model, its own Qwen3-8B anger − neutral vector from E03's frozen file.
4. **Freeze**: both sets and the overlap between them go into `TRANSFER_FREEZE.json`, with hashes.
5. **Stop rule**: if the source set has fewer than 4 heads, stop.

## Conditions on Qwen3-8B (validation and test)

Steering is continuous: at the anchor positions and at every generated position. The recipient input is the neutral lead + anchor.

| Condition | What it is |
|---|---|
| `none` | no steering |
| `primary_prefix` | the anger lead itself; this is the ceiling |
| `target` | the transfer set, λ·v_h. Its summed residual-change norm T sets the push for every head-set control below |
| `native` | the 8B native two-stage set, rescaled to total push T |
| `rheads_j` | 12 random heads outside the transfer set (seeds 301–305), each with its own vector, rescaled to total push T |
| `rdir_i` | the transfer heads with random directions, norm-matched per head (seeds 101–105) |

## Validation (choose λ)

- 20 scenarios, paraphrase A.
- λ grid {1, 2, 4, 8}.
- rdir seeds 101–102 and rheads seeds 301–302 at validation.
- Two blind LLM raters, E01's continuation-fragment protocol.
- **Quality gates**: three-valued, as in E04 (fluency threshold −0.3, consistency threshold −5 pp; FAIL if the mean is already worse than the threshold).

**Pick**: among λ that do not FAIL, the largest `target − mean(rdir)` on anger. If two are within 0.1, take the smaller λ.

**Freeze**: `FREEZE_E05.json`, written before test generation.

## Test (opened once): 40 scenarios, paraphrase A

| Gate | Test (anger unless stated) | Requirement |
|---|---|---|
| **G1** | target − mean(rdir) | ≥ 0.5 / 4 and CI lower bound > 0 |
| **G2** | target − none | CI lower bound > 0 |
| **G3** | target − mean(rheads) | CI lower bound > 0 |
| **G4** (non-inferiority to selecting in the target model itself) | target − native | **PASS** if the CI lower bound is > −0.5; **FAIL** if the mean is < −0.5; otherwise INCONCLUSIVE |
| **G5** (specificity) | (target − none) on anger minus the same on joy, and minus the same on sadness | each CI lower bound > 0 |
| **G6** (quality) | fluency and consistency | three-valued; FAIL blocks |

All tests use scenario-level means and a bootstrap over scenarios (10,000 draws, 95% CI).

**Reported, not gated**:
- the index overlap between the transfer and native sets;
- each set's layers;
- native − none.

**Interpretation, fixed now**:

| Outcome | Reading |
|---|---|
| G1–G3, G5, G6 PASS and G4 PASS | Index transfer works and is not meaningfully worse than selecting in the new model. This supports the dependent claim |
| G1–G3 PASS, G4 FAIL | Transfer steers, but clearly worse than native selection |
| G1–G3 FAIL | The claim is not supported |

## Known limits

- One family pair.
- LLM raters.
- The same dataset as E01–E04.
