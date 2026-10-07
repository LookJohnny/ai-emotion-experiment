# E03 pre-registration: does the anchor head map pick causal heads in an 8B model?

Written 2026-10-07. At the time of writing:
- no Qwen3-8B feature had been extracted;
- no Qwen3-8B continuation had been generated;
- nothing had been rated.

## Question

E02 showed one thing on Qwen3-4B: a head ranking needing only a forward pass recovers the heads that E01 found by causal patching. The ranking is the within-scenario emotion share of each head's output, read at token-identical anchor positions.

E03 asks a stronger question on a new, larger model: **when we steer the heads the map picks, does the model's generated text change more than when we steer other heads with the same push?**

## Model and data

- **Model**: Qwen3-8B (ModelScope `Qwen/Qwen3-8B`), bf16, chat template with `enable_thinking=False`. Greedy decoding, 64 new tokens.
- **Data**: E01's `formal_960.jsonl`.

| Split | Scenarios | Used for |
|---|---|---|
| discovery | 60 | the map and the steering vectors |
| validation | 20 | choosing λ |
| test | 40 | the single final evaluation |

## Step 1: map (discovery only), frozen before any generation

- **Features**: `2026-10-06/feats.py` on discovery records (paraphrases A + B).
- **Score**: for every head, `within_share = SS_attr / (SS_tot − SS_scn)` at the anchor site, as in E02.
- **Selection rule**: the **top 12 heads** by within_share, with no layer restriction. 12 matches E01.
- **Steering vectors**: for each selected head, the mean over discovery (A + B) of its o_proj input slice, averaged over anchor tokens, for anger minus the same for neutral.
- **Freeze**: the selection, vectors and ranking are written to `MAP_FREEZE.json` with sha256 hashes before validation generation.
- **Reported, not gated**: the residual anchor emo_auc depth profile (E02's H1) on discovery.

## Step 2: conditions

Every steered condition adds the vector to the head's o_proj input slice at the anchor positions **and at every generated position**, as in E01 §5b. The recipient input is the neutral lead + anchor.

| Condition | What it is |
|---|---|
| `none` | no steering |
| `primary_prefix` | the anger lead itself; this is the ceiling |
| `target` | λ·v_h on the 12 map heads |
| `rdir_i` | the same 12 heads with random Gaussian directions; each head is norm-matched to the target's residual change ‖W_O[:, h]·v‖ (seeds 101–105) |
| `rheads_j` | 12 heads drawn uniformly at random from all heads not in the map set (seeds 301–305), each steered with **its own** true anger − neutral vector |

**rheads_j is the key new control.** Each random set is scaled by one common factor, so that its summed residual-change norm equals the target set's. It pushes in the true emotion direction with the same total strength, but at heads the map did not choose.

## Step 3: validation (choose λ)

- 20 scenarios, paraphrase A only.
- λ grid {1, 2, 4, 8}.
- 2 rdir seeds (101, 102) and 2 rheads sets (301, 302) per λ.
- Two blind LLM raters, using E01's continuation-fragment protocol (`2026-09-28/stageD/RATER_PROMPT_FRAGMENT.md`), on anonymous shuffled packets.

**Quality gates** against `none`, three-valued as in E01's approved deviation:
- fluency threshold −0.3;
- factual consistency threshold −5 pp;
- **PASS** if the CI lower bound is above the threshold; **FAIL** if the mean is already worse than it; otherwise **INCONCLUSIVE**, which does not block.

**Pick**: among λ that do not FAIL, the largest `target − mean(rdir)` on anger. If two are within 0.1, take the smaller λ. If every λ fails, stop and do not open test.

**Freeze**: λ, rater files and script hashes go into `FREEZE_E03.json` before test generation.

## Step 4: test (opened once)

- 40 scenarios, paraphrase A.
- Conditions: none, primary_prefix, target, rdir × 5, rheads × 5. That is 13 × 40 = 520 continuations, each rated by 2 raters.
- Scenario-level means; random seeds or sets pooled per scenario.
- Bootstrap over scenarios, 10,000 draws, 95% CI.

| Gate | Test | Requirement |
|---|---|---|
| **G1** | target − mean(rdir), anger | ≥ 0.5 / 4 and CI lower bound > 0 |
| **G2** | target − none, anger | CI lower bound > 0 |
| **G3** (the map's causal claim) | target − mean(rheads), anger | CI lower bound > 0 |
| **G4** (specificity) | (target − none) on anger minus the same on joy, and minus the same on sadness | each CI lower bound > 0 |
| **G5** (quality) | fluency and consistency vs none | three-valued, as in validation; FAIL blocks |

**Reported, not gated**:
- the effect size of target − mean(rheads);
- primary_prefix − none, as the ceiling;
- the anger level of each rheads set;
- the overlap of the map heads with the top 10% by emo_auc;
- the layers of the selected heads.

**Interpretation, fixed now**:
- **G3 PASS** supports "the anchor map selects heads that carry emotion causally, beyond any heads pushed with the same strength in the true direction". This is the patent's claim 1 and claim 10 on a second model.
- **G1 + G2 PASS with G3 FAIL**: steering works, but the map's head choice adds nothing over random heads at equal push.

## Known limits

- Raters are LLMs; human calibration has not been done.
- One 8B model.
- The test split was opened for E01 (Qwen3-4B steering) and used in E02 (analysis only). It has never been used with Qwen3-8B.
