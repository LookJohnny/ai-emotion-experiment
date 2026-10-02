# Pre-registration: universal probes via automatic adaptation (2026-10-02)

Written before any download, feature extraction or training. Product question:
- When Nous meets a local open model it has no pack for, can a pack be generated **automatically, with no new labels**?
- Is it good enough to ship?

## Background
Probes are linear directions in one model's activation space and do not transfer as-is. Two label-free routes are compared:
- **(a) Native retrain.** The labels we hold (hostility, sycophancy, fabrication) describe *text*, so the existing labeled dev texts are teacher-forced through the new model and its own probes are trained. This needs no new rating, but it does need the labeled corpus and some GPU hours.
- **(b) Align-transfer.** The new model and the reference model (Qwen3-4B-Instruct-2507) read the same **unlabeled** anchor texts. A linear map from the new model's states to the reference's is fitted, and the reference's frozen probe is applied through it. This needs no labels at all, and could later run on a user's machine.

## Models
**Target models**, open, ≤ 10B, popular, deliberately diverse:
1. Qwen3-8B (same family, non-2507) — already on disk.
2. Llama-3.1-8B-Instruct (different family, English-centric).
3. DeepSeek-R1-Distill-Qwen-7B (reasoning distillation).
4. GLM-4-9B-0414 or glm-4-9b-chat-hf (Chinese, different lab). Whichever loads natively in transformers 4.57; recorded before results.

Models are loaded in bf16, with chat templates as shipped. Thinking is disabled where the template supports it.

**Reference:** Qwen3-4B-Instruct-2507 and its frozen probes:
- `c1_token`, layer 16;
- `syco_tok`, layer 20;
- `c3h_token`, layer 19.

## Data (all labels already exist; no new rating)

| Concept | Dev, used by (a) and as (b)'s anchors with labels ignored | Test (sealed for this question) |
|---|---|---|
| C1 hostility | V1.1 C1 dev, Qwen replies (400) | V1.1 C1 test (240, 38 positive); secondary: Kimi C1 test (143, 18 positive) |
| SYCO | Qwen syco dev (727 labeled) | Qwen syco test (506, 94 positive); secondary: Kimi syco test (498, 17 positive) |
| C3-H fabrication | C3-H dev, Qwen answers (591) | C3-H test, Qwen answers (354); secondary: Kimi-val (223) |

Every text is read by the target model exactly as by the reference: same system prompt and history, reply teacher-forced. Pooling is the mean over reply tokens (first 120; first 64 for C3-H).

## Methods, fixed in advance
- **(a) Native.** For each concept, standardise + LR (C = 0.05) on the target's dev features. The layer is chosen by 5-fold CV grouped by template/category, over 7 layers at relative depths {0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70}.
- **(b) Align-transfer.**
  - Target layer = round(ref_layer / 36 × N_target).
  - Map W: ridge regression (α chosen by 5-fold CV on anchors, from {1, 10, 100, 1000}) from standardised target states to reference states at the probe layer.
  - Anchors = dev texts of all three concepts, pooled the same way. Labels are never used.
  - Score = reference probe(W · h_target).
- **(b′) reported, not gated:** the same with anchors restricted to 300 dev texts per concept, the size a user's machine can do in minutes.

## Gates (per target model × concept, on the main test set; bootstrap by template/category, 2000 draws; strict rule)

| # | Content | Threshold |
|---|---|---|
| U1 | AUROC(b) − AUROC(a) | CI lower bound ≥ −0.03 (non-inferior) |
| U2 | AUROC(b) absolute | C1 and SYCO ≥ 0.80; C3-H ≥ 0.70 |

**Product decision rule** (written now):
- A target-model × concept with U1 and U2 both PASS can be auto-adapted by alignment.
- If (a) passes U2 but (b) fails U1, ship native retrain as a server-side "auto pack" service instead.
- If both fail, that model needs a full pack build.

**Reported:** the reference reading the same test texts; secondary test sets; (b′); every per-layer number.

## Deviations
(none)
