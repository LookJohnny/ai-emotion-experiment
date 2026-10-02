# Universal probes via automatic adaptation — report (PREREG_TRANSFER.md) — 2026-10-02

Numbers: `transfer/report_<model>.json`. GPU: RTX 5090.

Two label-free routes for a new local model are compared:
- **(a) native retrain**: the existing labeled texts are teacher-forced through the new model and its own probes are trained;
- **(b) align-transfer**: the new model and Qwen3-4B read the same unlabeled texts, a linear map is fitted, and the frozen reference probes are reused through it. **Labels are never used.**

## Results (main test sets; AUROC; bootstrap by category, strict rule)

| Model | Concept | (a) native | (b) align | (b′) 300 anchors | U1: (b) not worse than (a) by > 0.03 | U2: (b) absolute |
|---|---|---|---|---|---|---|
| Qwen3-8B | Hostility | 0.933 | 0.925 | 0.923 | PASS | PASS |
| | Sycophancy | 0.776 | **0.828** | 0.832 | PASS | INCONCLUSIVE |
| | Fabrication | 0.805 | 0.821 | 0.818 | PASS | PASS |
| Llama-3.1-8B | Hostility | 0.924 | **0.933** | 0.933 | PASS | PASS |
| | Sycophancy | 0.753 | 0.795 | 0.747 | PASS | FAIL (0.795 < 0.80) |
| | Fabrication | 0.731 | 0.756 | 0.758 | PASS | PASS |
| DeepSeek-R1-Distill-Qwen-7B | Hostility | 0.927 | 0.923 | 0.923 | PASS | PASS |
| | Sycophancy | 0.768 | **0.832** | 0.812 | PASS | INCONCLUSIVE |
| | Fabrication | 0.731 | 0.748 | 0.750 | PASS | INCONCLUSIVE |
| GLM-4-9B-0414 | Hostility | 0.941 | 0.936 | 0.936 | PASS | PASS |
| | Sycophancy | 0.768 | **0.831** | 0.830 | PASS | INCONCLUSIVE |
| | Fabrication | 0.793 | 0.807 | 0.808 | PASS | PASS |

Reference, Qwen3-4B reading the same test texts: hostility 0.926, sycophancy 0.833, fabrication 0.811.

## Conclusions

1. **Label-free alignment is never worse than native retraining: U1 passes in all 12 of 12 model × concept cells.**
   - Hostility and fabrication are on par.
   - For sycophancy, alignment is **significantly better** than retraining on every model (+0.04 to +0.06).
   - Likely reason: the reference probe was learned on a strong, clean signal. A target model retrained from a few hundred dev examples learns a noisier probe.
   - This holds across four labs and four architecture lineages (Qwen, Meta Llama, DeepSeek distillation, Zhipu GLM).
2. **Absolute quality (U2) depends on the concept.**
   - Hostility passes on 4 of 4 models.
   - Fabrication passes on 3 of 4; DeepSeek-distill is INCONCLUSIVE.
   - **Sycophancy reaches a firm PASS on none**: three are INCONCLUSIVE at 0.83 points with CI lower bounds around 0.79, and Llama FAILs at 0.795. Native retraining is no better.
   - Sycophancy is the weakest concept, consistent with the earlier sycophancy study.
3. **300 anchors per concept are enough.** The small-anchor variant matches the full one except Llama sycophancy (0.747). So the alignment can plausibly run on a user's machine in minutes.
4. **Product decision, by the pre-registered rule:**
   - Every hostility cell, and fabrication on Qwen3-8B, Llama and GLM, is "auto-adaptable by alignment".
   - The Nous app can generate these readouts itself for a detected local model with no pack, from ~900 unlabeled anchor texts, without our team building a pack.
   - Sycophancy stays experimental everywhere.

## Limits

- **What has not been shown:**
  - All four targets are 7–9B open models read in teacher-forcing (reader) mode on our Chinese datasets.
  - "Pre-reply" probes, which read the model's state before it speaks, were not transferred here.
  - Models above 10B and quantized (GGUF) weights are untested.
- **The anchors are the dev texts of the same three tasks**, with labels unused. Anchors from a different distribution, for example generic chat, are untested.
- **Labels** come from blind LLM raters with no human spot check (see the earlier reports).
- **Run order:** Qwen3-8B results were seen before the other three models ran. The method and thresholds were not changed (recorded in Deviations).
