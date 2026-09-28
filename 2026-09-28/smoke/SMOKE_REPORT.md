# E01 Smoke Test Report — 2026-09-28

Status: the debug loop runs end to end. **Debug data only (12 scenarios / 48 records). No research conclusions.** Raw artifacts are in `run_smoke_1201/`.

## Environment (recorded)

| Item | Value |
|---|---|
| GPU | RTX 5090, 31.36 GiB available, driver 595.71.05 |
| Software | Python 3.12.3 · torch 2.8.0+cu128 · transformers 4.57.6 · peft 0.21.0 (full list in `requirements.lock`) |
| Model | Qwen/Qwen3-4B-Instruct-2507, downloaded from ModelScope; sha256 of all 3 safetensors files **matches HF commit `cdbee75f17c01a7cc42f958dc650907174af0554`** exactly |
| Instance path | `/root/autodl-tmp/E01/` (models, runs, logs) |

## Today's completion criteria

| Criterion | Result | Evidence |
|---|---|---|
| Model runs correctly | ✅ | CUDA matmul, forward pass, and generation all succeed |
| 48-record data/template check | ✅ | Anchor and prompt identical within each scenario; labels absent from input; no tokenization merge at the anchor boundary; anchor token sequences identical across the 4 emotions; max 87 tokens |
| Blank-prefix check | ✅ | At the last shared-prefix token, activations are **bitwise identical** across the 4 emotions (max_abs_diff = 0 at all 4 layers); nearest-centroid classification = 0.25 (chance) |
| MLP intermediate capture | ✅ | down_proj forward_pre_hook, shape [48, 4 layers, 9728] |
| Zero-intensity intervention | ✅ | λ=0 vs. no hook: logit difference = 0, generations identical |
| Continuations with random controls | ✅ | 4 layers × 4 held-out scenarios × 8 conditions = 128 continuations |
| 10-step LoRA backward | ✅ | r=8 q/v, 2.95M trainable params, losses finite |

Time and memory: the main smoke run took 103 s end to end, with peak allocated memory 8.14 GiB (including LoRA training). About 25 min elapsed from connecting to finishing the analysis, mostly spent downloading the model. Actual GPU compute was about 5 min.

## Key finding: the primary intervention design is probably near-floor

In the main smoke run, patching the top-32 units at λ=1 on the last anchor token gave a next-token KL of only about 0.001, which is no larger than random sets. I then ran a dose/positive-control check (`dose.py`, `dose_check.json`) on all 12 debug scenarios. The reference ceiling is feeding the full joy prefix directly, which gives a mean KL of 0.652:

| Intervention | L17 mean KL | L26 mean KL | Share of ceiling |
|---|---|---|---|
| top-32 ×1, last token | 0.0011 | 0.0011 | ~0.2% |
| top-32 ×16, last token | 0.0133 | 0.0122 | ~2% |
| **All 9728 MLP units**, last token | 0.0153 | 0.0151 | ~2% |
| All 9728 MLP units, all anchor tokens | 0.0456 | 0.0217 | 3–7% |
| **Full residual stream**, all anchor tokens | 0.3637 | **0.6349** | 56% / **97%** |

What this means:
1. **Even swapping one layer's entire MLP only reaches about 2% of the effect.** So a single-layer, single-position, 32-unit patch will very likely fail the 0.5/4 gate *because of the method*, not because no signal exists. That would be a false negative.
2. The tone information *is* present at the anchor positions: swapping the residual stream at L26 recovers 97%. But it is written into the residual stream jointly by attention (reading back to the style_lead) and by many layers, not by one layer's MLP.
3. **The ceiling itself is low.** Even with the real joy prefix, continuations after "我的看法是：" often drift back to a factual, neutral register (for example d09 and d12). Joy ratings of continuations may not show a 0.5/4 difference even at the ceiling. In addition, the **baseline itself invents facts** (for example "北京市朝阳区", "市人民医院"), so the fact-error metric must be measured relative to the no-intervention baseline.

## Suggested revisions before the formal run (not yet applied to EXPERIMENT.md; your call)

1. **Add a "ceiling / positive control" stage first:** full joy prefix → human joy score for the continuation. If the ceiling itself is < 0.5/4, the gate is unreachable, and the anchor wording (e.g. "我的看法是：") or the continuation length needs changing first.
2. **Localize before going sparse:** run a residual-stream patching sweep over 36 layers × (last token / all anchor tokens) to find where the tone lives. Then split per layer into attention output vs. MLP output. Only then search for sparse units inside the component that matters.
3. **Make the unit-level candidate cross-layer:** top-k per layer over the key layer range, patched jointly at all anchor positions, instead of single-layer top-32.
4. Report every effect both in the form "share of ceiling KL / logit difference" and as human ratings. KL is cheap and computable in real time, so it suits validation-set tuning. Human ratings are used only on the test set.
5. Random controls: norm-matched scaling at L35 went as high as 8×. Flag this as OOD per the plan; it's worth adding a cap.

## Files

- `build_debug_data.py` / `debug_48.jsonl` (sha256 prefix 99c94b14): 12 debug scenarios, avoiding the 6 formal topics
- `smoke.py`: main smoke test; `dose.py`: dose/positive-control check
- `run_smoke_1201/`: smoke_log.json (environment, timings, memory, all checks), generations.jsonl, unit_sets.json, dose_check.json, debug_activations.npz, requirements.lock

---

## Addendum: layer-wise residual localization (debug set, `locate.py` / `windows.py`, `run_locate_debug/`)

Metric: teacher-forced continuation recovery rate. The donor emotion's 24-token greedy continuation is attached after the neutral recipient: 0 = neutral baseline, 1 = full donor prefix. All three emotions' continuations are scored in cross to check specificity. Runtime: 3 min for the sweep and 1 min for the windows, peak 8.2 GiB. See `run_locate_debug/layer_sweep.png`.

| Site (all anchor tokens) | Joy own/cross | Sadness | Anger |
|---|---|---|---|
| Residual stream, best layer | **0.34/0.11 (L20)** | 0.42/0.22 (L21) | 0.44/0.16 (L19) |
| Attention output, best single layer | 0.14 (L19) | 0.12 (L22) | 0.15 (L19) |
| MLP output, best single layer | 0.11 (L20) | 0.09 (L22) | 0.06 (L22) |
| Attention output L8–31 together | 0.58/0.14 | 0.61/0.29 | 0.66/0.15 |
| MLP output L8–31 together | **−0.43**/−0.21 | −0.06 | −0.14 |
| Attention + MLP L0–35 (= full anchor substitution) | 0.56/0.16 | 0.64/0.31 | 0.71/0.15 |

Conclusions (to be re-measured on the discovery set):
1. **Emotion information enters the anchor positions via attention**, and it is distributed across many layers, peaking in the L19–21 residual stream. It is not stored in a few MLP units.
2. Replacing only the MLPs across multiple layers gives negative recovery. The MLPs are processing the attention's input; transplanting them alone creates an inconsistent state.
3. The anchor positions can carry at most about 0.56–0.71 of the effect (the anchor-mediated ceiling). The rest reaches the continuation directly from the style_lead.
4. Own recovery is clearly above cross, so this is not simply "moving away from neutral". Sadness has higher cross recovery (0.22–0.31), which suggests it contains a generic "non-neutral" component.

These findings have been written into `../EXPERIMENT.md` as v2 (§0 revision record, §4.3 coarse-to-fine localization, §5 ceiling → frozen object → sealed test).
