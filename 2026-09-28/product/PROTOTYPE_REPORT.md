# Pre-synchronised expression prototype: batch latency test (2026-09-29)

**Concept**: before the model outputs any text, read its internal state to predict the upcoming tone and drive the face in advance (pre-sync), then correct the expression per token during generation. Compared against the existing approach: "analyse sentiment after the text / sentence is complete".
Code: `readout_server.py` (v3), `avatar/index.html`, `bench_latency.py`; results: `results/bench/bench_latency.json`.

## Setup
- Model: Qwen3-4B-Instruct-2507, RTX 5090; per-token readout probe v2 (L19); pre-readout probe v2 (L22); expression intensity calibration `avatar/calib.json`.
- Prompts: the **held-out test split** of the pre-readout experiment, 80 user messages (never used to train either probe): good news, grievance, sad news, neutral task (18 each) and mixed (8).
- Three modes, greedy decoding (same reply text in every mode):
  - `pre`: pre-reply readout + per-token correction (this proposal)
  - `token`: per-token readout only
  - `sentence`: the expression is updated only when a sentence is complete (models the existing post-hoc sentiment approach)
- "Expression onset" = the first moment any of happy/sad/angry reaches intensity ≥ 0.3 (ms from receiving the request).

## Results (54 emotional messages)

| Mode | Expression started | **Onset median** | Lead over first character, median | **Face before first character** | Dominant expression correct | Neutral-task false alarms |
|---|---|---|---|---|---|---|
| **Pre-sync + per-token** (v1, before fix) | 54/54 | **24 ms** | **+22 ms** | **54/54** | 53/54 | 12/18 ⚠️ |
| Per-token only | 54/54 | 68 ms | −22 ms | 0/54 | 53/54 | 12/18 ⚠️ |
| Post-sentence analysis (existing) | 53/54 | 286 ms | −241 ms | 0/54 | 53/54 | 3/18 |

Onset median by category (pre / token / sentence): good news 24 / 67 / 111 ms · sad 24 / 46 / 304 ms · **grievance 24 / 132 / 642 ms**.
First-character latency median ≈ 46 ms in all modes (the pre-readout adds almost no overhead).

## Conclusions

1. **Face before text is achievable**: in pre-sync mode, 54/54 emotional messages started their expression before the first character (median 22 ms early). Onset is about 12× faster than the "post-sentence analysis" baseline (24 vs 286 ms), and about 27× faster for grievances (24 vs 642 ms), because an angry reply often opens with neutral words like "我理解……".
2. **Accurate tone judgement**: the dominant expression is correct in 53/54 (good news → happy, sad → sad, grievance → angry).
3. **Absolute times are small because this GPU generates fast** (first character 46 ms). In real deployments with networking, TTS and robot servo latency, the full chain is slower and the pre-sync lead grows in proportion to the duration of "the first sentence". The actual end-to-end gain needs measuring on real hardware.

## False-alarm fix (tuned on validation, frozen, re-checked on test)

**Problem**: in v1, per-token spikes caused neutral-task false alarms 12/18 (pre-reply readout itself was fine).
**Method** (rule pre-registered before tuning and committed as `50d5535`; see `SMOOTHING_PREREG.md`):
- record raw frames once (pre-readout + per-token readout + timestamps), 80 validation + 80 test prompts;
- `smoothing.py` is shared by the server and the offline tuner; grid of 27 configs: α × sustain N × pre-gate γ;
- select on validation only → freeze (`smoothing_freeze.json`) → one re-check on test.
- **Selected**: α = 0.15, N = 1, **γ = 0.3 (pre-gate)**. If the pre-reply readout says "neutral", token-phase emotion intensity counts at only 30%. N had almost no effect, so the simplest value was kept.

**Live service on held-out test (`results/bench/bench_latency_v2.json`)**:

| Mode | Onset median | Face before first char | Dominant correct | Neutral-task false alarms |
|---|---|---|---|---|
| **Pre-sync + per-token + pre-gate (this proposal)** | **23 ms** | **54/54** | **54/54** | **0/18** |
| Per-token only | 92 ms | 0/54 | 54/54 | 5/18 |
| Post-sentence analysis (existing approach) | 274 ms | 0/54 | 53/54 | 3/18 |

The pre-reply readout plays two roles: ① it drives the expression before any text; ② it **gates** per-token corrections, suppressing spikes. With the gate, false alarms drop below the existing approach (0 vs 3).

**Process note**: on the first auto-tuning pass the wait condition was wrong and a half-collected dataset (36/80) was pulled. That freeze is void (`results/raw/void/`, not used); the test re-check in that pass ran on an empty file, so no test data was seen.

**Side effect / not evaluated**: if the pre-readout mislabels a genuinely emotional reply as neutral, the per-token expression is also damped. The 8 "mixed" prompts were not scored; mixed and emotional-shift conversations (e.g. a user whose tone turns mid-conversation) need separate testing.

## Patent write-up notes
- Technical effect: the first-moment expression onset of the existing approach is limited by "sentence completion". This proposal moves onset to before the first character (≈0 dependence on generation progress), with no added first-character delay.
- Embodiments: VRM avatar (`avatar/index.html`), 6-channel servo mapping (brows / eyelids / mouth corners, HTTP 10 Hz).
- Calibration: floor/ceil intensity calibration (reduces misreadings of task replies) + **pre-gate** (pre-reply readout constrains the gain of per-token corrections) — a candidate dependent claim.
