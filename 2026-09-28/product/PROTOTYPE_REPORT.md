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
| **Pre-sync + per-token** | 54/54 | **24 ms** | **+22 ms** | **54/54** | 53/54 | 12/18 ⚠️ |
| Per-token only | 54/54 | 68 ms | −22 ms | 0/54 | 53/54 | 12/18 ⚠️ |
| Post-sentence analysis (existing) | 53/54 | 286 ms | −241 ms | 0/54 | 53/54 | 3/18 |

Onset median by category (pre / token / sentence): good news 24 / 67 / 111 ms · sad 24 / 46 / 304 ms · **grievance 24 / 132 / 642 ms**.
First-character latency median ≈ 46 ms in all modes (the pre-readout adds almost no overhead).

## Conclusions

1. **Face before text is achievable**: in pre-sync mode, 54/54 emotional messages started their expression before the first character (median 22 ms early). Onset is about 12× faster than the "post-sentence analysis" baseline (24 vs 286 ms), and about 27× faster for grievances (24 vs 642 ms), because an angry reply often opens with neutral words like "我理解……".
2. **Accurate tone judgement**: the dominant expression is correct in 53/54 (good news → happy, sad → sad, grievance → angry).
3. **Absolute times are small because this GPU generates fast** (first character 46 ms). In real deployments with networking, TTS and robot servo latency, the full chain is slower and the pre-sync lead grows in proportion to the duration of "the first sentence". The actual end-to-end gain needs measuring on real hardware.

## Known problem (must fix before release)

**Neutral-task false alarms 12/18**: the pre-reply readout itself is fine; the false alarms come from **short per-token spikes** during the reply (peaks 0.33–1.0). Emotional replies peak at ≥0.73, but some neutral-task replies also reach 0.97–1.0, so a single threshold cannot separate them.
Fix options: stronger smoothing (EMA α 0.3 → 0.1), a "sustained confirmation" rule (N consecutive tokens over threshold) or hysteresis, or down-weighting per-token corrections that go against the pre-readout direction.
**Tuning must use the validation split of prompts; the test split is only for re-confirmation**, otherwise the numbers above get overfitted.

## Patent write-up notes
- Technical effect: the first-moment expression onset of the existing approach is limited by "sentence completion". This proposal moves onset to before the first character (≈0 dependence on generation progress), with no added first-character delay.
- Embodiments: VRM avatar (`avatar/index.html`), 6-channel servo mapping (brows / eyelids / mouth corners, HTTP 10 Hz).
- Calibration: floor/ceil intensity calibration from neutral-task and per-emotion distributions (reduces misreadings of task replies); plus the post-hoc spike suppression still to be done.
