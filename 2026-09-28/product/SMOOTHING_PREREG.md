# Expression anti-spike tuning — pre-registration (2026-09-29, written before any tuning result was computed)

Problem: per-token spikes cause neutral-task false alarms of 12/18 on the test split (pre-sync mode).
Data: raw frames in pre-sync mode (pre-readout + per-token probs + ms timestamps), greedy, validation split 80 prompts / test split 80 prompts.
Tuned on validation only; the test split is used only for the one final re-check of the frozen config.

Grid (27): alpha in {0.3, 0.15, 0.08} × sustain N in {1, 3, 6} × pre-gate gamma in {1, 0.6, 0.3} (smoothing.py).
Metrics (visibility threshold 0.3): emotional messages (good_news / sad / grievance)
- started: max intensity ≥ 0.3 at some point
- correct: dominant expression = the category's emotion
- before first char: expression onset earlier than the first character
- neutral-task false alarm: any emotional expression ≥ 0.3 at any point

Selection rule:
1. Eligible: started ≥ 95% of emotional messages, correct ≥ 95%, before first char ≥ 90%
2. Among eligible configs: fewest neutral-task false alarms
3. Tie: smaller median onset; still tied: simpler config (larger alpha, smaller N, larger gamma)
4. If nothing is eligible: report that and do not move on to the test re-check

Test re-check (reported once, no further tuning): the same four metrics for the frozen config vs the current default (alpha 0.3, N 1, gamma 1).
