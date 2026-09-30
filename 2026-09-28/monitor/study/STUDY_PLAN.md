# User study pre-registration: does the internal-state monitor help people judge alerts? (2026-09-30)

Following the survey (`../LLM_VIZ_SURVEY.md`): Transformer Explainer's controlled study found that people who only watched a video "felt they understood but got the quiz wrong"; Kaur (CHI 2020) found analysts over-trust explainability tools; Hase & Bansal found subjective ratings do not predict actual usefulness. So this study **uses task performance as the primary outcome, not satisfaction**, and measures whether trust is calibrated.

## 1. Research questions and hypotheses
- **RQ1**: Compared with text + score only, do the 2D heatmap / 3D pathway views help people judge more accurately whether a "hostility alert" is correct?
- **RQ2**: Do they help people locate the trigger span of the hostility more accurately?
- **RQ3**: Do they improve trust calibration (confidence matching correctness), or cause over-trust?
- H1 (main): judgement accuracy under **T+2D > T**.
- H2: localisation IoU under **T+2D > T**.
- H3 (exploratory, two-sided): the difference between T+3D and T+2D.
- H4 (calibration): under the visualisation conditions, the rate of "high confidence but wrong" is **not higher** than under T.

## 2. Design
- **Within-subjects, 3 conditions**:
  - T: conversation + reply + monitor score/tier.
  - T+2D: T plus the layer × token heatmap replay.
  - T+3D: T plus the 3D pathway replay.
- **Blocks**: each participant does 3 blocks × 8 trials. The 24 formal trials are split into 3 fixed sets (true/false alerts 5/3, 5/3, 6/2). Condition order follows a 3×3 Latin square by `participant number mod 3`. The set for block b is `(b + ⌊participant number / 3⌋) mod 3`. Every 9 participants therefore complete a full condition × set balance. Trial order within a block is shuffled by participant number.
- **Practice**: 2 trials before the formal blocks (from the val set, not analysed), one per condition type, to learn how to operate each view.

## 3. Stimuli (fixed, pre-recorded)
- Drawn from the V1.1 C1 **sealed test set** (companion domain; the reply text has two blind LLM hostility ratings). Selection script: `select_stimuli.py`, random seed fixed.
- 24 formal trials (at most 3 per template), all "cases where the monitor alerted" (monitor item score ≥ demo threshold 0.237):
  - **16 true alerts**: rating label = hostile.
  - **8 false alerts**: rating label = not hostile. These test whether people can spot a false alarm; over-trusting participants will call them all real.
- Replays: the fixed reply text is teacher-forced back into the same model (Qwen3-4B, RTX 4090), and every x_risk frame per token is recorded (the same fields the live monitor shows). Script: `record_replays.py`. The replay is a faithful reproduction of the monitor's readings on that exact text.
- **Borderline trials (recorded before data collection)**: the two span annotators marked mild hostile spans in false-alert trials s21, s22 and s23 (and practice p25). These trials' original ratings sit just below the threshold. A **sensitivity analysis** is pre-registered: the primary analysis includes all trials, then a secondary analysis excludes s21–s23, and both are reported.
- **Trigger-span gold standard**: two blind LLM raters independently mark the character spans of the true-alert replies that "blame, belittle or sneer at the user". The union of the two raters' marks is the gold span (`gold_spans.json`). Inter-rater character IoU = 0.80 (21 trials with spans). Human spot-check rules follow V1.1.

## 4. Per-trial task (3 steps)
1. **Judgement**: "Is this alert correct? Did the reply really show hostility toward the user?" Yes / No.
2. **Localisation**: select the span in the reply that most expresses hostility (mouse selection; choosing "no hostility" is allowed).
3. **Confidence**: 1–7.

Time is logged automatically (from trial display to submit). The answer is never shown during the trial.

## 5. Measures and analysis (decided before any data)
- **Primary**: judgement accuracy (per trial 0/1). Mixed-effects logistic regression: `correct ~ condition + (1|participant) + (1|item)`. The key contrast is T+2D vs T (H1), with α = 0.05 two-sided.
- **Secondary**:
  - Localisation IoU on true-alert trials only (character level). Linear mixed model with the same structure.
  - Time (log seconds).
  - Calibration: per condition, the Brier score and the "confidence ≥ 6 and wrong" rate (H4).
  - False-alert detection: the share of false-alert trials judged "no".
- **Subjective (report only, not used for conclusions)**: after each block, "How well do you think this view helped you understand the model?" (1–7). The difference between the subjective score and the objective accuracy is reported, testing for an "illusion of understanding".
- **Exclusion rules (pre-registered)**:
  - Accuracy on practice trials < 50% and every formal answer the same → excluded.
  - Median time per trial < 3 seconds → excluded.
  - Exclusion counts are reported.
- **Sample size**: 27 people (a multiple of the 3 × 3 full balance). Rough estimate: with a within-subjects effect of about 10 percentage points and 24 × 8 trials per condition, power is about 0.7. The sample size is not changed after seeing results.

## 6. Participants and ethics
- Adults who use AI products at work (support / operations / product / engineering). No knowledge of LLM internals is required.
- A consent page before the start; no personally identifying information is collected (only a random participant number). Participants may quit at any time.
- The stimuli contain AI replies with belittling wording (companion scenarios). This is disclosed on the consent page.

## 7. Materials and data
- Task page: `/study` (served by the monitor service or as a static page). Answers are written to `study/responses/*.jsonl` and can also be downloaded as JSON locally.
- Stimuli and replays: `study/trials.json`, `study/replays/*.json`, `study/gold_spans.json`.
- The analysis script `analyze.py` was committed before data collection. It was dry-run on 9 simulated participants (results meaningless, only to confirm the code runs). The automated self-test session was excluded by the RT rule and moved out of the data directory.

## Deviations
(none)
