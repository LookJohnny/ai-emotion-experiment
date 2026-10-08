# Nous Tone 0.1: evaluation on Qwen3-1.7B (2026-10-08)

## Setup

- **Packs**: four attributes (anger, joy, sadness, warmth), each built from 40 built-in user messages. Each pack takes about 4.5 minutes on an Apple M4.
- **Readout probe**: 5-way (reference + the four attributes), CV accuracy 0.835 (chance 0.20), layer 14 (relative depth 0.52).
- **Held-out set**: 20 user messages the packs never saw. Each attribute was generated at strength +1, +2 and −1 next to the untouched reply: 260 replies of 64 tokens.
- **Raters**: two blind LLM raters scored joy, sadness, anger and warmth (0–4) and fluency (1–5). Rater agreement (Spearman):

| Dimension | Agreement |
|---|---|
| joy | 0.97 |
| sadness | 0.94 |
| anger | 0.97 |
| warmth | 0.89 |
| fluency | 0.73 |

Raw data: `eval/heldout_1.7b.jsonl`, `eval/rating/`, `eval/results.json`, `eval/analyze.py`.

## Results

Change in the target attribute's rating against the untouched reply for the same message (bootstrap 95% CI):

| Attribute | Strength +1 | Strength +2 | Strength −1 | Fluency at +2 | Readout vs rating (Spearman) | Status |
|---|---|---|---|---|---|---|
| **warmth** | +0.57 [0.28, 0.90] | **+1.20** [0.85, 1.58] | **−0.70** [−1.03, −0.40] | −0.38 | **0.66** | usable |
| **joy** | +0.42 [0.18, 0.70] | **+1.12** [0.83, 1.43] | **−0.38** [−0.60, −0.18] | −0.53 | 0.48 | usable |
| sadness | +0.10 [0.00, 0.25] | +0.33 [0.08, 0.63] | −0.12 | −0.07 | 0.22 | experimental |
| anger | 0.00 | +0.15 [0.00, 0.30] | 0.00 | −0.47 | 0.06 | not working on this model |

Further observations:

- **Warmth and joy steer in both directions.** At strength 1 there is no clear fluency cost; at strength 2 it is about −0.4 to −0.5.
- **They spill over into each other.** Joy +2 also raises warmth by +0.47, and warmth +1 also raises joy by +0.23.
- **Sadness mostly raises warmth** (+0.62 at strength 2), more than it raises sadness.
- **Anger barely moves.** Qwen3-1.7B describes the user's anger ("让人很不爽") but does not become angry itself. The research's anger results came from 4B/14B models (E01, E04).
- **The readout tracks the raters best on warmth (0.66).** On sadness and anger it tracks poorly, partly because those tones are almost absent from the replies, so there is little variance to track.

## Target mode

The first engine version adjusted lambda token by token from the readout. That did not work: in demos, lambda hit its cap while the readout stayed flat. A reply's tone is set in its first tokens, and later steering barely moves it.

The engine now does two things instead:

- it picks the strength up front by inverting each pack's held-out calibration, the mean readout at strengths −1, 0, +1 and +2, stored in `pack.json → meta.calibration`;
- within the reply, it only lowers lambda if the readout overshoots the target.

## Bugs found and fixed

**Split characters.** Emoji and some characters split across tokens left "�" in the streamed text. Raters flagged this; it lowered fluency scores in every condition alike. The engine now holds a delta back until the character is complete, and a regression test covers it.
