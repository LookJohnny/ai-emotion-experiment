# E01 Formal Data (draft-v2, 2026-09-28)

**Status: draft v2. Automated checks and blind plausibility checks are done; human review is still pending.** Do not use it for formal localization or testing until the review is complete.

## Files

| File | Contents |
|---|---|
| `formal_960.jsonl` | 120 scenarios × 4 emotions × paraphrases A/B = 960 main records |
| `controls_48.jsonl` | Scenarios 01–04 in each topic × 2 controls (quoting an emotion word neutrally / describing another person's joy) = 48 records |
| `review_sheet.csv` | **Human review sheet** (open in Excel or WPS). Includes the blind-rating scores and a "priority review" column |
| `manifest.json` | Split rules, sha256 hashes, balance statistics, tokenization check, blind-rating rounds and summary |
| `plausibility_combined.json` | Final per-scenario emotion plausibility (mean across raters) |
| `plausibility_blind_v1..v4.json`, `plausibility_v5a/b` … `plausibility_v8a/b.json` | Raw scores from each blind rating round |
| `finalize_plausibility.py` | Combines the rounds' scores → combined file, manifest, and review-sheet columns (**re-run after every `build_formal.py`**) |
| `BRIEF.md` | Writing brief for the drafting agents, including the v2 addendum (the three-element structure) |
| `draft/*.json` | Drafts by topic (**edit these when revising**, then re-run `build_formal.py`) |
| `draft/v1_backup/` | v1 drafts (pure good-news/bad-news events), kept for comparison |
| `draft/generators/` | Agent generator scripts, archived for provenance only. **Do not re-run them** |
| `validate_topic.py` / `build_formal.py` / `tok_check.py` | Hard-rule validation per topic / merge / tokenization check |

## Splits

| Split | Topics | Scenarios | Records |
|---|---|---|---|
| Discovery | Shopping (shop), travel (trav), family arrangements (fam) | 60 | 480 |
| Validation | Study arrangements (study) | 20 | 160 |
| **Sealed test** | Work collaboration (work), hobby activities (hobby) | 40 | 320 |

The test set may be proofread only. It must not be used for localization, layer selection, or tuning.

## How v2 differs from v1

A blind check of v1 found that **114/120 events were pure good news or pure bad news**, so the four emotions were forced onto them (e.g. "joy" about a train delay). This would have mixed the main joy-vs-neutral contrast with "contradiction/irony".

In v2, every event is rewritten to contain three elements:
1. **A real gain** (supports joy)
2. **Losing something you valued** (supports sadness)
3. **A decision made by someone else, with a specific impropriety** (supports anger)

Each scenario includes an `emotion_basis` explaining the source of each emotion.

## Blind plausibility check

Method: each round a new agent sees only the facts (in shuffled order) and rates, for each emotion, how natural it would be for an ordinary person to feel it, on a 1–5 scale. The final score is the mean over all raters who saw the current version.

| Round | All three emotions ≥3 |
|---|---|
| v1 (single rater) | 6/120 |
| v2 round 2 | 103/120 |
| v2 round 4 | 102/120 |
| After round-4 fixes, rater mean ≥2.5 | 116/120 |
| After round 6, rater mean ≥2.5 | 120/120 (≥3: 108/120) |
| **Final (after round 8), rater mean ≥3** | **120/120 (100%)**; the lowest score of any emotion in any scenario is 3 |

Mean score by split (joy/sadness/anger):
- Discovery: 3.56/3.81/3.66
- Validation: 3.83/3.82/3.40
- Test: 3.88/3.89/3.70
- Overall: 3.71/3.84/3.63

Rater agreement: on unchanged scenarios, two raters differ by 0.24 points on average and never by more than 1. The judgments are stable, but a single rater is noisy near the 2/3 boundary.

**Caveat**: the raters are all LLMs. This check catches structural mismatches but does not replace human judgment.

## Automated checks (all passing)

- All 6 topics: 0 hard-rule violations, covering:
  - Format and length balance
  - No category emotion words in B / neutral leads
  - No ！？…
  - No new numbers in suffixes
  - `emotion_basis` complete
- Across topics: no near-duplicate facts, no duplicate anchors.
- Tokenization: no boundary merges in any of the 1008 records; the anchor tokenizes identically across all 8 records of every scenario; the longest record is 155 tokens.
- Balance:
  - Mean lead length is 13.1–14.1 characters across the four emotions.
  - Neutral leads are about 1 token shorter, and neutral suffixes about 5 characters shorter. The analysis should use a length baseline.
- Paraphrase A leads contain category emotion words at a high rate for emotional records and never for neutral ones, so **A ≈ explicit, B ≈ implicit**. This can serve as a lexical-level control.

## Review guide

In `review_sheet.csv`, fill in the four review columns for every record: tone correct, natural (1–5), facts correct, suggested edits. After changing a draft, run:

```bash
python3 validate_topic.py draft/<topic>.json <prefix>
python3 build_formal.py
python3 finalize_plausibility.py   # restore the blind-score columns in the review sheet
```

### About the blind-rating results
Every scenario scores ≥3 on all three emotions (multi-rater mean). The emotion-to-event fit has been checked by LLM raters, but **human review is still needed for naturalness, factual consistency, and whether suffixes add new facts**. It's easiest to review scenario by scenario; the 8 records of one scenario are adjacent in the sheet.

Fix history: the 20 scenarios whose rewrite went through several rounds (fam01/03/08/10, shop05/11, trav16/20, study06, hobby05, etc.) are recorded in `manifest.json` → `plausibility.history` and in the `raters` field of each entry in `plausibility_combined.json`.

### Other things to check
- Whether suffixes add small subjective inferences beyond the facts (this rewrite round did not audit these one by one). Suggested standard: allow subjective inference, forbid new objects, events, or places.
- Whether "procedural impropriety" wording repeats too often. 23/120 facts use phrases like "没通知 / 没征求 / 背着 / 未经 / 突然" (without notice / without consulting / behind my back / without permission / suddenly), mostly in shop and trav (6 each). Not a large share, but check whether it reads templated, so the model doesn't learn "anger ≈ someone didn't consult me".

## Not done yet

- Human review
- Measuring the tone gap on the model (stage A of the plan: ceiling measurement on the validation set, run after review)
