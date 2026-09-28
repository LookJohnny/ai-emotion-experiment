# Stage A grid — selection rule (written 2026-09-28 before any grid rating was seen)

Candidates: anchor tail T0「我的看法是：」/ T1 (none) / T2「我的感受是：」/ T3「总之，」 × continuation length 64 / 32 tokens.
Metric: validation joy ceiling = joy(prefix_joy) − joy(prefix_neutral), scenario-level mean of 2 blind LLM raters, bootstrap 95% CI (10,000).

Selection (validation only):
1. Eligible: ceiling ≥ 0.5 and CI lower > 0; mean fluency ≥ 4.0; mean new_fact_error not above T0_64's by more than 0.10.
2. Among eligible, pick the largest ceiling point estimate. If two are within 0.10, prefer 64 tokens over 32 (more text for later rating), then T0 > T1 > T3 > T2 (least change from the original design first).
3. If none is eligible: pick the largest ceiling anyway for the discovery run, but report stage A as NOT passed and stop before B-stage; return to the user.

Confirmation (discovery, 60 scenarios, generated once with the chosen tail):
- Passes if discovery ceiling ≥ 0.5 and CI lower > 0 with the same 2-rater protocol.
- No further tuning on discovery. Test split stays sealed.

## Addendum (2026-09-28, written after joy failed on validation and discovery, before any discovery anger ratings were seen)
Candidate switch of primary contrast to anger vs neutral (validation anger effect +1.08 [0.71,1.45] was seen before this rule was written).
Discovery confirmation: prefix_anger vs prefix_neutral, original tail T0, 64 tokens, the 240 already-generated continuations in run_v2 (60 scenarios × A/B).
Two fresh blind LLM raters; items shuffled; condition hidden. Metric: anger score difference, scenario-level mean, bootstrap 95% CI (10,000).
Passes if anger effect ≥ 0.5 and CI lower > 0. Also report: sadness/joy in anger continuations (specificity), fluency, new_fact_error vs neutral.
If it passes: the plan revision (primary contrast → anger) must still be logged and approved by the user before B-stage runs.
