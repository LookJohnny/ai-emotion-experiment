# Stage A (ceiling measurement) report — 2026-09-28

**Conclusion: gate not passed. Per the pre-registration (GRID_PREREG.md, item 3), stop before the B stages and bring the decision back to the user.**
Provisional ratings came from blind scoring by two independent LLM raters. They are not human-validated.

## Generation (run_v2/)
- 80 scenarios (discovery 60 + validation 20) × paraphrases A/B × 5 conditions = 800 continuations, greedy, 64 tokens. Test split stays sealed.
- Sanity checks (after the fix): self-substitution 160/160 identical; first token matches 160/160; reruns reproduce 320/320.
- Fix record: in the first run, donor states were captured with a use_cache=False forward pass, while generation used a cached prefill. The bf16 numerics differ between the two (see diag.py), and self-substitution matched only 2/8. Capture was switched to the same generate-prefill path and the 160 anchor-substitution samples were rerun.

## Validation, initial ending "我的看法是：" (rating/)
| Contrast (vs. neutral prefix, matching emotion) | Mean /4 | 95% CI |
|---|---|---|
| Joy prefix → joy (gate) | +0.35 | [0.19, 0.51] |
| Full anchor substitution → joy | +0.16 | [0.01, 0.35] |
| Sadness prefix → sadness | +0.45 | [0.23, 0.68] |
| Anger prefix → anger | +1.08 | [0.71, 1.45] |
Inter-rater r = 0.87–0.98. The neutral baseline already has sadness ≈ 1.0; 25–36% of continuations contain invented facts.

## Validation, anchor ending × length grid (rating_grid/, 4 raters)
T0「我的看法是：」64: 0.43 [0.25,0.61] | T0 32: 0.34 | T2「我的感受是：」64: 0.28 | T1 no ending 64: 0.26 | T3「总之，」: 0.23 | others 0.15–0.19. **No cell passes**; the original setup is best.

## Discovery confirmation (rating_disc/, T0_64, 60 scenarios)
Joy ceiling **+0.20 [0.08, 0.32]**; joy after neutral prefix 0.45 → after joy prefix 0.65; sadness after joy prefix 1.22. Inter-rater r = 0.79–0.92.

## Interpretation
- The joy opening barely carries into the continuation, while anger carries strongly. Possible reasons: (1) every v2 event has a "loss", and the continuation mentions it on its own, so all conditions drift toward "有点遗憾"; (2) the anchor puts the facts back at the centre of attention, and the continuation tends to evaluate them.
- Changing the anchor ending does not help, so the issue is more likely in the data design (emotion in a single opening sentence is too weak vs. an event that includes a loss) than in the anchor wording.
