# Anger-Neuron Causal Ablation (2026-09-28)

**Question**: The census found a few strongly tuned "anger neurons" (e.g. L12#8223, L14#6962). If we erase them, does the model's anger expression drop?
Script: `ablate.py`; results: `results/ablate/`.

## Method

- **Neuron selection (discovery set only)**: over the opening-sentence tokens, rank all 350k MLP neurons (`down_proj` input) by the within-scenario standardized difference "anger − other emotions", and take the top k ∈ {32, 256, 1024}. The top 1024 cluster in the middle layers (L12–27); the top-ranked ones include L22#6294, L12#8285, L17#9359, **L14#6962** and **L12#8223**. The last two also appear in the census, which used discovery + validation, so the selection is stable.
- **Ablation**: replace the selected neurons' activations with their mean under the neutral condition, at every position (prompt and generation).
- **Evaluation (validation set, test set untouched)**: 20 scenarios × {anger, sadness, joy} × paraphrases A/B = 120 records; greedy continuation of 48 tokens. Emotion is measured by the L19 readout probe running on the **intact model** over the generated text. This readout correlates with blind ratings at 0.71 for anger.
- **Controls**: no ablation; random neurons matched in count and per-layer distribution (2 seeds each). **Sanity check**: k=0 matches no-ablation exactly (passed).

## Results (scenario-level paired differences, bootstrap 95% CI; baseline anger readout 0.68)

| Ablated | Anger: target − no ablation | **Anger: target − random** | Sadness: target − random | Joy: target − random | Anger texts changed |
|---|---|---|---|---|---|
| top-32 | −0.006 [−0.05, 0.03] | −0.029 [−0.07, 0.01] | +0.014 | −0.014 [−0.03, −0.00] | 34/40 |
| top-256 | −0.039 [−0.09, 0.01] | **−0.063 [−0.11, −0.02]** | −0.004 | −0.002 | 37/40 |
| top-1024 | −0.017 [−0.07, 0.03] | −0.019 [−0.06, 0.02] | **+0.054 [0.02, 0.10]** | +0.006 | 40/40 |

## Conclusions

1. **Erasing the strongest anger neurons barely changes anger expression.** Only k=256 shows a small drop against the random control (≈9% of baseline). k=32 and k=1024 are not significant, and the effect does not grow with k. With 9 comparisons, the k=256 result should be treated as a weak signal.
2. **The neurons change the wording, not the tone**: 85–100% of the continuations change their text, yet the anger readout stays about the same. The model says the same thing with different words, a sign that the representation is redundant or self-repairing.
3. Ablating the top 1024 neurons nudges sadness up slightly (+0.05): erasing the neurons most specific to anger may push negative expression toward sadness. Weak signal, needs replication.
4. **Consistent with the other evidence**: the census shows emotion information is dense (70–76% of neurons per layer carry it); B1/B2 patching shows the emotion is carried across many layers by attention into the residual stream. Together they point the same way: **in Qwen3-4B, anger expression does not depend on a small set of "anger neurons"**. It is a distributed, redundant representation. Strongly selective single neurons exist, but they are mostly readouts or detectors, not necessary switches.

## Limitations

- The outcome metric is the model's own readout probe (validated against blind LLM ratings), not human judgement.
- The fill value for mean-ablation comes from the opening-sentence positions but is applied at every position, so later positions may be out of distribution. A more precise version would use per-position means, or ablate only the opening-sentence positions.
- Only 20 validation scenarios, so the confidence intervals are fairly wide.

## Next steps

- Train a sparse autoencoder (SAE) on L12–27 to find features rather than neurons, and run causal tests at the feature level.
- Test a "distributed direction" in contrast: B4's residual direction d, or attention-head sets (B3).
