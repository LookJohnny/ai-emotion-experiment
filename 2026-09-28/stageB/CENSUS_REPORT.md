# Neuron Selectivity Census (2026-09-28)

**Question**: Do MLP neurons follow a regular organisation, grouped by emotion, by knowledge domain (topic), or by task type?
**Nature**: correlational analysis, no intervention. Data: discovery + validation splits (the test split stays sealed).
Script: `neuron_census.py`. Results: `results/census/` (paraphrases A+B, 640 records) and `results/census_B/` (paraphrase B only, 320 records; the style_lead contains no emotion-category words).

## Method

- **Unit**: the MLP intermediate value z (input to `down_proj`), 36 layers × 9728 = 350,208 neurons.
- **Segments**: the mean of z over the opening sentence (lead), the fact anchor (anchor), and the reference continuation (suffix), with the continuation teacher-forced.
- **Emotion selectivity** η²_emotion: after subtracting each scenario's mean, the share of variance explained by the 4 emotions (within-scenario design, so the event itself is controlled).
- **Topic selectivity** η²_topic: the share of variance in scenario means explained by the 4 topics (shopping, travel, family, study).
- **Task type**: 400 user messages in 5 categories (good news, grievance, sadness, neutral task, mixed); η² over the mean of the user-message tokens.
- **Significance**: permutation null distribution (emotion shuffled within scenario, topic shuffled across scenarios, category shuffled); the threshold is the 99.9th percentile of the null for that layer.
- **Generalisation**: correlation, and top-1% Jaccard, between neuron selectivity for "the speaker is angry" (lead) and for "the user describes a grievance".

## Results (paraphrase B, lead position)

| Layer | Emotion-significant (A+B → B) | Topic-significant | User-message category significant | Speaker anger vs user grievance: r / J (chance 0.005) |
|---|---|---|---|---|
| L2 | 8133 → 6828 | 1643 | 3454 | 0.15 / 0.026 |
| L10 | 8515 → 6896 | 2795 | 5818 | 0.15 / 0.032 |
| L17 | 8872 → 7398 | 3776 | 7075 | 0.20 / 0.026 |
| L22 | 8420 → 7061 | 2361 | 7238 | 0.20 / 0.037 |
| L35 | 8238 → 7024 | 3054 | 6637 | 0.18 / 0.026 |

- At the anchor position (paraphrase B), 1.3k–6.4k emotion-significant neurons per layer; at the continuation, 5.2k–8.0k.
- Median η²_emotion at L17 is 0.087 and median η²_topic is 0.046.
- Among the 60 most selective neurons: preferred emotion is neutral 30, anger 20, sadness 9, joy 1; **49/60 keep the same preferred emotion across all 4 topics**.
- Examples of strongly tuned neurons (η²_emotion 0.6–0.7, η²_topic < 0.05): **L12#8223 and L14#6962 (anger)**, L19#7801 and L30#4393 (sadness), L8#3145 and L11#7612 (neutral).

## Conclusions

1. **Emotion information is dense, not sparse**: even after removing emotion-category words, 70–76% of neurons per layer vary consistently with emotion, but each neuron explains little variance (median 9%). This fits the patching results: replacing 32 neurons has almost no effect, and the effect only appears when the residual stream / attention across many layers is replaced (B1/B2).
2. **A small number of strongly tuned neurons do exist and are stable across topics.** Many of them prefer "neutral", which most likely detects a calm, planning style of speech rather than emotion itself.
3. **Mixed selectivity dominates**: 2,000–3,700 neurons per layer are selective for both emotion and topic. Organisation by knowledge and organisation by emotion sit in the same neurons; both peak in the middle layers L14–17.
4. **"Whose emotion" is encoded separately**: selectivity for "the speaker is angry" and for "the user describes something infuriating" correlates at only r≈0.15–0.22, with top-1% overlap at 2–7× chance. Expressing emotion and recounting emotion do not share one set of "anger neurons". Product implication: the readout should be trained on states from the model's own speech (the current approach).

## Limitations and next steps

- Correlational only. Paraphrase B still uses emotionally coloured wording ("说变就变", "太离谱"), so lexical and semantic effects cannot be fully separated.
- **Task difficulty was not measured**: the 5 user-message categories mix content and form (translation vs. confiding), so they cannot be read as difficulty. A dedicated set of same-type prompts at varying difficulty is needed.
- Next: (1) **causal ablation** — mean-ablate strong anger neurons such as L12#8223 / L14#6962 on the lead, check whether anger expression drops, with random neurons as the control; (2) train a **sparse autoencoder (SAE)** on L12–27 to decompose mixed-selectivity neurons into cleaner features.
