# E01 Formal Data Draft: Shared Writing Brief (provenance)

Used for: parallel drafting by 6 Claude subagents (Opus 5.5), one topic each; 2026-09-28. Drafts require line-by-line human review before use.

## Task background
Research on how a language model internally represents emotional expression in Chinese. For each scenario, the model gets the same neutral question: "请根据下面事实，以第一人称写一段简短回应，保留时间、地点和数量等信息。事实：{facts}"
The answer is split into three parts: `style_lead` (one opening sentence in a given tone) + `anchor_fact` (a restatement of the facts, **identical word for word** across all 8 answers in the scenario; the program appends "我的看法是：") + `suffix` (a natural continuation in the same tone).
Four tones: joy / sadness / anger (dissatisfaction only; no insults, threats or profanity) / neutral (acknowledging and planning, with no evaluation or feeling). Two paraphrases per tone, A and B.

## Hard rules (the validator checks these automatically)
1. Exactly 20 scenarios; scenario_id = prefix + two-digit number (01–20).
2. facts: 1–2 sentences, 15–80 characters (70 in v1, raised to 80 in the v2 addendum), with specific times/places/quantities; a realistic small event from everyday life.
3. required_facts: 2–4 key facts a full answer must preserve (short phrases).
4. anchor_fact: one sentence ending in "。", 8–30 characters, a neutral, concise restatement of the core facts; no evaluative words, no emotion words, no "看法"; any numbers must come from facts.
5. leads: 8 in total (4 tones × A/B). Each is **one sentence** ending in "。", 10–22 characters, containing "我", **no digits, and no restatement of specific facts** (only tone and a general reaction). Within the same paraphrase, the 4 tones differ in length by ≤4 characters.
6. **Paraphrase B and all neutral leads must not contain emotion-category words**: 高兴 开心 愉快 快乐 喜悦 愉悦 欣喜 兴奋 悲伤 难过 伤心 悲哀 难受 沮丧 失落 愤怒 生气 气愤 恼火 火大 不满 中性 平静 情绪. B conveys tone through situation, evaluation and word choice only. A may contain category words, but doesn't have to.
7. Punctuation: only "，", "。", "、", and quotation marks; **no "！？…~"** (to balance punctuation across the four categories).
8. suffix: 25–70 characters, continuing directly after "我的看法是：" in a tone consistent with the lead; **it must not introduce numbers/times/places/people not in facts** (no new verifiable facts); the ending may be a complete sentence.
9. Scenarios 01–04 also get controls:
   - `quote_word`: the speaker is neutral and **quotes an emotion word in quotation marks** (e.g. mentioning, explaining or relaying the word "高兴") without feeling that emotion. lead is 10–24 characters.
   - `other_person`: describes **another person's** joyful emotion while the speaker stays neutral (e.g. "室友听了很开心，我这边照常安排。"). lead is 10–24 characters.
   - The control emotion is always **joy** (the main contrast). The suffix is neutral and follows the same rules. Scenarios 05–20 must not have controls.
10. Facts across the 20 scenarios must differ clearly: **no reusing the same template with a different name, time or quantity**. Cover as many distinct sub-types within the topic as possible.
11. Avoid the events already used in the debug set: water outage, dental follow-up, library book renewal, elevator repair, health-check report, weather forecast, broadband upgrade, vaccination site, rent/property fees, phone-screen repair, gym opening hours, driver's-license annual inspection.

## Quality requirements (reviewed by people; strongly recommended)
- The four tones should be distinct and natural, like something a real person would say, not stilted. Anger should be restrained dissatisfaction.
- The tone must fit the facts: every event should be something a reasonable person could feel happy, sad or dissatisfied about (ideally a change, a result or a notice).
- A and B should differ clearly in wording and sentence structure, not just swap one or two words.
- Use a mix of sentence patterns across the 20 scenarios; don't let every joy lead start with "太好了".
- Neutral means "I know about it / I'll handle it this way", with no evaluative words such as "还不错" or "可惜".

## JSON format (one array per file)
```json
[{"scenario_id":"shop01","topic":"购物",
  "facts":"...", "required_facts":["...","..."], "anchor_fact":"...。",
  "leads":{"joy":{"A":"...","B":"..."},"sadness":{"A":"...","B":"..."},"anger":{"A":"...","B":"..."},"neutral":{"A":"...","B":"..."}},
  "suffix":{"joy":{"A":"...","B":"..."},"sadness":{...},"anger":{...},"neutral":{...}},
  "controls":{"quote_word":{"lead":"...","suffix":"..."},"other_person":{"lead":"...","suffix":"..."}}   // 01-04 only
}]
```

## v2 addendum (2026-09-28, after the blind plausibility check)

The blind check found that 114/120 events had at least one emotion rated ≤2 for naturalness (most events were purely good or purely bad news). **All events are to be rewritten as mixed events with three elements**:
1. **A real gain**: something a reasonable person would genuinely be glad about (supports joy);
2. **Loss of something already valued**: a habit, a relationship, an expectation, a place, a chance (supports sadness; not just added hassle, which only produces annoyance);
3. **A decision by someone else**: made by a platform, institution, organiser or other person, so the speaker has a target for dissatisfaction (supports anger).
Each scenario must include `emotion_basis` (one sentence per emotion, derived from the facts themselves). Target: in an independent blind rating, all three emotions score ≥3/5 in ≥90% of scenarios.
