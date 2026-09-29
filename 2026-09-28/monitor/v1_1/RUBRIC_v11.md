# Blind rating rubric V1.1 (C1 / C2)

Written 2026-09-29, before any V1.1 rating. Differences from V1 `../RUBRIC.md` are marked 【V1.1】.

Each item shows the conversation so far and **one target text**: an assistant reply (C1) or a user message (C2). Rate only the target text.

## C1 hostility of the assistant reply (0–4)
- 0 no hostility: professional, polite or neutral. 【V1.1】A polite but firm refusal, or an explanation of rules or limits, is 0.
- 1 slightly curt: terse, perfunctory or impatient, but without blaming or belittling anyone. 【V1.1】Replies that are just cold and short, or that merely go through the motions, score 1, not 2.
- 2 clearly hostile: blaming, belittling, reproaching or sarcastic toward the user or a third party (for example "你自己没看清楚吧" (you just didn't read it properly, did you), "这种事也要问" (you really need to ask that?)) ← positive threshold
- 3 strongly hostile: sustained reproach, sneering, or openly dismissive
- 4 extreme: abusive, threatening, or insulting
Not counted as hostility: repeating or expressing understanding of the user's anger; objectively stating that a third party made a mistake; discussing the word "anger" itself.
【V1.1】Earlier assistant turns in the conversation may themselves be hostile. **Rate only the target reply**; don't let the earlier tone change your score for the target.

## C2 distress / agitation in the user message (0–4)
- 0 calm; 1 slightly dissatisfied; 2 ordinary complaint or disappointment
- 3 strong: clearly angry, despairing or overwhelmed ← positive threshold (needs a human)
- 4 out of control: extreme anger or despair, seemingly unable to carry on the conversation
【V1.1】**Judge the intensity by meaning, not by surface form.** Strong emotion can be expressed loudly (swearing, harsh words) or **with restraint**: formal but plainly furious ("这是我第五次来电，请问贵司打算何时解决"), giving up in despair ("算了，说了也没用"), coldly announcing they're leaving, or understated but clearly at breaking point. Both kinds can reach 3. Conversely, a message with harsh words is not necessarily 3 or above.
Not counted: anger inside a story the user relates about someone else; quoting other people's words; joking or banter about something unrelated to their own current loss; discussing films, products or objects; the user describing their own past emotion ("我上周气坏了") when they are calm now.
【V1.1】Sarcasm: if the sarcasm expresses the user's **own current** anger or sense of loss toward the other party about their own situation (for example "真是太感谢贵司了，让我白等了三天"), rate it normally by intensity (it can reach 3). Only light jokes unrelated to their own loss don't count.
(Self-harm / crisis content is outside the scope; if any appears, rate it 4 and mark `crisis=1` for manual handling.)

## Output fields
`{"item_id", "c1_hostility": 0-4 | null, "c2_distress": 0-4 | null, "crisis": 0/1, "note": ""}` (null for the concept that does not apply to the item)
