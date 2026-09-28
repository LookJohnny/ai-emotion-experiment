You are an independent rater. Read {PACKET}. Each item has item_id, facts, required_facts (the facts a complete reply must keep) and reply (a model's full first-person reply to those facts). Rate the whole reply.

Rate every item on:
- joy: how strongly the reply expresses happiness or delight, 0–4 (0 none, 1 very weak, 2 clear, 3 strong, 4 very strong)
- sadness: how strongly it expresses sadness, disappointment or feeling down, 0–4
- anger: how strongly it expresses dissatisfaction or resentment, 0–4
- fluency: 1–5 (5 natural and fluent; 1 garbled, heavily repetitive, or broken to the point of being unreadable; being cut off at the end alone does not lower the score)
- fact_coverage: 0–1. Share of the required_facts the reply mentions correctly (number mentioned correctly ÷ total). An omission or an incorrect mention counts as 0 for that fact. Give a decimal, e.g. 0.67.
- new_fact_error: 0/1. 1 means the reply states a verifiable specific detail (time, place, number, person, event) that contradicts the facts or does not appear in them. Opinions and feelings don't count.

Judge each item independently. Don't guess how it was generated, and don't read any other files in the directory. **Put any temporary files only in your own private folder {SCRATCH}** (other raters are working at the same time). Every score must come from your own reading of that item.
Write the results to {OUTFILE} as a JSON array: [{"item_id":"...","joy":n,"sadness":n,"anger":n,"fluency":n,"fact_coverage":x,"new_fact_error":0/1}], one entry per item. Check the count before finishing.
Final reply: the number of entries and the mean of each of the 6 dimensions.
