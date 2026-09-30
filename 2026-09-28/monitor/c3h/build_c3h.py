"""Sample 1200 Chinese SimpleQA questions and split by secondary category (PREREG_C3H.md). python build_c3h.py"""
import json, math, random, collections
R = [json.loads(l) for l in open("chinese_simpleqa.jsonl", encoding="utf-8")]
by = collections.defaultdict(list)
for r in R: by[r["secondary_category"]].append(r)
rng = random.Random(7); pick = []
for c in sorted(by):
    rs = by[c][:]; rng.shuffle(rs); pick += rs[:math.ceil(1200 * len(rs) / len(R))]
rng.shuffle(pick); pick = pick[:1200]
cnt = collections.Counter(r["secondary_category"] for r in pick); cats = sorted(cnt); random.Random(8).shuffle(cats)
split, acc = {}, 0
for c in cats:
    split[c] = "dev" if acc < 0.5 * len(pick) else ("val" if acc < 0.7 * len(pick) else "test"); acc += cnt[c]
SYS = "你是知识问答助手，请直接、简洁地回答问题。"
with open("items.jsonl", "w", encoding="utf-8") as f:
    for r in pick:
        f.write(json.dumps(dict(item_id="h-" + r["id"][:12], concept="C3H", template_id=r["secondary_category"], primary=r["primary_category"],
                                split=split[r["secondary_category"]], system=SYS, messages=[{"role": "user", "content": r["question"]}],
                                gold=r["answer"]), ensure_ascii=False) + "\n")
print(collections.Counter(split[r["secondary_category"]] for r in pick), "categories", collections.Counter(split.values()))
