"""Pick study stimuli (STUDY_PLAN.md §3): from V1.1 C1 sealed TEST, monitor-alerted items (item_score >= demo thr):
16 true alerts (label 1) + 8 false alerts (label 0); at most 3 per template; practice = 1 TP + 1 FP from VAL. Greedy + samples allowed.
python select_stimuli.py  -> trials.json"""
import json, random, collections
V = "../v1_1"
lab = {json.loads(l)["item_id"]: json.loads(l) for l in open(f"{V}/c1/labels_all.jsonl")}
items = {json.loads(l)["item_id"]: json.loads(l) for l in open(f"{V}/c1/items_all.jsonl")}
S = {json.loads(l)["item_id"]: json.loads(l) for l in open(f"{V}/c1_out/c1_scores.jsonl")}
thr = json.load(open(f"{V}/demo_thresholds.json"))["item_score"]["thr"]
rng = random.Random(20260930)
def pick(split, label, n, cap=3):
    pool = [k for k, s in S.items() if s["split"] == split and k in lab and lab[k]["label"] == label and s["item_score"] >= thr]
    rng.shuffle(pool); out, per = [], collections.Counter()
    for k in pool:
        t = items[k]["template_id"]
        if per[t] < cap: out.append(k); per[t] += 1
        if len(out) == n: break
    return out
sel = [("formal", k, 1) for k in pick("test", 1, 16)] + [("formal", k, 0) for k in pick("test", 0, 8)] + \
      [("practice", k, 1) for k in pick("val", 1, 1)] + [("practice", k, 0) for k in pick("val", 0, 1)]
trials = []
for n, (phase, k, y) in enumerate(sel):
    it = items[k]
    trials.append(dict(trial_id=f"{'p' if phase == 'practice' else 's'}{n:02d}", phase=phase, source_item=k, truth=y, rater_scores=lab[k]["rater_scores"],
                       system=it["system"], messages=it["messages"], reply=it["target_text"], item_score=round(S[k]["item_score"], 4)))
json.dump(trials, open("trials.json", "w"), ensure_ascii=False, indent=1)
print(collections.Counter((t["phase"], t["truth"]) for t in trials), "templates", len({items[t["source_item"]]["template_id"] for t in trials}))
