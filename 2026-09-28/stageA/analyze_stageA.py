"""Stage A analysis: join ratings with blind key, compute ceilings with scenario-level bootstrap CI.
python3 analyze_stageA.py blind_key.json rating_a.json [rating_b.json ...]
Rating file: JSON array [{item_id, joy, sadness, anger, fluency, new_fact_error, no_fact_claim}]"""
import json, sys, random, statistics as st
key = json.load(open(sys.argv[1]))
R = [{x["item_id"]: x for x in json.load(open(p))} for p in sys.argv[2:]]
DIMS = ["joy", "sadness", "anger", "fluency", "new_fact_error"]
# mean over raters per item
item = {}
for iid, k in key.items():
    got = [r[iid] for r in R if iid in r]
    if got:
        item[iid] = dict(k, **{d: st.mean(float(g[d]) for g in got) for d in DIMS})
# per scenario x condition: mean over paraphrases
cell = {}
for v in item.values():
    cell.setdefault((v["split"], v["scenario_id"], v["condition"]), []).append(v)
S = {k: {d: st.mean(x[d] for x in vs) for d in DIMS} for k, vs in cell.items()}
rng = random.Random(0)


def boot(diffs, B=10000):
    n = len(diffs); m = [st.mean(rng.choice(diffs) for _ in range(n)) for _ in range(B)]
    m.sort(); return st.mean(diffs), m[int(0.025 * B)], m[int(0.975 * B)]


out = {}
for split in ["validation", "discovery"]:
    scen = sorted({k[1] for k in S if k[0] == split})
    if not scen: continue
    conds = sorted({k[2] for k in S if k[0] == split})
    res = {"n_scenarios": len(scen), "means": {c: {d: round(st.mean(S[(split, s, c)][d] for s in scen if (split, s, c) in S), 3) for d in DIMS} for c in conds}}
    for name, a, b, d in [("prefix_ceiling_joy", "prefix_joy", "prefix_neutral", "joy"),
                          ("anchor_ceiling_joy", "anchor_sub_joy", "prefix_neutral", "joy"),
                          ("prefix_sadness_effect", "prefix_sadness", "prefix_neutral", "sadness"),
                          ("prefix_anger_effect", "prefix_anger", "prefix_neutral", "anger"),
                          ("anchor_sub_fact_error_delta", "anchor_sub_joy", "prefix_neutral", "new_fact_error"),
                          ("anchor_sub_fluency_delta", "anchor_sub_joy", "prefix_neutral", "fluency")]:
        diffs = [S[(split, s, a)][d] - S[(split, s, b)][d] for s in scen if (split, s, a) in S and (split, s, b) in S]
        if diffs:
            m, lo, hi = boot(diffs); res[name] = dict(mean=round(m, 3), ci95=[round(lo, 3), round(hi, 3)], n=len(diffs))
    if "prefix_ceiling_joy" in res:
        pc = res["prefix_ceiling_joy"]
        res["gate_A"] = dict(rule="prefix ceiling joy >= 0.5/4 and CI lower > 0 (validation)",
                             passed=pc["mean"] >= 0.5 and pc["ci95"][0] > 0)
    out[split] = res
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open("stageA_results.json", "w"), ensure_ascii=False, indent=1)
