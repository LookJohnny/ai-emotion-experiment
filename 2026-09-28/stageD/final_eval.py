"""§5.4 test-set evaluation against the frozen gates.
python3 final_eval.py rating_test FREEZE.json
Uses rating_test/blind_key.json + ratings_{A,B}_{1,2}.json (made by select_freeze.py make on main_final_test.jsonl)."""
import json, sys, os, glob, random, statistics as st
out, fzp = sys.argv[1:3]; fz = json.load(open(fzp)); P = fz["primary"]; lam = fz["main_lambda"]
key = json.load(open(f"{out}/blind_key.json")); streams = {}
for f in glob.glob(f"{out}/ratings_*_*.json"):
    s = os.path.basename(f).split("_")[1]
    for x in json.load(open(f)): streams.setdefault(s, {})[x["item_id"]] = x
D = ["joy", "sadness", "anger", "fluency", "new_fact_error"]
item = {i: dict(k, **{d: st.mean(float(streams[s][i][d]) for s in streams) for d in D}) for i, k in key.items() if all(i in streams[s] for s in streams)}
assert len(item) == len(key), f"incomplete ratings {len(item)}/{len(key)}"
cell = {}
for v in item.values():
    c = "random" if v["condition"].startswith("random_") else v["condition"]; cell.setdefault((v["scenario_id"], c), []).append(v)
S = {k: {d: st.mean(x[d] for x in vs) for d in D} for k, vs in cell.items()}
scen = sorted({k[0] for k in S}); rng = random.Random(0)
def boot(x, B=10000):
    m = sorted(st.mean(rng.choice(x) for _ in x) for _ in range(B)); return round(st.mean(x), 3), [round(m[int(.025 * B)], 3), round(m[int(.975 * B)], 3)]
g = lambda c, d: [S[(s, c)][d] for s in scen]
diff = lambda c1, c2, d: [a - b for a, b in zip(g(c1, d), g(c2, d))]
others = [e for e in ["joy", "sadness", "anger"] if e != P]
R = dict(n_scenarios=len(scen), primary=P, frozen_lambda=lam, object=fz["main_object"],
         target_minus_random=boot(diff("target", "random", P)), target_minus_none=boot(diff("target", "none", P)),
         anchor_ceiling=boot(diff("anchor_sub", "none", P)), prefix_ceiling=boot(diff("primary_prefix", "none", P)),
         specificity={e: boot([a - b for a, b in zip(diff("target", "none", P), diff("target", "none", e))]) for e in others},
         consistency_target_minus_none=boot([(1 - a) - (1 - b) for a, b in zip(g("target", "new_fact_error"), g("none", "new_fact_error"))]),
         fluency_target_minus_none=boot(diff("target", "none", "fluency")),
         means={c: {d: round(st.mean(g(c, d)), 3) for d in D} for c in sorted({k[1] for k in S})})
if any((s, "shuffled") in S for s in scen): R["target_minus_shuffled"] = boot(diff("target", "shuffled", P))
need = 0.5 * R["anchor_ceiling"][0]
R["gates"] = {"main_effect>=0.5*anchor_ceiling & CI>0": R["target_minus_random"][0] >= need and R["target_minus_random"][1][0] > 0,
              "target_vs_none CI>0": R["target_minus_none"][1][0] > 0,
              "specificity CI>0 (all other emotions)": all(v[1][0] > 0 for v in R["specificity"].values()),
              "consistency CI>-5pp": R["consistency_target_minus_none"][1][0] > -0.05,
              "fluency CI>-0.3": R["fluency_target_minus_none"][1][0] > -0.3}
R["required_main_effect"] = round(need, 3); R["all_gates_passed"] = all(R["gates"].values())
R["note"] = "Ratings are blind LLM raters (provisional), not the two human raters of plan §5.4. CI too wide -> 'inconclusive', not pass/fail."
print(json.dumps(R, ensure_ascii=False, indent=1)); json.dump(R, open(f"{out}/final_results.json", "w"), ensure_ascii=False, indent=1)
