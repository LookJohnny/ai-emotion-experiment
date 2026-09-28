"""Combine blind plausibility rounds -> plausibility_combined.json, manifest.plausibility, review_sheet columns.
Each scenario uses only raters that saw its CURRENT version. Run after build_formal.py."""
import json, statistics as st, csv
E = ["joy", "sadness", "anger"]
def load(f): return {x["id"]: x for x in json.load(open(f))}
R = {"v2": load("plausibility_blind_v2.json"), "v3": load("plausibility_blind_v3.json"), "v4": load("plausibility_blind_v4.json"),
     "v5a": load("plausibility_v5a.json"), "v5b": load("plausibility_v5b.json"), "v6a": load("plausibility_v6a.json"), "v6b": load("plausibility_v6b.json"),
     "v7a": load("plausibility_v7a.json"), "v7b": load("plausibility_v7b.json")}
import os
if os.path.exists("plausibility_v8a.json"): R["v8a"] = load("plausibility_v8a.json"); R["v8b"] = load("plausibility_v8b.json")
# which scenarios were edited after each round (so later raters only)
edits = [({"fam01","fam04","fam08","fam14","fam15","hobby03","shop02","shop06","shop07","study06","trav07","trav15","trav16","trav18","trav20","work02","work04"}, ["v3","v4"]),
         ({"fam01","fam08","study06","trav16","trav20"}, ["v4"]),
         ({"shop08","shop09","shop11","trav02","trav12","trav14","fam03","fam10","work01","work17","hobby05"}, ["v5a","v5b"]),
         ({"fam03","fam10","shop11","hobby05"}, ["v6a","v6b"]),
         ({"fam02","fam09","hobby02","hobby11","shop05","shop15","shop17","study05","trav01","trav08","work03","work16"}, ["v7a","v7b"])]
if "v8a" in R: edits.append(({"shop05"}, ["v8a","v8b"]))
comb = {}
for i in R["v4"]:
    rs = ["v2", "v3", "v4"]
    for ids, raters in edits:
        if i in ids: rs = raters
    comb[i] = dict(n_raters=len(rs), raters=rs, **{e: round(st.mean(R[r][i][e] for r in rs), 2) for e in E})
json.dump(comb, open("plausibility_combined.json", "w"), ensure_ascii=False, indent=0)
ok25 = sum(min(c[e] for e in E) >= 2.5 for c in comb.values()); ok3 = sum(min(c[e] for e in E) >= 3 for c in comb.values())
weak = sorted(i for i, c in comb.items() if min(c[e] for e in E) < 2.5)
split = {}
for sp, pre in [("discovery", ("shop", "trav", "fam")), ("validation", ("study",)), ("test", ("work", "hobby"))]:
    g = [c for i, c in comb.items() if i.startswith(pre)]
    split[sp] = dict(mean={e: round(st.mean(c[e] for c in g), 2) for e in E}, pass_2_5=f"{sum(min(c[e] for e in E) >= 2.5 for c in g)}/{len(g)}")
m = json.load(open("manifest.json"))
m["version"] = "draft-v2"; m["status"] = "DRAFT_v2_needs_human_review"
m["plausibility"] = {"method": "independent blind LLM raters (fresh agent each round, facts only, shuffled order), 1-5 per emotion; combined = mean of raters that saw the current version",
    "criterion_agreed": "combined mean >=2.5 on all three emotions in >=90% of scenarios",
    "final_combined_mean_ge_2.5": f"{ok25}/120", "final_combined_mean_ge_3": f"{ok3}/120", "by_split": split, "weak_scenarios": weak,
    "history": {"v1_single_all_ge3": "6/120", "v2_round2": "103/120", "v3_round3": "98/120 (rater noise, inter-rater |Δ|=0.24)", "v4_round4": "102/120"},
    "rater_files": ["plausibility_blind_v1..v4.json", "plausibility_v5a/b..v8a/b.json"]}
if "v2: all scenarios rewritten" not in m["provenance"]:
    m["provenance"] += "; v2: all scenarios rewritten to 3-element mixed events (BRIEF.md v2 addendum), iterated against blind plausibility raters"
json.dump(m, open("manifest.json", "w"), ensure_ascii=False, indent=1)
rows = list(csv.reader(open("review_sheet.csv", encoding="utf-8-sig")))
if "盲评joy" not in rows[0]:
    out = [rows[0][:10] + ["盲评joy", "盲评sad", "盲评ang", "盲评最弱项"] + rows[0][10:]]
    for r in rows[1:]:
        c = comb[r[0].split("-")[0]]; lo = min(E, key=lambda e: c[e])
        out.append(r[:10] + [c["joy"], c["sadness"], c["anger"], f"{lo}={c[lo]}" if c[lo] < 3 else ""] + r[10:])
    csv.writer(open("review_sheet.csv", "w", newline="", encoding="utf-8-sig")).writerows(out)
print(f"mean>=2.5: {ok25}/120  mean>=3: {ok3}/120  weak(<2.5): {weak}"); print(json.dumps(split, ensure_ascii=False))
print("scenarios with some emotion in [2.5,3):", sorted((i, {e: c[e] for e in E if c[e] < 3}) for i, c in comb.items() if 2.5 <= min(c[e] for e in E) < 3))
