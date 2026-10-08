"""Analyse blind ratings of held-out tone generations.  python eval/analyze.py eval/rating > eval/results.json"""
import glob, json, os, random, statistics as st, sys
rd = sys.argv[1]; key = json.load(open(f"{rd}/blind_key.json")); S = {}
for f in glob.glob(f"{rd}/ratings_*_*.json"):
    for x in json.load(open(f)): S.setdefault(os.path.basename(f).split("_")[1], {})[x["item_id"]] = x
D = ["joy", "sadness", "anger", "warmth", "fluency"]; rng = random.Random(0)
item = {i: {**k, **{d: st.mean(float(S[s][i][d]) for s in S) for d in D}} for i, k in key.items() if all(i in S[s] for s in S)}
assert len(item) == len(key), f"incomplete {len(item)}/{len(key)}"
def boot(x, B=10000):
    m = sorted(st.mean(rng.choice(x) for _ in x) for _ in range(B)); return [round(st.mean(x), 3), [round(m[int(.025 * B)], 3), round(m[int(.975 * B)], 3)]]
def spearman(a, b):
    ra = {i: r for r, i in enumerate(sorted(range(len(a)), key=lambda i: a[i]))}; rb = {i: r for r, i in enumerate(sorted(range(len(b)), key=lambda i: b[i]))}
    n = len(a); return round(1 - 6 * sum((ra[i] - rb[i]) ** 2 for i in range(n)) / (n * (n * n - 1)), 3)
base = {v["msg_id"]: v for v in item.values() if v["attribute"] == "none"}; R = {"n_items": len(item), "raters": sorted(S)}
ia = [S["A"][i] for i in key]; ib = [S["B"][i] for i in key]
R["rater_agreement_spearman"] = {d: spearman([x[d] for x in ia], [x[d] for x in ib]) for d in D}
for a in ("anger", "joy", "sadness", "warmth"):
    r = {}
    for s in (1.0, 2.0, -1.0):
        rows = [v for v in item.values() if v["attribute"] == a and v["strength"] == s]
        r[str(s)] = {d: boot([v[d] - base[v["msg_id"]][d] for v in rows]) for d in D}
    r["readout_vs_rating_spearman_all_items"] = spearman([v["readout"].get(a, 0) for v in item.values()], [v[a] for v in item.values()])
    R[a] = r
print(json.dumps(R, ensure_ascii=False, indent=1))
