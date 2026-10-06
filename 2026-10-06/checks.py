"""Evaluate CLAIMS.md H1-H10 on holdout outputs, exactly as written.  python checks.py > out/checks_holdout.json"""
import json, random
import numpy as np
from scipy.linalg import subspace_angles
from scipy.stats import spearmanr

M = {"L": "llama1b", "Q1": "qwen17b", "Q4": "qwen4b"}
load = lambda m, p: json.load(open(f"out/{M[m]}_{p}.json"))
ws = lambda x: x["eta_emo"] / max(1e-9, 1 - x["eta_scn"])
EMOS = ["joy", "sadness", "anger", "neutral"]; res = {}


def null_p95(d):
    r = np.random.default_rng(0)
    return float(np.percentile([np.cos(subspace_angles(r.standard_normal((d, 3)), r.standard_normal((d, 3))).min()) for _ in range(200)], 95))


for m in M:
    o = load(m, "holdout"); L = o["layers"]; NL = o["meta"]["layers"]
    auc = [l["anchor"]["emo_auc"] for l in L]; pk = int(np.argmax(auc)); dpk = pk / (NL - 1)
    res[f"H1/{m}"] = dict(peak_layer=pk, depth=round(dpk, 3), drop=round(auc[pk] - auc[-1], 4), passed=bool(0.35 <= dpk <= 0.65 and auc[pk] - auc[-1] >= 0.02))
    res[f"H2/{m}"] = dict(eta_emo=L[pk]["anchor"]["eta_emo"], auc=auc[pk], passed=bool(L[pk]["anchor"]["eta_emo"] < 0.05 and auc[pk] >= 0.95))
    p95 = null_p95(o["meta"]["hidden"])
    for site in ("lead", "anchor"):
        frac = float(np.mean([l[site]["emo_vs_content_cos"] > p95 for l in L]))
        res[f"H3/{m}/{site}"] = dict(share_above_null=round(frac, 3), null_p95=round(p95, 3), passed=bool(frac <= 0.2))
    below = float(np.mean([l["lead"]["anger_std"]["top1pct"] < l["lead"]["null_std_mean_p95"]["top1pct"][1] for l in L]))
    res[f"H4/{m}"] = dict(share_layers_below_null=round(below, 3), passed=bool(below >= 0.9))
    z = np.load(f"feats/{M[m]}_holdout.npz"); y = np.array([EMOS.index(e) for e in z["emotion"]])
    cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)); gaps = []
    for l in range(NL):
        X = z["res"][1, :, l].astype(np.float64); d = [X[y == k].mean(0) - X[y == 3].mean(0) for k in range(3)]
        if l / (NL - 1) >= 0.25: gaps.append(cos(d[1], d[2]) - cos(d[0], d[2]))
    res[f"H5/{m}"] = dict(min_gap=round(min(gaps), 4), passed=bool(min(gaps) > 0))

ex, ho = load("Q4", "explore")["heads"], load("Q4", "holdout")["heads"]
for name, f in (("within_share", lambda h: ws(h["anchor"])), ("emo_auc", lambda h: h["anchor"]["emo_auc"])):
    rho = spearmanr([f(h) for h in ex], [f(h) for h in ho]).correlation
    res[f"H6/{name}"] = dict(rho=round(float(rho), 3), passed=bool(rho >= 0.6))
    E01 = json.load(open("../2026-09-28/stageD/FREEZE_5b.json"))["heads"]
    order = sorted(ho, key=lambda h: -f(h)); rank = {f"L{h['layer']}H{h['head']}": i + 1 for i, h in enumerate(order)}; top = len(ho) // 10
    k = sum(rank[x] <= top for x in E01)
    pool = [f"L{h['layer']}H{h['head']}" for h in ho if 12 <= h["layer"] <= 27]; random.seed(0)
    sims = [sum(rank[x] <= top for x in random.sample(pool, 12)) for _ in range(10000)]
    p = sum(s >= k for s in sims) / len(sims)
    res[f"H7/{name}"] = dict(in_top10pct=f"{k}/12", ranks=sorted(rank[x] for x in E01), perm_p=p, passed=bool(p < 0.05))
NL = 36; top = sorted(ho, key=lambda h: -ws(h["anchor"]))[:len(ho) // 20]
share = float(np.mean([0.3 <= h["layer"] / (NL - 1) <= 0.7 for h in top]))
res["H8/Q4"] = dict(share_mid=round(share, 3), passed=bool(share >= 0.7))
g = json.load(open("out/qwen4b_geometry_holdout.json"))["probe_vs_emotion"]
cells = [(n, k, v) for n in ("c1_token", "c3h_token", "syco_token") for k, v in g[n].items() if k != "layer"]
bad = [f"{n}/{k}" for n, k, v in cells if abs(v["cos"]) > v["null_abs_p95"]]
res["H9/Q4"] = dict(cells=len(cells), above_null=bad, passed=not bad)
c = g["c2"]["lead/anger"]; res["H10/Q4"] = dict(cos=c["cos"], null_abs_p95=c["null_abs_p95"], passed=bool(c["cos"] > c["null_abs_p95"]))
print(json.dumps(res, ensure_ascii=False, indent=1))
for k, v in res.items(): print(("PASS " if v["passed"] else "FAIL ") + k, {a: b for a, b in v.items() if a != "passed"})
