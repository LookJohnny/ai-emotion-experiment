"""Fabrication v2 gates F1-F3 (PREREG_SYCO_FAB2.md Part F2). Run on the GPU host (sklearn), all features precomputed by gpu_feats.py.
python fab2_eval.py FAB2_DIR C3H_DIR READER_DIR
  FAB2_DIR : qwen8b_answers.jsonl, labels_8b.jsonl, feats_8b_own.npz, kimi_{dev,val}.jsonl, labels_kimi_{dev,val}.jsonl,
             feats4_kimi_{dev,val,test}.npz (4B, layer 19, 64 tok), feats8_kimi_{dev,val,test}.npz (8B, candidate layers, 64 tok)
  C3H_DIR  : out/c3h_token.json (frozen 4B probe), labels_all.jsonl (4B own grades)
  READER_DIR: kimi_c3h_test.jsonl, labels_c3h.jsonl (Kimi-test, secondary)"""
import json, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer
D, C3, RD = sys.argv[1:4]
load = lambda p: {json.loads(l)["item_id"]: json.loads(l) for l in open(p, encoding="utf-8")}
rng = np.random.default_rng(2026); auc = lambda y, s: roc_auc_score(y, s) if len(set(y)) == 2 else np.nan
tok = AutoTokenizer.from_pretrained(os.environ.get("TOK", "/root/autodl-tmp/E01/models/models/Qwen--Qwen3-4B-Instruct-2507/snapshots/master"))


def boot(groups, fn, n=2000):
    ug = sorted(set(groups)); idx = {g: np.where(np.array(groups) == g)[0] for g in ug}; o = []
    for _ in range(n):
        v = fn(np.concatenate([idx[g] for g in rng.choice(ug, len(ug))]))
        if not np.isnan(v): o.append(v)
    return [round(float(np.percentile(o, 2.5)), 3), round(float(np.percentile(o, 97.5)), 3)]


def gate(point, ci, thr): return "PASS" if ci[0] >= thr else ("FAIL" if point < thr else "INCONCLUSIVE")


# ---- 8B probe on 8B's own answers (same procedure as c3h_token: mean over answer tokens, candidate layers, CV by category)
A8, L8 = load(f"{D}/qwen8b_answers.jsonl"), load(f"{D}/labels_8b.jsonl"); z = np.load(f"{D}/feats_8b_own.npz"); iz = {k: i for i, k in enumerate(z["ids"])}
dev = [k for k in A8 if A8[k]["split"] == "dev" and L8.get(k, {}).get("grade") in ("CORRECT", "INCORRECT")]
yd = np.array([L8[k]["grade"] == "INCORRECT" for k in dev], int); gd = np.array([A8[k]["template_id"] for k in dev]); LAY = list(z["layers"])
def cv(X):
    ug = np.array(sorted(set(gd))); r = np.random.default_rng(0); r.shuffle(ug); a = []
    for f in np.array_split(ug, 5):
        te = np.isin(gd, f)
        if len(set(yd[te])) < 2: continue
        sc = StandardScaler().fit(X[~te]); c = LogisticRegression(max_iter=1000, C=0.05).fit(sc.transform(X[~te]), yd[~te]); a.append(roc_auc_score(yd[te], c.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.mean(a))
res = {L: cv(z["tok"][[iz[k] for k in dev], j].astype(np.float32)) for j, L in enumerate(LAY)}; L8b = max(res, key=res.get); j8 = LAY.index(L8b)
Xd = z["tok"][[iz[k] for k in dev], j8].astype(np.float32); sc8 = StandardScaler().fit(Xd); c8 = LogisticRegression(max_iter=1000, C=0.05).fit(sc8.transform(Xd), yd)
print("8B probe layer", L8b, {k: round(v, 3) for k, v in res.items()}, flush=True)
p4 = json.load(open(f"{C3}/out/c3h_token.json")); mu4, sd4, w4, b4 = (np.array(p4[k]) for k in ("mean", "scale", "coef", "intercept"))


def readers(split):
    f4, f8 = np.load(f"{D}/feats4_kimi_{split}.npz"), np.load(f"{D}/feats8_kimi_{split}.npz")
    r4 = 1 / (1 + np.exp(-(((f4["tok"][:, 0].astype(np.float32) - mu4) / sd4) @ w4 + b4)))
    r8 = c8.predict_proba(sc8.transform(f8["tok"][:, list(f8["layers"]).index(L8b)].astype(np.float32)))[:, 1]
    return {k: (a, b) for k, a, b in zip(f4["ids"], r4, r8)}


def kimi_lp(r):
    cut = tok.decode(tok(r["content"] or "", add_special_tokens=False)["input_ids"][:64], skip_special_tokens=True); nll, ent, n = [], [], 0
    for t in r.get("logprobs") or []:
        if n >= len(cut): break
        n += len(t["t"] or ""); nll.append(-t["lp"]); ps = np.exp(np.array([x for x in t["top"] if x is not None] or [t["lp"]])); ps = ps / ps.sum(); ent.append(float(-(ps * np.log(ps + 1e-12)).sum()))
    return (float(np.mean(nll)) if nll else 0.0, float(np.mean(ent)) if ent else 0.0)


def table(split, kimi_file, label_file):
    K, L, R = load(kimi_file), load(label_file), readers(split); keys = [k for k in K if L.get(k, {}).get("grade") in ("CORRECT", "INCORRECT") and k in R]
    y = np.array([L[k]["grade"] == "INCORRECT" for k in keys], int); g = [K[k]["template_id"] for k in keys]
    F = np.array([[R[k][0], R[k][1], *kimi_lp(K[k])] for k in keys]); return keys, y, g, F, K


kd, yd_, gd_, Fd, _ = table("dev", f"{D}/kimi_dev.jsonl", f"{D}/labels_kimi_dev.jsonl")
scb = StandardScaler().fit(Fd); comb = LogisticRegression(max_iter=1000, C=1.0).fit(scb.transform(Fd), yd_)
G4 = {json.loads(l)["item_id"]: json.loads(l)["grade"] for l in open(f"{C3}/labels_all.jsonl")}
rep = {"8b_probe_layer": int(L8b), "8b_cv": {str(k): round(v, 3) for k, v in res.items()}, "combo_coef(r4,r8,nll,ent)": [round(x, 3) for x in comb.coef_[0]]}
for split, kf, lf in [("val", f"{D}/kimi_val.jsonl", f"{D}/labels_kimi_val.jsonl"), ("test", f"{RD}/kimi_c3h_test.jsonl", f"{RD}/labels_c3h.jsonl")]:
    keys, y, g, F, K = table(split, kf, lf); cb = comb.predict_proba(scb.transform(F))[:, 1]; o = {"n": len(y), "incorrect": int(y.sum())}
    r4, r8 = F[:, 0], F[:, 1]
    if split == "val" and y.sum() < 30: o["verdict"] = "INCONCLUSIVE (<30 incorrect)"
    else:
        a8 = auc(y, r8); c = boot(g, lambda i: auc(y[i], r8[i])); o["F1_8b_reader"] = dict(point=round(a8, 3), ci=c, thr=0.75, verdict=gate(a8, c, 0.75))
        d = a8 - auc(y, r4); cd = boot(g, lambda i: auc(y[i], r8[i]) - auc(y[i], r4[i])); o["F2_8b_minus_4b"] = dict(point=round(d, 3), ci=cd, thr=0.05, r4_auroc=round(auc(y, r4), 3), verdict=gate(d, cd, 0.05))
        ac = auc(y, cb); cc = boot(g, lambda i: auc(y[i], cb[i])); o["F3_combination"] = dict(point=round(ac, 3), ci=cc, thr=0.78, verdict=gate(ac, cc, 0.78))
    o["reported"] = dict(kimi_nll=round(auc(y, F[:, 2]), 3), kimi_entropy=round(auc(y, F[:, 3]), 3))
    k4 = np.array([G4.get(K[k].get("src_item", k[2:])) for k in keys]); k8 = np.array([L8.get(K[k].get("src_item", k[2:]), {}).get("grade") for k in keys])
    o["by_reader_knowledge"] = {f"{nm}_{v}": dict(n=int((arr == v).sum()), incorrect=int(y[arr == v].sum()), r4=round(auc(y[arr == v], r4[arr == v]), 3) if (arr == v).sum() > 5 else None,
                                                  r8=round(auc(y[arr == v], r8[arr == v]), 3) if (arr == v).sum() > 5 else None)
                                for nm, arr in (("4b", k4), ("8b", k8)) for v in ("CORRECT", "INCORRECT")}
    rep["kimi_" + split] = o
json.dump(rep, open(f"{D}/report_fab2.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(rep, ensure_ascii=False, indent=1))
