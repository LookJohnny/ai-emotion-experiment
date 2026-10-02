"""PREREG_TRANSFER.md: (a) native retrain vs (b) label-free align-transfer, gates U1/U2.  python tr_eval.py NAME N_LAYERS
Reads feats/ref/*.npz (Qwen3-4B at 16/19/20), feats/NAME/*.npz, data/labels.json, data/*.jsonl, frozen reference probes."""
import json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.preprocessing import StandardScaler
NAME, N = sys.argv[1], int(sys.argv[2]); D = "/root/autodl-tmp/E01/tr"
LAB = json.load(open(f"{D}/data/labels.json")); rng = np.random.default_rng(2026)
auc = lambda y, s: roc_auc_score(y, s) if len(set(y)) == 2 else np.nan
C = {"c1": ("c1_token.json", 16, 0.80), "sy": ("syco_tok.json", 20, 0.80), "fa": ("c3h_token.json", 19, 0.70)}
NATIVE = sorted({round(f * N) for f in (.30, .40, .45, .50, .55, .60, .70)})


def load(model, ds):
    z = np.load(f"{D}/feats/{model}/{ds}.npz"); rows = {json.loads(l)["item_id"]: json.loads(l) for l in open(f"{D}/data/{ds}.jsonl", encoding="utf-8")}
    return list(z["ids"]), list(z["layers"]), z["tok"], rows


def feat(model, ds, layer):
    ids, lay, tok, rows = load(model, ds); return ids, tok[:, lay.index(layer)].astype(np.float32), rows


def labeled(c, ids):
    keep = [i for i, k in enumerate(ids) if LAB.get(f"{c}:{k}") is not None]
    return keep, np.array([LAB[f"{c}:{ids[i]}"] for i in keep])


def boot(g, fn, n=2000):
    ug = sorted(set(g)); idx = {x: np.where(np.array(g) == x)[0] for x in ug}; o = []
    for _ in range(n):
        v = fn(np.concatenate([idx[x] for x in rng.choice(ug, len(ug))]))
        if not np.isnan(v): o.append(v)
    return [round(float(np.percentile(o, 2.5)), 3), round(float(np.percentile(o, 97.5)), 3)]


def ref_probe(c):
    p = json.load(open(f"{D}/data/{C[c][0]}")); mu, sd, w, b = (np.array(p[k], dtype=np.float32) for k in ("mean", "scale", "coef", "intercept"))
    return lambda X: 1 / (1 + np.exp(-(((X - mu) / sd) @ w + b)))


rep = {"model": NAME, "n_layers": N, "native_candidates": NATIVE}
for c, (pf, Lr, thr) in C.items():
    P = ref_probe(c); La = round(Lr / 36 * N); o = {"ref_layer": Lr, "align_layer": La}
    ids, Xd, rows = feat(NAME, f"{c}_dev", NATIVE[0]); kd, yd = labeled(c, ids); gd = np.array([rows[ids[i]]["template_id"] for i in kd])
    # (a) native: layer by grouped 5-fold CV on dev
    def cv(L):
        _, X, _ = feat(NAME, f"{c}_dev", L); X = X[kd]; ug = np.array(sorted(set(gd))); r = np.random.default_rng(0); r.shuffle(ug); a = []
        for f in np.array_split(ug, 5):
            te = np.isin(gd, f)
            if len(set(yd[te])) < 2 or len(set(yd[~te])) < 2: continue
            s = StandardScaler().fit(X[~te]); m = LogisticRegression(max_iter=2000, C=0.05).fit(s.transform(X[~te]), yd[~te]); a.append(roc_auc_score(yd[te], m.predict_proba(s.transform(X[te]))[:, 1]))
        return float(np.mean(a))
    cvs = {L: cv(L) for L in NATIVE}; Ln = max(cvs, key=cvs.get)
    _, X, _ = feat(NAME, f"{c}_dev", Ln); s = StandardScaler().fit(X[kd]); m = LogisticRegression(max_iter=2000, C=0.05).fit(s.transform(X[kd]), yd)
    native = lambda X: m.predict_proba(s.transform(X))[:, 1]
    o["native_layer"], o["native_cv"] = int(Ln), {str(k): round(v, 3) for k, v in cvs.items()}
    # (b) align-transfer: ridge map target(La) -> reference(Lr) on ALL dev texts of all three concepts, labels unused
    def anchors(limit=None):
        XT, XR = [], []
        for cc in C:
            it, xt, _ = feat(NAME, f"{cc}_dev", La); ir, xr, _ = feat("ref", f"{cc}_dev", Lr); pos = {k: i for i, k in enumerate(ir)}
            sel = [i for i, k in enumerate(it) if k in pos]
            if limit: sel = list(np.random.default_rng(7).permutation(sel)[:limit])
            XT.append(xt[sel]); XR.append(xr[[pos[it[i]] for i in sel]])
        return np.concatenate(XT), np.concatenate(XR)
    def fit_map(XT, XR):
        sT = StandardScaler().fit(XT); Z = sT.transform(XT); best = None
        for a in (1, 10, 100, 1000):
            folds = np.array_split(np.random.default_rng(1).permutation(len(Z)), 5); r2 = []
            for f in folds:
                tr = np.setdiff1d(np.arange(len(Z)), f); r2.append(r2_score(XR[f], Ridge(alpha=a).fit(Z[tr], XR[tr]).predict(Z[f])))
            if best is None or np.mean(r2) > best[1]: best = (a, float(np.mean(r2)))
        R = Ridge(alpha=best[0]).fit(Z, XR); return (lambda X: P(R.predict(sT.transform(X)))), best
    tb, (alpha, r2) = fit_map(*anchors()); tb3, (alpha3, r23) = fit_map(*anchors(300))
    o["align_alpha"], o["align_r2"], o["align300_r2"] = alpha, round(r2, 3), round(r23, 3)
    # evaluate
    for split in ("test", "kimi"):
        ds = f"{c}_{split}"; ids, XN, rows = feat(NAME, ds, Ln); _, XA, _ = feat(NAME, ds, La); k, y = labeled(c, ids)
        g = [rows[ids[i]].get("template_id", "x") for i in k]; sa, sb, sb3 = native(XN[k]), tb(XA[k]), tb3(XA[k])
        ir, XR, _ = feat("ref", ds, Lr); pos = {kk: i for i, kk in enumerate(ir)}; sr = P(XR[[pos[ids[i]] for i in k]])
        r = {"n": len(y), "pos": int(y.sum()), "native_a": round(auc(y, sa), 3), "align_b": round(auc(y, sb), 3), "align300_b2": round(auc(y, sb3), 3), "reference_qwen3_4b": round(auc(y, sr), 3)}
        if split == "test":
            d = auc(y, sb) - auc(y, sa); cd = boot(g, lambda i: auc(y[i], sb[i]) - auc(y[i], sa[i])); cb = boot(g, lambda i: auc(y[i], sb[i]))
            r["U1_b_minus_a"] = {"point": round(d, 3), "ci": cd, "verdict": "PASS" if cd[0] >= -0.03 else ("FAIL" if d < -0.03 else "INCONCLUSIVE")}
            r["U2_b_abs"] = {"point": r["align_b"], "ci": cb, "thr": thr, "verdict": "PASS" if cb[0] >= thr else ("FAIL" if r["align_b"] < thr else "INCONCLUSIVE")}
            ca = boot(g, lambda i: auc(y[i], sa[i])); r["native_ci"] = ca; r["native_meets_U2"] = "PASS" if ca[0] >= thr else ("FAIL" if r["native_a"] < thr else "INCONCLUSIVE")
        o[split] = r
    rep[c] = o
json.dump(rep, open(f"{D}/report_{NAME}.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(rep, ensure_ascii=False, indent=1))
