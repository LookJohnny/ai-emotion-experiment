"""E02 analyses A-C (PLAN.md) on one model's features.  python analyze.py FEATS.npz OUT.json
A head map (anchor site; lead reported too): eta_emo (within-scenario emotion share of variance), eta_scn, emo_auc (4-way macro
  one-vs-rest AUROC, standardise + LR C=0.05, 5-fold CV grouped by scenario), class (emotion / content / mixed / weak).
B layer profile (residual, both sites): eta_emo, eta_scn, emo_auc, and cos of the smallest principal angle between the
  within-scenario emotion LDA subspace (3 dims, shrinkage) and the top-3 scenario-mean PCs.
C basis alignment of anger-neutral (residual, lead and anchor): participation ratio, top-1% coordinate share of the squared
  norm, share on massive-activation coordinates (mean |x| > 20x median), each against a within-scenario label-permutation null."""
import json, sys, time
import numpy as np
from scipy.linalg import subspace_angles
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

FE, OUT = sys.argv[1:3]
z = np.load(FE); meta = json.loads(str(z["meta"]))
emo, scn = z["emotion"], z["scenario"]; EMOS = ["joy", "sadness", "anger", "neutral"]
y = np.array([EMOS.index(e) for e in emo]); us = np.array(sorted(set(scn))); g = np.searchsorted(us, scn)
rng = np.random.default_rng(0); folds = np.array_split(rng.permutation(len(us)), 5)


def eta(X):
    X = X.astype(np.float64); grand = X.mean(0)
    mu_s = np.zeros((len(us), X.shape[1])); np.add.at(mu_s, g, X); mu_s /= np.bincount(g)[:, None]
    R = X - mu_s[g]; m_e = np.stack([R[y == k].mean(0) for k in range(4)])
    tot = ((X - grand) ** 2).sum()
    return float((((m_e[y]) ** 2).sum()) / tot), float(((mu_s[g] - grand) ** 2).sum() / tot), R


def emo_auc(X):
    P = np.zeros((len(y), 4))
    for f in folds:
        te = np.isin(g, f); sc = StandardScaler().fit(X[~te])
        c = LogisticRegression(max_iter=500, C=0.05).fit(sc.transform(X[~te]), y[~te]); P[te] = c.predict_proba(sc.transform(X[te]))
    return float(np.mean([roc_auc_score(y == k, P[:, k]) for k in range(4)]))


def klass(a, ee, es):
    if a >= 0.75 and ee > es: return "emotion"
    if es >= 0.5 and a < 0.6: return "content"
    if a >= 0.75: return "mixed"
    return "weak"


def subspace_cos(X, R):
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(R, y)
    E = lda.scalings_[:, :3]
    mu_s = np.stack([X[g == k].mean(0) for k in range(len(us))]); mu_s -= mu_s.mean(0)
    S = np.linalg.svd(mu_s, full_matrices=False)[2][:3].T
    return float(np.cos(subspace_angles(E, S).min()))


def basis(d, massive):
    d2 = d.astype(np.float64) ** 2; s = d2.sum(); k = max(1, len(d) // 100)
    return {"pr": float(s ** 2 / (d2 ** 2).sum() / len(d)), "top1pct": float(np.sort(d2)[-k:].sum() / s), "massive": float(d2[massive].sum() / s)}


def perm_diff(X):
    """null: anger-vs-neutral labels shuffled within each scenario (same counts, same scenarios)."""
    yp = y.copy()
    for k in range(len(us)):
        ix = np.where((g == k) & np.isin(y, [2, 3]))[0]; yp[ix] = rng.permutation(y[ix])
    return X[yp == 2].mean(0) - X[yp == 3].mean(0)


t0 = time.time(); NL, NH, HD = meta["layers"], meta["heads"], meta["head_dim"]; out = {"meta": meta, "heads": [], "layers": []}
for L in range(NL):
    for h in range(NH):
        row = {"layer": L, "head": h}
        for s, site in enumerate(("lead", "anchor")):
            X = z["head"][s, :, L, h * HD:(h + 1) * HD].astype(np.float32)
            ee, es, _ = eta(X); a = emo_auc(X)
            row[site] = {"eta_emo": round(ee, 4), "eta_scn": round(es, 4), "emo_auc": round(a, 4), "class": klass(a, ee, es)}
        out["heads"].append(row)
    lay = {"layer": L, "depth": round(L / (NL - 1), 3)}
    for s, site in enumerate(("lead", "anchor")):
        X = z["res"][s, :, L].astype(np.float32); ee, es, R = eta(X)
        massive = np.abs(X).mean(0) > 20 * np.median(np.abs(X).mean(0))
        d = X[y == 2].mean(0) - X[y == 3].mean(0); sd = X.std(0) + 1e-6
        nulls = [perm_diff(X) for _ in range(20)]
        bz = lambda f: {k: [round(float(np.mean([f(n)[k] for n in nulls])), 4), round(float(np.percentile([f(n)[k] for n in nulls], 95)), 4)] for k in ("pr", "top1pct", "massive")}
        lay[site] = {"eta_emo": round(ee, 4), "eta_scn": round(es, 4), "emo_auc": round(emo_auc(X), 4), "emo_vs_content_cos": round(subspace_cos(X, R), 4),
                     "n_massive": int(massive.sum()), "massive_dims": np.where(massive)[0][:10].tolist(),
                     "anger_raw": {k: round(v, 4) for k, v in basis(d, massive).items()}, "null_raw_mean_p95": bz(lambda n: basis(n, massive)),
                     "anger_std": {k: round(v, 4) for k, v in basis(d / sd, massive).items()}, "null_std_mean_p95": bz(lambda n: basis(n / sd, massive))}
    out["layers"].append(lay); print(L, NL, f"{time.time() - t0:.0f}s", flush=True)
json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=0)
print("DONE", round(time.time() - t0))
