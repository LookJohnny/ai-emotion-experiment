"""E03 step 1 (PREREG_E03.md): the anchor head map on discovery, frozen before any generation.
python map_select.py FEATS_discovery.npz OUT_DIR
-> OUT_DIR/MAP_FREEZE.json (top-12 heads by within_share, full ranking, residual depth profile, sha256 of the vectors file)
-> OUT_DIR/anger_minus_neutral.npy  [layers, heads*head_dim]: per head, mean anchor head output (A+B) anger - neutral.
   Used for the target heads and, as their own true vectors, for the random-head control."""
import hashlib, json, os, sys
import numpy as np

FE, OUT = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True)
z = np.load(FE); meta = json.loads(str(z["meta"])); NL, NH, HD = meta["layers"], meta["heads"], meta["head_dim"]
emo, scn = z["emotion"], z["scenario"]; EMOS = ["joy", "sadness", "anger", "neutral"]
y = np.array([EMOS.index(e) for e in emo]); us = np.array(sorted(set(scn))); g = np.searchsorted(us, scn)


def within_share(X):
    X = X.astype(np.float64); grand = X.mean(0)
    mu = np.zeros((len(us), X.shape[1])); np.add.at(mu, g, X); mu /= np.bincount(g)[:, None]
    R = X - mu[g]; m = np.stack([R[y == k].mean(0) for k in range(4)])
    tot = ((X - grand) ** 2).sum(); scn_ss = ((mu[g] - grand) ** 2).sum()
    return float((m[y] ** 2).sum() / (tot - scn_ss))


A = z["head"][1]                                                      # anchor site: [N, layers, heads*hd]
score = {(L, h): within_share(A[:, L, h * HD:(h + 1) * HD]) for L in range(NL) for h in range(NH)}
rank = sorted(score, key=lambda k: -score[k]); top = rank[:12]
vec = (A[y == 2].astype(np.float64).mean(0) - A[y == 3].astype(np.float64).mean(0)).astype(np.float32)   # [layers, heads*hd]
vp = os.path.join(OUT, "anger_minus_neutral.npy"); np.save(vp, vec)

prof = []                                                             # residual: within-scenario emotion share by depth (reported)
for L in range(NL):
    prof.append({"layer": L, "depth": round(L / (NL - 1), 3), "within_share": round(within_share(z["res"][1, :, L]), 4)})

fz = {"model": meta["model"], "split": "discovery", "n_records": int(len(y)), "rule": "top 12 heads by within_share at the anchor site, no layer restriction",
      "heads": [f"L{L}H{h}" for L, h in top], "scores": {f"L{L}H{h}": round(score[(L, h)], 5) for L, h in top},
      "ranking": [f"L{L}H{h}" for L, h in rank], "residual_profile": prof, "layers": NL, "heads_per_layer": NH, "head_dim": HD,
      "vectors_file": os.path.basename(vp), "vectors_sha256": hashlib.sha256(open(vp, "rb").read()).hexdigest(),
      "feats_sha256": hashlib.sha256(open(FE, "rb").read()).hexdigest()}
json.dump(fz, open(os.path.join(OUT, "MAP_FREEZE.json"), "w"), ensure_ascii=False, indent=1)
print("top-12:", fz["heads"]); print("scores:", fz["scores"])
print("residual within_share peak:", max(prof, key=lambda p: p["within_share"]))
