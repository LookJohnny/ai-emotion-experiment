"""E02 analysis E (PLAN.md) + emotion-direction geometry, on Qwen3-4B features.
python geometry.py FEATS.npz PACK_DIR OUT.json
- probe vs emotion: cosine between each product probe's raw-space weight (coef / scale) and the emotion - neutral directions
  (residual, lead and anchor sites) at the probe's layer; null = within-scenario label-permutation directions (50 draws).
- emotion vs emotion: cosines between joy/sadness/anger - neutral directions at every layer (a valence axis would show anger
  and sadness closer to each other than to joy)."""
import json, os, sys
import numpy as np

FE, PACK, OUT = sys.argv[1:4]
z = np.load(FE); emo, scn = z["emotion"], z["scenario"]; EMOS = ["joy", "sadness", "anger", "neutral"]
y = np.array([EMOS.index(e) for e in emo]); us = np.array(sorted(set(scn))); g = np.searchsorted(us, scn)
rng = np.random.default_rng(0); cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
diff = lambda X, k, yy=y: X[yy == k].mean(0) - X[yy == 3].mean(0)


def perm(k):
    yp = y.copy()
    for s in range(len(us)):
        ix = np.where((g == s) & np.isin(y, [k, 3]))[0]; yp[ix] = rng.permutation(y[ix])
    return yp


man = json.load(open(os.path.join(PACK, "manifest.json")))
out = {"probe_vs_emotion": {}, "emotion_vs_emotion": []}
for name, f in man["probes"].items():
    p = json.load(open(os.path.join(PACK, f))); L = p["layer"]; w = np.array(p["coef"]) / np.array(p["scale"])
    row = {"layer": L}
    for s, site in enumerate(("lead", "anchor")):
        X = z["res"][s, :, L].astype(np.float64)
        for k, e in enumerate(EMOS[:3]):
            nul = [cos(w, diff(X, k, perm(k))) for _ in range(50)]
            row[f"{site}/{e}"] = {"cos": round(cos(w, diff(X, k)), 4), "null_mean": round(float(np.mean(nul)), 4),
                                  "null_abs_p95": round(float(np.percentile(np.abs(nul), 95)), 4)}
    out["probe_vs_emotion"][name] = row
for L in range(z["res"].shape[2]):
    row = {"layer": L}
    for s, site in enumerate(("lead", "anchor")):
        X = z["res"][s, :, L].astype(np.float64); d = [diff(X, k) for k in range(3)]
        row[site] = {"joy~sad": round(cos(d[0], d[1]), 4), "joy~anger": round(cos(d[0], d[2]), 4), "sad~anger": round(cos(d[1], d[2]), 4)}
    out["emotion_vs_emotion"].append(row)
json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
for n, r in out["probe_vs_emotion"].items():
    print(n, "L", r["layer"], " ".join(f"{k}:{v['cos']:+.3f}(null|p95| {v['null_abs_p95']:.3f})" for k, v in r.items() if k != "layer"))
print("layer  lead joy~sad joy~anger sad~anger | anchor joy~sad joy~anger sad~anger")
for r in out["emotion_vs_emotion"]:
    a, b = r["lead"], r["anchor"]
    print(f"{r['layer']:2d} | {a['joy~sad']:+.2f} {a['joy~anger']:+.2f} {a['sad~anger']:+.2f} | {b['joy~sad']:+.2f} {b['joy~anger']:+.2f} {b['sad~anger']:+.2f}")
