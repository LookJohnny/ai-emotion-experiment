"""Print the main numbers of an analyze.py output.  python summarize.py OUT.json"""
import json, sys, collections
import numpy as np
from scipy.linalg import subspace_angles
o = json.load(open(sys.argv[1])); m = o["meta"]; print(m["model"], m["layers"], "layers", m["heads"], "heads")
r = np.random.default_rng(0); d = m["hidden"]
nc = [np.cos(subspace_angles(r.standard_normal((d, 3)), r.standard_normal((d, 3))).min()) for _ in range(200)]
print(f"null subspace cos (random 3-d pairs in {d} dims): mean {np.mean(nc):.3f}, p95 {np.percentile(nc, 95):.3f}")
ws = lambda x: x["eta_emo"] / max(1e-9, 1 - x["eta_scn"])
print("layer depth | lead: auc eta_e eta_s cos | anchor: auc eta_e eta_s cos | anger top1% raw (null p95) std (null p95) massive-share raw (null p95) n_massive")
for l in o["layers"]:
    a, b = l["lead"], l["anchor"]
    print(f"{l['layer']:2d} {l['depth']:.2f} | {a['emo_auc']:.3f} {a['eta_emo']:.3f} {a['eta_scn']:.3f} {a['emo_vs_content_cos']:.2f} | "
          f"{b['emo_auc']:.3f} {b['eta_emo']:.3f} {b['eta_scn']:.3f} {b['emo_vs_content_cos']:.2f} | "
          f"{a['anger_raw']['top1pct']:.3f} ({a['null_raw_mean_p95']['top1pct'][1]:.3f}) {a['anger_std']['top1pct']:.3f} ({a['null_std_mean_p95']['top1pct'][1]:.3f}) "
          f"{a['anger_raw']['massive']:.3f} ({a['null_raw_mean_p95']['massive'][1]:.3f}) {a['n_massive']}")
H = o["heads"]
for site in ("lead", "anchor"):
    c = collections.Counter(h[site]["class"] for h in H); print(site, "classes", dict(c))
    top = sorted(H, key=lambda h: -h[site]["emo_auc"])[:12]
    print(" top within_share:", ", ".join(f"L{h['layer']}H{h['head']} {ws(h[site]):.2f}" for h in sorted(H, key=lambda h: -ws(h[site]))[:8]))
    print(" top emo_auc:", ", ".join(f"L{h['layer']}H{h['head']} {h[site]['emo_auc']:.2f}/{h[site]['eta_emo']:.2f}/{h[site]['eta_scn']:.2f}" for h in top))
    byL = collections.defaultdict(list)
    for h in H: byL[h["layer"]].append(h[site]["emo_auc"])
    print(" per-layer max/median head auc:", " ".join(f"{L}:{max(v):.2f}/{sorted(v)[len(v)//2]:.2f}" for L, v in sorted(byL.items())))
