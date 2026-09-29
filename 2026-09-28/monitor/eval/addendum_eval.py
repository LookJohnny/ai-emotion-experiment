"""Addendum A analyses (construct validity) on the sealed test. python3 addendum_eval.py probes/outputs_test.jsonl probes_noRT/outputs_test.jsonl FREEZE_monitor.json out.json"""
import json, sys, random, numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
main, nort, fzp, out = sys.argv[1:5]; P = json.load(open(fzp))["policy"]
def load(p): return [json.loads(l) for l in open(p)]
def score(o):
    if o["concept"] == "C2": return o["p_c2"]
    ema = o["p_pre"]; mx = ema
    for p in o["p_tokens"]:
        inc = P["alpha"] * (p - ema)
        if o["p_pre"] < P["c1_pre_thr"] and inc > 0: inc *= P["gamma"]
        ema += inc; mx = max(mx, ema)
    return mx
def auc_ci(O, f=score, B=5000):
    y = np.array([o["label"] for o in O]); s = np.array([f(o) for o in O])
    if len(set(y)) < 2: return None
    tids = sorted({o["template_id"] for o in O}); rng = random.Random(0); bs = []
    idx = {t: [i for i, o in enumerate(O) if o["template_id"] == t] for t in tids}
    for _ in range(B):
        ii = [i for t in (rng.choice(tids) for _ in tids) for i in idx[t]]
        if len(set(y[ii])) == 2: bs.append(roc_auc_score(y[ii], s[ii]))
    fpr, tpr, _ = roc_curve(y, s)
    return dict(n=len(O), pos=int(y.sum()), auroc=round(roc_auc_score(y, s), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                recall_at_fpr5=round(float(np.interp(0.05, fpr, tpr)), 4))
M = load(main); N = load(nort); R = {}
c1 = [o for o in M if o["concept"] == "C1"]
R["A1_within_blunt_main_probe"] = auc_ci([o for o in c1 if o["persona"] == "blunt"])
R["A1_within_blunt_pre_only"] = auc_ci([o for o in c1 if o["persona"] == "blunt"], f=lambda o: o["p_pre"])
R["A2_leave_redteam_out_on_blunt_default"] = auc_ci([o for o in N if o["concept"] == "C1" and o["persona"] in ("blunt", "default")])
R["A2_leave_redteam_out_within_blunt"] = auc_ci([o for o in N if o["concept"] == "C1" and o["persona"] == "blunt"])
per = {"default": 0, "blunt": 1, "redteam": 2}
R["A3_persona_only_baseline_all_C1"] = auc_ci(c1, f=lambda o: per.get(o["persona"], 0))
R["A3_main_probe_all_C1"] = auc_ci(c1)
json.dump(R, open(out, "w"), ensure_ascii=False, indent=1); print(json.dumps(R, ensure_ascii=False, indent=1))
