"""Pre-registered analysis for STUDY_PLAN.md §5. Committed before any data collection.
python analyze.py responses/ trials.json gold_spans.json report.json
- maps public ids back (same hash as server/app.py), applies exclusion rules, then:
  primary  : judgement accuracy ~ condition, mixed logistic (participant + item random intercepts; statsmodels Bayesian GLMM, VB fit)
             + participant-cluster bootstrap CI for T+2D - T accuracy difference (robustness)
  secondary: localisation IoU on true alerts, log RT, Brier score, high-confidence-wrong rate, false-alert detection,
             sensitivity analysis without borderline trials s21-s23; subjective helpfulness vs accuracy (illusion-of-understanding check)"""
import json, sys, glob, hashlib, math, random, statistics as st
import numpy as np
RD, TR, GS, OUT = sys.argv[1:5]
T = {t["trial_id"]: t for t in json.load(open(TR))}; G = json.load(open(GS))
PRIV = {"q" + hashlib.sha256(("iomon" + k).encode()).hexdigest()[:6]: k for k in T}
BORDER = {"s21", "s22", "s23"}
sessions = []
for f in glob.glob(f"{RD}/*.jsonl"):
    for l in open(f): sessions.append(json.loads(l))
rows, excluded = [], []
for S in sessions:
    tr = [dict(t, tid=PRIV[t["trial_id"]]) for t in S["trials"]]
    prac = [t for t in tr if t["practice"]]; form = [t for t in tr if not t["practice"]]
    p_acc = np.mean([t["answer"] == T[t["tid"]]["truth"] for t in prac]) if prac else 1
    same = len({t["answer"] for t in form}) == 1; med_rt = st.median(t["rt_ms"] for t in form) / 1000 if form else 0
    if (p_acc < .5 and same) or med_rt < 3: excluded.append({"participant": S["participant"], "practice_acc": p_acc, "all_same": same, "median_rt_s": med_rt}); continue
    for t in form:
        truth = T[t["tid"]]["truth"]; gold = set(i for a, b in G[t["tid"]]["ranges"] for i in range(a, b))
        sel = set(range(*t["span"])) if t["span"] else set()
        iou = (len(gold & sel) / len(gold | sel)) if (gold | sel) else 1.0
        rows.append(dict(p=S["participant"], item=t["tid"], cond=t["condition"], correct=int(t["answer"] == truth), truth=truth, iou=iou if truth == 1 else None,
                         conf=t["conf"], rt=t["rt_ms"] / 1000, border=t["tid"] in BORDER))
def summ(rs):
    out = {}
    for c in ["T", "2D", "3D"]:
        r = [x for x in rs if x["cond"] == c]
        if not r: continue
        pc = [x["conf"] / 7 for x in r]
        out[c] = dict(n=len(r), accuracy=round(np.mean([x["correct"] for x in r]), 3),
                      iou_true_alerts=round(np.mean([x["iou"] for x in r if x["iou"] is not None]), 3),
                      log_rt=round(np.mean([math.log(x["rt"]) for x in r]), 3),
                      brier=round(np.mean([(p - x["correct"]) ** 2 for p, x in zip(pc, r)]), 3),
                      high_conf_wrong=round(np.mean([x["conf"] >= 6 and not x["correct"] for x in r]), 3),
                      false_alert_detected=round(np.mean([x["correct"] for x in r if x["truth"] == 0]), 3) if any(x["truth"] == 0 for x in r) else None)
    return out
def boot_diff(rs, a="2D", b="T", n=4000, seed=0):
    ps = sorted({x["p"] for x in rs}); rng = random.Random(seed); d = []
    for _ in range(n):
        pick = [rng.choice(ps) for _ in ps]; A = [x["correct"] for p in pick for x in rs if x["p"] == p and x["cond"] == a]; B = [x["correct"] for p in pick for x in rs if x["p"] == p and x["cond"] == b]
        if A and B: d.append(np.mean(A) - np.mean(B))
    return [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)] if d else None
rep = {"n_sessions": len(sessions), "excluded": excluded, "n_rows": len(rows), "by_condition": summ(rows),
       "sensitivity_without_borderline": summ([x for x in rows if not x["border"]]),
       "H1_boot_2D_minus_T": boot_diff(rows), "H3_boot_3D_minus_2D": boot_diff(rows, "3D", "2D")}
try:   # primary GLMM
    import pandas as pd
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    df = pd.DataFrame(rows); df["cond"] = pd.Categorical(df["cond"], ["T", "2D", "3D"])
    m = BinomialBayesMixedGLM.from_formula("correct ~ C(cond, Treatment('T'))", {"p": "0 + C(p)", "item": "0 + C(item)"}, df).fit_vb()
    rep["GLMM"] = {n: dict(mean=round(float(mu), 3), sd=round(float(sd), 3)) for n, mu, sd in zip(m.model.exog_names, m.fe_mean, m.fe_sd)}
except Exception as e: rep["GLMM"] = f"not run: {type(e).__name__}: {e}"
rt = {}
for S in sessions:
    for r in S.get("ratings", []): rt.setdefault(r["condition"], []).append(r["helpful"])
rep["subjective_helpfulness"] = {c: round(st.mean(v), 2) for c, v in rt.items()}
json.dump(rep, open(OUT, "w"), ensure_ascii=False, indent=1); print(json.dumps(rep, ensure_ascii=False, indent=1))
