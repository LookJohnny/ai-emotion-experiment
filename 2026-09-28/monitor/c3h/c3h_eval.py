"""C3-H gates G1-G3 + reported metrics (PREREG_C3H.md). python c3h_eval.py answers.jsonl labels_all.jsonl scores.jsonl split OUT.json"""
import json, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
ANS, LAB, SC, SPLIT, OUT = sys.argv[1:6]
it = {json.loads(l)["item_id"]: json.loads(l) for l in open(ANS, encoding="utf-8")}
lab = {json.loads(l)["item_id"]: json.loads(l)["grade"] for l in open(LAB, encoding="utf-8")}
sc = {json.loads(l)["item_id"]: json.loads(l) for l in open(SC, encoding="utf-8")}
rng = np.random.default_rng(2026)
dev = [k for k in it if it[k]["split"] == "dev" and lab.get(k) in ("CORRECT", "INCORRECT")]
te = [k for k in it if it[k]["split"] == SPLIT and lab.get(k) in ("CORRECT", "INCORRECT")]
tea = [k for k in it if it[k]["split"] == SPLIT and k in lab]
y = np.array([lab[k] == "INCORRECT" for k in te], int); g = [it[k]["template_id"] for k in te]
q = lambda k: it[k]["messages"][-1]["content"]
v = TfidfVectorizer(analyzer="char", ngram_range=(1, 3)).fit([q(k) for k in dev])
tq = LogisticRegression(max_iter=2000).fit(v.transform([q(k) for k in dev]), [lab[k] == "INCORRECT" for k in dev]).predict_proba(v.transform([q(k) for k in te]))[:, 1]
va = TfidfVectorizer(analyzer="char", ngram_range=(1, 3)).fit([it[k]["target_text"] for k in dev])
ta = LogisticRegression(max_iter=2000).fit(va.transform([it[k]["target_text"] for k in dev]), [lab[k] == "INCORRECT" for k in dev]).predict_proba(va.transform([it[k]["target_text"] for k in te]))[:, 1]
er = {}
for k in dev: er.setdefault(it[k]["primary"], []).append(lab[k] == "INCORRECT")
cat = np.array([np.mean(er.get(it[k]["primary"], [0.5])) for k in te])
p = np.array([sc[k]["c3h_pre"] for k in te]); pt = np.array([sc[k]["c3h_token"] for k in te])
A = lambda yy, ss: roc_auc_score(yy, ss) if len(set(yy)) == 2 else np.nan


def boot(fn, n=2000):
    ug = sorted(set(g)); idx = {c: np.where(np.array(g) == c)[0] for c in ug}; o = []
    for _ in range(n):
        i = np.concatenate([idx[c] for c in rng.choice(ug, len(ug))]); x = fn(i)
        if not np.isnan(x): o.append(x)
    return [round(float(np.percentile(o, 2.5)), 3), round(float(np.percentile(o, 97.5)), 3)]


def verdict(pt_, ci, thr, diff=False):
    if ci[0] >= thr and (not diff or ci[0] > 0): return "PASS"
    return "FAIL" if pt_ < thr else "INCONCLUSIVE"


a = A(y, p); ci = boot(lambda i: A(y[i], p[i])); rep = {"n": len(y), "incorrect": int(y.sum()), "G1_pre_auroc": dict(point=round(a, 3), ci=ci, thr=0.70, verdict=verdict(a, ci, 0.70))}
for nm, b in [("G2_minus_question_text", tq), ("G3_minus_category", cat)]:
    d = a - A(y, b); cd = boot(lambda i: A(y[i], p[i]) - A(y[i], b[i]))
    rep[nm] = dict(point=round(d, 3), ci=cd, thr=0.05, baseline_auroc=round(A(y, b), 3), verdict="PASS" if (cd[0] > 0 and cd[0] >= 0.05) else ("FAIL" if d < 0.05 else "INCONCLUSIVE"))
ya = np.array([lab[k] == "INCORRECT" for k in tea], int); pa_ = np.array([sc[k]["c3h_pre"] for k in tea])
bins = np.quantile(p, [0, .2, .4, .6, .8, 1]); cal = [dict(bin=i, mean_p=round(float(p[(p >= bins[i]) & (p <= bins[i + 1])].mean()), 3), err_rate=round(float(y[(p >= bins[i]) & (p <= bins[i + 1])].mean()), 3)) for i in range(5)]
rep["reported"] = dict(auroc_incl_not_attempted=round(A(ya, pa_), 3), token_auroc=round(A(y, pt), 3), answer_text_auroc=round(A(y, ta), 3),
                       question_text_auroc=round(A(y, tq), 3), category_auroc=round(A(y, cat), 3), quintiles=cal)
json.dump(rep, open(OUT, "w"), ensure_ascii=False, indent=1); print(json.dumps(rep, ensure_ascii=False, indent=1))
