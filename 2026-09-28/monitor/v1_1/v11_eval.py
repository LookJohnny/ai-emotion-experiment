"""V1.1 gates (EVAL_PREREG_v11.md §4). Bootstrap 2000 draws by template/topic; three-valued verdicts.
c2: python v11_eval.py c2 m4/transfer_train.jsonl m4/labels.jsonl c2_out/c2_scores.jsonl c2_out/report_c2.json
c1: python v11_eval.py c1 c1/items_all.jsonl c1/labels_all.jsonl c1_out/c1_scores.jsonl c1_out/report_c1.json   (test labels only after freeze)"""
import json, sys, itertools
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
MODE, A, B, S, OUT = sys.argv[1:6]
load = lambda p: [json.loads(l) for l in open(p)]
KEYS = ["垃圾", "骗子", "气死", "烦死", "滚", "废物", "混蛋", "该死", "傻", "蠢", "恶心"]
M4DEV = set("k102 k103 k106 k107 k108 k109 k112 k116 k119 k120 k121 k126 k127 k128 k132 k133 k135 k137 k138 k141 k144 k146 k147 k148 k149 k150 k153 k155 k156 k157".split())
M4TEST = set("k101 k110 k115 k118 k123 k124 k125 k130 k131 k134 k136 k139 k142 k143 k145 k158 k159 k160".split())
rng = np.random.default_rng(2026)


def text_model(tr_x, tr_y, te_x):
    v = TfidfVectorizer(analyzer="char", ngram_range=(1, 3)).fit(tr_x)
    return LogisticRegression(max_iter=2000).fit(v.transform(tr_x), tr_y).predict_proba(v.transform(te_x))[:, 1]


def auc(y, s): return roc_auc_score(y, s) if len(set(y)) == 2 else np.nan


def boot(groups, fn, n=2000):
    ug = sorted(set(groups)); idx = {g: np.where(np.array(groups) == g)[0] for g in ug}; out = []
    for _ in range(n):
        pick = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))]); v = fn(pick)
        if not np.isnan(v): out.append(v)
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]


def verdict(point, ci, thr):
    if ci[0] >= thr: return "PASS"
    if point < thr: return "FAIL"
    return "INCONCLUSIVE"


rep = {}
if MODE == "c2":
    tr = load(A); m4 = {r["item_id"]: r for r in load(B)}; sc = load(S)
    y = np.array([r["label"] for r in sc]); g = [r["template_id"] for r in sc]; cell = np.array([r["cell"] for r in sc])
    txt = [m4[r["item_id"]]["target_text"] for r in sc]
    p = np.array([r["p_transfer"] for r in sc]); t = text_model([r["target_text"] for r in tr], [r["label"] for r in tr], txt)
    kw = np.array([sum(k in x for k in KEYS) for x in txt], float)
    a_p, a_t = auc(y, p), auc(y, t)
    ci_p = boot(g, lambda i: auc(y[i], p[i])); ci_d = boot(g, lambda i: auc(y[i], p[i]) - auc(y[i], t[i]))
    rep["G5_transfer_auroc"] = dict(point=round(a_p, 3), ci=ci_p, thr=0.80, verdict=verdict(a_p, ci_p, 0.80))
    rep["G6_minus_same_data_text"] = dict(point=round(a_p - a_t, 3), ci=ci_d, thr=0.10, text_auroc=round(a_t, 3),
                                          verdict="PASS" if ci_d[0] > 0 and (a_p - a_t) >= 0.10 and verdict(a_p - a_t, ci_d, 0.10) == "PASS" else
                                          ("FAIL" if a_p - a_t < 0.10 else "INCONCLUSIVE"))
    off = np.isin(cell, ["pos_nokw", "neg_kw"])
    rep["reported"] = dict(
        keyword_list_auroc=round(auc(y, kw), 3),
        offdiag_probe=round(auc(y[off], p[off]), 3), offdiag_text=round(auc(y[off], t[off]), 3), offdiag_keyword=round(auc(y[off], kw[off]), 3),
        by_cell_mean_probe={c: round(float(p[cell == c].mean()), 3) for c in sorted(set(cell))},
        by_cell_mean_text={c: round(float(t[cell == c].mean()), 3) for c in sorted(set(cell))})
    dev = [r for r in sc if r["template_id"] in M4DEV]; te = np.array([r["template_id"] in M4TEST for r in sc])
    ti = text_model([m4[r["item_id"]]["target_text"] for r in dev], [r["label"] for r in dev], [x for x, k in zip(txt, te) if k])
    pin = np.array([r["p_indomain"] for r in sc])
    rep["reported"]["indomain_test"] = dict(probe=round(auc(y[te], pin[te]), 3), text=round(auc(y[te], ti), 3), n=int(te.sum()), pos=int(y[te].sum()))
else:
    items = {r["item_id"]: r for r in load(A)}; lab = {r["item_id"]: r["label"] for r in load(B)}; sc = [r for r in load(S)]
    conv = lambda r: "\n".join(m["content"] for m in r["messages"]) + "\n<reply>\n" + r["target_text"]
    dev = [r for r in sc if r["split"] == "dev" and r["item_id"] in lab]; te = [r for r in sc if r["split"] == "test" and r["item_id"] in lab]
    y = np.array([lab[r["item_id"]] for r in te]); g = [r["template_id"] for r in te]
    p = np.array([r["item_score"] for r in te]); pre = np.array([r["p_pre"] for r in te]); e16 = np.array([r["early16"] for r in te])
    t = text_model([conv(items[r["item_id"]]) for r in dev], [lab[r["item_id"]] for r in dev], [conv(items[r["item_id"]]) for r in te])
    first16 = lambda r: "\n".join(m["content"] for m in items[r["item_id"]]["messages"]) + "\n<reply>\n" + items[r["item_id"]]["target_text"][:24]
    t16 = text_model([first16(r) for r in dev], [lab[r["item_id"]] for r in dev], [first16(r) for r in te])
    lvl = np.array([{"polite": 0, "curt": 1, "sarcastic": 2, "insulting": 3}[r["level"]] for r in te], float)
    a = auc(y, p); ci = boot(g, lambda i: auc(y[i], p[i]))
    rep["G1_c1_token_auroc"] = dict(point=round(a, 3), ci=ci, thr=0.85, verdict=verdict(a, ci, 0.85), n=len(y), pos=int(y.sum()))
    for name, base in [("G2_minus_fullconv_text", t), ("G3_minus_level_only", lvl)]:
        d = a - auc(y, base); cd = boot(g, lambda i: auc(y[i], p[i]) - auc(y[i], base[i]))
        rep[name] = dict(point=round(d, 3), ci=cd, thr=0.05, baseline_auroc=round(auc(y, base), 3),
                         verdict="PASS" if cd[0] > 0 and verdict(d, cd, 0.05) == "PASS" else ("FAIL" if d < 0.05 else "INCONCLUSIVE"))
    # G4: within-prompt pairs (same template + level; mixed labels)
    P = {}
    for k, r in enumerate(te): P.setdefault((r["template_id"], r["level"]), []).append(k)
    mixed = {k: v for k, v in P.items() if 0 < y[v].sum() < len(v)}
    def pair_acc(ks, s):
        ok = n = 0
        for key in ks:
            v = mixed[key]; pos = [i for i in v if y[i]]; neg = [i for i in v if not y[i]]
            for i, j in itertools.product(pos, neg): ok += (s[i] > s[j]) + 0.5 * (s[i] == s[j]); n += 1
        return ok / n if n else np.nan
    keys = sorted(mixed); pa = pair_acc(keys, p)
    if len(keys) >= 3:
        bs = [pair_acc([keys[i] for i in rng.integers(0, len(keys), len(keys))], p) for _ in range(2000)]
        cp = [round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]
        v4 = "PASS" if cp[0] > 0.5 and pa >= 0.65 and verdict(pa, cp, 0.65) == "PASS" else ("FAIL" if pa < 0.65 else "INCONCLUSIVE")
    else: cp, v4 = None, "INCONCLUSIVE (fewer than 3 mixed prompts)"
    rep["G4_within_prompt_pairwise"] = dict(point=round(pa, 3) if not np.isnan(pa) else None, ci=cp, thr=0.65, verdict=v4, mixed_prompts=len(keys),
                                            text_pairwise=round(pair_acc(keys, t), 3) if keys else None, early16_pairwise=round(pair_acc(keys, e16), 3) if keys else None)
    rep["reported"] = dict(c1_pre_auroc=round(auc(y, pre), 3), early16_auroc=round(auc(y, e16), 3), text_first24chars_auroc=round(auc(y, t16), 3),
                           fullconv_text_auroc=round(auc(y, t), 3), level_only_auroc=round(auc(y, lvl), 3))
json.dump(rep, open(OUT, "w"), ensure_ascii=False, indent=1); print(json.dumps(rep, ensure_ascii=False, indent=1))
