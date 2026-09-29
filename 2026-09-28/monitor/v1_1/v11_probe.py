"""V1.1 probes (EVAL_PREREG_v11.md §3). Probes are trained and ALL items are scored in one pass, so scores are fixed before test labels exist.
c1: python v11_probe.py c1 MODEL items_all.jsonl labels_dev.jsonl OUTDIR
    train C1-pre / C1-token on dev labels (layer by template-grouped 5-fold CV), score every item (dev/val/test) -> OUTDIR/c1_scores.jsonl
c2: python v11_probe.py c2 MODEL transfer_train.jsonl m4_labels.jsonl OUTDIR
    transfer probe: train on transfer set, score all M4; in-domain probe: train on M4 dev topics, score all M4 -> OUTDIR/c2_scores.jsonl
C2 features never include a system prompt."""
import json, sys, os, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core.probe import ProbeSpec

MODE, MODEL, A, B, OUT = sys.argv[1:6]; os.makedirs(OUT, exist_ok=True)
LAYERS = [12, 16, 19, 22, 26]
M4DEV = set("k102 k103 k106 k107 k108 k109 k112 k116 k119 k120 k121 k126 k127 k128 k132 k133 k135 k137 k138 k141 k144 k146 k147 k148 k149 k150 k153 k155 k156 k157".split())
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
load = lambda p: [json.loads(l) for l in open(p)]


@torch.no_grad()
def c2_feat(r):
    msgs = [m for m in r["messages"] if m["role"] != "system"]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    e = tok(text, return_offsets_mapping=True, add_special_tokens=False); u = msgs[-1]["content"]
    c0 = text.rindex(u); c1 = c0 + len(u); idx = [i for i, (s, t) in enumerate(e["offset_mapping"]) if s >= c0 and t <= c1 and t > s]
    hs = model(torch.tensor([e["input_ids"]], device="cuda"), output_hidden_states=True, use_cache=False).hidden_states
    return {L: hs[L + 1][0, idx[0]:idx[-1] + 1].float().mean(0).cpu().numpy() for L in LAYERS}


@torch.no_grad()
def c1_feat(r):
    msgs = ([{"role": "system", "content": r["system"]}] if r.get("system") else []) + r["messages"]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pids = tok(text, add_special_tokens=False)["input_ids"]; rids = tok(r["target_text"], add_special_tokens=False)["input_ids"]
    hs = model(torch.tensor([pids + rids], device="cuda"), output_hidden_states=True, use_cache=False).hidden_states
    n = len(pids)
    return {L: (hs[L + 1][0, n - 1].float().cpu().numpy(), hs[L + 1][0, n:n + len(rids)].float().cpu().numpy()) for L in LAYERS}


def fit(X, y, w=None):
    sc = StandardScaler().fit(X); return sc, LogisticRegression(max_iter=1000, C=0.05).fit(sc.transform(X), y, sample_weight=w)


def cv_auc(X, y, g, w=None, pool=None):
    ug = np.array(sorted(set(g))); rng = np.random.default_rng(0); rng.shuffle(ug); aucs = []
    for f in np.array_split(ug, 5):
        te = np.isin(g, f)
        if len(set(y[te])) < 2 or len(set(y[~te])) < 2: continue
        sc, clf = fit(X[~te], y[~te], None if w is None else w[~te]); p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        if pool is not None:
            it = pool[te]; u = sorted(set(it)); p = np.array([p[it == i].mean() for i in u]); yy = np.array([y[te][it == i][0] for i in u])
        else: yy = y[te]
        aucs.append(roc_auc_score(yy, p))
    return float(np.mean(aucs)) if aucs else float("nan")


def choose(build, name, concept, kind):
    res = {}; best = None
    for L in LAYERS:
        X, y, g, w, pool = build(L); a = cv_auc(X, y, g, w, pool); res[L] = round(a, 4)
        if best is None or a > best[0] + 1e-9: best = (a, L, X, y, w)
    a, L, X, y, w = best; sc, clf = fit(X, y, w)
    spec = ProbeSpec.from_sklearn(concept, name, L, kind, sc, clf, meta={"cv_auroc": round(a, 4), "cv_by_layer": res})
    spec.save(f"{OUT}/{name}.json"); print(name, "layer", L, "CV", round(a, 4), res, flush=True); return spec


t0 = time.time()
if MODE == "c1":
    items = load(A); lab = {r["item_id"]: r["label"] for r in load(B) if r["split"] == "dev"}
    F = {r["item_id"]: c1_feat(r) for r in items}; print("features", len(F), f"{time.time()-t0:.0f}s", flush=True)
    dev = [r for r in items if r["item_id"] in lab]; y0 = np.array([lab[r["item_id"]] for r in dev]); g0 = np.array([r["template_id"] for r in dev])
    pre = choose(lambda L: (np.stack([F[r["item_id"]][L][0] for r in dev]), y0, g0, None, None), "c1_pre", "C1", "last_prompt")
    def tokb(L):
        lens = [len(F[r["item_id"]][L][1]) for r in dev]
        return (np.concatenate([F[r["item_id"]][L][1] for r in dev]), np.repeat(y0, lens), np.repeat(g0, lens),
                np.concatenate([[1.0 / n] * n for n in lens]), np.repeat([r["item_id"] for r in dev], lens))
    tk = choose(tokb, "c1_token", "C1", "token")
    with open(f"{OUT}/c1_scores.jsonl", "w") as f:
        for r in items:
            pt = tk.prob(torch.tensor(F[r["item_id"]][tk.layer][1])).tolist()
            f.write(json.dumps(dict(item_id=r["item_id"], template_id=r["template_id"], split=r["split"], level=r["level"], sample=r["sample"],
                                    p_pre=float(pre.prob(torch.tensor(F[r["item_id"]][pre.layer][0]))), p_tokens=pt,
                                    item_score=float(np.mean(pt)), early16=float(np.mean(pt[:16]))), ensure_ascii=False) + "\n")
else:
    tr = load(A); m4 = load(B)
    F = {r["item_id"]: c2_feat(r) for r in tr + m4}; print("features", len(F), f"{time.time()-t0:.0f}s", flush=True)
    def bld(rs): return lambda L: (np.stack([F[r["item_id"]][L] for r in rs]), np.array([r["label"] for r in rs]), np.array([r["template_id"] for r in rs]), None, None)
    s_tr = choose(bld(tr), "c2_transfer", "C2", "mean_span")
    s_in = choose(bld([r for r in m4 if r["template_id"] in M4DEV]), "c2_indomain", "C2", "mean_span")
    with open(f"{OUT}/c2_scores.jsonl", "w") as f:
        for r in m4:
            f.write(json.dumps(dict(item_id=r["item_id"], template_id=r["template_id"], cell=r["cell"], label=r["label"],
                                    p_transfer=float(s_tr.prob(torch.tensor(F[r["item_id"]][s_tr.layer]))),
                                    p_indomain=float(s_in.prob(torch.tensor(F[r["item_id"]][s_in.layer])))), ensure_ascii=False) + "\n")
print("DONE", MODE, round(time.time() - t0, 1))
