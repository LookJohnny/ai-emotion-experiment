"""Train C2-user, C1-pre and C1-token probes on DEV; pick each probe's layer by template-grouped 5-fold CV AUROC.
Also dumps per-item probe outputs for VAL (and later TEST, when --dump_split test is given with a frozen spec dir).
python train_probe.py MODEL labels.jsonl OUTDIR                     (train on dev, dump val)
python train_probe.py MODEL labels_test.jsonl OUTDIR --dump_only test (after freeze: dump test outputs with saved specs)"""
import json, sys, os, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core.probe import ProbeSpec

MODEL, LAB, OUT = sys.argv[1:4]; os.makedirs(OUT, exist_ok=True)
DUMP_ONLY = sys.argv[sys.argv.index("--dump_only") + 1] if "--dump_only" in sys.argv else None
EXCL = sys.argv[sys.argv.index("--exclude_persona") + 1] if "--exclude_persona" in sys.argv else None   # Addendum A.2 (leave-redteam-out)
LAYERS = [12, 16, 19, 22, 26]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
R = [json.loads(l) for l in open(LAB)]


def prompt_of(r):
    return ([{"role": "system", "content": r["system"]}] if r.get("system") else []) + r["messages"]


@torch.no_grad()
def feats(r):
    """C2: mean state over last-user-message tokens. C1: state at last prompt position + per-token states of the reply."""
    msgs = prompt_of(r); text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if r["concept"] == "C2":
        e = tok(text, return_offsets_mapping=True, add_special_tokens=False); u = r["messages"][-1]["content"]
        c0 = text.rindex(u); c1 = c0 + len(u); idx = [i for i, (s, t) in enumerate(e["offset_mapping"]) if s >= c0 and t <= c1 and t > s]
        hs = model(torch.tensor([e["input_ids"]], device="cuda"), output_hidden_states=True, use_cache=False).hidden_states
        return {L: hs[L + 1][0, idx[0]:idx[-1] + 1].float().mean(0).cpu().numpy() for L in LAYERS}
    pids = tok(text, add_special_tokens=False)["input_ids"]; rids = tok(r["target_text"], add_special_tokens=False)["input_ids"]
    hs = model(torch.tensor([pids + rids], device="cuda"), output_hidden_states=True, use_cache=False).hidden_states
    n = len(pids)
    return {L: (hs[L + 1][0, n - 1].float().cpu().numpy(), hs[L + 1][0, n:n + len(rids)].float().cpu().numpy()) for L in LAYERS}


t0 = time.time(); F = {r["item_id"]: feats(r) for r in R}; print(f"features {len(F)} {time.time()-t0:.0f}s", flush=True)


def cv_auc(X, y, g, w=None, pool_items=None):
    ug = np.array(sorted(set(g))); rng = np.random.default_rng(0); rng.shuffle(ug); folds = np.array_split(ug, 5); aucs = []
    for f in folds:
        te = np.isin(g, f)
        if len(set(y[te])) < 2 or len(set(y[~te])) < 2: continue
        sc = StandardScaler().fit(X[~te]); clf = LogisticRegression(max_iter=500, C=0.05).fit(sc.transform(X[~te]), y[~te], sample_weight=None if w is None else w[~te])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        if pool_items is not None:     # token-level probe: score each item by its mean token probability
            it = pool_items[te]; u = sorted(set(it)); p = np.array([p[it == i].mean() for i in u]); yy = np.array([y[te][it == i][0] for i in u])
        else: yy = y[te]
        aucs.append(roc_auc_score(yy, p))
    return float(np.mean(aucs)) if aucs else float("nan")


specs = {}
if not DUMP_ONLY:
    dev = [r for r in R if r["split"] == "dev" and not (EXCL and r.get("persona") == EXCL)]; res = {}
    for name, concept in [("c2_user", "C2"), ("c1_pre", "C1"), ("c1_token", "C1")]:
        rs = [r for r in dev if r["concept"] == concept]; best = None; res[name] = {}
        for L in LAYERS:
            if name == "c2_user":   X = np.stack([F[r["item_id"]][L] for r in rs]); y = np.array([r["label"] for r in rs]); g = np.array([r["template_id"] for r in rs]); w = None; pi = None
            elif name == "c1_pre":  X = np.stack([F[r["item_id"]][L][0] for r in rs]); y = np.array([r["label"] for r in rs]); g = np.array([r["template_id"] for r in rs]); w = None; pi = None
            else:
                X = np.concatenate([F[r["item_id"]][L][1] for r in rs]); lens = [len(F[r["item_id"]][L][1]) for r in rs]
                y = np.concatenate([[r["label"]] * n for r, n in zip(rs, lens)]); g = np.concatenate([[r["template_id"]] * n for r, n in zip(rs, lens)])
                w = np.concatenate([[1.0 / n] * n for n in lens]); pi = np.concatenate([[r["item_id"]] * n for r, n in zip(rs, lens)])
            auc = cv_auc(X, y, g, w, pi); res[name][L] = round(auc, 4)
            if best is None or auc > best[0] + 1e-9: best = (auc, L, X, y, w)
        auc, L, X, y, w = best; sc = StandardScaler().fit(X); clf = LogisticRegression(max_iter=1000, C=0.05).fit(sc.transform(X), y, sample_weight=w)
        specs[name] = ProbeSpec.from_sklearn(concept, name, L, {"c2_user": "mean_span", "c1_pre": "last_prompt", "c1_token": "token"}[name], sc, clf,
                                             meta={"dev_cv_auroc": round(auc, 4), "cv_by_layer": res[name]})
        specs[name].save(f"{OUT}/{name}.json"); print(name, "layer", L, "dev CV AUROC", round(auc, 4), res[name], flush=True)
    dump_split = "val"
else:
    specs = {n: ProbeSpec.load(f"{OUT}/{n}.json") for n in ["c2_user", "c1_pre", "c1_token"]}; dump_split = DUMP_ONLY

# per-item outputs for calibration / evaluation
with open(f"{OUT}/outputs_{dump_split}.jsonl", "w") as f:
    for r in R:
        if r["split"] != dump_split: continue
        o = dict(item_id=r["item_id"], template_id=r["template_id"], concept=r["concept"], kind=r["kind"], persona=r.get("persona"), label=r["label"], score=r["score"])
        if r["concept"] == "C2":
            s = specs["c2_user"]; o["p_c2"] = float(s.prob(torch.tensor(F[r["item_id"]][s.layer])))
        else:
            s = specs["c1_pre"]; o["p_pre"] = float(s.prob(torch.tensor(F[r["item_id"]][s.layer][0])))
            s = specs["c1_token"]; o["p_tokens"] = s.prob(torch.tensor(F[r["item_id"]][s.layer][1])).tolist()
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print("DONE dump", dump_split, round(time.time() - t0, 1))
