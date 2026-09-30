"""C3-H probes (PREREG_C3H.md): train pre (last prompt position) and token (mean over answer tokens) probes on dev labels
(INCORRECT=1 vs CORRECT=0), layer by 5-fold CV grouped by secondary category; score EVERY item (incl. sealed test) in one pass.
python c3h_probe.py MODEL answers.jsonl labels_devval.jsonl OUTDIR"""
import json, sys, os, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core.probe import ProbeSpec
MODEL, ANS, LAB, OUT = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True); LAYERS = [12, 16, 19, 22, 26]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
items = [json.loads(l) for l in open(ANS, encoding="utf-8")]
lab = {json.loads(l)["item_id"]: json.loads(l)["grade"] for l in open(LAB, encoding="utf-8")}


@torch.no_grad()
def feat(r):
    text = tok.apply_chat_template([{"role": "system", "content": r["system"]}] + r["messages"], tokenize=False, add_generation_prompt=True)
    pids = tok(text, add_special_tokens=False)["input_ids"]; rids = tok(r["target_text"], add_special_tokens=False)["input_ids"] or [tok.eos_token_id]
    hs = model(torch.tensor([pids + rids], device="cuda"), output_hidden_states=True, use_cache=False).hidden_states; n = len(pids)
    return {L: (hs[L + 1][0, n - 1].float().cpu().numpy(), hs[L + 1][0, n:n + len(rids)].float().mean(0).cpu().numpy()) for L in LAYERS}


t0 = time.time(); F = {r["item_id"]: feat(r) for r in items}; print("features", len(F), f"{time.time()-t0:.0f}s", flush=True)
dev = [r for r in items if r["split"] == "dev" and lab.get(r["item_id"]) in ("CORRECT", "INCORRECT")]
y = np.array([lab[r["item_id"]] == "INCORRECT" for r in dev], int); g = np.array([r["template_id"] for r in dev])


def cv(X):
    ug = np.array(sorted(set(g))); rng = np.random.default_rng(0); rng.shuffle(ug); a = []
    for f in np.array_split(ug, 5):
        te = np.isin(g, f)
        if len(set(y[te])) < 2: continue
        sc = StandardScaler().fit(X[~te]); c = LogisticRegression(max_iter=1000, C=0.05).fit(sc.transform(X[~te]), y[~te])
        a.append(roc_auc_score(y[te], c.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.mean(a))


S = {}
for name, k in [("c3h_pre", 0), ("c3h_token", 1)]:
    res = {L: cv(np.stack([F[r["item_id"]][L][k] for r in dev])) for L in LAYERS}; L = max(res, key=res.get)
    X = np.stack([F[r["item_id"]][L][k] for r in dev]); sc = StandardScaler().fit(X); c = LogisticRegression(max_iter=1000, C=0.05).fit(sc.transform(X), y)
    S[name] = (ProbeSpec.from_sklearn("C3H", name, L, "last_prompt" if k == 0 else "mean_answer", sc, c, meta={"cv_by_layer": {str(a): round(b, 4) for a, b in res.items()}}), k)
    S[name][0].save(f"{OUT}/{name}.json"); print(name, "layer", L, {a: round(b, 3) for a, b in res.items()}, flush=True)
with open(f"{OUT}/c3h_scores.jsonl", "w", encoding="utf-8") as f:
    for r in items:
        o = dict(item_id=r["item_id"], template_id=r["template_id"], primary=r["primary"], split=r["split"])
        for name, (sp, k) in S.items(): o[name] = float(sp.prob(torch.tensor(F[r["item_id"]][sp.layer][k])))
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print("DONE", round(time.time() - t0, 1))
