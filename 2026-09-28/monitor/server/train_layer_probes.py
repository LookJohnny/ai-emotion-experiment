"""Visualization-only (NOT pre-registered): one C1 per-token probe per layer, for the dashboard's layer x token heatmap and
per-head attribution. Trained on V1.1 C1 dev labels (<=24 sampled reply tokens per item), checked on val (item = mean token prob).
python train_layer_probes.py MODEL items_all.jsonl labels_devval.jsonl OUT.npz"""
import json, sys
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
MODEL, ITEMS, LAB, OUT = sys.argv[1:5]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
NL = model.config.num_hidden_layers
lab = {json.loads(l)["item_id"]: json.loads(l) for l in open(LAB, encoding="utf-8")}
items = [json.loads(l) for l in open(ITEMS, encoding="utf-8")]; items = [r for r in items if r["item_id"] in lab]
rng = np.random.default_rng(0); X = {s: [[] for _ in range(NL)] for s in ("dev", "val")}; Y = {"dev": [], "val": []}; G = {"dev": [], "val": []}
with torch.no_grad():
    for k, r in enumerate(items):
        msgs = [{"role": "system", "content": r["system"]}] + r["messages"]
        pids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
        rids = tok(r["target_text"], add_special_tokens=False)["input_ids"]
        hs = model(torch.tensor([pids + rids], device="cuda"), output_hidden_states=True, use_cache=False).hidden_states
        n = len(pids); m = len(rids); sp = r["split"]; pick = np.sort(rng.choice(m, min(24, m), replace=False))
        for L in range(NL): X[sp][L].append(hs[L + 1][0, n + pick].float().cpu().numpy())
        Y[sp] += [lab[r["item_id"]]["label"]] * len(pick); G[sp] += [k] * len(pick)
        if k % 100 == 0: print("features", k, len(items), flush=True)
yd, yv, gv = np.array(Y["dev"]), np.array(Y["val"]), np.array(G["val"])
res = {"mean": [], "scale": [], "coef": [], "intercept": [], "val_auroc": []}
for L in range(NL):
    Xd = np.concatenate(X["dev"][L]); sc = StandardScaler().fit(Xd); clf = LogisticRegression(max_iter=1000, C=0.05).fit(sc.transform(Xd), yd)
    p = clf.predict_proba(sc.transform(np.concatenate(X["val"][L])))[:, 1]; u = np.unique(gv)
    auc = roc_auc_score([yv[gv == i][0] for i in u], [p[gv == i].mean() for i in u])
    for k2, v in [("mean", sc.mean_), ("scale", sc.scale_), ("coef", clf.coef_[0]), ("intercept", clf.intercept_[0]), ("val_auroc", auc)]: res[k2].append(v)
    print("layer", L, "val AUROC", round(auc, 3), flush=True)
np.savez(OUT, **{k: np.array(v, dtype=np.float32) for k, v in res.items()})
print("DONE", OUT)
